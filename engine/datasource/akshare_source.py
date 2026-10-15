"""AkShare 数据源适配器（T-06）。

设计要点：

1. **akshare 调用收敛到 ``AkShareClient``，且延迟导入**：没有安装 akshare 的环境里
   engine 其余部分照常可用、离线测试照常跑（akshare 依赖较重，不该成为内核的硬依赖）。
2. **字段映射与单位换算只在 ``_normalize_daily`` 一处做**。akshare 有两个高频坑：
   - 列名是中文（日期/开盘/收盘…）；
   - ``stock_zh_a_hist`` 的成交量单位是「**手**」= 100 股，不换算会让所有成交量类因子和
     「单笔不超过当日成交量 X%」的撮合约束全部差 100 倍。
3. 每次外部请求前过 ``RateLimiter``，失败走 ``retry_call`` 指数退避。
4. 日志用 stdlib ``logging``（engine 不得依赖 app；app 侧已把 stdlib 日志接入 structlog）。

已知缺口（不假装支持）：akshare 免费接口**不提供**涨跌停价、真实上市日、历史指数成分。
分别处理为：涨跌停价留给撮合/质量层按前收 × 板块阈值计算；``list_date`` 明确留空并告警；
历史成分必须由调用方用 ``fetch_index_members()`` 做带生效日的快照，否则 ``universe()`` 直接抛错。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Protocol

import pandas as pd

from engine.datahub.calendar import TradingCalendarImpl
from engine.datahub.rules import infer_board
from engine.datahub.schema import (
    DAILY_COLUMNS,
    DAILY_REQUIRED_COLUMNS,
    ensure_columns,
)
from engine.datahub.universe import UniverseHistory
from engine.datasource.base import RateLimiter, retry_call
from engine.errors import DataNotReadyError, DataQualityError, DataSourceError
from engine.ports import AdjType

_logger = logging.getLogger(__name__)

#: 我们的复权口径 -> akshare 的 adjust 参数
_AK_ADJUST: Final[dict[str, str]] = {"none": "", "pre": "qfq", "post": "hfq"}

#: akshare 中文列名 -> 本项目列名。未列入的列会被丢弃（如振幅/涨跌幅/换手率，可由价格与成交量推导）
#: 注：``adj_factor`` 列不是来自上游，而是由 _with_true_factor() 计算得出
_DAILY_ALIASES: Final[dict[str, str]] = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "股票代码": "source_code",
}


class AkShareClientLike(Protocol):
    """akshare 调用的最小接口（便于注入离线替身）。"""

    def trade_calendar(self) -> pd.DataFrame: ...

    def daily(self, symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame: ...

    def stock_list(self) -> pd.DataFrame: ...

    def index_members(self, index_code: str) -> pd.DataFrame: ...


class AkShareClient:
    """akshare 薄封装：全项目唯一允许 import akshare 的地方。"""

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    @property
    def module(self) -> Any:
        if self._module is None:
            try:
                import akshare
            except ImportError as exc:  # pragma: no cover —— 依赖缺失时的友好报错
                raise DataSourceError(
                    "未安装 akshare，无法使用 AkShare 数据源",
                    detail={"hint": 'pip install -e ".[source]"'},
                ) from exc
            self._module = akshare
        return self._module

    def trade_calendar(self) -> pd.DataFrame:
        frame: pd.DataFrame = self.module.tool_trade_date_hist_sina()
        return frame

    def daily(self, symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame:
        frame: pd.DataFrame = self.module.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        return frame

    def stock_list(self) -> pd.DataFrame:
        frame: pd.DataFrame = self.module.stock_info_a_code_name()
        return frame

    def index_members(self, index_code: str) -> pd.DataFrame:
        frame: pd.DataFrame = self.module.index_stock_cons(symbol=index_code)
        return frame


@dataclass
class AkShareDataSource:
    """实现 ``engine.ports.DataSource`` 的 AkShare 适配器。"""

    name: str = "akshare"
    request_interval: float = 0.3
    max_retries: int = 3
    retry_backoff: float = 1.5
    retry_base_delay: float = 0.5
    #: akshare 成交量单位为「手」，换算成股
    volume_multiplier: float = 100.0
    #: 请求后复权时是否额外取一次不复权数据以计算真实复权因子。
    #: 代价是每只标的每次同步多一次请求；收益是下单价可以可靠还原——
    #: 因子缺失时若默认为 1，还原出的下单价会偏离盘口，那是真实亏损。故默认开启。
    with_adj_factor: bool = True
    client: AkShareClientLike = field(default_factory=AkShareClient)
    universe_history: UniverseHistory = field(default_factory=UniverseHistory)
    sleep: Callable[[float], None] = time.sleep
    _limiter: RateLimiter = field(init=False, repr=False)
    _calendar: TradingCalendarImpl | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(min_interval=self.request_interval, sleep=self.sleep)

    # ============================================================ 交易日历
    def trading_calendar(self) -> TradingCalendarImpl:
        if self._calendar is None:
            frame = self._call("交易日历", self.client.trade_calendar)
            ensure_columns(frame, ("trade_date",), dataset="trade_calendar")
            days = pd.to_datetime(frame["trade_date"]).dt.date.tolist()
            self._calendar = TradingCalendarImpl.from_dates(days, name=self.name)
        return self._calendar

    # ============================================================ 日线
    def daily(
        self,
        codes: Sequence[str],
        start: date,
        end: date,
        *,
        adj: AdjType = "post",
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for code in codes:

            def fetch(
                symbol: str = code,
                _start: date = start,
                _end: date = end,
                _adjust: str = _AK_ADJUST[adj],
            ) -> pd.DataFrame:
                return self.client.daily(symbol=symbol, start=_start, end=_end, adjust=_adjust)

            frame = self._normalize_daily(self._call(f"日线[{code}]", fetch), code=code)
            if adj == "post" and self.with_adj_factor and not frame.empty:
                frame = self._with_true_factor(frame, code=code, start=start, end=end)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=list(DAILY_COLUMNS))
        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values(["code", "date"]).reset_index(drop=True)

    def adjust_factors(self, codes: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """复权因子 = 后复权收盘 / 不复权收盘。

        akshare 没有直接的因子接口，故用同一标的两种口径的比值推导；
        而 ``daily(adj="post")`` 已经把该因子合并进结果，这里直接取列即可。
        """
        columns = ["code", "date", "adj_factor"]
        frame = self.daily(codes, start, end, adj="post")
        if frame.empty:
            return pd.DataFrame(columns=columns)
        return frame.loc[:, columns]

    # ============================================================ 基础数据
    def stock_info(self) -> pd.DataFrame:
        frame = self._call("股票列表", self.client.stock_list)
        ensure_columns(frame, ("code", "name"), dataset="stock_list")
        result = pd.DataFrame(
            {
                "code": frame["code"].astype(str).str.zfill(6),
                "name": frame["name"].astype(str),
            }
        )
        result["is_st"] = result["name"].str.upper().str.contains("ST")
        result["board"] = result["code"].map(infer_board)
        result["list_date"] = pd.NaT
        result["delist_date"] = pd.NaT
        _logger.warning(
            "akshare 免费接口不提供真实上市日期，stock_info.list_date 暂为空；"
            "「次新股过滤」在补充该字段前不会生效"
        )
        return result

    def index_members(self, index_code: str) -> tuple[str, ...]:
        """指数当前成分（DataSource 端口方法，供成分快照任务调用）。"""
        return self.fetch_index_members(index_code)

    def fetch_index_members(self, index_code: str) -> tuple[str, ...]:
        """抓取指数**当前**成分（调用方负责带上生效日登记进 ``universe_history``）。"""
        def fetch() -> pd.DataFrame:
            return self.client.index_members(index_code)

        frame = self._call(f"指数成分[{index_code}]", fetch)
        ensure_columns(frame, ("品种代码",), dataset=f"index_members[{index_code}]")
        return tuple(frame["品种代码"].astype(str).str.zfill(6).tolist())

    def universe(self, index_code: str, on: date) -> tuple[str, ...]:
        if self.universe_history.effective_date(index_code, on) is None:
            raise DataNotReadyError(
                "指数成分历史未装载，禁止用最新成分回填历史",
                detail={
                    "index_code": index_code,
                    "on": on.isoformat(),
                    "hint": "先用 fetch_index_members() 取当前成分并以当日为生效日登记（T-10）",
                },
            )
        return self.universe_history.resolve(index_code, on)

    # ============================================================ 内部
    def _call(self, description: str, operation: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        self._limiter.acquire()
        return retry_call(
            operation,
            description=f"akshare {description}",
            retries=self.max_retries,
            backoff=self.retry_backoff,
            base_delay=self.retry_base_delay,
            sleep=self.sleep,
        )

    def _normalize_daily(self, raw: pd.DataFrame | None, *, code: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=list(DAILY_COLUMNS))

        frame = raw.rename(columns=_DAILY_ALIASES)

        # 必须先补 code 再校验：akshare 原表里只有「股票代码」（映射为 source_code，会被丢弃），
        # 以「请求的代码」为准也更安全——不信任上游返回的代码列（避免错位）。
        frame["code"] = code
        ensure_columns(frame, DAILY_REQUIRED_COLUMNS, dataset=f"akshare daily[{code}]")

        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        # 手 -> 股
        frame["volume"] = (
            pd.to_numeric(frame["volume"], errors="coerce").fillna(0) * self.volume_multiplier
        ).astype("int64")
        for column in ("open", "high", "low", "close", "amount"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # 占位因子：后复权口径会由 _with_true_factor() 覆盖为真实因子；
        # 不复权口径下因子本就是 1（hfq = 真实价 × 1），故这里的 1.0 是正确值而非占位。
        frame["adj_factor"] = 1.0
        frame["is_suspended"] = frame["volume"] <= 0
        frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        # 前收只能由窗口内推导，窗口首日为 NaN（不猜）
        frame["prev_close"] = frame["close"].shift(1)
        return frame.loc[:, [column for column in DAILY_COLUMNS if column in frame.columns]]

    def _with_true_factor(self, frame: pd.DataFrame, *, code: str, start: date, end: date) -> pd.DataFrame:
        """把真实复权因子 ``F(t) = 后复权价 / 不复权价`` 合并进来。

        没有这一步就无法把后复权价还原成下单价；而任何"缺因子就按 1 算"的兜底
        都会让下单价偏离盘口。因此这里宁可因为多取一次数据失败，也不放弃因子。
        """

        def fetch() -> pd.DataFrame:
            return self.client.daily(symbol=code, start=start, end=end, adjust=_AK_ADJUST["none"])

        raw = self._normalize_daily(self._call(f"不复权日线[{code}]", fetch), code=code)
        if raw.empty:
            raise DataQualityError(
                "取得后复权数据但无法取得不复权数据，无法计算复权因子",
                detail={"code": code, "start": start.isoformat(), "end": end.isoformat()},
            )
        merged = frame.merge(
            raw[["date", "close"]].rename(columns={"close": "_raw_close"}), on="date", how="left"
        )
        divisor = merged["_raw_close"].where(merged["_raw_close"] > 0)
        factor = merged["close"] / divisor
        missing = int(factor.isna().sum())
        if missing:
            raise DataQualityError(
                "部分交易日缺少不复权价，无法计算复权因子",
                detail={"code": code, "missing_rows": missing, "rows": len(merged)},
            )
        merged["adj_factor"] = factor.astype(float)
        return merged.drop(columns=["_raw_close"])


__all__ = ["AkShareClient", "AkShareClientLike", "AkShareDataSource", "infer_board"]
