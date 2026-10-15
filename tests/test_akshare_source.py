"""T-06 AkShare 适配器测试（全部离线）。

线上取数不可控也不可复现，所以把 akshare 收敛到可注入的 ``AkShareClientLike`` 后面，
这里注入假 client，用「akshare 真实返回的中文列名 + 手为单位的成交量」来验证映射逻辑。
真正的联网冒烟在 tests/test_akshare_live.py（默认不跑）。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.datahub.adjust import FACTOR_COLUMN, restore_real_frame
from engine.datasource.akshare_source import AkShareDataSource, infer_board
from engine.errors import DataNotReadyError, DataQualityError, DataSourceError


class FakeAkClient:
    """模拟 akshare：返回中文列名、成交量单位为「手」的原始数据，并像真实接口一样按区间过滤。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.fail_times = 0
        #: 按 akshare 的 adjust 参数分别准备数据（"" 不复权 / hfq 后复权 / qfq 前复权）
        self.frames: dict[str, pd.DataFrame] = {"": _raw_daily(), "hfq": _raw_daily(), "qfq": _raw_daily()}
        self.stock_list_frame = pd.DataFrame(
            [{"code": "600000", "name": "浦发银行"}, {"code": "1", "name": "*ST 平安"}]
        )
        self.index_frame = pd.DataFrame([{"品种代码": "1", "品种名称": "平安银行"}])

    def trade_calendar(self) -> pd.DataFrame:
        self.calls.append(("calendar", "", ""))
        return pd.DataFrame([{"trade_date": date(2026, 10, 12)}, {"trade_date": date(2026, 10, 13)}])

    def daily(self, symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame:
        self.calls.append(("daily", symbol, adjust))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("数据源抖动")
        frame = self.frames[adjust].copy()
        if frame.empty:
            return frame
        days = pd.to_datetime(frame["日期"]).dt.date
        return frame.loc[(days >= start) & (days <= end)].copy()

    def stock_list(self) -> pd.DataFrame:
        self.calls.append(("stock_list", "", ""))
        return self.stock_list_frame.copy()

    def index_members(self, index_code: str) -> pd.DataFrame:
        self.calls.append(("index_members", index_code, ""))
        return self.index_frame.copy()


def _raw_daily(*, factor: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "日期": "2026-10-13",
                "股票代码": "600000",
                "开盘": 10.2 * factor,
                "收盘": 10.5 * factor,
                "最高": 10.8 * factor,
                "最低": 10.0 * factor,
                "成交量": 1234,
                "成交额": 1_234_500.0,
                "振幅": 1.0,
                "涨跌幅": 0.5,
                "换手率": 0.3,
            },
            {
                "日期": "2026-10-12",
                "股票代码": "600000",
                "开盘": 10.0 * factor,
                "收盘": 10.1 * factor,
                "最高": 10.2 * factor,
                "最低": 9.8 * factor,
                "成交量": 1000,
                "成交额": 1_000_000.0,
                "振幅": 1.0,
                "涨跌幅": 0.1,
                "换手率": 0.2,
            },
            {
                "日期": "2026-10-14",
                "股票代码": "600000",
                "开盘": 10.5 * factor,
                "收盘": 10.5 * factor,
                "最高": 10.5 * factor,
                "最低": 10.5 * factor,
                "成交量": 0,  # 停牌
                "成交额": 0.0,
                "振幅": 0.0,
                "涨跌幅": 0.0,
                "换手率": 0.0,
            },
        ]
    )


@pytest.fixture
def client() -> FakeAkClient:
    return FakeAkClient()


@pytest.fixture
def source(client: FakeAkClient) -> AkShareDataSource:
    # 注入零等待 sleep，避免限速重试把测试拖慢
    return AkShareDataSource(client=client, request_interval=0.3, retry_base_delay=0.0, sleep=lambda _: None)


def test_中文列名映射与成交量手转股(source: AkShareDataSource) -> None:
    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 14))
    first = frame.iloc[0]

    assert list(frame.columns) >= ["code", "date", "open", "high", "low", "close", "volume", "amount"]
    assert first["open"] == pytest.approx(10.0)
    assert first["close"] == pytest.approx(10.1)
    assert first["volume"] == 100_000, "akshare 成交量单位是「手」，必须 ×100 换成股"


def test_以请求的代码为准而不是上游返回的列(source: AkShareDataSource, client: FakeAkClient) -> None:
    client.frames["hfq"]["股票代码"] = "000001"  # 上游错位
    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    assert set(frame["code"]) == {"600000"}


def test_结果按日期升序且无重复(source: AkShareDataSource) -> None:
    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 14))
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2026-10-12", "2026-10-13", "2026-10-14"]


def test_前收由窗口内推导_首日为空(source: AkShareDataSource) -> None:
    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 14))
    assert pd.isna(frame.iloc[0]["prev_close"]), "窗口首日的前收不可推导，应为空而不是猜测值"
    assert frame.iloc[1]["prev_close"] == pytest.approx(10.1)


def test_零成交量标记为停牌(source: AkShareDataSource) -> None:
    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 14))
    assert bool(frame.iloc[2]["is_suspended"])
    assert not bool(frame.iloc[0]["is_suspended"])


def test_缺少必需列时拒绝使用(source: AkShareDataSource, client: FakeAkClient) -> None:
    client.frames["hfq"] = client.frames["hfq"].drop(columns=["最高"])
    with pytest.raises(DataQualityError):
        source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))


def test_标的无数据时返回带列的空表(source: AkShareDataSource, client: FakeAkClient) -> None:
    client.frames["hfq"] = pd.DataFrame()
    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    assert frame.empty
    assert "code" in frame.columns


def test_复权口径映射到akshare参数(source: AkShareDataSource, client: FakeAkClient) -> None:
    source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13), adj="post")
    source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13), adj="pre")
    source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13), adj="none")

    adjusts = [call[2] for call in client.calls if call[0] == "daily"]
    # 后复权额外取一次不复权数据以计算真实复权因子
    assert adjusts == ["hfq", "", "qfq", ""]


def test_后复权结果合并真实复权因子并可还原真实价(source: AkShareDataSource, client: FakeAkClient) -> None:
    client.frames["hfq"] = _raw_daily(factor=2.0)  # 后复权价 = 不复权价 × 2

    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13), adj="post")
    assert list(frame[FACTOR_COLUMN]) == pytest.approx([2.0, 2.0])

    real = restore_real_frame(frame)
    assert list(real["close"]) == pytest.approx([10.1, 10.5]), "还原后应与不复权价一致"


def test_不复权数据取不到时整体失败而不是把因子当作1(source: AkShareDataSource, client: FakeAkClient) -> None:
    client.frames[""] = pd.DataFrame()
    with pytest.raises(DataQualityError):
        source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13), adj="post")


def test_可显式关闭因子合并以节省请求(client: FakeAkClient) -> None:
    source = AkShareDataSource(client=client, request_interval=0.0, sleep=lambda _: None, with_adj_factor=False)

    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13), adj="post")

    assert [call[2] for call in client.calls if call[0] == "daily"] == ["hfq"]
    assert list(frame[FACTOR_COLUMN]) == [1.0, 1.0], "关闭后因子为占位值，调用方需自行承担还原风险"


def test_复权因子由后复权与不复权收盘比值推导(source: AkShareDataSource, client: FakeAkClient) -> None:
    client.frames["hfq"] = _raw_daily(factor=2.0)
    factors = source.adjust_factors(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    assert len(factors) == 2
    assert list(factors["adj_factor"]) == pytest.approx([2.0, 2.0])


def test_交易日历构建(source: AkShareDataSource) -> None:
    calendar = source.trading_calendar()
    assert calendar.is_trade_day(date(2026, 10, 12))
    assert calendar.shift(date(2026, 10, 12), 1) == date(2026, 10, 13)
    # 第二次调用应命中缓存，不再请求
    source.trading_calendar()
    assert len([call for call in source.client.calls if call[0] == "calendar"]) == 1  # type: ignore[attr-defined]


def test_股票基础信息与ST识别(source: AkShareDataSource) -> None:
    info = source.stock_info()
    assert list(info["code"]) == ["600000", "000001"], "代码必须补齐 6 位"
    assert list(info["is_st"]) == [False, True]
    assert list(info["board"]) == ["沪市主板", "深市主板"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("688981", "科创板"),
        ("600000", "沪市主板"),
        ("300750", "创业板"),
        ("000001", "深市主板"),
        ("830799", "北交所"),
        ("430047", "北交所"),
        ("999999", "未知"),
    ],
)
def test_板块推断(code: str, expected: str) -> None:
    assert infer_board(code) == expected


def test_成分历史未装载时拒绝回填(source: AkShareDataSource) -> None:
    with pytest.raises(DataNotReadyError):
        source.universe("000300", date(2026, 10, 13))


def test_成分按生效日解析(source: AkShareDataSource) -> None:
    members = source.fetch_index_members("000300")
    assert members == ("000001",), "成分代码必须补齐 6 位"

    source.universe_history.add("000300", date(2026, 7, 1), members)
    assert source.universe("000300", date(2026, 7, 1)) == ("000001",)
    with pytest.raises(DataNotReadyError):
        source.universe("000300", date(2026, 6, 30))  # 早于首次快照


def test_限速器在连续请求之间生效() -> None:
    waits: list[float] = []
    source = AkShareDataSource(
        client=FakeAkClient(),
        request_interval=0.3,
        sleep=waits.append,
        retry_base_delay=0.0,
    )
    source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    source.daily(["000001"], date(2026, 10, 12), date(2026, 10, 13))

    assert waits, "第二次请求前必须限速等待"
    assert waits[0] <= 0.3


def test_失败自动重试后成功() -> None:
    client = FakeAkClient()
    client.fail_times = 2
    source = AkShareDataSource(client=client, request_interval=0.0, retry_base_delay=0.0, sleep=lambda _: None)

    frame = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    assert not frame.empty
    # 后复权请求失败 2 次后成功（共 3 次），随后仅 1 次不复权请求用于算因子
    assert len([call for call in client.calls if call[2] == "hfq"]) == 3
    assert len([call for call in client.calls if call[2] == ""]) == 1


def test_重试耗尽后抛数据源异常() -> None:
    client = FakeAkClient()
    client.fail_times = 99
    source = AkShareDataSource(
        client=client, request_interval=0.0, max_retries=2, retry_base_delay=0.0, sleep=lambda _: None
    )

    with pytest.raises(DataSourceError):
        source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))
