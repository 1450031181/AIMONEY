"""因子服务（P2-F3）：从本地行情到"因子评价报告"的完整用例。

编排顺序（每一步都有它必须在前的理由）::

    1. 数据就绪门禁        —— 没同步过数据直接拒绝，而不是算出空报告
    2. 带**预热期**读取行情 —— 没有预热，窗口前 N 天的因子值全是 NaN，
                              评价窗口开头会凭空少掉一段（且不报错）
    3. 计算因子            —— 逐标的独立，历史不足处为 NaN
    4. 截面预处理          —— winsorize → rank → zscore（默认）
    5. 逐因子评价          —— IC / ICIR / 分层 / 多空 / 换手 / 单调性
    6. 因子相关性矩阵      —— 多因子合成前必须知道因子是否在重复表达同一信号
    7. 报告落盘            —— summary.json + 每因子 IC 序列与分层净值 CSV
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine.datahub.calendar import TradingCalendarImpl
from engine.datahub.store import ParquetStore
from engine.datasource.local_source import LocalDataSource
from engine.errors import ValidationError
from engine.factor.base import get_factor
from engine.factor.engine import compute_factors
from engine.factor.evaluate import EvaluationConfig, FactorEvaluation, evaluate_factor, factor_correlation
from engine.factor.preprocess import PreprocessConfig, preprocess_cross_section

_logger = logging.getLogger(__name__)

DEFAULT_PREPROCESS_STEPS: tuple[str, ...] = ("winsorize", "rank", "zscore")


@dataclass(frozen=True, slots=True)
class FactorEvaluationRequest:
    factors: tuple[str, ...]
    start: date
    end: date
    codes: tuple[str, ...] = ()
    horizon: int = 1
    quantiles: int = 5
    min_cross_section: int = 5
    preprocess_steps: tuple[str, ...] = DEFAULT_PREPROCESS_STEPS
    dataset: str = "daily"

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValidationError("至少指定一个因子")
        if self.start > self.end:
            raise ValidationError(
                "评价区间非法：start 晚于 end",
                detail={"start": self.start.isoformat(), "end": self.end.isoformat()},
            )


@dataclass
class FactorEvaluationReport:
    output_dir: Path
    start: date
    end: date
    evaluations: list[FactorEvaluation]
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    warmup_days: int = 0
    loaded_rows: int = 0

    @property
    def rankable(self) -> list[FactorEvaluation]:
        """可比较的因子（有效截面足够），按 |IC| 降序。"""
        return sorted(
            [item for item in self.evaluations if item.sufficient],
            key=lambda item: abs(item.ic_mean),
            reverse=True,
        )

    def summary(self) -> str:
        lines = [
            f"因子评价 {self.start}~{self.end}：{len(self.evaluations)} 个因子，"
            f"{self.loaded_rows} 行数据（预热 {self.warmup_days} 个交易日）"
        ]
        if self.rankable:
            lines.append(f"按 |IC| 排序：{' > '.join(item.factor for item in self.rankable)}")
        for item in self.evaluations:
            lines.append("  " + item.summary())
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "warmup_days": self.warmup_days,
            "loaded_rows": self.loaded_rows,
            "factors": [item.to_jsonable() for item in self.evaluations],
            "correlation": {
                str(row): {str(column): float(value) for column, value in values.items()}
                for row, values in self.correlation.to_dict(orient="index").items()
            },
        }


class FactorService:
    """因子工厂的应用层编排（只依赖 Settings）。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def run(self, request: FactorEvaluationRequest) -> FactorEvaluationReport:
        store = ParquetStore(self.settings.parquet_dir)
        store.require_data(request.dataset)

        definitions = [get_factor(name) for name in request.factors]
        warmup = max(definition.window for definition in definitions)
        local = LocalDataSource(store, dataset=request.dataset)

        warmup_start = self._warmup_start(local, request.start, warmup)
        raw = local.daily(
            list(request.codes) if request.codes else self._all_codes(local, request.end),
            warmup_start,
            request.end,
            adj="post",
        )
        if raw.empty:
            raise ValidationError(
                "评价区间内没有行情数据",
                detail={
                    "dataset": request.dataset,
                    "start": warmup_start.isoformat(),
                    "end": request.end.isoformat(),
                    "hint": "检查区间与本地数据覆盖范围（aimoney-data coverage）",
                },
            )

        values = compute_factors(raw, list(request.factors))
        values["date"] = pd.to_datetime(values["date"])
        processed = preprocess_cross_section(
            values,
            list(request.factors),
            PreprocessConfig(steps=request.preprocess_steps),
        )

        # 评价窗口裁掉预热期：预热数据只用于让窗口开头的因子值可用，不参与统计
        evaluated = processed.loc[processed["date"] >= pd.Timestamp(request.start)].reset_index(drop=True)
        prices = raw.copy()
        prices["date"] = pd.to_datetime(prices["date"])

        config = EvaluationConfig(
            horizon=request.horizon,
            quantiles=request.quantiles,
            min_cross_section=request.min_cross_section,
        )
        evaluations = [
            evaluate_factor(prices, evaluated, factor=name, config=config) for name in request.factors
        ]
        correlation = (
            factor_correlation(evaluated, list(request.factors), min_cross_section=request.min_cross_section)
            if len(request.factors) > 1
            else pd.DataFrame()
        )

        output_dir = self._write_report(
            evaluations,
            correlation=correlation,
            definitions={definition.name: definition for definition in definitions},
            request=request,
            warmup_days=warmup,
            loaded_rows=len(raw),
        )
        _logger.info("因子评价完成：%s 个因子，报告目录 %s", len(evaluations), output_dir)
        return FactorEvaluationReport(
            output_dir=output_dir,
            start=request.start,
            end=request.end,
            evaluations=evaluations,
            correlation=correlation,
            warmup_days=warmup,
            loaded_rows=len(raw),
        )

    # ============================================================ 内部
    def _warmup_start(self, local: LocalDataSource, start: date, warmup: int) -> date:
        """把读取起点前推到预热期。

        本地数据不足以覆盖完整预热期时**从最早交易日开始**（而不是报错）：
        此时窗口开头的因子值会是 NaN，评价会自动跳过这些截面——
        这是诚实降级；若强行用短窗口硬算，会得到系统性失真的因子值。
        """
        if warmup <= 0:
            return start
        try:
            calendar: TradingCalendarImpl = local.trading_calendar()
            anchor = calendar.next_trade_day(start, include_self=True)
            return calendar.shift(anchor, -warmup)
        except ValidationError:
            calendar = local.trading_calendar()
            _logger.warning("本地数据不足以覆盖 %s 个交易日的预热期，从最早交易日开始", warmup)
            return calendar.first_day

    def _all_codes(self, local: LocalDataSource, end: date) -> list[str]:
        info = local.stock_info()
        codes = info["code"].astype(str)
        if "delist_date" in info.columns:
            alive = info["delist_date"].isna() | (pd.to_datetime(info["delist_date"]) > pd.Timestamp(end))
            codes = codes[alive]
        return sorted(set(codes))

    def _write_report(
        self,
        evaluations: Sequence[FactorEvaluation],
        *,
        correlation: pd.DataFrame,
        definitions: Mapping[str, Any],
        request: FactorEvaluationRequest,
        warmup_days: int,
        loaded_rows: int,
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.settings.data.root_dir) / "factors" / f"evaluate_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "warmup_days": warmup_days,
            "loaded_rows": loaded_rows,
            "config": {
                "horizon": request.horizon,
                "quantiles": request.quantiles,
                "min_cross_section": request.min_cross_section,
                "preprocess_steps": list(request.preprocess_steps),
                "codes": list(request.codes) or "universe",
                "dataset": request.dataset,
            },
            "factor_definitions": {
                name: {
                    "category": definition.category,
                    "description": definition.description,
                    "window": definition.window,
                }
                for name, definition in definitions.items()
            },
            "evaluations": [item.to_jsonable() for item in evaluations],
            "correlation": {
                str(row): {str(column): float(value) for column, value in values.items()}
                for row, values in correlation.to_dict(orient="index").items()
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for item in evaluations:
            if not item.ic_series.empty:
                frame = item.ic_series.rename("ic").to_frame()
                frame.index.name = "date"
                frame.to_csv(output_dir / f"ic_{item.factor}.csv", encoding="utf-8")
            if not item.quantile_cum_returns.empty:
                nav = item.quantile_cum_returns.copy()
                nav.index.name = "date"
                nav.columns = [f"q{int(column)}" for column in nav.columns]
                nav.to_csv(output_dir / f"quantile_{item.factor}.csv", encoding="utf-8")

        if not correlation.empty:
            correlation.to_csv(output_dir / "correlation.csv", encoding="utf-8")
        return output_dir


__all__ = ["DEFAULT_PREPROCESS_STEPS", "FactorEvaluationReport", "FactorEvaluationRequest", "FactorService"]
