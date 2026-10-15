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
from engine.factor.base import FactorDefinition, get_factor
from engine.factor.combine import (
    DEFAULT_CORRELATION_THRESHOLD,
    composite_score,
    deduplicate_factors,
    rolling_ic_weights,
    select_top,
)
from engine.factor.engine import compute_factors
from engine.factor.evaluate import (
    EvaluationConfig,
    FactorEvaluation,
    evaluate_factor,
    factor_correlation,
    forward_returns,
)
from engine.factor.preprocess import PreprocessConfig, preprocess_cross_section

_logger = logging.getLogger(__name__)

DEFAULT_PREPROCESS_STEPS: tuple[str, ...] = ("winsorize", "rank", "zscore")
#: 合成因子在报告中的列名（与单因子区分开，避免同名混淆）
COMPOSITE_FACTOR = "composite"


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


@dataclass(frozen=True, slots=True)
class FactorCombineRequest(FactorEvaluationRequest):
    """多因子合成请求：在评价请求之上增加"怎么合成"的配置。"""

    #: 合成方法：equal / ic / icir
    method: str = "icir"
    #: 估计权重用的历史 IC 观测数（滚动窗口，严格使用 t 之前的信息）
    weight_window: int = 20
    #: 选股名单数量
    top_k: int = 10
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    #: |IC| 低于该值的因子权重置零（过滤噪声因子，避免因子池越大噪声越多）
    min_abs_ic: float = 0.0

    def __post_init__(self) -> None:
        # 刻意不用零参 super()：slots=True 的 dataclass 会重建类对象，
        # 导致父类方法里的 __class__ 闭包失效并抛
        # "super(type, obj): obj is not an instance or subtype of type"（CPython 已知问题）。
        FactorEvaluationRequest.__post_init__(self)
        if self.weight_window <= 0:
            raise ValidationError("权重估计窗口必须为正", detail={"weight_window": self.weight_window})
        if self.top_k <= 0:
            raise ValidationError("top_k 必须为正", detail={"top_k": self.top_k})


@dataclass(frozen=True, slots=True)
class _Prepared:
    """评价与合成共用的中间产物。"""

    prices: pd.DataFrame
    evaluated: pd.DataFrame
    definitions: Mapping[str, FactorDefinition]
    warmup_days: int
    loaded_rows: int


@dataclass
class FactorCombineReport:
    output_dir: Path
    start: date
    end: date
    method: str
    kept_factors: tuple[str, ...]
    dropped_factors: tuple[str, ...]
    composite: FactorEvaluation | None
    singles: list[FactorEvaluation]
    weights: pd.DataFrame
    selections: pd.DataFrame
    correlation: pd.DataFrame
    top_k: int = 10
    warmup_days: int = 0
    loaded_rows: int = 0

    def summary(self) -> str:
        # 输入数取 singles 的长度：dropped 是它的子集，两者相加会重复计数
        lines = [
            f"多因子合成（{self.method}）{self.start}~{self.end}："
            f"输入 {len(self.singles)} 个因子，"
            f"保留 {len(self.kept_factors)} 个（{', '.join(self.kept_factors)}）"
        ]
        if self.dropped_factors:
            lines.append(
                f"  去重剔除：{', '.join(self.dropped_factors)}"
                f"（与保留因子相关性超过阈值，属于重复下注）"
            )
        if self.composite is not None:
            lines.append("  合成因子：" + self.composite.summary())
            best_single = max(
                (item for item in self.singles if item.sufficient),
                key=lambda item: abs(item.ic_mean),
                default=None,
            )
            if best_single is not None:
                gain = abs(self.composite.ic_mean) - abs(best_single.ic_mean)
                lines.append(
                    f"  对比最强单因子 {best_single.factor}（IC {best_single.ic_mean:+.4f}）："
                    f"IC 变化 {gain:+.4f}"
                )
        if not self.selections.empty:
            last_day = self.selections["date"].max()
            latest = self.selections.loc[self.selections["date"] == last_day]
            lines.append(
                f"  最新名单（{pd.Timestamp(last_day).date()}，{len(latest)} 只）："
                + ", ".join(str(code) for code in latest["code"])
            )
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "method": self.method,
            "kept_factors": list(self.kept_factors),
            "dropped_factors": list(self.dropped_factors),
            "warmup_days": self.warmup_days,
            "loaded_rows": self.loaded_rows,
            "composite": self.composite.to_jsonable() if self.composite else None,
            "singles": [item.to_jsonable() for item in self.singles],
            "weights_last": {
                str(pd.Timestamp(str(day)).date()): {str(name): float(value) for name, value in row.items()}
                for day, row in self.weights.tail(1).to_dict(orient="index").items()
            },
            "selections_last": [
                {
                    "code": str(row["code"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "weight": float(row["weight"]),
                }
                for _, row in self.selections.tail(self.top_k).iterrows()
            ]
            if not self.selections.empty
            else [],
        }


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
        prepared = self._prepare(request)
        config = self._evaluation_config(request)
        evaluations = [
            evaluate_factor(prepared.prices, prepared.evaluated, factor=name, config=config)
            for name in request.factors
        ]
        correlation = (
            factor_correlation(prepared.evaluated, list(request.factors), min_cross_section=request.min_cross_section)
            if len(request.factors) > 1
            else pd.DataFrame()
        )

        output_dir = self._write_report(
            evaluations,
            correlation=correlation,
            definitions=prepared.definitions,
            request=request,
            warmup_days=prepared.warmup_days,
            loaded_rows=prepared.loaded_rows,
        )
        _logger.info("因子评价完成：%s 个因子，报告目录 %s", len(evaluations), output_dir)
        return FactorEvaluationReport(
            output_dir=output_dir,
            start=request.start,
            end=request.end,
            evaluations=evaluations,
            correlation=correlation,
            warmup_days=prepared.warmup_days,
            loaded_rows=prepared.loaded_rows,
        )

    def run_combine(self, request: FactorCombineRequest) -> FactorCombineReport:
        """多因子合成：去重 → 滚动权重 → 合成分数 → 选股名单 → 报告。

        顺序不可调换：**先去重再估权重**。若先按 IC 给相关性 0.95 的两个因子各 0.5 权重，
        等于把同一个信号按 1.0 押注；去重后留下的因子才配拿满权重。
        """
        prepared = self._prepare(request)
        config = self._evaluation_config(request)
        singles = [
            evaluate_factor(prepared.prices, prepared.evaluated, factor=name, config=config)
            for name in request.factors
        ]

        # 去重与权重估计都只允许使用"评价窗口最早一段"的信息（对后续日期是纯历史）
        estimation = self._estimation_window(prepared, request.weight_window)
        correlation = (
            factor_correlation(estimation, list(request.factors), min_cross_section=request.min_cross_section)
            if len(request.factors) > 1
            else pd.DataFrame()
        )
        strength = {
            item.factor: abs(item.ic_mean) for item in singles if item.sufficient
        } or dict.fromkeys(request.factors, 0.0)
        kept = (
            list(deduplicate_factors(correlation, strength, threshold=request.correlation_threshold))
            if len(request.factors) > 1
            else list(request.factors)
        )
        if not kept:
            raise ValidationError(
                "去重后没有可用因子（所有因子之间相关性都超过阈值）",
                detail={"threshold": request.correlation_threshold, "factors": list(request.factors)},
            )

        # IC 估计需要"因子值 + 未来收益"两张表对齐：因子表本身不含 forward_return
        merged = prepared.evaluated.merge(
            forward_returns(prepared.prices, horizon=request.horizon),
            on=["date", "code"],
            how="inner",
        )
        weights = rolling_ic_weights(
            merged,
            kept,
            window=request.weight_window,
            method=request.method,
            min_cross_section=request.min_cross_section,
            min_abs_ic=request.min_abs_ic,
        )
        scores = composite_score(prepared.evaluated, kept, weights)
        composite = scores.rename(columns={"score": COMPOSITE_FACTOR}) if not scores.empty else scores
        composite_evaluation = (
            evaluate_factor(prepared.prices, composite, factor=COMPOSITE_FACTOR, config=config)
            if not composite.empty
            else None
        )
        selections = select_top(scores, top_k=request.top_k)

        output_dir = self._write_combine_report(
            composite_evaluation=composite_evaluation,
            singles=singles,
            correlation=correlation,
            weights=weights,
            selections=selections,
            kept=tuple(kept),
            dropped=tuple(name for name in request.factors if name not in kept),
            request=request,
            warmup_days=prepared.warmup_days,
            loaded_rows=prepared.loaded_rows,
        )
        _logger.info("多因子合成完成：保留 %s，报告目录 %s", kept, output_dir)
        return FactorCombineReport(
            output_dir=output_dir,
            start=request.start,
            end=request.end,
            method=request.method,
            kept_factors=tuple(kept),
            dropped_factors=tuple(name for name in request.factors if name not in kept),
            composite=composite_evaluation,
            singles=singles,
            weights=weights,
            selections=selections,
            correlation=correlation,
            top_k=request.top_k,
            warmup_days=prepared.warmup_days,
            loaded_rows=prepared.loaded_rows,
        )

    # ============================================================ 共享管线
    def _prepare(self, request: FactorEvaluationRequest) -> _Prepared:
        """数据门禁 → 带预热期读取 → 计算因子 → 截面预处理（评价与合成共用）。"""
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
        return _Prepared(
            prices=prices,
            evaluated=evaluated,
            definitions={definition.name: definition for definition in definitions},
            warmup_days=warmup,
            loaded_rows=len(raw),
        )

    @staticmethod
    def _evaluation_config(request: FactorEvaluationRequest) -> EvaluationConfig:
        return EvaluationConfig(
            horizon=request.horizon,
            quantiles=request.quantiles,
            min_cross_section=request.min_cross_section,
        )

    @staticmethod
    def _estimation_window(prepared: _Prepared, window: int) -> pd.DataFrame:
        """用于「去重决策」的估计窗口：评价区间最早的 ``window`` 个截面。

        只用最早一段，是因为去重结论一旦确定就固定用于整段评价；
        若用全区间相关性去重，等于用未来信息挑选了因子组合。
        """
        days = sorted(pd.Index(prepared.evaluated["date"]).unique())
        if len(days) <= window:
            # 区间太短：退化为使用全部可用截面（并在日志中明示）
            _logger.warning("评价区间不足 %s 个截面，去重决策改用全部 %s 个截面", window, len(days))
            return prepared.evaluated
        return prepared.evaluated.loc[prepared.evaluated["date"].isin(days[:window])].reset_index(drop=True)

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

    def _write_combine_report(
        self,
        *,
        composite_evaluation: FactorEvaluation | None,
        singles: Sequence[FactorEvaluation],
        correlation: pd.DataFrame,
        weights: pd.DataFrame,
        selections: pd.DataFrame,
        kept: tuple[str, ...],
        dropped: tuple[str, ...],
        request: FactorCombineRequest,
        warmup_days: int,
        loaded_rows: int,
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.settings.data.root_dir) / "factors" / f"combine_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "warmup_days": warmup_days,
            "loaded_rows": loaded_rows,
            "config": {
                "method": request.method,
                "weight_window": request.weight_window,
                "top_k": request.top_k,
                "correlation_threshold": request.correlation_threshold,
                "min_abs_ic": request.min_abs_ic,
                "horizon": request.horizon,
                "quantiles": request.quantiles,
                "preprocess_steps": list(request.preprocess_steps),
                "codes": list(request.codes) or "universe",
                "dataset": request.dataset,
            },
            "kept_factors": list(kept),
            "dropped_factors": list(dropped),
            "composite": composite_evaluation.to_jsonable() if composite_evaluation else None,
            "singles": [item.to_jsonable() for item in singles],
            "correlation": {
                str(row): {str(column): float(value) for column, value in values.items()}
                for row, values in correlation.to_dict(orient="index").items()
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not weights.empty:
            frame = weights.copy()
            frame.index.name = "date"
            frame.to_csv(output_dir / "weights.csv", encoding="utf-8")
        if not selections.empty:
            selections.to_csv(output_dir / "selections.csv", index=False, encoding="utf-8")
        if not correlation.empty:
            correlation.to_csv(output_dir / "correlation.csv", encoding="utf-8")
        if composite_evaluation is not None and not composite_evaluation.ic_series.empty:
            frame = composite_evaluation.ic_series.rename("ic").to_frame()
            frame.index.name = "date"
            frame.to_csv(output_dir / f"ic_{COMPOSITE_FACTOR}.csv", encoding="utf-8")
        for item in singles:
            if not item.ic_series.empty:
                frame = item.ic_series.rename("ic").to_frame()
                frame.index.name = "date"
                frame.to_csv(output_dir / f"ic_{item.factor}.csv", encoding="utf-8")
        return output_dir


__all__ = [
    "COMPOSITE_FACTOR",
    "DEFAULT_PREPROCESS_STEPS",
    "FactorCombineReport",
    "FactorCombineRequest",
    "FactorEvaluationReport",
    "FactorEvaluationRequest",
    "FactorService",
]
