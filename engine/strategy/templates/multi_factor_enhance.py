"""多因子增强策略模板（P2-F5）：把因子层的合成结果接到策略层。

逻辑：每 ``rebalance_interval`` 个交易日调仓一次——
- 从 ``Context`` 读取**当日**各标的的因子值（截面标准化后的值，由回测引擎按日切片注入）；
- 按 ``weights``（缺省等权）线性合成得分；
- 取得分最高的 ``top_k`` 只等权买入，已持有但跌出名单的清仓；
- **缺任一必需因子的标的直接跳过**（历史不足、次新股、数据缺口都会造成缺值），
  而不是拿 0 顶替——0 在标准化后的因子里是"平均水平"，会凭空造出中性信号。

与 ``multi_factor_enhance`` 在需求文档里的定位一致：这是"因子 → 选股 → 回测"闭环的载体，
因子权重可以来自 ``aimoney-factor combine`` 的合成结果（``weights.csv``）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from engine.errors import StrategyError
from engine.portfolio.allocator import realized_volatility
from engine.portfolio.builder import PortfolioBuilder, PortfolioDecision
from engine.portfolio.constraints import PortfolioConstraints
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class MultiFactorEnhanceStrategy(BaseStrategy):
    """多因子选股策略（定期调仓、top_k 等权）。"""

    def __init__(self, *, strategy_id: str = "multi_factor_enhance", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            strategy_id=strategy_id,
            params={
                "factors": [],
                "weights": None,
                "top_k": 20,
                "rebalance_interval": 5,
                "total_weight": 0.95,
                "min_score": None,
                "require_all_factors": True,
                # ---- 组合层（默认全部关闭，保持"等权满仓"的朴素行为）----
                "weight_scheme": "equal",
                "max_weight": None,
                "min_weight": 0.0,
                "no_trade_band": 0.0,
                "max_turnover": None,
                "volatility_window": 20,
                **(params or {}),
            },
        )
        self._day_counter = 0
        #: 最近一次调仓的得分与组合决策（供测试与排查使用）
        self.last_scores: dict[str, float] = {}
        self.last_decision: PortfolioDecision | None = None

    # -------------------------------------------------- 协议方法
    def on_bar(self, context: Any, bars: Any) -> list[Signal]:
        factors = self._factor_list()
        top_k = int(self.param("top_k"))
        interval = int(self.param("rebalance_interval"))
        total_weight = float(self.param("total_weight"))
        min_score = self.param_or("min_score", None)
        require_all = bool(self.param_or("require_all_factors", True))

        if top_k <= 0 or interval <= 0:
            raise StrategyError(
                "top_k 与 rebalance_interval 必须为正",
                detail={"top_k": top_k, "rebalance_interval": interval},
            )

        self._day_counter += 1
        if (self._day_counter - 1) % interval != 0:
            return []  # 非调仓日不动，控制换手

        scores = self._score_universe(context, bars, factors, require_all=require_all)
        if min_score is not None:
            threshold = float(min_score)
            scores = {code: value for code, value in scores.items() if value >= threshold}
        self.last_scores = dict(scores)

        # 组合层：选股 → 分配方案 → 上下限/碎仓/无交易带/换手上限
        decision = self._build_weights(context, bars, scores, total_weight=total_weight)
        self.last_decision = decision

        signals: list[Signal] = []
        for code, weight in sorted(decision.weights.items(), key=lambda item: item[1], reverse=True):
            signals.append(
                Signal(
                    code=code,
                    strategy_id=context.strategy_id,
                    trade_date=context.trade_date,
                    direction=Direction.BUY,
                    target_weight=weight,
                    reason=f"多因子得分前 {top_k}（{decision.scheme} 权重 {weight:.2%}）",
                )
            )
        # 持仓中不在目标权重里的（跌出名单或被约束剔除）一律清仓
        for code in sorted(context.positions()):
            if code not in decision.weights:
                signals.append(
                    Signal(
                        code=code,
                        strategy_id=context.strategy_id,
                        trade_date=context.trade_date,
                        direction=Direction.SELL,
                        target_weight=-1.0,
                        reason="跌出多因子名单，清仓",
                    )
                )
        return signals

    # -------------------------------------------------- 内部
    def _factor_list(self) -> tuple[str, ...]:
        raw = self.params.get("factors")
        if isinstance(raw, str):
            names = tuple(part.strip() for part in raw.split(",") if part.strip())
        elif isinstance(raw, Sequence):
            names = tuple(str(item) for item in raw)
        else:
            names = ()
        if not names:
            raise StrategyError(
                "多因子策略必须配置 factors 列表",
                detail={"strategy_id": self.strategy_id, "given": raw},
            )
        return names

    def _weights(self, factors: Sequence[str]) -> dict[str, float]:
        raw = self.params.get("weights")
        if not isinstance(raw, Mapping):
            return {name: 1.0 / len(factors) for name in factors}
        weights = {name: float(raw.get(name, 0.0)) for name in factors}
        if all(value == 0.0 for value in weights.values()):
            raise StrategyError("weights 配置了但全部为 0", detail={"factors": list(factors)})
        return weights

    def _build_weights(
        self,
        context: Any,
        bars: Mapping[str, Any],
        scores: Mapping[str, float],
        *,
        total_weight: float,
    ) -> PortfolioDecision:
        """把得分交给组合层，得到带约束的目标权重。"""
        account = context.account()
        total_asset = float(account.total_asset)
        current_weights = self._current_weights(context, bars, total_asset)

        builder = PortfolioBuilder(
            scheme=str(self.param_or("weight_scheme", "equal")),
            top_k=int(self.param("top_k")),
            constraints=PortfolioConstraints(
                max_weight=self.param_or("max_weight", None),
                min_weight=float(self.param_or("min_weight", 0.0)),
                cash_buffer=max(1.0 - total_weight, 0.0),
                max_turnover=self.param_or("max_turnover", None),
                no_trade_band=float(self.param_or("no_trade_band", 0.0)),
            ),
        )
        volatilities = (
            self._volatilities(context, bars) if builder.scheme == "inverse_vol" else None
        )
        return builder.build(scores, current_weights=current_weights, volatilities=volatilities)

    def _current_weights(
        self, context: Any, bars: Mapping[str, Any], total_asset: float
    ) -> dict[str, float]:
        """当前持仓权重（按最新价盯市）。无价可依的持仓不参与，避免用臆测价格算权重。"""
        if total_asset <= 0:
            return {}
        weights: dict[str, float] = {}
        for code, position in context.positions().items():
            # 优先用当日收盘价（策略在收盘时点决策）；停牌等无当日 bar 时退回持仓最新价。
            # 注意不能用 position.last_price 代替当日价：它是**昨日**盯市的结果，
            # 会引入一天的滞后。
            bar = bars.get(code)
            price = float(bar.close) if bar is not None else float(getattr(position, "last_price", 0.0) or 0.0)
            if price <= 0.0:
                continue
            weights[code] = float(position.qty) * price / total_asset
        return weights

    def _volatilities(self, context: Any, bars: Mapping[str, Any]) -> dict[str, float]:
        """各标的的已实现波动率（逆波动率加权用）。"""
        window = int(self.param_or("volatility_window", 20))
        volatilities: dict[str, float] = {}
        for code in sorted(bars):
            closes = [float(value) for value in context.history(code, field="close", window=window + 1)]
            if len(closes) < 3:
                continue
            returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes)) if closes[index - 1] > 0]
            volatilities[code] = realized_volatility(returns)
        return volatilities

    def _score_universe(
        self,
        context: Any,
        bars: Mapping[str, Any],
        factors: Sequence[str],
        *,
        require_all: bool,
    ) -> dict[str, float]:
        weights = self._weights(factors)
        # 只有"回测压根没装载因子快照"才算配置错误；"今日无值"（预热期、停牌、缺数据）
        # 属于正常情况，应当静默跳过——否则每次回测都会在预热期被误判为故障。
        if require_all and not context.has_factors():
            raise StrategyError(
                "回测未装载因子快照（请用 --factors 指定因子），多因子策略无法工作",
                detail={"factors": list(factors)},
            )

        scores: dict[str, float] = {}
        for code in sorted(bars):
            values = {name: context.factor(name, code) for name in factors}
            present = {name: value for name, value in values.items() if value is not None}
            if require_all and len(present) != len(factors):
                continue  # 缺值不参与打分（拿 0 顶替会造出假的"中性"信号）
            if not present:
                continue
            score = sum(weights.get(name, 0.0) * float(value) for name, value in present.items())
            scores[code] = score
        return scores


__all__ = ["MultiFactorEnhanceStrategy"]
