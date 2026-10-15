# 多因子增强示例策略（P2-F5）：因子 → 选股 → 回测 闭环的入口
#
# 运行（先把因子喂给回测，策略才能在 Context 里读到）：
#   aimoney-backtest --strategy strategies/multi_factor_enhance \
#       --start 2026-09-01 --end 2026-10-30 \
#       --factors mom_20,vol_20 --top-k 3
#
# 权重可以取自 `aimoney-factor combine` 产出报告里的 weights.csv 最新一行，
# 也可以直接在 config.yaml 里写死；不写则等权。

from engine.strategy.templates.multi_factor_enhance import MultiFactorEnhanceStrategy


def create_strategy(params: dict) -> MultiFactorEnhanceStrategy:
    return MultiFactorEnhanceStrategy(params=params)
