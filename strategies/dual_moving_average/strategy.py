# 双均线示例策略：演示用户策略目录的标准结构
#
# 目录约定：
#   strategies/<name>/
#     ├── strategy.py   # 必须定义 create_strategy(params) -> Strategy
#     └── config.yaml   # 可选：默认参数（CLI --params 可覆盖）
#
# 运行：
#   aimoney-backtest --strategy strategies/dual_moving_average \
#       --start 2026-09-01 --end 2026-10-30 --params fast=3 --params slow=10

from engine.strategy.templates.dual_moving_average import DualMovingAverageStrategy


def create_strategy(params: dict) -> DualMovingAverageStrategy:
    return DualMovingAverageStrategy(params=params)
