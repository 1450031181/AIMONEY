"""因子工厂：定义、计算、预处理与评价。"""

from __future__ import annotations

# 必须显式导入 library：内置因子靠 import 副作用完成注册（漏掉这行 = 注册表全空）
from engine.factor import library  # noqa: F401
from engine.factor.base import (
    FactorDefinition,
    factor,
    factor_names,
    get_factor,
    list_factors,
    register_factor,
)
from engine.factor.engine import compute_factors
from engine.factor.evaluate import (
    EvaluationConfig,
    FactorEvaluation,
    evaluate_factor,
    factor_correlation,
    forward_returns,
    information_coefficient,
)
from engine.factor.preprocess import (
    PreprocessConfig,
    demean,
    preprocess_cross_section,
    rank_pct,
    winsorize_mad,
    zscore,
)

__all__ = [
    "EvaluationConfig",
    "FactorDefinition",
    "FactorEvaluation",
    "PreprocessConfig",
    "compute_factors",
    "demean",
    "evaluate_factor",
    "factor",
    "factor_correlation",
    "factor_names",
    "forward_returns",
    "get_factor",
    "information_coefficient",
    "list_factors",
    "preprocess_cross_section",
    "rank_pct",
    "register_factor",
    "winsorize_mad",
    "zscore",
]
