"""AIMoney 计算内核。

约束（见 docs/02-系统设计文档.md）：
- engine 只依赖第三方库与自身，**禁止 import app.***；
- engine 不做 IO 决策，数据进出统一走 engine.ports 中的 Protocol。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
