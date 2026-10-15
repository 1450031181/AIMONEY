"""应用层异常入口：统一异常来源 + HTTP 映射。

异常定义集中在 ``engine.errors``（engine 不能依赖 app，故异常体系放在内核侧），
本模块只做三件事：
1. 对 API 层**显式重导出**所有异常，避免各处散落 ``from engine.errors import ...``；
2. 提供异常 -> HTTP 状态码 / 响应体的映射，保证 FastAPI 异常处理器只有一处逻辑；
3. 兜底未预期异常（不泄漏内部实现细节）。
"""

from __future__ import annotations

from typing import Any

from engine.errors import (
    AimoneyError,
    BrokerAuthError,
    BrokerError,
    BrokerUnavailable,
    ConfigError,
    CryptoError,
    DataNotReadyError,
    DataQualityError,
    DataSourceError,
    InsufficientFunds,
    InsufficientPosition,
    LookaheadError,
    NotFoundError,
    OrderRejected,
    RiskRejected,
    StateTransitionError,
    StrategyError,
    TradeGateClosed,
    ValidationError,
)

INTERNAL_ERROR_CODE = "INTERNAL_ERROR"


def http_status_of(exc: BaseException) -> int:
    """异常对应的 HTTP 状态码；未知异常一律 500。"""
    if isinstance(exc, AimoneyError):
        return exc.http_status
    return 500


def error_payload(exc: BaseException) -> dict[str, Any]:
    """构造统一错误响应体；未预期异常不返回堆栈与内部信息。"""
    if isinstance(exc, AimoneyError):
        return exc.to_dict()
    return {"code": INTERNAL_ERROR_CODE, "message": "服务内部错误", "detail": {}}


__all__ = [
    "INTERNAL_ERROR_CODE",
    "AimoneyError",
    "BrokerAuthError",
    "BrokerError",
    "BrokerUnavailable",
    "ConfigError",
    "CryptoError",
    "DataNotReadyError",
    "DataQualityError",
    "DataSourceError",
    "InsufficientFunds",
    "InsufficientPosition",
    "LookaheadError",
    "NotFoundError",
    "OrderRejected",
    "RiskRejected",
    "StateTransitionError",
    "StrategyError",
    "TradeGateClosed",
    "ValidationError",
    "error_payload",
    "http_status_of",
]
