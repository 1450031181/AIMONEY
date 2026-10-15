"""AIMoney 统一异常体系（共享内核，engine 与 app 均可依赖，engine 零外部依赖）。

约定：
1. 每个异常携带稳定的 ``code`` 字符串常量，便于告警规则匹配与前端映射；
2. ``http_status`` 只服务 app 层的翻译，engine 自身不关心 HTTP；
3. ``detail`` 放结构化上下文（code / date / qty / rule ...），日志与 API 响应可直接复用。
"""

from __future__ import annotations

from typing import Any


class AimoneyError(Exception):
    """所有业务异常的基类。"""

    code: str = "AIMONEY_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}

    def __str__(self) -> str:
        if not self.detail:
            return self.message
        return f"{self.message} | {self.detail}"


# ---------------------------------------------------------------- 配置与安全
class ConfigError(AimoneyError):
    code = "CONFIG_ERROR"
    http_status = 500


class CryptoError(ConfigError):
    code = "CRYPTO_ERROR"


# ---------------------------------------------------------------- 通用校验
class ValidationError(AimoneyError):
    """领域数据校验失败（非 pydantic 校验错误）。"""

    code = "VALIDATION_ERROR"
    http_status = 422


class StateTransitionError(AimoneyError):
    """状态机非法迁移。"""

    code = "STATE_TRANSITION_ERROR"
    http_status = 409


class NotFoundError(AimoneyError):
    """请求的资源不存在（如某日尚未生成质量报告）。"""

    code = "NOT_FOUND"
    http_status = 404


# ---------------------------------------------------------------- 数据
class DataSourceError(AimoneyError):
    code = "DATA_SOURCE_ERROR"
    http_status = 502


class DataNotReadyError(DataSourceError):
    """数据未就绪（未同步完成 / 质量门禁关闭），下游应禁止交易而非降级使用。"""

    code = "DATA_NOT_READY"
    http_status = 503


class DataQualityError(AimoneyError):
    code = "DATA_QUALITY_ERROR"


class LookaheadError(AimoneyError):
    """未来函数越界：读取了超出当前时点的数据。"""

    code = "LOOKAHEAD_VIOLATION"


class TradeGateClosed(AimoneyError):
    """交易门禁关闭（数据质量 FAIL / 对账不平），当日禁止自动交易。"""

    code = "TRADE_GATE_CLOSED"
    http_status = 423


# ---------------------------------------------------------------- 策略
class StrategyError(AimoneyError):
    code = "STRATEGY_ERROR"


# ---------------------------------------------------------------- 交易与风控
class OrderRejected(AimoneyError):
    code = "ORDER_REJECTED"
    http_status = 409


class RiskRejected(OrderRejected):
    code = "RISK_REJECTED"


class InsufficientFunds(RiskRejected):
    code = "RISK_INSUFFICIENT_FUNDS"


class InsufficientPosition(RiskRejected):
    code = "RISK_INSUFFICIENT_POSITION"


class BrokerError(AimoneyError):
    code = "BROKER_ERROR"
    http_status = 502


class BrokerUnavailable(BrokerError):
    code = "BROKER_UNAVAILABLE"
    http_status = 503


class BrokerAuthError(BrokerError):
    code = "BROKER_AUTH_ERROR"
    http_status = 401


__all__ = [
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
]
