"""结构化日志（T-02）。

要点：
- 走 stdlib logging 作为后端，因此**同时**支持控制台与滚动文件输出，第三方库日志也统一格式；
- structlog 负责结构化字段（trace_id / strategy_id / order_id ...），便于按链路 ID 检索；
- 敏感信息一律用 ``crypto.mask`` 打码后再入日志，日志里绝不允许出现券商账号明文。
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

_configured = False
_LOG_FILE_NAME = "aimoney.log"
_MAX_BYTES = 32 * 1024 * 1024
_BACKUP_COUNT = 7


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_dir: Path | None = None,
    *,
    force: bool = False,
) -> None:
    """初始化日志。重复调用默认幂等，``force=True`` 可强制重配（测试用）。"""
    global _configured
    if _configured and not force:
        return

    log_level = _resolve_level(level)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False, pad_event_to=32)
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=shared_processors,
    )

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / _LOG_FILE_NAME,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(log_level)

    # 第三方库降噪：HTTP 请求日志在 DEBUG 下才输出
    for noisy in ("urllib3", "httpx", "akshare", "matplotlib"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))

    _configured = True
    get_logger(__name__).debug("logging_configured", level=level, json=json_output)


def get_logger(name: str = "aimoney") -> BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_context(**kwargs: Any) -> None:
    """把上下文（trace_id / task_name / strategy_id）绑到当前协程/线程后续所有日志。"""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def _resolve_level(level: str) -> int:
    value = logging.getLevelName(level.upper())
    if not isinstance(value, int):
        raise ValueError(f"非法日志级别：{level}")
    return value


__all__ = [
    "bind_context",
    "clear_context",
    "get_logger",
    "setup_logging",
]
