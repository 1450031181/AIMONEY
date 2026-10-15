"""FastAPI 应用入口（T-12 首次搭建）。

启动顺序刻意固定：**先校验配置，再建目录，再初始化日志**。
配置非法时必须立刻失败（fail-fast），不能带着"半懂"的配置提供服务。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.data import router as data_router
from app.core.config import Settings, get_settings
from app.core.errors import AimoneyError, error_payload, http_status_of
from app.core.logging import get_logger, setup_logging
from app.services.data_service import DataService

_logger = get_logger(__name__)
API_VERSION = "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造应用（工厂形式，便于测试注入独立配置）。"""
    resolved = settings or get_settings()
    resolved.ensure_dirs()
    setup_logging(
        level=resolved.log.level,
        json_output=resolved.log.json_output,
        log_dir=resolved.log_dir,
    )

    app = FastAPI(
        title="AIMoney",
        version=API_VERSION,
        description="全自动量化交易系统 —— 数据与交易服务",
    )
    app.state.settings = resolved
    app.state.data_service = DataService(settings=resolved)

    app.include_router(data_router)

    @app.exception_handler(AimoneyError)
    async def _handle_aimoney_error(request: Request, exc: AimoneyError) -> JSONResponse:
        """统一异常出口：业务异常按其自带 ``http_status`` 返回结构化错误体。"""
        _logger.warning("接口异常：%s %s | %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=http_status_of(exc), content=error_payload(exc))

    @app.get("/health", tags=["system"], summary="健康检查")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "env": resolved.app.env,
            "data_root": str(resolved.data.root_dir),
            "gate": app.state.data_service.gate_state(),
        }

    _logger.info("AIMoney 应用已创建：env=%s，数据目录=%s", resolved.app.env, resolved.data.root_dir)
    return app


app = create_app()


__all__ = ["app", "create_app"]
