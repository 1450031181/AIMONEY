"""数据相关接口（T-12）。

路由层职责边界：只做参数校验与调用服务，不含业务语义；
异常统一抛 ``AimoneyError`` 子类，由 ``app.main`` 的异常处理器翻译成 HTTP 响应。
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from app.services.data_service import DataService

router = APIRouter(prefix="/api/v1/data", tags=["data"])

DatasetQuery = Annotated[str | None, Query(description="数据集名，默认 daily")]


def _service(request: Request) -> DataService:
    service: DataService = request.app.state.data_service
    return service


@router.get("/coverage", summary="数据覆盖情况与交易门禁")
def get_coverage(request: Request, dataset: DatasetQuery = None) -> dict[str, Any]:
    """返回数据集的行数/标的数/起止日期/分区数/数据版本/最新同步时间，以及门禁状态。"""
    return _service(request).coverage_payload(dataset)


@router.get("/gate", summary="交易门禁状态")
def get_gate(request: Request) -> dict[str, Any]:
    return _service(request).gate_state()


@router.get("/quality/{as_of}", summary="指定日期的数据质量报告")
def get_quality_report(request: Request, as_of: date) -> dict[str, Any]:
    """``as_of`` 用 ISO 日期（如 2026-10-14）；当日未跑质量校验时返回 404。"""
    return _service(request).quality_report(as_of)


__all__ = ["router"]
