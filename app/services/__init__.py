"""应用服务层：编排 engine 能力，承载业务语义（事务边界在此）。"""

from __future__ import annotations

from app.services.data_service import DataService

__all__ = ["DataService"]
