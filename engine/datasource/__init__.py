"""数据源适配器：把外部数据统一成 engine.ports.DataSource 契约。"""

from __future__ import annotations

from engine.datasource.akshare_source import AkShareClient, AkShareDataSource, infer_board
from engine.datasource.base import RateLimiter, retry_call
from engine.datasource.memory import InMemoryDataSource

__all__ = [
    "AkShareClient",
    "AkShareDataSource",
    "InMemoryDataSource",
    "RateLimiter",
    "infer_board",
    "retry_call",
]
