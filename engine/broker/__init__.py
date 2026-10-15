"""券商适配器：模拟盘（离线全链路）+ 真实券商（P5）。"""

from __future__ import annotations

from engine.broker.paper import DEFAULT_MAX_DEFER_DAYS, PaperBrokerAdapter

__all__ = ["DEFAULT_MAX_DEFER_DAYS", "PaperBrokerAdapter"]
