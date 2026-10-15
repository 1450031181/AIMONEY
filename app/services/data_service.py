"""数据服务（T-12）：把 engine 的只读查询能力包装成应用层用例。

分层约定：路由层只做参数校验与编排，**业务语义都在这里**，
engine 侧只提供纯技术能力（查询、统计），不感知 HTTP/配置。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from engine.datahub.quality import TradeGate
from engine.datahub.reader import DEFAULT_DATASET, DuckDBReader
from engine.errors import NotFoundError


@dataclass
class DataService:
    """数据侧的只读用例集合（覆盖率、交易门禁、质量报告）。"""

    settings: Settings
    dataset: str = DEFAULT_DATASET

    # -------------------------------------------------- 查询入口
    def reader(self) -> DuckDBReader:
        """按需创建查询连接（默认内存库：查询直接读 Parquet，不依赖持久化库文件）。"""
        return DuckDBReader(self.settings.parquet_dir)

    def gate(self) -> TradeGate:
        return TradeGate(self.settings.gate_file)

    # -------------------------------------------------- 用例
    def attributes(self, dataset: str | None = None) -> dict[str, Any]:
        with self.reader() as reader:
            return dict(reader.attributes(dataset or self.dataset))

    def coverage_payload(self, dataset: str | None = None) -> dict[str, Any]:
        resolved = dataset or self.dataset
        attributes = self.attributes(resolved)
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": attributes,
            "gate": self.gate_state(),
        }

    def gate_state(self) -> dict[str, Any]:
        state = self.gate().state()
        return {
            "open": bool(state.get("open", False)),
            "reason": state.get("reason"),
            "as_of": state.get("as_of"),
            "updated_at": state.get("updated_at"),
        }

    def quality_report(self, as_of: date) -> dict[str, Any]:
        path = self._quality_report_path(as_of)
        if not path.is_file():
            raise NotFoundError(
                f"{as_of.isoformat()} 尚无质量报告（可能当日质量校验未执行）",
                detail={"as_of": as_of.isoformat(), "path": str(path)},
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise NotFoundError(f"质量报告格式非法：{path}", detail={"path": str(path)})
        return payload

    def _quality_report_path(self, as_of: date) -> Path:
        return self.settings.quality_dir / f"quality_{as_of:%Y%m%d}.json"


__all__ = ["DataService"]
