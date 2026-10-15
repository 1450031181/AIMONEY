"""急停开关（P3-2）。

**与交易门禁刻意相反的默认值**，这一点必须说清楚，否则很容易被"统一一下"改错：

| 机制 | 文件缺失时的语义 | 理由 |
| --- | --- | --- |
| 交易门禁 ``trade_gate`` | **关闭**（不许交易） | 它是"发车前必须确认数据可用"的闸门：没跑过质量校验就不该开车 |
| 急停开关 ``kill_switch`` | **未激活**（允许交易） | 它是"紧急制动"：全新部署时它当然不该是踩下的状态 |

两者的**唯一共同点**是读取异常时都倒向安全侧：急停开关文件损坏 / 内容非法时按
**已激活**处理——此时我们连"它到底该不该停"都无法判断，只能选择不动。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

DEFAULT_KILL_SWITCH_FILE = "state/kill_switch.json"


class KillSwitch:
    """持久化的急停开关（文件实现；接 Redis 后替换实现即可，接口不变）。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # -------------------------------------------------- 查询
    def is_active(self) -> bool:
        """是否处于急停状态。

        - 文件不存在 → 未激活（全新部署的正常状态）；
        - 文件存在但读不出来 / 格式非法 → **激活**（安全侧：不确定就不要动）。
        """
        if not self.path.is_file():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.error("急停开关状态文件无法读取，按已激活处理：%s | %s", self.path, exc)
            return True
        if not isinstance(payload, dict):
            _logger.error("急停开关状态文件格式非法，按已激活处理：%s", self.path)
            return True
        return bool(payload.get("active", False))

    def state(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"active": False, "reason": None, "updated_at": None, "by": None}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"active": True, "reason": f"状态文件不可读：{exc}", "updated_at": None, "by": "unknown"}
        if not isinstance(payload, dict):
            return {"active": True, "reason": "状态文件格式非法", "updated_at": None, "by": "unknown"}
        return payload

    # -------------------------------------------------- 变更
    def activate(self, reason: str, *, by: str = "manual") -> None:
        if not reason.strip():
            raise ValueError("激活急停必须给出原因（事后复盘唯一依据）")
        self._write(active=True, reason=reason, by=by)
        _logger.critical("急停开关已激活：%s（操作者 %s）", reason, by)

    def deactivate(self, *, by: str = "manual", note: str = "") -> None:
        """解除急停。必须由人显式调用——自动恢复会让"紧急制动"形同虚设。"""
        self._write(active=False, reason=note or "人工解除", by=by)
        _logger.warning("急停开关已解除（操作者 %s，备注 %s）", by, note or "-")

    def _write(self, *, active: bool, reason: str, by: str) -> None:
        payload = {
            "active": active,
            "reason": reason,
            "by": by,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)


__all__ = ["DEFAULT_KILL_SWITCH_FILE", "KillSwitch"]
