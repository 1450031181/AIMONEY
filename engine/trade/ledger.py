"""账本持久化（P4-2）：账户 / 持仓 / 委托的 JSON 快照。

模拟盘按**每个交易日跑一次**（定时任务），进程会退出，因此三类状态必须落盘：

- **账户与持仓**：否则每天从初始资金重新开始，净值曲线断掉；
- **委托账本**：否则重启后幂等键集合为空，同一笔计划会被**重复报出**；
- **最近价**：券商侧冻结资金需要参考价（没有参考价的市价单会被拒绝）。

**诚实的能力边界**：JSON 文件不是账本系统——没有事务、没有并发控制、没有对账凭证。
它只适合"单人、单进程、模拟盘"这个场景。接真实资金前必须换成事务型存储，
并在文档里明确写出这一点，避免它被误当成生产级账本。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from engine.datahub.jsonio import to_jsonable
from engine.errors import ValidationError
from engine.types import Account, Order, OrderStatus, OrderType, Position, Side

_logger = logging.getLogger(__name__)

DEFAULT_LEDGER_FILE: Final[str] = "paper_ledger.json"
_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class BrokerSnapshot:
    """券商侧状态快照（账户 / 持仓 / 最近价 / 在途委托冻结台账）。

    ``open_orders`` 用的是**原始映射**（primitives），因为那份台账是券商自己的状态，
    本模块只负责"原样存取"，不理解它的字段语义。
    """

    account: Account
    positions: tuple[Position, ...]
    last_prices: Mapping[str, Decimal]
    open_orders: tuple[Mapping[str, Any], ...] = ()
    #: 最后成功处理的交易日。用于**日级防重**：重复跑同一天不是幂等的
    #: （上次的挂单会在同一天行情上再撮一次，产生本不该有的成交）
    last_trade_date: date | None = None
    saved_at: datetime | None = None


class JsonLedger:
    """账户、持仓、委托的 JSON 快照读写（原子写）。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    # ============================================================ 读写
    def save_broker(
        self,
        *,
        account: Account,
        positions: Sequence[Position],
        last_prices: Mapping[str, Decimal],
        open_orders: Sequence[Mapping[str, Any]] = (),
        last_trade_date: date | None = None,
    ) -> Path:
        payload = self._read() or {"schema_version": _SCHEMA_VERSION, "orders": []}
        payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
        if last_trade_date is not None:
            payload["last_trade_date"] = last_trade_date.isoformat()
        payload["account"] = _account_to_dict(account)
        payload["positions"] = [_position_to_dict(position) for position in positions]
        payload["last_prices"] = {code: str(price) for code, price in last_prices.items()}
        # 在途委托的冻结台账必须一起落盘：漏掉它会让昨天的挂单今天无人撮合，
        # 账户里的冻结资金变成永远解不开的悬空额度
        payload["open_orders"] = [dict(item) for item in open_orders]
        return self._write(payload)

    def load_broker(self) -> BrokerSnapshot | None:
        payload = self._read()
        if payload is None or "account" not in payload:
            return None
        return BrokerSnapshot(
            account=_account_from_dict(payload["account"]),
            positions=tuple(_position_from_dict(item) for item in payload.get("positions", [])),
            last_prices={code: Decimal(str(price)) for code, price in payload.get("last_prices", {}).items()},
            open_orders=tuple(dict(item) for item in payload.get("open_orders", [])),
            last_trade_date=_parse_date(payload.get("last_trade_date")),
            saved_at=_parse_datetime(payload.get("saved_at")),
        )

    def save_orders(self, orders: Sequence[Order]) -> Path:
        payload = self._read() or {"schema_version": _SCHEMA_VERSION}
        payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
        payload["orders"] = [_order_to_dict(order) for order in orders]
        return self._write(payload)

    def load_orders(self) -> list[Order]:
        payload = self._read()
        if payload is None:
            return []
        return [_order_from_dict(item) for item in payload.get("orders", [])]

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    # ============================================================ 内部
    def _read(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("账本文件无法读取或格式非法", detail={"path": str(self.path), "error": str(exc)}) from exc
        if not isinstance(payload, dict):
            raise ValidationError("账本文件根节点必须是映射", detail={"path": str(self.path)})
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            # 版本不匹配宁可报错：静默按新格式解析老文件会造成"持仓凭空消失"
            raise ValidationError(
                "账本文件版本不匹配", detail={"path": str(self.path), "found": version, "expected": _SCHEMA_VERSION}
            )
        return payload

    def _write(self, payload: dict[str, Any]) -> Path:
        payload.setdefault("schema_version", _SCHEMA_VERSION)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        try:
            tmp.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return self.path


# ============================================================ 序列化
def _account_to_dict(account: Account) -> dict[str, Any]:
    return {
        "account_id": account.account_id,
        "initial_cash": str(account.initial_cash),
        "cash": str(account.cash),
        "market_value": str(account.market_value),
        "frozen_cash": str(account.frozen_cash),
        "nav": account.nav,
        "trade_date": account.trade_date.isoformat() if account.trade_date else None,
    }


def _account_from_dict(payload: Mapping[str, Any]) -> Account:
    return Account(
        account_id=str(payload.get("account_id", "default")),
        initial_cash=Decimal(str(payload["initial_cash"])),
        cash=Decimal(str(payload["cash"])),
        market_value=Decimal(str(payload.get("market_value", 0))),
        frozen_cash=Decimal(str(payload.get("frozen_cash", 0))),
        nav=float(payload.get("nav", 1.0)),
        trade_date=_parse_date(payload.get("trade_date")),
    )


def _position_to_dict(position: Position) -> dict[str, Any]:
    return {
        "code": position.code,
        "account_id": position.account_id,
        "qty": position.qty,
        "avail_qty": position.avail_qty,
        "frozen_qty": position.frozen_qty,
        "avg_cost": str(position.avg_cost),
        "last_price": str(position.last_price),
    }


def _position_from_dict(payload: Mapping[str, Any]) -> Position:
    return Position(
        code=str(payload["code"]),
        account_id=str(payload.get("account_id", "default")),
        qty=int(payload.get("qty", 0)),
        avail_qty=int(payload.get("avail_qty", 0)),
        frozen_qty=int(payload.get("frozen_qty", 0)),
        avg_cost=Decimal(str(payload.get("avg_cost", 0))),
        last_price=Decimal(str(payload.get("last_price", 0))),
    )


def _order_to_dict(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "code": order.code,
        "side": str(order.side),
        "qty": order.qty,
        "price": None if order.price is None else str(order.price),
        "order_type": str(order.order_type),
        "plan_id": order.plan_id,
        "strategy_id": order.strategy_id,
        "broker_order_id": order.broker_order_id,
        "filled_qty": order.filled_qty,
        "avg_price": str(order.avg_price),
        "status": str(order.status),
        "reject_reason": order.reject_reason,
        "idem_key": order.idem_key,
        "created_at": order.created_at.isoformat(timespec="seconds"),
    }


def _order_from_dict(payload: Mapping[str, Any]) -> Order:
    order = Order(
        code=str(payload["code"]),
        side=Side(str(payload["side"])),
        qty=int(payload["qty"]),
        plan_id=str(payload.get("plan_id", "")),
        strategy_id=str(payload.get("strategy_id", "")),
        price=None if payload.get("price") in (None, "") else Decimal(str(payload["price"])),
        order_type=OrderType(str(payload.get("order_type", OrderType.LIMIT))),
        order_id=str(payload["order_id"]),
        broker_order_id=payload.get("broker_order_id"),
        idem_key=str(payload.get("idem_key", "")),
    )
    # 状态与成交直接回填（不做迁移校验）：它们来自本地账本，属于"恢复"而非"迁移"
    order.filled_qty = int(payload.get("filled_qty", 0))
    if not 0 <= order.filled_qty <= order.qty:
        raise ValidationError(
            "账本中的委托成交量越界，拒绝加载",
            detail={"order_id": order.order_id, "filled_qty": order.filled_qty, "qty": order.qty},
        )
    order.avg_price = Decimal(str(payload.get("avg_price", 0)))
    order.status = OrderStatus(str(payload.get("status", OrderStatus.PENDING)))
    order.reject_reason = payload.get("reject_reason")
    created = _parse_datetime(payload.get("created_at"))
    if created is not None:
        order.created_at = created
    return order


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_date(value: object) -> date | None:
    parsed = _parse_datetime(value if isinstance(value, str) else None)
    return parsed.date() if parsed is not None else None


__all__ = ["DEFAULT_LEDGER_FILE", "BrokerSnapshot", "JsonLedger"]
