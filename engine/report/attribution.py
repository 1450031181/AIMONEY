"""收益归因（P4-4）：把"净值涨了多少"拆成"哪只标的赚的、成本吃掉多少"。

**为什么必须按成交逐笔重放成本**：只看 ``持仓.avg_cost`` 只能算**当前**持仓的浮盈，
已清仓标的的已实现盈亏就永久丢失了——而"哪个标的赚了钱"恰恰是最常被问的问题。
因此这里按时间顺序重放成交流水，用**移动加权平均成本**维护每个标的的持仓与成本基础：

    买入：qty += n, cost_basis += 金额 + 费用
    卖出：realized += 金额 − 费用 − 均价 × n, qty -= n, cost_basis -= 均价 × n

三个容易被忽略的细节：

1. **费用属于成本**：买入费用计入成本基础，卖出费用从卖出收入扣减；否则归因会把费用
   算成"凭空消失的钱"，与净值对不上。
2. **均价用「含费成本基础 ÷ 数量」**：这样卖出后的剩余成本与券商口径一致。
3. **重放结果要与持仓账本对得上**：剩余数量必须等于 ``Position.qty``，均价必须与
   ``Position.avg_cost`` 在一个最小价位内一致——不一致说明账本与流水已经分叉，
   此时**宁可报错**，也不要输出一份"看起来很专业"的错误归因。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from engine.errors import ValidationError
from engine.types import Position, Side, Trade

_ZERO: Final[Decimal] = Decimal(0)
#: 均价一致性容差：账本里的 avg_cost 是 4 位小数四舍五入后的值
COST_TOLERANCE: Final[Decimal] = Decimal("0.005")


@dataclass
class SymbolPnl:
    """单标的盈亏（已实现 + 未实现）。"""

    code: str
    buy_amount: Decimal = _ZERO
    sell_amount: Decimal = _ZERO
    fees: Decimal = _ZERO
    realized_pnl: Decimal = _ZERO
    qty: int = 0
    #: 含费成本均价（归因口径：买入费用属于成本）
    avg_cost: Decimal = _ZERO
    #: **不含费**成交均价（券商/账本口径，与 ``Position.avg_cost`` 对齐）
    gross_avg_cost: Decimal = _ZERO
    last_price: Decimal = _ZERO
    buy_count: int = 0
    sell_count: int = 0

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.qty <= 0 or self.last_price <= 0:
            return _ZERO
        return (self.last_price - self.avg_cost) * self.qty

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def market_value(self) -> Decimal:
        return self.last_price * self.qty if self.qty > 0 else _ZERO

    @property
    def closed(self) -> bool:
        return self.qty == 0 and self.sell_count > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "buy_amount": str(self.buy_amount),
            "sell_amount": str(self.sell_amount),
            "fees": str(self.fees),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_pnl": str(self.total_pnl),
            "qty": self.qty,
            "avg_cost": str(self.avg_cost),
            "gross_avg_cost": str(self.gross_avg_cost),
            "last_price": str(self.last_price),
            "market_value": str(self.market_value),
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
        }


@dataclass
class PortfolioPnl:
    """组合级归因结果。"""

    symbols: tuple[SymbolPnl, ...] = ()
    realized_pnl: Decimal = _ZERO
    unrealized_pnl: Decimal = _ZERO
    total_fees: Decimal = _ZERO
    buy_turnover: Decimal = _ZERO
    sell_turnover: Decimal = _ZERO
    trade_count: int = 0

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def turnover(self) -> Decimal:
        return self.buy_turnover + self.sell_turnover

    @property
    def fee_ratio(self) -> float:
        """费用 / 成交额：衡量"这套换手节奏被交易成本吃掉多少"。"""
        return float(self.total_fees / self.turnover) if self.turnover > 0 else 0.0

    def winners(self, top: int = 3) -> tuple[SymbolPnl, ...]:
        ranked = sorted(self.symbols, key=lambda item: item.total_pnl, reverse=True)
        return tuple(item for item in ranked if item.total_pnl > 0)[:top]

    def losers(self, top: int = 3) -> tuple[SymbolPnl, ...]:
        ranked = sorted(self.symbols, key=lambda item: item.total_pnl)
        return tuple(item for item in ranked if item.total_pnl < 0)[:top]

    def summary(self) -> str:
        if not self.symbols:
            return "归因：无成交记录"
        parts = [
            f"归因（{len(self.symbols)} 只标的，{self.trade_count} 笔成交）："
            f"已实现 {self.realized_pnl:+.2f}，浮动 {self.unrealized_pnl:+.2f}，"
            f"合计 {self.total_pnl:+.2f}",
            f"费用合计 {self.total_fees:.2f}（占成交额 {self.fee_ratio:.3%}）",
        ]
        best = self.winners(1)
        worst = self.losers(1)
        if best:
            parts.append(f"贡献最大：{best[0].code} {best[0].total_pnl:+.2f}")
        if worst:
            parts.append(f"拖累最大：{worst[0].code} {worst[0].total_pnl:+.2f}")
        return "；".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_pnl": str(self.total_pnl),
            "total_fees": str(self.total_fees),
            "buy_turnover": str(self.buy_turnover),
            "sell_turnover": str(self.sell_turnover),
            "fee_ratio": self.fee_ratio,
            "trade_count": self.trade_count,
            "symbols": [item.to_dict() for item in sorted(self.symbols, key=lambda i: i.total_pnl, reverse=True)],
        }


def compute_attribution(
    trades: Sequence[Trade],
    positions: Mapping[str, Position] | Sequence[Position] = (),
    *,
    check_reconcile: bool = True,
) -> PortfolioPnl:
    """按成交逐笔重放得到各标的盈亏。

    ``check_reconcile=True`` 时会校验"重放结果 vs 持仓账本"，不一致即报错：
    分叉的账本算出来的归因只会误导人，早点暴露比输出漂亮数字更有价值。
    """
    position_map = _as_position_map(positions)
    books: dict[str, SymbolPnl] = {}
    # 两套成本基础必须并行维护，不能混用：
    # - ``cost_basis``（含费）：算盈亏，因为买入费是真金白银花掉的；
    # - ``gross_basis``（不含费）：与账本 Position.avg_cost 对账，因为券商口径不含费。
    cost_basis: dict[str, Decimal] = {}
    gross_basis: dict[str, Decimal] = {}

    for trade in sorted(trades, key=lambda item: item.trade_time):
        book = books.setdefault(trade.code, SymbolPnl(code=trade.code))
        amount = trade.price * trade.qty
        fee = trade.total_fee
        book.fees += fee
        if trade.side is Side.BUY:
            book.buy_amount += amount
            book.buy_count += 1
            book.qty += trade.qty
            cost_basis[trade.code] = cost_basis.get(trade.code, _ZERO) + amount + fee
            gross_basis[trade.code] = gross_basis.get(trade.code, _ZERO) + amount
            book.avg_cost = cost_basis[trade.code] / book.qty
            book.gross_avg_cost = gross_basis[trade.code] / book.qty
        else:
            book.sell_amount += amount
            book.sell_count += 1
            if trade.qty > book.qty:
                raise ValidationError(
                    "成交流水不平：卖出数量超过累计买入（账本或流水被篡改）",
                    detail={"code": trade.code, "sell_qty": trade.qty, "held": book.qty},
                )
            average = book.avg_cost if book.avg_cost > 0 else _ZERO
            gross_average = book.gross_avg_cost if book.gross_avg_cost > 0 else _ZERO
            book.realized_pnl += amount - fee - average * trade.qty
            book.qty -= trade.qty
            cost_basis[trade.code] = max(cost_basis.get(trade.code, _ZERO) - average * trade.qty, _ZERO)
            gross_basis[trade.code] = max(gross_basis.get(trade.code, _ZERO) - gross_average * trade.qty, _ZERO)
            if book.qty == 0:
                book.avg_cost = _ZERO
                book.gross_avg_cost = _ZERO
                cost_basis[trade.code] = _ZERO
                gross_basis[trade.code] = _ZERO

    # 未平仓标的的最新价：优先用持仓账本（已由券商盯市），否则退回最后成交价
    for code, book in books.items():
        position = position_map.get(code)
        if position is not None and position.qty > 0:
            book.last_price = position.last_price
            if check_reconcile:
                _assert_reconciled(book, position)

    return PortfolioPnl(
        symbols=tuple(books.values()),
        realized_pnl=sum((book.realized_pnl for book in books.values()), _ZERO),
        unrealized_pnl=sum((book.unrealized_pnl for book in books.values()), _ZERO),
        total_fees=sum((book.fees for book in books.values()), _ZERO),
        buy_turnover=sum((book.buy_amount for book in books.values()), _ZERO),
        sell_turnover=sum((book.sell_amount for book in books.values()), _ZERO),
        trade_count=len(trades),
    )


def _as_position_map(positions: Mapping[str, Position] | Sequence[Position]) -> dict[str, Position]:
    if isinstance(positions, Mapping):
        return dict(positions)
    return {position.code: position for position in positions}


def _assert_reconciled(book: SymbolPnl, position: Position) -> None:
    """重放结果必须与持仓账本一致，否则宁可报错也不输出错误归因。

    比的是**不含费**均价：``Position.avg_cost`` 是成交均价（券商口径不含费），
    而归因的 ``avg_cost`` 含费——拿含费均价比账本必然不等，那是口径差异不是错误。
    """
    if book.qty != position.qty:
        raise ValidationError(
            "归因重放与持仓账本不一致（数量）",
            detail={"code": book.code, "replayed_qty": book.qty, "ledger_qty": position.qty},
        )
    if abs(book.gross_avg_cost - position.avg_cost) > COST_TOLERANCE:
        raise ValidationError(
            "归因重放与持仓账本不一致（成交均价）",
            detail={
                "code": book.code,
                "replayed_gross_avg_cost": str(book.gross_avg_cost),
                "ledger_avg_cost": str(position.avg_cost),
                "hint": "检查成交流水是否完整（是否漏存了某些成交）",
            },
        )


def per_symbol_returns(trades: Iterable[Trade], starting_cash: Decimal) -> dict[str, float]:
    """各标的对**初始资金**的贡献率（便于横向看谁在赚钱）。"""
    attribution = compute_attribution(list(trades), check_reconcile=False)
    if starting_cash <= 0:
        return {}
    return {book.code: float(book.total_pnl / starting_cash) for book in attribution.symbols}


__all__ = [
    "COST_TOLERANCE",
    "PortfolioPnl",
    "SymbolPnl",
    "compute_attribution",
    "per_symbol_returns",
]
