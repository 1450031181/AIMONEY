"""模拟盘券商适配器（P4-1）。

它用**真实的撮合规则**在本地扮演一个"交易所"。价值不在于省下接口费，而在于
让实盘链路的全部代码在离线状态下被完整执行：

    风控 → 报单 → 冻结资金/持仓 → 撮合 → 成交通知 → 持仓资金更新 → 查单校准 → 日切

这些路径如果只在实盘第一次运行，第一次运行就是"带着真钱调试"。

三条与真实券商对齐的语义：

1. **报单幂等**（端口硬要求）：同一 ``Order.idem_key`` 重复调用只产生一笔委托，
   返回同一个券商委托号。断线重连后的重试正是靠它避免重复下单。
2. **冻结与释放**：买入冻结资金（含预估费用）、卖出冻结可卖数量；
   撤单/废单/顺延超限都必须把冻结**还回去**，否则资金与持仓会缓慢泄漏。
3. **查询返回副本**：``query_positions()`` 等返回的是快照。若把内部可变对象直接吐出去，
   调用方一次无心修改就会污染账本，且极难定位。
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Final

from engine.backtest.cost import FeeModel
from engine.backtest.match import FillStatus, MatchEngine
from engine.backtest.slippage import NoSlippage, SlippageModel
from engine.errors import (
    BrokerError,
    BrokerUnavailable,
    InsufficientFunds,
    InsufficientPosition,
    OrderRejected,
    ValidationError,
)
from engine.ids import new_id
from engine.num import round_cost, to_decimal
from engine.ports import OrderCallback, TradeCallback
from engine.types import Account, Bar, Order, OrderPlan, Position, Side, Trade

_logger = logging.getLogger(__name__)

_ZERO: Final[Decimal] = Decimal(0)
DEFAULT_MAX_DEFER_DAYS: Final[int] = 2


class _OpenOrder:
    """券商侧的在途委托（含冻结资源台账，撤单/废单时按此归还）。"""

    __slots__ = ("broker_order_id", "defer_days", "frozen_cash", "frozen_qty", "order")

    def __init__(self, broker_order_id: str, order: Order) -> None:
        self.broker_order_id = broker_order_id
        self.order = order
        self.frozen_cash: Decimal = _ZERO
        self.frozen_qty: int = 0
        self.defer_days = 0


class PaperBrokerAdapter:
    """模拟盘券商（实现 ``engine.ports.BrokerAdapter``）。"""

    def __init__(
        self,
        *,
        initial_cash: Decimal | float | str = "1000000",
        fee_model: FeeModel | None = None,
        matcher: MatchEngine | None = None,
        slippage: SlippageModel | None = None,
        max_defer_days: int = DEFAULT_MAX_DEFER_DAYS,
        account_id: str = "default",
        adapter_name: str = "paper",
    ) -> None:
        start_cash = round_cost(to_decimal(initial_cash))
        if start_cash <= 0:
            raise ValidationError("模拟盘初始资金必须为正", detail={"initial_cash": str(start_cash)})
        self._name = adapter_name
        self._account = Account(account_id=account_id, initial_cash=start_cash, cash=start_cash)
        self._positions: dict[str, Position] = {}
        self._fee_model = fee_model or FeeModel()
        self._matcher = matcher or MatchEngine()
        self._slippage = slippage or NoSlippage()
        self._max_defer_days = max_defer_days
        self._open: dict[str, _OpenOrder] = {}
        self._idem_index: dict[str, str] = {}
        self._orders: list[Order] = []
        self._trades: list[Trade] = []
        self._last_prices: dict[str, Decimal] = {}
        self._connected = False
        self._order_callbacks: list[OrderCallback] = []
        self._trade_callbacks: list[TradeCallback] = []

    # ============================================================ 端口：连接
    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        _logger.info("模拟盘券商已连接（初始资金 %s）", self._account.initial_cash)

    def disconnect(self) -> None:
        self._connected = False

    def on_order_update(self, callback: OrderCallback) -> None:
        self._order_callbacks.append(callback)

    def on_trade_update(self, callback: TradeCallback) -> None:
        self._trade_callbacks.append(callback)

    # ============================================================ 端口：查询
    def query_account(self) -> Account:
        self._mark_to_market()
        return copy.deepcopy(self._account)

    def query_positions(self) -> list[Position]:
        self._mark_to_market()
        return [copy.deepcopy(position) for position in self._positions.values()]

    def query_orders(self) -> list[Order]:
        """全量委托（含已终结）。断线后校准以它为准。"""
        return [copy.deepcopy(order) for order in self._orders]

    # ============================================================ 端口：报单
    def place_order(self, order: Order) -> str:
        """报单，返回券商委托号。**幂等**：同一 ``idem_key`` 只会产生一笔委托。"""
        self._require_connected()
        if order.idem_key:
            existing = self._idem_index.get(order.idem_key)
            if existing is not None:
                _logger.info("幂等拦下重复报单：%s → %s", order.idem_key, existing)
                return existing

        # 先建台账再冻结：冻结额必须记在**同一个**条目上，否则撤单时无从归还
        entry = _OpenOrder(new_id("bo"), order)
        try:
            self._freeze(entry)
        except (InsufficientFunds, InsufficientPosition, OrderRejected) as exc:
            self._release(entry)  # 归还可能已冻结的部分（防御性，正常路径下冻结是原子的）
            self._orders.append(order)
            order.reject(str(exc))
            self._emit_order(order)
            raise OrderRejected(str(exc), detail={"code": order.code, "qty": order.qty}) from exc

        order.mark_submitted(entry.broker_order_id)
        self._open[entry.broker_order_id] = entry
        if order.idem_key:
            self._idem_index[order.idem_key] = entry.broker_order_id
        self._orders.append(order)
        self._emit_order(order)
        return entry.broker_order_id

    def cancel_order(self, broker_order_id: str) -> bool:
        entry = self._open.pop(broker_order_id, None)
        if entry is None:
            return False
        self._release(entry)
        entry.order.cancel("人工撤单")
        self._emit_order(entry.order)
        return True

    # ============================================================ 模拟盘专属：行情推进
    def push_bars(self, bars: Mapping[str, Bar]) -> list[Trade]:
        """推进行情：用当日 bar 撮合所有在途委托，返回本次成交。"""
        trades: list[Trade] = []
        for bar in bars.values():
            if bar.close > 0:
                self._last_prices[bar.code] = bar.close
        for broker_order_id in list(self._open):
            entry = self._open[broker_order_id]
            trades.extend(self._match(entry, bars.get(entry.order.code)))
        return trades

    def push_bar(self, bar: Bar) -> list[Trade]:
        return self.push_bars({bar.code: bar})

    def roll_to_next_day(self) -> None:
        """日切：T+1 解冻（昨日买入转为可卖）。"""
        for position in self._positions.values():
            position.roll_to_next_day()
        self._account.trade_date = date.today() if self._account.trade_date is None else self._account.trade_date

    # ============================================================ 内部：冻结与释放
    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerUnavailable("券商未连接，禁止报单", detail={"adapter": self._name})

    def _reference_price(self, order: Order) -> Decimal:
        """冻结资金所需的参考价：限价单用限价，市价单用最近一次行情价。"""
        if order.price is not None:
            return order.price
        price = self._last_prices.get(order.code)
        if price is None or price <= 0:
            raise OrderRejected(
                "市价单需要已知参考价才能冻结资金（请先推送该标的行情）", detail={"code": order.code}
            )
        return price

    def _freeze(self, entry: _OpenOrder) -> None:
        """按整单冻结资源：买入冻结资金（含预估费用），卖出冻结可卖数量。"""
        order = entry.order
        if order.side is Side.BUY:
            price = self._reference_price(order)
            amount = round_cost(price * order.qty)
            fees = self._fee_model.estimate(amount, Side.BUY)
            total = amount + fees.total  # total 是 property，不是方法
            # 先真正冻结成功再记账：若先记账后失败，回滚时会去解冻一笔从未冻结的金额
            self._account.freeze_cash(total)
            entry.frozen_cash = total
            return

        position = self._positions.get(order.code)
        available = position.avail_qty if position is not None else 0
        if position is None or available < order.qty:
            raise InsufficientPosition(
                "可卖数量不足（注意 T+1：当日买入不可卖）",
                detail={"code": order.code, "requested": order.qty, "available": available},
            )
        position.freeze(order.qty)
        entry.frozen_qty = order.qty

    def _release(self, entry: _OpenOrder) -> None:
        if entry.frozen_cash > 0:
            self._account.unfreeze_cash(entry.frozen_cash)
            entry.frozen_cash = _ZERO
        if entry.frozen_qty > 0:
            position = self._positions.get(entry.order.code)
            if position is not None:
                position.unfreeze(entry.frozen_qty)
            entry.frozen_qty = 0

    # ============================================================ 内部：撮合与结算
    def _match(self, entry: _OpenOrder, bar: Bar | None) -> list[Trade]:
        order = entry.order
        if not order.is_open:
            self._open.pop(entry.broker_order_id, None)
            return []
        plan = OrderPlan(
            code=order.code,
            strategy_id=order.strategy_id,
            trade_date=self._account.trade_date or date.today(),
            side=order.side,
            target_qty=order.remaining_qty,
            limit_price=order.price,
            plan_id=order.plan_id or order.order_id,
        )
        account_view, positions_view = self._matching_views(entry)
        attempt = self._matcher.attempt(plan, bar, account=account_view, positions=positions_view)

        if attempt.status is FillStatus.EXPIRED:
            self._open.pop(entry.broker_order_id, None)
            self._release(entry)
            order.cancel(f"撮合判定过期：{attempt.reason or '顺延超限'}")
            self._emit_order(order)
            return []

        if attempt.status is FillStatus.DEFERRED or not attempt.has_fill:
            # 撮合器把"本根不成交"（停牌、一字板、限价未触及、资金不足）统一报为 DEFERRED，
            # 由本层按天数决定何时放弃——这与回测引擎的处理一致
            entry.defer_days += 1
            if entry.defer_days > self._max_defer_days:
                self._open.pop(entry.broker_order_id, None)
                self._release(entry)
                order.cancel(f"顺延超过 {self._max_defer_days} 日未成交")
                self._emit_order(order)
            return []

        assert attempt.price is not None  # has_fill 保证
        trade = self._settle(entry, attempt.qty, attempt.price)
        if order.remaining_qty <= 0:
            self._open.pop(entry.broker_order_id, None)
            self._release(entry)  # 归还剩余冻结（费用估算的零头）
        self._emit_order(order)
        self._emit_trade(trade)
        return [trade]

    def _matching_views(self, entry: _OpenOrder) -> tuple[Account, dict[str, Position]]:
        """给撮合器的"把本笔委托的冻结额度放回去"的视图。

        撮合器判断上限用的是 ``avail_qty``（可卖）与 ``available_cash``（可用资金），
        而本笔委托在报单时**已经**把额度冻结了——不还原就等于自己吃掉自己的额度：
        实测表现为"冻结 600 股去卖，只成交 400 股"，以及大额买单永远只能部分成交。
        """
        positions_view = self._positions
        if entry.frozen_qty > 0:
            position = self._positions.get(entry.order.code)
            if position is not None:
                mirror = copy.copy(position)
                mirror.avail_qty += entry.frozen_qty
                mirror.frozen_qty -= entry.frozen_qty
                positions_view = {**self._positions, entry.order.code: mirror}
        if entry.frozen_cash > 0:
            account_view = copy.copy(self._account)
            account_view.frozen_cash -= entry.frozen_cash
            return account_view, positions_view
        return self._account, positions_view

    def _settle(self, entry: _OpenOrder, qty: int, price: Decimal) -> Trade:
        order = entry.order
        amount = round_cost(price * qty)
        fees = self._fee_model.estimate(amount, order.side)
        if order.side is Side.BUY:
            # 成交部分先把冻结额度解掉，再按实际金额扣款（额度是按整单预冻的）
            ratio = Decimal(qty) / Decimal(entry.order.remaining_qty)
            released = round_cost(entry.frozen_cash * ratio)
            self._account.unfreeze_cash(released)
            entry.frozen_cash -= released
            self._account.settle_cash(-(amount + fees.total))
            position = self._positions.get(order.code)
            if position is None:
                position = Position(code=order.code, account_id=self._account.account_id)
                self._positions[order.code] = position
            position.apply_buy(qty, price)
        else:
            position = self._positions.get(order.code)
            if position is None:
                raise BrokerError("卖出成交但无持仓记录", detail={"code": order.code})
            position.unfreeze(qty)  # 先解冻再卖出（apply_sell 校验的是 avail_qty）
            entry.frozen_qty -= qty
            position.apply_sell(qty, price)
            self._account.settle_cash(amount - fees.total)
            if position.qty == 0:
                self._positions.pop(order.code, None)

        order.apply_fill(qty, price)
        trade = Trade(
            code=order.code,
            side=order.side,
            qty=qty,
            price=price,
            order_id=order.order_id,
            strategy_id=order.strategy_id,
            commission=fees.commission,
            stamp_tax=fees.stamp_tax,
            transfer_fee=fees.transfer_fee,
            trade_time=datetime.now(),
        )
        self._trades.append(trade)
        return trade

    def _mark_to_market(self) -> None:
        for code, position in self._positions.items():
            price = self._last_prices.get(code)
            if price is not None and price > 0:
                position.last_price = price
        self._account.mark_to_market(self._positions.values())
        self._account.refresh_nav()

    def _emit_order(self, order: Order) -> None:
        for callback in self._order_callbacks:
            callback(copy.deepcopy(order))

    def _emit_trade(self, trade: Trade) -> None:
        for callback in self._trade_callbacks:
            callback(trade)


__all__ = ["DEFAULT_MAX_DEFER_DAYS", "PaperBrokerAdapter"]
