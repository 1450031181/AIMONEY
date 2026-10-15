"""P4-2 模拟盘运行器测试。

核心是**跨日/跨进程的正确性**——这类错误在单日运行里完全看不出来：

1. 重启后不能重复下单（幂等键必须持久化）；
2. 账户与持仓必须延续（否则净值曲线每天从头开始）；
3. 门禁未开/急停时必须"不交易但留证据"，且与"故障"用不同退出码区分。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from app.core.config import Settings
from app.paper_cli import EXIT_OK, EXIT_SKIPPED, main
from app.services.paper_service import PaperRunRequest, PaperTradingService
from engine.datahub.quality import TradeGate
from engine.datahub.store import ParquetStore
from engine.risk.kill_switch import KillSwitch
from engine.trade.ledger import DEFAULT_LEDGER_FILE, JsonLedger
from engine.types import OrderStatus

DAY1 = date(2026, 10, 12)
DAY2 = date(2026, 10, 13)
DAY3 = date(2026, 10, 14)
CODES = ("600000", "000001")

_STRATEGY_BODY = """
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class HalfPosition(BaseStrategy):
    \"\"\"每个调仓日把目标权重固定为 30%：便于断言数量与仓位。\"\"\"

    def on_bar(self, context, bars):
        if not bars:
            return []
        code = sorted(bars)[0]
        return [
            Signal(
                code=code,
                strategy_id=self.strategy_id,
                trade_date=context.trade_date,
                direction=Direction.BUY,
                target_weight=0.30,
                reason="固定 30% 仓位",
            )
        ]


def create_strategy(params):
    return HalfPosition(strategy_id=str(params.get("strategy_id", "half_position")), params=params)
"""


def _frame() -> pd.DataFrame:
    rows = []
    for offset in range(6):
        day = DAY1 + timedelta(days=offset)
        for index, code in enumerate(CODES):
            price = 10.0 + index * 5
            rows.append(
                {
                    "code": code,
                    "date": day.isoformat(),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 5_000_000,
                    "amount": price * 5_000_000,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # 放宽两项默认限额，让"撮合/续跑"类用例聚焦自身逻辑：
    # 单票上限默认 20%、单笔金额上限默认 20 万，二者都会截断 30% 仓位的样本单
    # （20,000 股 × 15 元 = 30 万）。风控生效本身另有专门用例。
    resolved = Settings(
        data={"root": str(tmp_path / "data")},
        risk={"max_weight": 0.5, "max_order_value": "5000000"},
    )
    resolved.ensure_dirs()
    ParquetStore(resolved.parquet_dir).write("daily", _frame(), data_version="20261014.1")
    reference = resolved.data.root_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"code": code, "name": f"标的{i}", "is_st": False, "list_date": "2010-01-01", "delist_date": None}
            for i, code in enumerate(CODES)
        ]
    ).to_parquet(reference / "stock_info.parquet", index=False)
    # 门禁默认关闭：测试里显式打开，避免"忘了开门"变成隐式前提
    TradeGate(resolved.gate_file).open(as_of=DAY1, note="测试用")
    return resolved


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "strategies" / "half_position"
    directory.mkdir(parents=True)
    (directory / "strategy.py").write_text(_STRATEGY_BODY, encoding="utf-8")
    (directory / "config.yaml").write_text(yaml.safe_dump({"strategy_id": "half_position"}), encoding="utf-8")
    return directory


def _request(strategy_dir: Path, day: date, **overrides: Any) -> PaperRunRequest:
    options: dict[str, Any] = {"strategy_dir": strategy_dir, "trade_date": day, "codes": CODES}
    options.update(overrides)
    return PaperRunRequest(**options)  # type: ignore[arg-type]


# ============================================================ 基本链路
def test_跑一天并落盘(settings: Settings, strategy_dir: Path) -> None:
    report = PaperTradingService(settings).run(_request(strategy_dir, DAY1))

    assert report.ok, report.skipped_reason
    # 信号在 DAY1 收盘产生，最早 DAY2 成交
    assert report.matched_trades == 0
    assert len(report.submission.submitted) == 1

    output_dir = report.output_dir
    assert (output_dir / DEFAULT_LEDGER_FILE).is_file()
    assert (output_dir / "nav.jsonl").is_file()
    assert list(output_dir.glob("run_*.json"))


def test_次日撮合昨日挂单(settings: Settings, strategy_dir: Path) -> None:
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))

    second = service.run(_request(strategy_dir, DAY2))

    assert second.matched_trades == 1, "DAY1 的挂单必须在 DAY2 成交"
    assert second.positions, "成交后应有持仓"
    # 策略选 sorted(bars)[0] = 000001（价 15 元）→ 30% × 100 万 ÷ 15 = 20,000 股
    assert second.positions[0] == ("000001", 20_000, 0)


def test_T加1_次日才可卖(settings: Settings, strategy_dir: Path) -> None:
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))
    second = service.run(_request(strategy_dir, DAY2))

    assert second.positions[0][2] == 0, "DAY2 买入当日不可卖"

    third = service.run(_request(strategy_dir, DAY3))

    assert third.positions[0][2] == 20_000, "日切后（DAY3）应可卖"


# ============================================================ 跨日续跑
def test_账户与持仓跨日延续(settings: Settings, strategy_dir: Path) -> None:
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))
    service.run(_request(strategy_dir, DAY2))

    ledger = JsonLedger(settings.data.root_dir / "paper" / strategy_dir.name / DEFAULT_LEDGER_FILE)
    snapshot = ledger.load_broker()

    assert snapshot is not None
    assert ledger.exists
    assert snapshot.account.cash < Decimal("1000000"), "现金必须延续（而不是每天重置为初始资金）"
    assert snapshot.account.total_asset == snapshot.account.cash + snapshot.account.market_value


def test_重复跑同一天被日级防重拦下(settings: Settings, strategy_dir: Path) -> None:
    """重复跑同一天**不是幂等的**：上次的挂单会在同一天行情上再撮一次。

    因此运行器必须做日级防重（定时任务天然安全），而不是依赖幂等键——
    幂等键只能防"报出两笔委托"，防不住"把挂单再成交一次"。
    """
    service = PaperTradingService(settings)
    first = service.run(_request(strategy_dir, DAY1))

    second = service.run(_request(strategy_dir, DAY1))

    assert len(first.submission.submitted) == 1
    assert not second.ok
    assert "已处理过" in (second.skipped_reason or "")
    assert second.submission.submitted == ()
    assert second.matched_trades == 0, "被防重拦下时不得有任何成交"

    ledger = JsonLedger(settings.data.root_dir / "paper" / strategy_dir.name / DEFAULT_LEDGER_FILE)
    assert len(ledger.load_orders()) == 1, "账本里不应出现第二笔委托"


def test_强制重跑会覆盖防重并可能多撮一次(settings: Settings, strategy_dir: Path) -> None:
    """``--force`` 的真实代价：挂单会在同一天行情上再撮一次——这就是它不该进定时任务的原因。"""
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))

    forced = service.run(_request(strategy_dir, DAY1, force=True))

    assert forced.ok, "显式 --force 时应真的重跑"
    assert forced.matched_trades == 1, "被强制的挂单在同一天成交——这正是 force 的危险之处"


def test_风控上限在运行器中生效(settings: Settings, strategy_dir: Path) -> None:
    """策略要 30% 仓位，单票上限 5% 时必须被缩量（运行器与回测共用同一套风控）。"""
    tight = Settings(data={"root": str(settings.data.root_dir)}, risk={"max_weight": 0.05})
    tight.ensure_dirs()

    report = PaperTradingService(tight).run(_request(strategy_dir, DAY1))

    assert report.ok, report.skipped_reason
    # 5% × 100 万 ÷ 15 元 = 3333 → 整手 3300
    assert [order.qty for order in report.submission.submitted] == [3_300]


def test_成交状态会落回委托账本(settings: Settings, strategy_dir: Path) -> None:
    """回归：账本曾被加载**两次**（券商一份、OMS 一份）。

    券商的成交只改到其中一份对象，另一份永远停在 SUBMITTED，
    于是落盘账本里出现"早已成交却仍显示在途"的委托——真实环境里这会让人误以为
    还有挂单没处理，进而做出错误的人工干预。
    """
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))
    second = service.run(_request(strategy_dir, DAY2))
    assert second.matched_trades == 1

    orders = JsonLedger(settings.data.root_dir / "paper" / strategy_dir.name / DEFAULT_LEDGER_FILE).load_orders()
    filled = [order for order in orders if order.status is OrderStatus.FILLED]

    assert filled, "已成交的委托必须反映到落盘账本"
    assert all(order.filled_qty > 0 for order in filled)
    assert not [order for order in orders if order.broker_order_id is None], "在途委托必须带上券商委托号"


def test_净值历史逐日追加(settings: Settings, strategy_dir: Path) -> None:
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))
    service.run(_request(strategy_dir, DAY2))
    report = service.run(_request(strategy_dir, DAY3))

    records = [
        json.loads(line)
        for line in (report.output_dir / "nav.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 3
    assert [item["trade_date"] for item in records] == [DAY1.isoformat(), DAY2.isoformat(), DAY3.isoformat()]


# ============================================================ 闸门
def test_门禁未开时不交易但留报告(settings: Settings, strategy_dir: Path) -> None:
    TradeGate(settings.gate_file).close(as_of=DAY1, reason="质量校验 FAIL")

    report = PaperTradingService(settings).run(_request(strategy_dir, DAY1))

    assert not report.ok
    assert "门禁" in (report.skipped_reason or "")
    assert report.submission.submitted == ()
    assert list(report.output_dir.glob("run_*.json")), "跳过也必须留证据，便于回答'今天为什么没动'"


def test_急停激活时跳过(settings: Settings, strategy_dir: Path) -> None:
    KillSwitch(settings.kill_switch_file).activate("人工急停", by="test")

    report = PaperTradingService(settings).run(_request(strategy_dir, DAY1))

    assert not report.ok
    assert "急停" in (report.skipped_reason or "")


def test_交易日无行情时跳过(settings: Settings, strategy_dir: Path) -> None:
    report = PaperTradingService(settings).run(_request(strategy_dir, date(2026, 11, 30)))

    assert not report.ok
    assert "行情" in (report.skipped_reason or "")


# ============================================================ 因子
def test_策略声明因子时自动准备快照(settings: Settings, strategy_dir: Path) -> None:
    """运行器应读取策略配置里的 factors 并准备因子快照（否则多因子策略读不到值）。"""
    config_path = strategy_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump({"strategy_id": "half_position", "factors": ["mom_5"]}), encoding="utf-8")

    report = PaperTradingService(settings).run(_request(strategy_dir, DAY1))

    assert report.ok, report.skipped_reason


# ============================================================ CLI
def test_CLI跑一天(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "run",
            "--strategy",
            str(strategy_dir),
            "--date",
            DAY1.isoformat(),
            "--codes",
            ",".join(CODES),
            "--json",
        ]
    )

    assert code == EXIT_OK, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "output_dir" in payload


def test_CLI被闸门拦下时返回跳过码(
    settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    KillSwitch(settings.kill_switch_file).activate("测试急停", by="test")

    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "run",
            "--strategy",
            str(strategy_dir),
            "--date",
            DAY1.isoformat(),
            "--json",
        ]
    )

    assert code == EXIT_SKIPPED, "跳过必须与故障区分（否则监控会对告警脱敏）"
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_CLI查看账本状态(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    service = PaperTradingService(settings)
    service.run(_request(strategy_dir, DAY1))
    service.run(_request(strategy_dir, DAY2))

    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "status",
            "--strategy",
            str(strategy_dir),
            "--json",
        ]
    )

    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["has_state"] is True
    assert payload["positions"] and payload["nav_points"] == 2


def test_CLI策略参数类型会被转换(settings: Settings, strategy_dir: Path) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "run",
            "--strategy",
            str(strategy_dir),
            "--date",
            DAY1.isoformat(),
            "--codes",
            ",".join(CODES),
            "--params",
            "top_k=3",
            "--params",
            "enabled=true",
        ]
    )

    assert code == EXIT_OK


def test_CLI参数格式非法时报错(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "run",
            "--strategy",
            str(strategy_dir),
            "--date",
            DAY1.isoformat(),
            "--params",
            "no_equals_sign",
        ]
    )

    assert code == 1
    assert "key=value" in capsys.readouterr().err
