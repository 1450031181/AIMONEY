"""模拟盘命令行（P4-2）：`aimoney-paper`。

用法::

    aimoney-paper run --strategy strategies/multi_factor_enhance --date 2026-10-15 [--json]
    aimoney-paper status --strategy strategies/multi_factor_enhance

``run`` 对**一个交易日**跑一次完整链路（盘前闸门 → 恢复账本 → 撮合昨日挂单 →
策略决策 → 风控 → 报单 → 盯市 → 落盘）。定时任务每天调用一次即可。

**退出码**：``0`` 正常 / ``1`` 出错 / ``3`` 跳过（门禁未开、急停、当日无行情）。
把"跳过"与"出错"分开是刻意的：门禁未开是预期内的状态，
若与故障共用一个退出码，监控很快就会对告警脱敏。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from app.cli import load_settings, parse_date, print_table
from app.core.logging import setup_logging
from app.services.paper_service import PaperRunRequest, PaperTradingService
from engine.errors import AimoneyError
from engine.trade.ledger import DEFAULT_LEDGER_FILE, JsonLedger

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimoney-paper", description="AIMoney 模拟盘运行器")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--root", help="数据根目录（覆盖 data.root）")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="跑一个交易日的完整模拟盘链路")
    run.add_argument("--strategy", required=True, help="策略目录（含 strategy.py / config.yaml）")
    run.add_argument("--date", help="交易日 YYYY-MM-DD（默认今天）")
    run.add_argument("--codes", action="append", help="限定标的：逗号分隔或重复传参（默认全部在市标的）")
    run.add_argument("--factors", action="append", help="因子（缺省取策略声明）")
    run.add_argument("--params", action="append", help="策略参数覆盖，形如 key=value")
    run.add_argument("--initial-cash", default="1000000", help="初始资金（仅首次运行生效）")
    run.add_argument(
        "--force",
        action="store_true",
        help="强制重跑已处理过的交易日（危险：上次挂单会在同一天行情上再撮一次，勿用于定时任务）",
    )
    run.add_argument("--dataset", default="daily")
    run.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status", help="查看模拟盘账本状态")
    status.add_argument("--strategy", required=True, help="策略目录")
    status.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = load_settings(args)
    setup_logging(
        level="DEBUG" if args.verbose else "WARNING",
        json_output=False,
        log_dir=settings.log_dir,
        force=True,
    )

    handlers = {"run": _cmd_run, "status": _cmd_status}
    try:
        return handlers[args.command](args, settings)
    except AimoneyError as exc:
        # str(exc) 携带 detail（含 hint），排障信息必须完整输出
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_FAILED


# ============================================================ 子命令
def _cmd_run(args: argparse.Namespace, settings: Any) -> int:
    trade_date = parse_date(getattr(args, "date", None)) or date.today()
    request = PaperRunRequest(
        strategy_dir=Path(str(args.strategy)),
        trade_date=trade_date,
        codes=tuple(_split(args.codes)),
        params_override=_parse_params(getattr(args, "params", None)),
        factors=tuple(_split(args.factors)),
        dataset=str(args.dataset),
        initial_cash=str(args.initial_cash),
        force=bool(args.force),
    )
    report = PaperTradingService(settings).run(request)

    if args.json:
        print(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        if report.positions:
            print_table(
                ["标的", "持仓", "可卖"],
                [[code, str(qty), str(avail)] for code, qty, avail in report.positions],
            )
        if not report.ok:
            print(f"\n（本次未交易，报告已写入 {report.output_dir}）")
    return EXIT_OK if report.ok else EXIT_SKIPPED


def _cmd_status(args: argparse.Namespace, settings: Any) -> int:
    strategy_dir = Path(str(args.strategy))
    output_dir = settings.data.root_dir / "paper" / strategy_dir.name
    ledger = JsonLedger(output_dir / DEFAULT_LEDGER_FILE)
    snapshot = ledger.load_broker()
    orders = ledger.load_orders()
    nav_history = _read_nav_history(output_dir)

    payload = {
        "strategy": strategy_dir.name,
        "ledger_file": str(output_dir / DEFAULT_LEDGER_FILE),
        "has_state": snapshot is not None,
        "cash": str(snapshot.account.cash) if snapshot else None,
        "total_asset": str(snapshot.account.total_asset) if snapshot else None,
        "nav": snapshot.account.nav if snapshot else None,
        "positions": (
            [{"code": p.code, "qty": p.qty, "avail_qty": p.avail_qty} for p in snapshot.positions]
            if snapshot
            else []
        ),
        "open_orders": sum(1 for order in orders if order.is_open),
        "orders": len(orders),
        "nav_points": len(nav_history),
        "last_nav": nav_history[-1] if nav_history else None,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    if snapshot is None:
        print(f"该策略还没有模拟盘账本（{output_dir}）")
        return EXIT_OK
    print_table(
        ["项目", "数值"],
        [
            ["现金", str(snapshot.account.cash)],
            ["总资产", str(snapshot.account.total_asset)],
            ["净值", f"{snapshot.account.nav:.4f}"],
            ["持仓", str(len(snapshot.positions))],
            ["委托", f"{len(orders)}（在途 {sum(1 for o in orders if o.is_open)}）"],
            ["净值记录", str(len(nav_history))],
        ],
    )
    return EXIT_OK


# ============================================================ 辅助
def _read_nav_history(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "nav.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _split(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(part.strip() for part in str(value).split(",") if part.strip())
    return result


def _parse_params(values: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise AimoneyError("策略参数格式非法，应为 key=value", detail={"given": item})
        key, raw = item.split("=", 1)
        parsed[key.strip()] = _coerce(raw.strip())
    return parsed


def _coerce(raw: str) -> Any:
    """把 CLI 字符串转成合适的类型（否则 `top_k=3` 会以字符串传给策略）。"""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
