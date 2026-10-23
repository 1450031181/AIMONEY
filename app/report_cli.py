"""报告命令行（P4-4）：`aimoney-report`。

用法::

    aimoney-report daily  --strategy strategies/multi_factor_enhance --date 2026-10-15
    aimoney-report period --strategy strategies/multi_factor_enhance --start 2026-10-12 --end 2026-10-16

报告同时写两份到 ``data/reports/<策略>/``（``.md`` 给人看、``.json`` 给程序读），
默认把 Markdown 打到 stdout，方便直接重定向或喂给其它工具。

**退出码**：``0`` 正常 / ``1`` 出错（没有账本、区间不足都会在这里明确报出来）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.cli import load_settings, parse_date, print_table
from app.core.logging import setup_logging
from app.services.report_service import (
    DailyReportRequest,
    PeriodReportRequest,
    ReportService,
)
from engine.errors import AimoneyError

EXIT_OK = 0
EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimoney-report", description="AIMoney 持仓与绩效报告")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--root", help="数据根目录（覆盖 data.root）")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("daily", help="某交易日的持仓/成交/归因日报")
    daily.add_argument("--strategy", required=True, help="策略目录（与模拟盘一致）")
    daily.add_argument("--date", help="交易日 YYYY-MM-DD（默认今天）")
    daily.add_argument("--json", action="store_true", help="以 JSON 输出（便于程序处理）")

    period = subparsers.add_parser("period", help="区间的绩效、交易成本与归因")
    period.add_argument("--strategy", required=True, help="策略目录（与模拟盘一致）")
    period.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    period.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    period.add_argument("--rf", type=float, default=0.02, help="无风险利率（年化，默认 2%%）")
    period.add_argument("--json", action="store_true")
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
    try:
        return _cmd_daily(args, settings) if args.command == "daily" else _cmd_period(args, settings)
    except AimoneyError as exc:
        # str(exc) 携带 detail（含 hint），排障信息必须完整输出
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_FAILED


def _cmd_daily(args: argparse.Namespace, settings: Any) -> int:
    from datetime import date

    day = parse_date(getattr(args, "date", None)) or date.today()
    report = ReportService(settings).daily(
        DailyReportRequest(strategy_dir=Path(str(args.strategy)), trade_date=day)
    )
    if args.json:
        payload = report.to_jsonable()
        payload["output_dir"] = str(report.output_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report.to_markdown(), end="")
        print_table(["项目", "数值"], [["报告目录", str(report.output_dir)]])
    return EXIT_OK


def _cmd_period(args: argparse.Namespace, settings: Any) -> int:
    start = parse_date(str(args.start))
    end = parse_date(str(args.end))
    if start is None or end is None:
        raise AimoneyError(
            "缺少或无法解析日期参数", detail={"start": str(args.start), "end": str(args.end)}
        )
    report = ReportService(settings).period(
        PeriodReportRequest(
            strategy_dir=Path(str(args.strategy)),
            start=start,
            end=end,
            risk_free_rate=float(args.rf),
        )
    )
    if args.json:
        payload = report.to_jsonable()
        payload["output_dir"] = str(report.output_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report.to_markdown(), end="")
        print_table(["项目", "数值"], [["报告目录", str(report.output_dir)]])
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
