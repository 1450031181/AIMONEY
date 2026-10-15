"""盘前/盘后调度命令行（P4-3）：`aimoney-schedule`。

用法::

    aimoney-schedule once --job pre  --date 2026-10-15
    aimoney-schedule once --job post --date 2026-10-15 --strategy strategies/multi_factor_enhance
    aimoney-schedule plan --days 5

``once`` 由**外部调度器**调用（Windows 任务计划 / cron / APScheduler），
进程跑完即退出，不需要常驻服务。``plan`` 用来核对接下来哪些天会真正触发——
交易日历守卫失效（例如日历数据没同步）时会在这里暴露，而不是等到跑了一堆空任务才发现。

**退出码**：``0`` 正常 / ``1`` 有步骤失败 / ``3`` 跳过（非交易日、未指定策略、该交易日已处理）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from app.cli import load_settings, parse_date, print_table
from app.core.logging import setup_logging
from app.services.schedule_service import (
    JobKind,
    ScheduleService,
    ensure_strategy_dir,
)
from engine.errors import AimoneyError
from engine.ports import DataSource

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimoney-schedule", description="AIMoney 盘前/盘后调度")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--root", help="数据根目录（覆盖 data.root）")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    once = subparsers.add_parser("once", help="执行一次盘前或盘后任务链（由外部调度器调用）")
    once.add_argument("--job", required=True, choices=[job.value for job in JobKind])
    once.add_argument("--date", help="交易日 YYYY-MM-DD（默认今天）")
    once.add_argument("--strategy", help="盘后要跑的模拟盘策略目录")
    once.add_argument("--codes", action="append", help="限定标的：逗号分隔或重复传参")
    once.add_argument("--params", action="append", help="策略参数覆盖，形如 key=value")
    once.add_argument("--factors", action="append", help="因子（多因子策略必需）")
    once.add_argument("--dataset", default="daily")
    once.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan", help="查看接下来会触发的交易日（核对日历守卫）")
    plan.add_argument("--days", type=int, default=5, help="查看未来多少个交易日（默认 5）")
    plan.add_argument("--start", help="起点 YYYY-MM-DD（默认今天）")
    plan.add_argument("--dataset", default="daily")
    plan.add_argument("--json", action="store_true")
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
        return _cmd_plan(args, settings) if args.command == "plan" else _cmd_once(args, settings)
    except AimoneyError as exc:
        # str(exc) 携带 detail（含 hint），排障信息必须完整输出
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_FAILED


def _cmd_once(args: argparse.Namespace, settings: Any) -> int:
    day = parse_date(getattr(args, "date", None)) or date.today()
    strategy_dir = ensure_strategy_dir(getattr(args, "strategy", None)) if args.strategy else None
    service = ScheduleService(
        settings=settings,
        source_factory=_source_factory(settings),
        dataset=str(args.dataset),
        strategy_dir=strategy_dir,
        codes=tuple(_split(args.codes)),
        params_override=_parse_params(getattr(args, "params", None)),
        factors=tuple(_split(args.factors)),
    )
    report = service.run(JobKind(str(args.job)), day=day)

    if args.json:
        print(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        if report.steps:
            print_table(
                ["步骤", "状态", "说明", "耗时(ms)"],
                [[step.name, step.status.value, step.detail, str(step.duration_ms)] for step in report.steps],
            )

    if report.skipped:
        return EXIT_SKIPPED
    return EXIT_OK if report.ok else EXIT_FAILED


def _cmd_plan(args: argparse.Namespace, settings: Any) -> int:
    start = parse_date(getattr(args, "start", None)) or date.today()
    service = ScheduleService(
        settings=settings, source_factory=_source_factory(settings), dataset=str(args.dataset)
    )
    days = service.upcoming_run_days(start=start, days=int(args.days))

    if args.json:
        print(json.dumps({"start": start.isoformat(), "days": [day.isoformat() for day in days]}, ensure_ascii=False))
        return EXIT_OK

    if not days:
        horizon = service.last_known_session()
        print(f"日历覆盖到 {horizon}，{start} 之后没有可调度的交易日")
        return EXIT_OK
    print_table(
        ["计划触发日", "距今(自然日)"],
        [[day.isoformat(), str((day - start).days)] for day in days],
    )
    return EXIT_OK


# ============================================================ 辅助
def _source_factory(settings: Any) -> Callable[[], DataSource]:
    """复用数据 CLI 的数据源装配（单一构造点，避免两处参数不一致）。"""
    from app.cli import build_source

    return lambda: build_source(settings, argparse.Namespace(no_factor=False))


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
