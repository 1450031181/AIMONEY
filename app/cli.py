"""命令行入口（T-13）：`aimoney-data <子命令>`。

为什么 CLI 放在 app 层而不是 engine：它要读配置（``Settings``）、要初始化日志、
要按环境选择数据源，这些都是应用层职责；engine 不依赖 app 是硬约束。

**退出码是设计的一部分**（定时任务与 CI 靠它判断）：
- ``0`` 正常
- ``1`` 数据侧问题（同步有失败标的 / 质量判定 FAIL）
- ``2`` 用法错误

子命令::

    aimoney-data list                                 列出本地所有数据集及覆盖情况
    aimoney-data coverage --dataset daily [--json]    单个数据集详细覆盖 + 交易门禁
    aimoney-data sync --start 2018-01-01 [--full]     日线增量同步
    aimoney-data index-members                        指数成分快照（按抓取日打点）
    aimoney-data check --date 2026-10-14              数据质量校验（写报告并开关门禁）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from engine.datahub.quality import QualityCheckTask, QualityStatus, QualityThresholds, TradeGate
from engine.datahub.reader import DuckDBReader
from engine.datahub.reference import ReferenceStore
from engine.datahub.store import ParquetStore
from engine.datahub.sync import DEFAULT_INDEX_CODES, DailySyncTask, IndexMembersSyncTask
from engine.datahub.tasklog import JsonlTaskLog
from engine.datahub.universe import JsonUniverseStore, UniverseFilter, select_universe
from engine.datasource.akshare_source import AkShareDataSource
from engine.errors import AimoneyError
from engine.ports import DataSource

EXIT_OK = 0
EXIT_DATA_PROBLEM = 1
EXIT_USAGE = 2

DEFAULT_DATASET = "daily"
_TASKS_LOG = "tasks.jsonl"
_UNIVERSE_FILE = "universe.json"


# ============================================================ 解析
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimoney-data",
        description="AIMoney 数据工具（行情同步 / 成分快照 / 质量校验 / 覆盖查询）",
    )
    parser.add_argument("--config", help="配置文件路径（默认 conf/aimoney.yaml）")
    parser.add_argument("--root", help="数据根目录（覆盖配置项 data.root）")
    parser.add_argument("--verbose", action="store_true", help="输出 DEBUG 日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="列出本地所有数据集及覆盖情况")

    coverage = subparsers.add_parser("coverage", help="查看单个数据集覆盖情况与交易门禁")
    coverage.add_argument("--dataset", default=DEFAULT_DATASET)
    coverage.add_argument("--json", action="store_true", help="以 JSON 输出")

    sync = subparsers.add_parser("sync", help="日线增量同步")
    sync.add_argument("--dataset", default=DEFAULT_DATASET)
    sync.add_argument("--start", help="起始日期 YYYY-MM-DD（首次同步必填）")
    sync.add_argument("--end", help="结束日期 YYYY-MM-DD（默认今天）")
    # action="append" + 逗号分隔双支持：PowerShell 会把 `a,b` 当数组、把 `000001` 解析成 `1`，
    # 因此推荐写成 --codes 600000 --codes 000001（或加引号），避免外壳吃掉前导零
    sync.add_argument("--codes", action="append", help="标的列表：逗号分隔或重复传参（默认全部在市标的）")
    sync.add_argument("--index", action="append", help="按指数成分限定标的（需已有成分快照）")
    sync.add_argument("--trade-date", help="锚定交易日 YYYY-MM-DD（用于补数据与测试）")
    sync.add_argument("--full", action="store_true", help="显式确认全量回填（配合 --start）")
    sync.add_argument("--no-factor", action="store_true", help="跳过真实复权因子拉取（会多省一次请求）")

    members = subparsers.add_parser("index-members", help="抓取指数当前成分并按抓取日打点")
    members.add_argument("--index", action="append", help="指数代码：逗号分隔或重复传参（默认沪深300/中证500/中证1000）")
    members.add_argument("--date", help="生效日 YYYY-MM-DD（默认今天）")

    check = subparsers.add_parser("check", help="数据质量校验（写报告 + 开关交易门禁）")
    check.add_argument("--date", help="校验日期 YYYY-MM-DD（默认今天）")
    check.add_argument("--dataset", default=DEFAULT_DATASET)
    check.add_argument("--index", action="append", help="以该指数当日成分作为期望标的，启用缺失率检查")
    check.add_argument("--no-gate", action="store_true", help="只出报告，不改动交易门禁")
    check.add_argument("--json", action="store_true", help="以 JSON 输出报告")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = load_settings(args)
    setup_logging(
        level="DEBUG" if args.verbose else settings.log.level,
        json_output=False,
        log_dir=settings.log_dir,
        force=True,
    )

    handlers = {
        "list": _cmd_list,
        "coverage": _cmd_coverage,
        "sync": _cmd_sync,
        "index-members": _cmd_index_members,
        "check": _cmd_check,
    }
    try:
        return handlers[args.command](args, settings)
    except AimoneyError as exc:
        # str(exc) 携带 detail（含 hint），排障信息必须完整输出
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_DATA_PROBLEM
    except KeyboardInterrupt:  # pragma: no cover
        print("已中断", file=sys.stderr)
        return EXIT_DATA_PROBLEM


def load_settings(args: argparse.Namespace) -> Settings:
    if args.config:
        config_path = Path(str(args.config)).expanduser()
        if not config_path.is_file():
            raise AimoneyError(f"配置文件不存在：{config_path}")
        os.environ["AIMONEY_CONFIG_FILE"] = str(config_path)
    if args.root:
        return Settings(data={"root": str(Path(str(args.root)).expanduser())})
    return get_settings()


# ============================================================ 子命令
def _cmd_list(_args: argparse.Namespace, settings: Settings) -> int:
    """扫描数据目录下的所有数据集（新增数据集无需改代码即可被发现）。

    签名与其它子命令保持一致（``(args, settings)``），便于统一分发。
    """
    root = settings.parquet_dir
    datasets = sorted(path.name for path in root.glob("*") if path.is_dir()) if root.exists() else []
    if not datasets:
        print(f"数据目录下没有任何数据集：{root}")
        print("提示：先执行 aimoney-data sync --start <起始日期> 做首次回填")
        return EXIT_OK

    rows: list[list[str]] = []
    with DuckDBReader(root) as reader:
        for dataset in datasets:
            attributes = reader.attributes(dataset)
            rows.append(
                [
                    dataset,
                    str(attributes["row_count"]),
                    str(attributes["code_count"]),
                    str(attributes["start"] or "-"),
                    str(attributes["end"] or "-"),
                    str(attributes["partitions"]),
                    str(attributes["data_version"] or "-"),
                ]
            )
    print_table(["数据集", "行数", "标的数", "起始", "结束", "分区", "数据版本"], rows)
    return EXIT_OK


def _cmd_coverage(args: argparse.Namespace, settings: Settings) -> int:
    service = _service(settings)
    payload = service.coverage_payload(str(args.dataset))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    dataset = payload["dataset"]
    gate = payload["gate"]
    print_table(
        ["数据集", "行数", "标的数", "起始", "结束", "分区", "数据版本", "最新同步"],
        [
            [
                str(dataset["dataset"]),
                str(dataset["row_count"]),
                str(dataset["code_count"]),
                str(dataset["start"] or "-"),
                str(dataset["end"] or "-"),
                str(dataset["partitions"]),
                str(dataset["data_version"] or "-"),
                str(dataset["updated_at"] or "-"),
            ]
        ],
    )
    print(f"交易门禁：{'开启' if gate['open'] else '关闭'}（{gate['reason'] or '-'}）")
    return EXIT_OK


def _cmd_sync(args: argparse.Namespace, settings: Settings) -> int:
    if args.full and not args.start:
        print("--full 必须与 --start 一起使用（显式声明回填起点）", file=sys.stderr)
        return EXIT_USAGE

    source = build_source(settings, args)
    store = ParquetStore(settings.parquet_dir)
    task = DailySyncTask(
        source=source,
        store=store,
        task_log=_task_log(settings),
        dataset=str(args.dataset),
        calendar=None,
        full_backfill_start=parse_date(args.start) if args.full else None,
        reference_store=_reference_store(settings),
    )
    codes = _resolve_codes(args, settings)
    result = task.run(
        codes=codes,
        start=parse_date(args.start),
        end=parse_date(args.end),
        trade_date=parse_date(args.trade_date),
    )

    print(f"同步结果：{result.summary()}")
    if result.empty_codes:
        print(f"无数据标的（{len(result.empty_codes)} 只）：{', '.join(result.empty_codes[:10])}")
    if result.failed:
        print("失败标的：", file=sys.stderr)
        for failure in result.failed[:20]:
            print(f"  {failure.code}: {failure.reason}", file=sys.stderr)
        return EXIT_DATA_PROBLEM
    return EXIT_OK


def _cmd_index_members(args: argparse.Namespace, settings: Settings) -> int:
    source = build_source(settings, args)
    universe_store = JsonUniverseStore(settings.data.root_dir / "universe" / _UNIVERSE_FILE)
    task = IndexMembersSyncTask(
        source=source,
        history=universe_store.load(),
        store=universe_store,
        task_log=_task_log(settings),
        index_codes=_split(args.index) or DEFAULT_INDEX_CODES,
    )
    result = task.run(on=parse_date(args.date))

    for index_code, count in result.added.items():
        print(f"{index_code}: 登记 {count} 只成分（生效日 {result.as_of}）")
    if result.skipped:
        print(f"当日已有快照，跳过：{', '.join(result.skipped)}")
    if result.empty:
        print(f"成分返回为空：{', '.join(result.empty)}", file=sys.stderr)
    if result.failed:
        for failure in result.failed:
            print(f"{failure.code}: {failure.reason}", file=sys.stderr)
    return EXIT_OK if result.ok else EXIT_DATA_PROBLEM


def _cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    as_of = parse_date(args.date) or date.today()
    expected: Sequence[str] | None = None
    if args.index:
        expected = _index_universe(settings, str(args.index), as_of)

    task = QualityCheckTask(
        store=ParquetStore(settings.parquet_dir),
        gate=TradeGate(settings.gate_file),
        task_log=_task_log(settings),
        report_dir=settings.quality_dir,
        thresholds=QualityThresholds.from_mapping(settings.data.quality.model_dump()),
        dataset=str(args.dataset),
        close_gate_on_fail=not args.no_gate,
    )
    report = task.run(as_of=as_of, expected_codes=expected)

    if args.json:
        print(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        for finding in report.findings:
            print(f"  [{finding.severity}] {finding.rule}：{finding.message}")
    if report.status is QualityStatus.FAIL:
        print("交易门禁已关闭" if not args.no_gate else "（--no-gate，未改动门禁）", file=sys.stderr)
        return EXIT_DATA_PROBLEM
    return EXIT_OK


# ============================================================ 辅助
def build_source(settings: Settings, args: argparse.Namespace) -> DataSource:
    """构造数据源。抽成独立函数便于测试注入替身（CLI 不该在测试里打外网）。"""
    return AkShareDataSource(
        request_interval=settings.data.request_interval,
        max_retries=settings.data.max_retries,
        retry_backoff=settings.data.retry_backoff,
        with_adj_factor=not bool(getattr(args, "no_factor", False)),
    )


def _service(settings: Settings) -> Any:
    from app.services.data_service import DataService

    return DataService(settings=settings)


def _task_log(settings: Settings) -> JsonlTaskLog:
    return JsonlTaskLog(settings.log_dir / _TASKS_LOG)


def _resolve_codes(args: argparse.Namespace, settings: Settings) -> list[str] | None:
    explicit = _split(getattr(args, "codes", None))
    if explicit:
        return explicit
    index_codes = _split(getattr(args, "index", None))
    if index_codes:
        as_of = (
            parse_date(getattr(args, "trade_date", None))
            or parse_date(getattr(args, "end", None))
            or date.today()
        )
        codes: set[str] = set()
        for index_code in index_codes:
            codes.update(_index_universe(settings, index_code, as_of))
        return sorted(codes)
    return None


def _index_universe(settings: Settings, index_code: str, as_of: date) -> Sequence[str]:
    history = JsonUniverseStore(settings.data.root_dir / "universe" / _UNIVERSE_FILE).load()
    # 基础信息取本地参考数据（由 sync 顺带落盘），保证 check/筛选不依赖网络
    stock_info = _reference_store(settings).load_stock_info()
    selection = select_universe(
        history,
        as_of=as_of,
        stock_info=stock_info,
        universe_filter=UniverseFilter(index_code=index_code),
    )
    return selection.codes


def _reference_store(settings: Settings) -> ReferenceStore:
    return ReferenceStore(settings.data.root_dir / "reference")


def _split(value: str | Sequence[str] | None) -> list[str]:
    """把 ``--xxx a,b`` 与 ``--xxx a --xxx b`` 两种写法统一成列表。

    之所以两种都支持：PowerShell 会把 ``600000,000001`` 当数组且把 ``000001`` 解析成 ``1``，
    只支持逗号写法时前导零会被外壳悄悄吃掉——这类"参数变了但程序不报错"的问题极难排查。
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    result: list[str] = []
    for item in items:
        result.extend(part.strip() for part in str(item).split(",") if part.strip())
    return result


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise AimoneyError(f"日期格式非法（应为 YYYY-MM-DD）：{value}", detail={"value": str(value)}) from exc


def _display_width(text: str) -> int:
    """终端显示宽度：中日韩字符占两列，用 len() 排表会错位。"""
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([_display_width(str(header)), *(_display_width(row[index]) for row in cells)])
        for index, header in enumerate(headers)
    ]
    line = "  ".join(_pad(str(header), width) for header, width in zip(headers, widths, strict=True))
    print(line)
    print("-" * _display_width(line))
    for row in cells:
        print("  ".join(_pad(cell, width) for cell, width in zip(row, widths, strict=True)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
