"""因子命令行（P2-F3）：`aimoney-factor`。

用法::

    aimoney-factor list [--category momentum]             查看内置因子库
    aimoney-factor evaluate --factors mom_20,vol_20 \\
        --start 2026-09-01 --end 2026-10-30 [--horizon 1] [--quantiles 5] [--json]

``evaluate`` 产出控制台摘要 + ``data/factors/evaluate_<时间戳>/`` 下的
``summary.json``、``ic_<因子>.csv``、``quantile_<因子>.csv``、``correlation.csv``。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from app.cli import load_settings, parse_date, print_table
from app.core.logging import setup_logging
from app.services.factor_service import (
    FactorCombineRequest,
    FactorEvaluationRequest,
    FactorService,
)
from engine.errors import AimoneyError
from engine.factor.base import list_factors

EXIT_OK = 0
EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimoney-factor", description="AIMoney 因子工厂")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--root", help="数据根目录（覆盖 data.root）")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="列出内置因子库")
    listing.add_argument("--category", help="只看某一类目")
    listing.add_argument("--json", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="计算并评价因子")
    evaluate.add_argument("--factors", required=True, action="append", help="因子名，逗号分隔或重复传参")
    evaluate.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    evaluate.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    evaluate.add_argument("--codes", action="append", help="限定标的（默认全部在市标的）")
    evaluate.add_argument("--horizon", type=int, default=1, help="未来收益持有天数（默认 1）")
    evaluate.add_argument("--quantiles", type=int, default=5, help="分层数（默认 5）")
    evaluate.add_argument("--min-cross-section", type=int, default=5, help="单截面最少标的数")
    evaluate.add_argument(
        "--preprocess",
        default="winsorize,rank,zscore",
        help="预处理步骤，逗号分隔（可选 winsorize/rank/demean/zscore；传 none 表示不做预处理）",
    )
    evaluate.add_argument("--dataset", default="daily")
    evaluate.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")

    combine = subparsers.add_parser("combine", help="多因子合成与选股名单")
    combine.add_argument("--factors", required=True, action="append", help="因子名，逗号分隔或重复传参")
    combine.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    combine.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    combine.add_argument("--codes", action="append", help="限定标的（默认全部在市标的）")
    combine.add_argument("--method", default="icir", choices=["equal", "ic", "icir"], help="合成方法（默认 icir）")
    combine.add_argument("--weight-window", type=int, default=20, help="权重估计的滚动窗口（默认 20 个截面）")
    combine.add_argument("--top-k", type=int, default=10, help="选股名单数量（默认 10）")
    combine.add_argument("--correlation-threshold", type=float, default=0.7, help="去重相关性阈值（默认 0.7）")
    combine.add_argument("--min-abs-ic", type=float, default=0.0, help="|IC| 低于该值的因子权重置零")
    combine.add_argument("--horizon", type=int, default=1)
    combine.add_argument("--quantiles", type=int, default=5)
    combine.add_argument("--min-cross-section", type=int, default=5)
    combine.add_argument("--preprocess", default="winsorize,rank,zscore")
    combine.add_argument("--dataset", default="daily")
    combine.add_argument("--json", action="store_true")
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

    handlers = {"list": _cmd_list, "evaluate": _cmd_evaluate, "combine": _cmd_combine}
    try:
        return handlers[args.command](args, settings)
    except AimoneyError as exc:
        # str(exc) 含 detail（含 hint），排障信息必须完整输出
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_FAILED


# ============================================================ 子命令
def _cmd_list(args: argparse.Namespace, _settings: Any) -> int:
    category = getattr(args, "category", None)
    factors = [item for item in list_factors() if category is None or item.category == category]
    if not factors:
        print(f"没有类目为 {category} 的因子", file=sys.stderr)
        return EXIT_FAILED

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": item.name,
                        "category": item.category,
                        "window": item.window,
                        "description": item.description,
                    }
                    for item in factors
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    print_table(
        ["因子", "类目", "最少历史", "说明"],
        [[item.name, item.category, str(item.window), item.description] for item in factors],
    )
    print(f"共 {len(factors)} 个因子")
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace, settings: Any) -> int:
    request = FactorEvaluationRequest(
        factors=tuple(_split(args.factors)),
        start=_require_date(str(args.start), "start"),
        end=_require_date(str(args.end), "end"),
        codes=tuple(_split(getattr(args, "codes", None))),
        horizon=int(args.horizon),
        quantiles=int(args.quantiles),
        min_cross_section=int(args.min_cross_section),
        preprocess_steps=_parse_steps(str(args.preprocess)),
        dataset=str(args.dataset),
    )
    report = FactorService(settings).run(request)

    if args.json:
        payload = report.to_jsonable()
        payload["output_dir"] = str(report.output_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    print(report.summary())
    if not report.correlation.empty:
        print("\n因子相关性（平均截面秩相关）：")
        names = [str(column) for column in report.correlation.columns]
        print_table(
            ["", *names],
            [
                [str(row), *[f"{float(value):+.2f}" for value in report.correlation.loc[row]]]
                for row in report.correlation.index
            ],
        )
    print(f"\n报告目录：{report.output_dir}")
    return EXIT_OK


def _cmd_combine(args: argparse.Namespace, settings: Any) -> int:
    request = FactorCombineRequest(
        factors=tuple(_split(args.factors)),
        start=_require_date(str(args.start), "start"),
        end=_require_date(str(args.end), "end"),
        codes=tuple(_split(getattr(args, "codes", None))),
        horizon=int(args.horizon),
        quantiles=int(args.quantiles),
        min_cross_section=int(args.min_cross_section),
        preprocess_steps=_parse_steps(str(args.preprocess)),
        dataset=str(args.dataset),
        method=str(args.method),
        weight_window=int(args.weight_window),
        top_k=int(args.top_k),
        correlation_threshold=float(args.correlation_threshold),
        min_abs_ic=float(args.min_abs_ic),
    )
    report = FactorService(settings).run_combine(request)

    if args.json:
        payload = report.to_jsonable()
        payload["output_dir"] = str(report.output_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    print(report.summary())
    if not report.weights.empty:
        active = report.weights.tail(1).iloc[0]
        weights_text = "，".join(f"{name} {float(value):+.2f}" for name, value in active.items())
        print(f"  最新权重（{pd.Timestamp(report.weights.index[-1]).date()}）：{weights_text}")
    print(f"\n报告目录：{report.output_dir}")
    return EXIT_OK


# ============================================================ 辅助
def _split(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(part.strip() for part in str(value).split(",") if part.strip())
    return result


def _parse_steps(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if text.lower() in ("none", "", "-"):
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _require_date(value: str, name: str) -> date:
    parsed = parse_date(value)
    if parsed is None:
        raise AimoneyError(f"缺少日期参数 --{name}", detail={"argument": name, "given": value})
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
