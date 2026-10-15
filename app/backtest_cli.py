"""回测命令行（P1-B6）：`aimoney-backtest`。

用法::

    aimoney-backtest --strategy strategies/dual_ma --start 2026-09-01 --end 2026-10-15 \\
        [--codes 600000 --codes 000001] [--cash 1000000] [--params fast=3 --params slow=10]

输出：控制台摘要 + ``data/backtests/<strategy_id>_<时间戳>/`` 下的
``metrics.json`` / ``nav.csv`` / ``trades.csv``。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from app.cli import load_settings, parse_date, print_table  # 复用数据 CLI 的解析与表格工具
from app.core.logging import setup_logging
from app.services.backtest_service import BacktestRequest, BacktestService
from engine.errors import AimoneyError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimoney-backtest", description="AIMoney 回测工具")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--root", help="数据根目录（覆盖 data.root）")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strategy", required=True, help="策略目录（含 strategy.py / config.yaml）")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--codes", action="append", help="限定标的（默认全部在市标的）")
    parser.add_argument("--cash", default="1000000", help="初始资金（默认 1000000）")
    parser.add_argument("--params", action="append", help="策略参数覆盖：key=value，可重复")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-vol-ratio", type=float, default=0.10)
    parser.add_argument("--max-delay-days", type=int, default=2)
    parser.add_argument("--dataset", default="daily")
    parser.add_argument(
        "--factors",
        action="append",
        help="提供给策略的因子（逗号分隔或重复传参）；多因子策略必需",
    )
    parser.add_argument(
        "--factor-preprocess",
        default="winsorize,rank,zscore",
        help="因子截面预处理步骤（传 none 表示不做）",
    )
    parser.add_argument(
        "--no-risk",
        action="store_true",
        help="关闭风控（仅研究调试用；实盘链路必须保持启用）",
    )
    parser.add_argument("--max-weight", type=float, default=None, help="单票集中度上限（覆盖配置值）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整指标")
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
        request = BacktestRequest(
            strategy_dir=Path(str(args.strategy)).expanduser(),
            start=_require_date(str(args.start), "start"),
            end=_require_date(str(args.end), "end"),
            initial_cash=str(args.cash),
            codes=tuple(_split_codes(args.codes)),
            params_override=_parse_params(args.params),
            slippage_bps=float(args.slippage_bps),
            max_vol_ratio=float(args.max_vol_ratio),
            max_delay_days=int(args.max_delay_days),
            dataset=str(args.dataset),
            factors=tuple(_split_codes(args.factors)),
            factor_preprocess=_parse_preprocess(str(args.factor_preprocess)),
            risk_enabled=not bool(args.no_risk),
            max_weight=float(args.max_weight) if args.max_weight is not None else None,
        )
        report = BacktestService(settings).run(request)
    except AimoneyError as exc:
        # str(exc) 会带上 detail（含 hint），排查"为什么跑不了"全靠它
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_FAILED

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "strategy_id": report.strategy_id,
                    "params": report.params,
                    "output_dir": str(report.output_dir),
                    "halted": report.result.halted,
                    "trading_days": report.result.trading_days,
                    "metrics": report.metrics.to_jsonable(),
                    "risk": {
                        "rejections": [
                            {"code": code, "reason": reason}
                            for code, reason in report.result.risk_rejections
                        ],
                        "shrinks": report.result.risk_shrinks,
                        "level_events": [
                            {"date": day, "level": level, "reason": reason}
                            for day, level, reason in report.result.risk_level_events
                        ],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK

    print(report.summary())
    metrics = report.metrics
    print_table(
        ["指标", "数值"],
        [
            ["总收益", f"{metrics.total_return:+.2%}"],
            ["年化收益", f"{metrics.annual_return:+.2%}"],
            ["年化波动", f"{metrics.annual_volatility:.2%}"],
            ["最大回撤", f"{metrics.max_drawdown:.2%}"],
            ["夏普比率", f"{metrics.sharpe_ratio:.2f}"],
            ["卡玛比率", f"{metrics.calmar_ratio:.2f}"],
            ["日胜率", f"{metrics.daily_win_rate:.2%}"],
            ["月度收益数", str(len(metrics.monthly_returns))],
        ],
    )
    print(f"报告目录：{report.output_dir}")
    if report.result.halted:
        print(f"⚠ 策略中途停仓：{report.result.halted}", file=sys.stderr)
    return EXIT_OK


def _require_date(value: str, name: str) -> date:
    parsed = parse_date(value)
    if parsed is None:
        raise AimoneyError(f"缺少日期参数 --{name}", detail={"argument": name, "given": value})
    return parsed


def _split_codes(values: list[str] | None) -> list[str]:
    codes: list[str] = []
    for value in values or []:
        codes.extend(part.strip() for part in str(value).split(",") if part.strip())
    return codes


def _parse_preprocess(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if text.lower() in ("none", "", "-"):
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _parse_params(values: list[str] | None) -> dict[str, Any]:
    """解析 ``key=value`` 形式的参数覆盖；数字自动转 int/float，其余保持字符串。"""
    params: dict[str, Any] = {}
    for value in values or []:
        key, _, raw = str(value).partition("=")
        if not key or not raw:
            raise AimoneyError(f"参数格式非法（应为 key=value）：{value}", detail={"value": str(value)})
        params[key.strip()] = _coerce(raw.strip())
    return params


def _coerce(raw: str) -> Any:
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
