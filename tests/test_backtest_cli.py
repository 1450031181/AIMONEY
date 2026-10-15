"""P1-B6 回测 CLI/服务测试：数据就绪 → 加载策略 → 跑回测 → 报告落盘。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.backtest_cli import main
from engine.datahub.store import ParquetStore

START = date(2026, 9, 1)
END = date(2026, 10, 30)

_STRATEGY_BODY = """
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class BuyAndHold(BaseStrategy):
    def on_bar(self, context, bars):
        if context.trade_date.isoformat() != "2026-09-01":
            return []
        return [
            Signal(
                code=code,
                strategy_id=self.strategy_id,
                trade_date=context.trade_date,
                direction=Direction.BUY,
                target_weight=0.95 / len(bars),
                reason="等权买入",
            )
            for code in sorted(bars)
        ]


def create_strategy(params):
    return BuyAndHold(strategy_id=str(params.get("strategy_id", "buy_and_hold")), params=params)
"""


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    store = ParquetStore(root / "parquet")
    rows = []
    for code, base in (("600000", 10.0), ("000001", 20.0)):
        for index in range(40):
            day = (pd.Timestamp(START) + pd.Timedelta(days=index)).date().isoformat()
            close = base + index * (0.05 if code == "600000" else 0.0)
            rows.append(
                {
                    "code": code,
                    "date": day,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                    "adj_factor": 1.0,
                }
            )
    store.write("daily", pd.DataFrame(rows), data_version="20261030.1")
    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"code": "600000", "name": "浦发银行", "is_st": False, "list_date": "1999-11-10", "delist_date": None},
            {"code": "000001", "name": "平安银行", "is_st": False, "list_date": "1991-04-03", "delist_date": None},
        ]
    ).to_parquet(reference / "stock_info.parquet", index=False)
    return root


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "strategies" / "buy_and_hold"
    directory.mkdir(parents=True)
    (directory / "strategy.py").write_text(_STRATEGY_BODY, encoding="utf-8")
    (directory / "config.yaml").write_text(yaml.safe_dump({"top_n_hint": 2}), encoding="utf-8")
    return directory


def _args(data_root: Path, strategy_dir: Path, *extra: str) -> list[str]:
    return ["--root", str(data_root), "--strategy", str(strategy_dir), "--start", START.isoformat(), "--end", END.isoformat(), *extra]


def test_回测跑通并产出报告文件(data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_args(data_root, strategy_dir))

    assert code == 0
    out = capsys.readouterr().out
    assert "总收益" in out and "报告目录" in out
    # 报告三件套齐全且可解析
    output_dir = Path(out.split("报告目录：")[-1].strip())
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy_id"] == "buy_and_hold"
    assert metrics["config"]["codes"] == "universe"
    assert metrics["metrics"]["total_return"] > 0, "600000 上行行情，买入持有应为正收益"
    nav = pd.read_csv(output_dir / "nav.csv")
    assert list(nav.columns) == ["date", "nav"]
    # CSV 无类型：股票代码必须按字符串读，否则 "000001" 会被推断成整数 1
    trades = pd.read_csv(output_dir / "trades.csv", dtype={"code": str})
    assert set(trades["code"]) == {"600000", "000001"}


def test_json输出包含完整指标(data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args(data_root, strategy_dir, "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy_id"] == "buy_and_hold"
    assert "sharpe_ratio" in payload["metrics"]
    assert payload["halted"] is None


def test_params覆盖进入策略并写入报告(data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args(data_root, strategy_dir, "--params", "strategy_id=renamed", "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy_id"] == "renamed"
    metrics = json.loads(
        (Path(payload["output_dir"]) / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["params"]["strategy_id"] == "renamed"


def test_本地无数据时拒绝回测(tmp_path: Path, strategy_dir: Path) -> None:
    empty_root = tmp_path / "empty_data"
    empty_root.mkdir()
    code = main(_args(empty_root, strategy_dir))

    assert code == 1


def test_区间无数据时退出码为1并给出原因(tmp_path: Path, data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_args(data_root, strategy_dir, "--start", "2030-01-01", "--end", "2030-02-01"))

    assert code == 1
    assert "VALIDATION_ERROR" in capsys.readouterr().err


def test_风控上限通过CLI生效(data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """策略想买 47.5% 仓位，把单票上限压到 5% 后必须被缩量，并在报告里留痕。"""
    assert main(_args(data_root, strategy_dir, "--max-weight", "0.05", "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    trades = pd.read_csv(Path(payload["output_dir"]) / "trades.csv", dtype={"code": str})
    assert not trades.empty
    assert trades["qty"].max() <= 5_000, f"风控未生效：成交 {trades['qty'].max()} 股"
    assert payload["risk"]["shrinks"] >= 1, "缩量必须在报告中留痕"

    report = json.loads((Path(payload["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    assert report["config"]["risk_limits"]["max_weight"] == 0.05
    assert report["config"]["risk_enabled"] is True


def test_关闭风控后仓位不再受限(data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args(data_root, strategy_dir, "--no-risk", "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    trades = pd.read_csv(Path(payload["output_dir"]) / "trades.csv", dtype={"code": str})
    assert trades["qty"].max() > 40_000, "关闭风控后应恢复原始仓位"

    report = json.loads((Path(payload["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    assert report["config"]["risk_enabled"] is False


def test_坏参数格式退出码为1(data_root: Path, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_args(data_root, strategy_dir, "--params", "no_value"))

    assert code == 1
    assert "参数格式非法" in capsys.readouterr().err
