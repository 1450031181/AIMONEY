"""P2-F3 因子服务/CLI 测试。

最核心的一条：**预热期**。因子有 window（mom_20 需要 21 根 K 线），
若不额外读取 start 之前的数据，评价窗口开头会有 N 天因子值为 NaN，
截面被静默跳过——结果"看起来正常"，但实际少评估了一段。这类缺失不报错，必须用测试钉住。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.factor_cli import main
from app.services.factor_service import FactorEvaluationRequest, FactorService
from engine.datahub.store import ParquetStore
from engine.errors import ValidationError
from engine.factor.evaluate import factor_correlation

START = date(2026, 8, 1)
END = date(2026, 10, 30)

#: 标的数必须 ≥ min_cross_section（默认 5），否则每个截面都会被跳过、评价"无有效样本"
CODES = ("600000", "000001", "300750", "601318", "000002", "600519", "002415", "300059")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """造 90 个自然日、3 只标的的行情（含真实波动，避免常数截面）。"""
    resolved = Settings(data={"root": str(tmp_path / "data")})
    resolved.ensure_dirs()
    rng = np.random.default_rng(2026)
    rows = []
    for index, code in enumerate(CODES):
        price = 10.0 + index * 5
        for offset in range(90):
            day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=offset)
            price *= 1.0 + float(rng.normal(0.001, 0.02))
            rows.append(
                {
                    "code": code,
                    "date": day.date().isoformat(),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000 + offset * 100,
                    "amount": price * 1_000_000,
                    "adj_factor": 1.0,
                }
            )
    ParquetStore(resolved.parquet_dir).write("daily", pd.DataFrame(rows), data_version="20261030.1")

    reference = resolved.data.root_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"code": code, "name": f"标的{index}", "is_st": False, "list_date": "2010-01-01", "delist_date": None}
            for index, code in enumerate(CODES)
        ]
    ).to_parquet(reference / "stock_info.parquet", index=False)
    return resolved


# ============================================================ 服务
def test_评价产出报告与明细文件(settings: Settings) -> None:
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_20", "vol_20"), start=START, end=END)
    )

    assert len(report.evaluations) == 2
    assert {item.factor for item in report.evaluations} == {"mom_20", "vol_20"}
    assert report.loaded_rows > 0
    assert report.correlation.shape == (2, 2)

    summary = json.loads((report.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["quantiles"] == 5
    assert summary["warmup_days"] == 21, "预热期取各因子 window 最大值（mom_20 与 vol_20 均为 21）"
    assert len(summary["evaluations"]) == 2
    assert (report.output_dir / "ic_mom_20.csv").is_file()
    assert (report.output_dir / "quantile_vol_20.csv").is_file()
    assert (report.output_dir / "correlation.csv").is_file()


def test_预热期让窗口开头的因子值可用(settings: Settings) -> None:
    """不带预热时，评价窗口前 20 天 mom_20 全是 NaN，会被静默跳过。"""
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_20",), start=date(2026, 9, 1), end=END)
    )

    evaluation = report.evaluations[0]
    assert evaluation.dropped_rows < 10, f"丢弃行数过多说明预热没生效：{evaluation.dropped_rows}"
    assert evaluation.sufficient
    assert evaluation.observations > 30, "60 个交易日的窗口应产出足够多的截面"


def test_本地数据不足预热期时降级而不是报错(settings: Settings) -> None:
    """从本地数据第一天开始评价：预热不足，应降级（前段 NaN）而非抛错。"""
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_60",), start=date(2026, 8, 1), end=END)
    )

    assert report.warmup_days == 61
    assert report.evaluations[0].sufficient or report.evaluations[0].dropped_rows > 0


def test_评价窗口裁掉预热期(settings: Settings) -> None:
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_5",), start=date(2026, 9, 1), end=END)
    )

    ic_series = report.evaluations[0].ic_series
    assert not ic_series.empty
    assert pd.Timestamp(ic_series.index.min()) >= pd.Timestamp(date(2026, 9, 1)), "IC 不应包含预热期"


def test_不做预处理时仍可评价(settings: Settings) -> None:
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_10",), start=START, end=END, preprocess_steps=())
    )
    assert report.evaluations[0].sufficient


def test_单因子时不出相关性矩阵(settings: Settings) -> None:
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_10",), start=START, end=END)
    )
    assert report.correlation.empty


def test_未注册因子被拒(settings: Settings) -> None:
    with pytest.raises(ValidationError):
        FactorService(settings).run(
            FactorEvaluationRequest(factors=("not_a_factor",), start=START, end=END)
        )


def test_本地无数据时拒绝(settings: Settings, tmp_path: Path) -> None:
    empty = Settings(data={"root": str(tmp_path / "empty")})
    empty.ensure_dirs()

    from engine.errors import DataNotReadyError

    with pytest.raises(DataNotReadyError):
        FactorService(empty).run(FactorEvaluationRequest(factors=("mom_5",), start=START, end=END))


def test_区间非法被拒() -> None:
    with pytest.raises(ValidationError):
        FactorEvaluationRequest(factors=("mom_5",), start=END, end=START)
    with pytest.raises(ValidationError):
        FactorEvaluationRequest(factors=(), start=START, end=END)


# ============================================================ 相关性
def test_相关性与自身为一且对称(settings: Settings) -> None:
    report = FactorService(settings).run(
        FactorEvaluationRequest(factors=("mom_5", "mom_20", "vol_10"), start=START, end=END)
    )
    correlation = report.correlation

    for name in ("mom_5", "mom_20", "vol_10"):
        assert correlation.loc[name, name] == pytest.approx(1.0), "因子与自身相关性必须为 1"
    assert correlation.loc["mom_5", "mom_20"] == pytest.approx(correlation.loc["mom_20", "mom_5"])


def test_相关性矩阵需要至少两个因子() -> None:
    frame = pd.DataFrame({"date": ["2026-10-12"], "code": ["600000"], "f": [1.0]})
    with pytest.raises(ValidationError):
        factor_correlation(frame, ["f"])


# ============================================================ CLI
def _args(settings: Settings, *extra: str) -> list[str]:
    return ["--root", str(settings.data.root_dir), *extra]


def test_CLI列出因子库(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args(settings, "list")) == 0
    out = capsys.readouterr().out

    assert "mom_20" in out and "共" in out
    assert main(_args(settings, "list", "--category", "reversal", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload and all(item["category"] == "reversal" for item in payload)


def test_CLI列出未知类目返回非零(settings: Settings) -> None:
    assert main(_args(settings, "list", "--category", "not_exists")) == 1


def test_CLI评价输出JSON(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        _args(
            settings,
            "evaluate",
            "--factors",
            "mom_20,vol_20",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--json",
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["factors"]) == 2
    assert payload["warmup_days"] == 21
    assert (Path(payload["output_dir"]) / "summary.json").is_file()


def test_CLI评价人类可读输出含相关性表(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        _args(
            settings,
            "evaluate",
            "--factors",
            "mom_20",
            "--factors",
            "vol_20",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
        )
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "因子相关性" in out
    assert "报告目录" in out


def test_CLI无数据时退出码为1(settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty_root = tmp_path / "no_data"
    empty_root.mkdir()

    code = main(
        [
            "--root",
            str(empty_root),
            "evaluate",
            "--factors",
            "mom_20",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
        ]
    )

    assert code == 1
    assert "DATA_NOT_READY" in capsys.readouterr().err


def test_CLI日期格式非法退出码为1(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        _args(settings, "evaluate", "--factors", "mom_20", "--start", "2026/08/01", "--end", END.isoformat())
    )

    assert code == 1
    assert "日期格式非法" in capsys.readouterr().err


def test_CLI_预处理传none表示不做(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        _args(
            settings,
            "evaluate",
            "--factors",
            "mom_20",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--preprocess",
            "none",
            "--json",
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["factors"][0]["sufficient"]
