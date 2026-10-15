"""P2-F4 多因子合成测试。

本文件最重要的一组断言是 **权重不许用未来信息**：
若用全样本 IC 定权重再回看整段区间，等于"提前知道哪个因子未来有效"，
合成因子的 IC 会被系统性高估，而回测曲线会漂亮得毫无意义。
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
from app.services.factor_service import FactorCombineRequest, FactorService
from engine.datahub.store import ParquetStore
from engine.errors import ValidationError
from engine.factor.combine import (
    composite_score,
    deduplicate_factors,
    normalize_weights,
    rolling_ic_weights,
    select_top,
    weights_from_ic,
)

START = date(2026, 8, 1)
END = date(2026, 10, 30)
CODES = ("600000", "000001", "300750", "601318", "000002", "600519", "002415", "300059")


# ============================================================ 权重
def test_归一化按绝对值之和() -> None:
    weights = normalize_weights({"a": 0.5, "b": -0.25})
    assert sum(abs(value) for value in weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(2 / 3)
    assert weights["b"] == pytest.approx(-1 / 3)


def test_全零权重不除零() -> None:
    assert normalize_weights({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


def test_等权与IC加权与ICIR加权() -> None:
    means = {"a": 0.10, "b": 0.05}
    stds = {"a": 0.20, "b": 0.01}

    equal = weights_from_ic(means, stds, method="equal")
    assert equal["a"] == equal["b"] == 0.5

    ic = weights_from_ic(means, stds, method="ic")
    assert ic["a"] == pytest.approx(2 / 3)

    # b 的 IC 更小但极稳定（ICIR 5.0）→ 在 ICIR 加权下权重反而更大
    icir = weights_from_ic(means, stds, method="icir")
    assert icir["b"] > icir["a"]


def test_负IC因子被反向使用() -> None:
    weights = weights_from_ic({"a": -0.1, "b": 0.1}, method="ic")
    assert weights["a"] < 0 and weights["b"] > 0, "IC 为负说明因子方向相反，应反向使用而不是丢弃"


def test_噪声因子权重置零() -> None:
    weights = weights_from_ic({"strong": 0.2, "noise": 0.001}, method="ic", min_abs_ic=0.02)
    assert weights["noise"] == 0.0
    assert abs(weights["strong"]) == pytest.approx(1.0)


def test_IC波动为零时退回IC() -> None:
    weights = weights_from_ic({"a": 0.1, "b": 0.1}, {"a": 0.0, "b": 0.0}, method="icir")
    assert weights["a"] == pytest.approx(0.5), "std=0 时不应除零，退回 IC 本身"


def test_未知合成方法被拒() -> None:
    with pytest.raises(ValidationError):
        weights_from_ic({"a": 0.1}, method="magic")


# ============================================================ 滚动权重（防未来函数）
def _merged_with_ic_pattern(periods: int = 10, codes: int = 8) -> pd.DataFrame:
    """构造一个"因子 f 与未来收益完全正相关"的截面数据（每天 IC 恒为 +1）。"""
    rows = []
    for index in range(periods):
        day = (pd.Timestamp("2026-09-01") + pd.Timedelta(days=index)).date().isoformat()
        for position in range(codes):
            rows.append({"date": day, "code": f"{position:06d}", "f": float(position), "forward_return": float(position)})
    return pd.DataFrame(rows)


def test_滚动权重前window个截面无权重() -> None:
    merged = _merged_with_ic_pattern(periods=10)
    weights = rolling_ic_weights(merged, ["f"], window=3, method="ic")

    assert weights["f"].isna().sum() == 3, "前 3 个截面没有足够历史，必须留空而不是用未来数据补"
    assert weights["f"].iloc[3:].notna().all()


def test_滚动权重与手工计算一致() -> None:
    merged = _merged_with_ic_pattern(periods=8)
    window = 3
    weights = rolling_ic_weights(merged, ["f"], window=window, method="ic")
    ic_series = pd.Series(1.0, index=sorted(pd.unique(merged["date"])))

    # 第 4 个截面（index 3）的权重应只由前 3 个截面的 IC 决定
    expected = weights_from_ic({"f": float(ic_series.iloc[:window].mean())}, method="ic")
    assert weights["f"].iloc[window] == pytest.approx(expected["f"])


def test_权重估计窗口不足时整体留空() -> None:
    merged = _merged_with_ic_pattern(periods=2)
    weights = rolling_ic_weights(merged, ["f"], window=5, method="ic")

    assert weights["f"].isna().all(), "样本不足时宁可不给权重"


def test_滚动权重参数校验() -> None:
    merged = _merged_with_ic_pattern(periods=5)
    with pytest.raises(ValidationError):
        rolling_ic_weights(merged, ["f"], window=0)
    with pytest.raises(ValidationError):
        rolling_ic_weights(merged, [], window=3)


# ============================================================ 去重
def test_去重保留更强因子剔除高相关() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    kept = deduplicate_factors(correlation, {"a": 0.10, "b": 0.30, "c": 0.05}, threshold=0.7)

    assert "b" in kept and "a" not in kept, "a 与更强的 b 相关性 0.95，属重复下注"
    assert "c" in kept, "低相关因子应保留"


def test_去重阈值必须合法() -> None:
    with pytest.raises(ValidationError):
        deduplicate_factors(pd.DataFrame(), {}, threshold=1.5)


# ============================================================ 合成与选股
def test_合成分数等于加权和() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-10-12"] * 3,
            "code": ["a", "b", "c"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0],
        }
    )
    weights = pd.DataFrame({"f1": [0.6], "f2": [0.4]}, index=[pd.Timestamp("2026-10-12")])

    scores = composite_score(frame, ["f1", "f2"], weights)

    assert list(scores["score"]) == pytest.approx([1.8, 2.0, 2.2])


def test_权重缺失的截面被跳过() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-10-12", "2026-10-13"],
            "code": ["a", "a"],
            "f1": [1.0, 1.0],
        }
    )
    weights = pd.DataFrame({"f1": [np.nan]}, index=[pd.Timestamp("2026-10-12")])

    scores = composite_score(frame, ["f1"], weights)
    assert scores.empty, "没有权重就没有合成分数，不允许用其他日期的权重顶上"


def test_合成分数对列缺失报错() -> None:
    frame = pd.DataFrame({"date": ["2026-10-12"], "code": ["a"], "f1": [1.0]})
    weights = pd.DataFrame({"f1": [1.0]}, index=[pd.Timestamp("2026-10-12")])
    with pytest.raises(ValidationError):
        composite_score(frame, ["f1", "f2"], weights)


def test_选股取前k且等权() -> None:
    scores = pd.DataFrame(
        {
            "date": ["2026-10-12"] * 5,
            "code": ["a", "b", "c", "d", "e"],
            "score": [0.1, 0.5, 0.3, 0.2, 0.4],
        }
    )
    selections = select_top(scores, top_k=3)

    assert list(selections["code"]) == ["b", "e", "c"], "按分数降序取前 3"
    assert list(selections["rank"]) == [1, 2, 3]
    assert selections["weight"].sum() == pytest.approx(1.0)
    assert (selections["weight"] - 1 / 3).abs().max() < 1e-12, "名单内等权"


def test_标的不足时取全部() -> None:
    scores = pd.DataFrame({"date": ["2026-10-12"] * 2, "code": ["a", "b"], "score": [0.1, 0.2]})
    assert len(select_top(scores, top_k=10)) == 2


def test_top_k非法被拒() -> None:
    with pytest.raises(ValidationError):
        select_top(pd.DataFrame(columns=["date", "code", "score"]), top_k=0)


# ============================================================ 服务与 CLI
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    resolved = Settings(data={"root": str(tmp_path / "data")})
    resolved.ensure_dirs()
    rng = np.random.default_rng(4242)
    rows = []
    for index, code in enumerate(CODES):
        price = 10.0 + index * 4
        drift = 0.002 * (index - 3.5)
        for offset in range(90):
            day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=offset)
            if day.dayofweek >= 5:
                continue
            price *= 1.0 + drift + float(rng.normal(0.0, 0.015))
            rows.append(
                {
                    "code": code,
                    "date": day.date().isoformat(),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000,
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


def _request(**overrides: object) -> FactorCombineRequest:
    options: dict[str, object] = {
        "factors": ("mom_20", "vol_20", "rev_5"),
        "start": START,
        "end": END,
        "weight_window": 10,
        "top_k": 3,
    }
    options.update(overrides)
    return FactorCombineRequest(**options)  # type: ignore[arg-type]


def test_合成产出完整报告(settings: Settings) -> None:
    report = FactorService(settings).run_combine(_request())

    assert report.composite is not None
    assert report.kept_factors, "至少要保留一个因子"
    assert set(report.dropped_factors) | set(report.kept_factors) == {"mom_20", "vol_20", "rev_5"}
    assert len(report.selections) > 0
    assert set(report.selections.columns) == {"date", "code", "score", "rank", "weight"}

    for name in ("summary.json", "weights.csv", "selections.csv", "correlation.csv", "ic_composite.csv"):
        assert (report.output_dir / name).is_file(), f"缺少报告文件 {name}"

    payload = json.loads((report.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["method"] == "icir"
    assert payload["config"]["weight_window"] == 10
    assert payload["composite"] is not None


def test_摘要里的因子计数不重复(settings: Settings) -> None:
    """回归：dropped 是 singles 的子集，两者相加会把输入数报大。"""
    report = FactorService(settings).run_combine(
        _request(factors=("mom_20", "rev_5", "vol_20"), correlation_threshold=0.2)
    )
    assert report.dropped_factors, "阈值 0.2 下应至少剔除一个因子"

    summary = report.summary()
    assert f"输入 {len(report.singles)} 个因子" in summary
    assert f"输入 {len(report.singles) + len(report.dropped_factors)} 个因子" not in summary


def test_合成权重前段不可用(settings: Settings) -> None:
    report = FactorService(settings).run_combine(_request(weight_window=10))

    assert report.weights.iloc[:10].isna().all().all(), "权重估计窗口内的截面必须留空（不可用）"
    assert report.weights.iloc[10:].notna().any().any()


def test_高相关因子被去重(settings: Settings) -> None:
    """mom_20 与 rev_10 在月度尺度上基本是镜像关系，阈值调低应剔除其中之一。"""
    report = FactorService(settings).run_combine(
        _request(factors=("mom_20", "rev_10", "vol_20"), correlation_threshold=0.2)
    )

    assert len(report.kept_factors) < 3, "阈值 0.2 下高度相关的因子应被剔除"
    assert "vol_20" in report.kept_factors or len(report.kept_factors) == 1


def test_等权合成不依赖估计(settings: Settings) -> None:
    report = FactorService(settings).run_combine(_request(method="equal", weight_window=5))

    active = report.weights.iloc[-1].dropna()
    assert len(set(active.round(6))) == 1, "等权方法下各因子权重应相同"


def test_合成请求参数校验(settings: Settings) -> None:
    with pytest.raises(ValidationError):
        _request(weight_window=0)
    with pytest.raises(ValidationError):
        _request(top_k=0)


def test_CLI合成输出JSON(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "combine",
            "--factors",
            "mom_20,vol_20",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--weight-window",
            "10",
            "--top-k",
            "3",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["composite"] is not None
    assert (Path(payload["output_dir"]) / "summary.json").is_file()


def test_CLI合成人类可读输出(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "combine",
            "--factors",
            "mom_20",
            "--factors",
            "vol_20",
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--weight-window",
            "10",
            "--top-k",
            "3",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "多因子合成" in out and "最新权重" in out and "报告目录" in out
