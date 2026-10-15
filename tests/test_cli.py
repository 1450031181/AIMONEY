"""T-13 CLI 测试（全离线）。

重点：**退出码是接口契约**——定时任务与 CI 靠它判断"今天数据能不能用"，
所以每个子命令的正常/异常退出码都要有断言。

联网路径通过 monkeypatch ``app.cli.build_source`` 注入内存数据源来隔离，
CLI 测试绝不该打外网。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app import cli
from engine.datahub.calendar import weekday_calendar
from engine.datahub.quality import TradeGate
from engine.datahub.reference import ReferenceStore
from engine.datahub.store import ParquetStore
from engine.datasource.memory import InMemoryDataSource
from engine.errors import DataSourceError

TRADE_DAY = date(2026, 10, 14)
START = date(2026, 10, 12)


def _source() -> InMemoryDataSource:
    source = InMemoryDataSource(calendar=weekday_calendar(date(2026, 1, 1), date(2026, 12, 31)))
    source.add_daily(
        pd.DataFrame(
            [
                {
                    "code": code,
                    "date": day,
                    "open": 10.0,
                    "high": 10.4,
                    "low": 9.9,
                    "close": 10.2,
                    "volume": 100_000,
                    "adj_factor": 1.0,
                }
                for code in ("600000", "000001")
                for day in ("2026-10-12", "2026-10-13", "2026-10-14")
            ]
        )
    )
    source.add_stock_info(
        pd.DataFrame(
            [
                {"code": "600000", "name": "浦发银行", "is_st": False, "list_date": "1999-11-10", "delist_date": None},
                {"code": "000001", "name": "平安银行", "is_st": False, "list_date": "1991-04-03", "delist_date": None},
            ]
        )
    )
    return source


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture(autouse=True)
def _inject_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_source", lambda settings, args: _source())


def _run(data_root: Path, *args: str) -> int:
    return cli.main(["--root", str(data_root), *args])


# ============================================================ sync
def test_同步成功后可用coverage查到(data_root: Path) -> None:
    assert _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat()) == cli.EXIT_OK

    assert _run(data_root, "coverage", "--json") == cli.EXIT_OK
    store = ParquetStore(data_root / "parquet")
    assert store.coverage("daily").code_count == 2


def test_同步顺带落盘股票基础信息(data_root: Path) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())

    info = ReferenceStore(data_root / "reference").load_stock_info()
    assert set(info["code"]) == {"600000", "000001"}, "顺序不作约定，只要求内容完整"


def test_首次同步必须显式给起点(data_root: Path) -> None:
    assert _run(data_root, "sync", "--end", TRADE_DAY.isoformat()) == cli.EXIT_DATA_PROBLEM


def test_full必须与start一起用(data_root: Path) -> None:
    assert _run(data_root, "sync", "--full", "--end", TRADE_DAY.isoformat()) == cli.EXIT_USAGE


def test_标的参数逗号与重复传参都支持(tmp_path: Path) -> None:
    """PowerShell 会把 `600000,000001` 当数组、把 `000001` 解析成 `1`，故两种写法都要能用。"""
    common = ["sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat()]
    comma_root = tmp_path / "comma"
    repeat_root = tmp_path / "repeat"

    assert _run(comma_root, *common, "--codes", "600000,000001") == cli.EXIT_OK
    assert _run(repeat_root, *common, "--codes", "600000", "--codes", "000001") == cli.EXIT_OK

    for root in (comma_root, repeat_root):
        assert ParquetStore(root / "parquet").coverage("daily").code_count == 2


def test_同步指定标的(data_root: Path) -> None:
    assert _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--codes", "600000", "--trade-date", TRADE_DAY.isoformat()) == cli.EXIT_OK

    assert ParquetStore(data_root / "parquet").coverage("daily").code_count == 1


class _FailingSource(InMemoryDataSource):
    """稳态失败的标的（用于验证 CLI 的失败退出码）。"""

    fail_codes: dict[str, int]

    def daily(
        self,
        codes: list[str],  # type: ignore[override]
        start: date,
        end: date,
        *,
        adj: str = "post",  # type: ignore[override]
    ) -> pd.DataFrame:
        for code in codes:
            if self.fail_codes.get(code, 0) > 0:
                self.fail_codes[code] -= 1
                raise DataSourceError(f"{code} 拉取失败")
        return super().daily(codes, start, end, adj=adj)  # type: ignore[arg-type]


def test_同步有失败标的时返回非零(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _FailingSource(calendar=weekday_calendar(date(2026, 1, 1), date(2026, 12, 31)))
    source.add_daily(
        pd.DataFrame(
            [
                {
                    "code": code,
                    "date": "2026-10-14",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                    "adj_factor": 1.0,
                }
                for code in ("600000", "300750")
            ]
        )
    )
    source.fail_codes = {"300750": 99}  # 一直失败
    monkeypatch.setattr(cli, "build_source", lambda settings, args: source)

    code = _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--codes", "600000,300750", "--trade-date", TRADE_DAY.isoformat())

    assert code == cli.EXIT_DATA_PROBLEM


# ============================================================ coverage / list
def test_覆盖率表与JSON(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())
    capsys.readouterr()

    assert _run(data_root, "coverage", "--json") == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"]["data_version"] == "20261014.1"
    assert payload["gate"]["open"] is False, "只同步未校验时门禁应保持关闭"

    assert _run(data_root, "coverage") == cli.EXIT_OK
    assert "交易门禁" in capsys.readouterr().out


def test_空数据目录时list给出引导(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(data_root, "list") == cli.EXIT_OK
    assert "没有任何数据集" in capsys.readouterr().out


def test_list在同步后列出数据集(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())
    capsys.readouterr()

    assert _run(data_root, "list") == cli.EXIT_OK
    assert "daily" in capsys.readouterr().out


# ============================================================ check
def test_质量校验通过时开门禁(data_root: Path) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())

    assert _run(data_root, "check", "--date", TRADE_DAY.isoformat()) == cli.EXIT_OK
    assert TradeGate(data_root / "state" / "trade_gate.json").is_open()
    assert (data_root / "quality" / "quality_20261014.json").is_file()


def test_质量校验失败时返回非零并关门禁(data_root: Path) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())
    gate = TradeGate(data_root / "state" / "trade_gate.json")
    gate.open(as_of=date(2026, 10, 13))

    # 用坏数据覆盖当日：直接写一份非法价格
    store = ParquetStore(data_root / "parquet")
    store.write("daily", pd.DataFrame([{"code": "600000", "date": TRADE_DAY.isoformat(), "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 100, "adj_factor": 1.0}]))

    assert _run(data_root, "check", "--date", TRADE_DAY.isoformat()) == cli.EXIT_DATA_PROBLEM
    assert not gate.is_open()


def test_no_gate只出报告不改门禁(data_root: Path) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())
    gate = TradeGate(data_root / "state" / "trade_gate.json")
    gate.open(as_of=date(2026, 10, 13))
    ParquetStore(data_root / "parquet").write("daily", pd.DataFrame([{"code": "600000", "date": TRADE_DAY.isoformat(), "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 100, "adj_factor": 1.0}]))

    code = _run(data_root, "check", "--date", TRADE_DAY.isoformat(), "--no-gate")

    assert code == cli.EXIT_DATA_PROBLEM
    assert gate.is_open(), "--no-gate 时门禁不应被改动"


def test_质量报告JSON输出(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(data_root, "sync", "--start", START.isoformat(), "--end", TRADE_DAY.isoformat(), "--trade-date", TRADE_DAY.isoformat())
    capsys.readouterr()

    assert _run(data_root, "check", "--date", TRADE_DAY.isoformat(), "--json") == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"


# ============================================================ index-members
def test_成分快照子命令(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    source.add_universe("000300", date(2020, 1, 1), ["600000", "000001"])  # 供 index_members 读取"当前成分"
    monkeypatch.setattr(cli, "build_source", lambda settings, args: source)

    assert _run(data_root, "index-members", "--index", "000300", "--date", TRADE_DAY.isoformat()) == cli.EXIT_OK
    assert (data_root / "universe" / "universe.json").is_file()


# ============================================================ 参数与错误
def test_日期格式非法时给出可读错误(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _run(data_root, "check", "--date", "2026/10/14")

    assert code == cli.EXIT_DATA_PROBLEM
    assert "日期格式非法" in capsys.readouterr().err


def test_缺少子命令时退出码为2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == 2
