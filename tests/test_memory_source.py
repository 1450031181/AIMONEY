"""T-04 内存数据源测试：区间过滤、复权口径、成分按生效日、数据契约校验。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.datahub.calendar import weekday_calendar
from engine.datahub.schema import DAILY_REQUIRED_COLUMNS, ensure_primary_key_unique
from engine.datasource.memory import InMemoryDataSource
from engine.errors import DataQualityError, ValidationError


@pytest.fixture
def source() -> InMemoryDataSource:
    calendar = weekday_calendar(date(2026, 10, 1), date(2026, 10, 31))
    return InMemoryDataSource(calendar=calendar)


def _daily_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_日线区间与代码过滤并排序(source: InMemoryDataSource) -> None:
    source.add_daily(
        _daily_frame(
            [
                {"code": "600000", "date": "2026-10-12", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
                {"code": "600000", "date": "2026-10-13", "open": 10, "high": 11, "low": 9, "close": 10.6, "volume": 1100},
                {"code": "000001", "date": "2026-10-13", "open": 20, "high": 21, "low": 19, "close": 20.5, "volume": 2000},
            ]
        )
    )

    result = source.daily(["600000"], date(2026, 10, 13), date(2026, 10, 13))
    assert len(result) == 1
    assert result.iloc[0]["code"] == "600000"
    assert result.iloc[0]["close"] == pytest.approx(10.6)

    both = source.daily(["600000", "000001"], date(2026, 10, 12), date(2026, 10, 13))
    assert list(both["date"].dt.strftime("%Y-%m-%d")) == ["2026-10-12", "2026-10-13", "2026-10-13"]


def test_查询不到数据时返回带列的空表(source: InMemoryDataSource) -> None:
    result = source.daily(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    assert result.empty
    assert "code" in result.columns


def test_后复权按因子缩放价格但不改成交量(source: InMemoryDataSource) -> None:
    source.add_daily(
        _daily_frame(
            [
                {
                    "code": "600000",
                    "date": "2026-10-13",
                    "open": 10,
                    "high": 12,
                    "low": 9,
                    "close": 11,
                    "volume": 1000,
                    "adj_factor": 2.0,
                }
            ]
        )
    )

    result = source.daily(["600000"], date(2026, 10, 13), date(2026, 10, 13), adj="post")
    row = result.iloc[0]
    assert row["close"] == pytest.approx(22.0)
    assert row["high"] == pytest.approx(24.0)
    assert row["volume"] == 1000, "复权只能改价格，成交量必须保持原值"

    raw = source.daily(["600000"], date(2026, 10, 13), date(2026, 10, 13), adj="none")
    assert raw.iloc[0]["close"] == pytest.approx(11.0)


def test_前复权明确报错而非返回含糊结果(source: InMemoryDataSource) -> None:
    source.add_daily(
        _daily_frame(
            [{"code": "600000", "date": "2026-10-13", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100}]
        )
    )
    with pytest.raises(ValidationError):
        source.daily(["600000"], date(2026, 10, 13), date(2026, 10, 13), adj="pre")


def test_缺少必要列时拒绝装载(source: InMemoryDataSource) -> None:
    with pytest.raises(DataQualityError):
        source.add_daily(pd.DataFrame([{"code": "600000", "date": "2026-10-13", "close": 10}]))


def test_重复装载同一主键以最后一次为准(source: InMemoryDataSource) -> None:
    row = {"code": "600000", "date": "2026-10-13", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100}
    source.add_daily(_daily_frame([row]))
    source.add_daily(_daily_frame([{**row, "close": 99}]))

    result = source.daily(["600000"], date(2026, 10, 13), date(2026, 10, 13))
    assert len(result) == 1
    assert result.iloc[0]["close"] == pytest.approx(99.0)
    source.assert_no_duplicated_keys(["600000"])


def test_指数成分按生效日生效(source: InMemoryDataSource) -> None:
    source.add_universe("000300", date(2026, 1, 1), ["600000", "000001"])
    source.add_universe("000300", date(2026, 7, 1), ["600000", "300750"])

    assert source.universe("000300", date(2026, 6, 30)) == ("600000", "000001")
    assert source.universe("000300", date(2026, 7, 1)) == ("600000", "300750")
    # 生效日之前没有成分，返回空（不得用最新成分回填历史）
    assert source.universe("000300", date(2025, 12, 31)) == ()
    assert source.universe("000905", date(2026, 7, 1)) == ()


def test_复权因子查询(source: InMemoryDataSource) -> None:
    source.add_daily(
        _daily_frame(
            [
                {"code": "600000", "date": "2026-10-12", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "adj_factor": 1.5},
                {"code": "600000", "date": "2026-10-13", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "adj_factor": 1.6},
            ]
        )
    )
    factors = source.adjust_factors(["600000"], date(2026, 10, 12), date(2026, 10, 13))
    assert list(factors["adj_factor"]) == [1.5, 1.6]


def test_基础信息返回副本不被外部修改(source: InMemoryDataSource) -> None:
    source.add_stock_info(pd.DataFrame([{"code": "600000", "name": "浦发银行"}]))
    info = source.stock_info()
    info.loc[0, "name"] = "被改了"
    assert source.stock_info().iloc[0]["name"] == "浦发银行"


def test_主键唯一性校验能抓出重复行() -> None:
    duplicated = pd.DataFrame(
        [
            {"code": "600000", "date": "2026-10-13", "close": 10},
            {"code": "600000", "date": "2026-10-13", "close": 11},
        ]
    )
    with pytest.raises(DataQualityError):
        ensure_primary_key_unique(duplicated, dataset="daily")


def test_数据契约列规范完整() -> None:
    assert {"code", "date", "open", "high", "low", "close", "volume"} == set(DAILY_REQUIRED_COLUMNS)
