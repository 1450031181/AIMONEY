"""T-10 股票池与指数成分测试。

三条关键性质：
1. **没有成分快照的日期必须报错**，不允许用最新成分回填历史（未来函数）；
2. 每条过滤规则都要能说清"排除了谁"，否则股票池异常时无从排查；
3. **数据缺失导致规则未生效时必须显式暴露**（akshare 不提供上市日，次新过滤实际不生效）。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.datahub.calendar import weekday_calendar
from engine.datahub.sync import IndexMembersSyncTask
from engine.datahub.tasklog import MemoryTaskLog, TaskStatus
from engine.datahub.universe import (
    RULE_DELISTED,
    RULE_MISSING_LIST_DATE,
    RULE_NEW_LISTING,
    RULE_NO_INFO,
    RULE_ST,
    RULE_SUSPENDED,
    JsonUniverseStore,
    UniverseFilter,
    UniverseHistory,
    select_universe,
)
from engine.datasource.memory import InMemoryDataSource
from engine.errors import DataNotReadyError, DataSourceError, ValidationError

AS_OF = date(2026, 10, 14)


def _stock_info(**overrides: object) -> pd.DataFrame:
    rows = [
        {"code": "600000", "name": "浦发银行", "is_st": False, "list_date": "1999-11-10", "delist_date": None},
        {"code": "600001", "name": "老牌股", "is_st": False, "list_date": "2000-01-01", "delist_date": None},
        {"code": "600002", "name": "ST 某股", "is_st": True, "list_date": "2001-01-01", "delist_date": None},
        {"code": "600003", "name": "刚上市", "is_st": False, "list_date": "2026-10-01", "delist_date": None},
        {"code": "600004", "name": "已退市", "is_st": False, "list_date": "2005-01-01", "delist_date": "2025-06-01"},
        {"code": "600005", "name": "无上市日", "is_st": False, "list_date": None, "delist_date": None},
    ]
    frame = pd.DataFrame(rows)
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _history(*, with_snapshot: bool = True) -> UniverseHistory:
    history = UniverseHistory()
    if with_snapshot:
        history.add("000300", date(2026, 7, 1), [row["code"] for row in _stock_info().to_dict("records")])
    return history


# ============================================================ 成分解析
def test_没有成分快照的日期直接报错() -> None:
    history = UniverseHistory()
    history.add("000300", date(2026, 7, 1), ["600000"])

    with pytest.raises(DataNotReadyError):
        select_universe(
            history,
            as_of=date(2026, 6, 30),
            stock_info=_stock_info(),
            universe_filter=UniverseFilter(index_code="000300"),
        )


def test_按指数成分构建股票池() -> None:
    history = UniverseHistory()
    history.add("000300", date(2026, 7, 1), ["600000", "600001"])

    selection = select_universe(
        history, as_of=AS_OF, stock_info=_stock_info(), universe_filter=UniverseFilter(index_code="000300")
    )

    assert selection.codes == ("600000", "600001")
    assert selection.index_code == "000300"
    assert selection.dropped == {}


# ============================================================ 过滤规则
def test_逐条规则排除且可追溯() -> None:
    selection = select_universe(
        _history(),
        as_of=AS_OF,
        stock_info=_stock_info(),
        universe_filter=UniverseFilter(min_list_days=90, list_date_policy="error"),
        suspended=["600001"],
    )

    assert selection.codes == ("600000",)
    assert selection.dropped[RULE_ST] == ("600002",)
    assert selection.dropped[RULE_DELISTED] == ("600004",)
    assert selection.dropped[RULE_NEW_LISTING] == ("600003",)
    assert selection.dropped[RULE_SUSPENDED] == ("600001",)
    assert selection.dropped[RULE_MISSING_LIST_DATE] == ("600005",)
    assert selection.dropped_total == 5


def test_不在基础信息中的标的被排除而不是默认保留() -> None:
    history = UniverseHistory()
    history.add("000300", date(2026, 7, 1), ["600000", "999999"])

    selection = select_universe(
        history, as_of=AS_OF, stock_info=_stock_info(), universe_filter=UniverseFilter(index_code="000300")
    )

    assert selection.codes == ("600000",)
    assert selection.dropped[RULE_NO_INFO] == ("999999",)


def test_关闭某条规则后不再排除() -> None:
    selection = select_universe(
        _history(),
        as_of=AS_OF,
        stock_info=_stock_info(),
        universe_filter=UniverseFilter(exclude_st=False, exclude_delisted=False, exclude_suspended=False),
        suspended=["600001"],
    )

    assert "600002" in selection.codes
    assert "600004" in selection.codes
    assert "600001" in selection.codes


# ============================================================ 数据缺失时必须显式暴露
def test_缺少上市日时次新过滤必须显式标注未生效() -> None:
    info = _stock_info()
    info["list_date"] = pd.NaT

    selection = select_universe(
        _history(), as_of=AS_OF, stock_info=info, universe_filter=UniverseFilter(min_list_days=90)
    )

    assert "未生效:次新过滤" in "".join(selection.rules_not_applied), "规则没生效必须能看出来"
    assert "600003" in selection.codes, "缺上市日时按『不排除』处理，而不是假装过滤了"


def test_严格模式下缺少上市日直接报错式排除() -> None:
    info = _stock_info()
    info["list_date"] = pd.NaT

    selection = select_universe(
        _history(),
        as_of=AS_OF,
        stock_info=info,
        universe_filter=UniverseFilter(min_list_days=90, list_date_policy="error"),
    )

    assert selection.dropped[RULE_MISSING_LIST_DATE]
    assert "600005" not in selection.codes


def test_基础信息为空时拒绝构建股票池() -> None:
    with pytest.raises(ValidationError):
        select_universe(
            _history(), as_of=AS_OF, stock_info=pd.DataFrame(columns=["code", "name"])
        )


def test_缺少ST列时标注该规则未生效() -> None:
    info = _stock_info().drop(columns=["is_st"])
    selection = select_universe(
        _history(), as_of=AS_OF, stock_info=info, universe_filter=UniverseFilter()
    )
    assert "未生效:ST 过滤" in "".join(selection.rules_not_applied)


def test_筛选摘要包含入选与排除数量() -> None:
    selection = select_universe(
        _history(), as_of=AS_OF, stock_info=_stock_info(), universe_filter=UniverseFilter(min_list_days=90)
    )
    summary = selection.summary()
    assert "只入选" in summary and "排除" in summary


# ============================================================ 成分持久化
def test_成分历史JSON往返(tmp_path) -> None:
    history = UniverseHistory()
    history.add("000300", date(2026, 7, 1), ["600000", "000001"])
    history.add("000300", date(2026, 10, 1), ["600000", "300750"])

    store = JsonUniverseStore(tmp_path / "universe.json")
    store.save(history)
    loaded = store.load()

    assert loaded.indices() == ("000300",)
    assert loaded.resolve("000300", date(2026, 9, 30)) == ("600000", "000001")
    assert loaded.resolve("000300", date(2026, 10, 14)) == ("600000", "300750")


def test_文件不存在时加载为空历史(tmp_path) -> None:
    assert JsonUniverseStore(tmp_path / "missing.json").load().indices() == ()


# ============================================================ 成分快照任务
class _MemberSource(InMemoryDataSource):
    def __init__(self, members: dict[str, tuple[str, ...]], fail: set[str] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.members = members
        self.fail = fail or set()
        self.calls: list[str] = []

    def index_members(self, index_code: str) -> tuple[str, ...]:
        self.calls.append(index_code)
        if index_code in self.fail:
            raise DataSourceError(f"{index_code} 抓取失败")
        return self.members.get(index_code, ())


def _member_task(tmp_path, source: _MemberSource, history: UniverseHistory | None = None):
    history = history if history is not None else UniverseHistory()
    store = JsonUniverseStore(tmp_path / "universe.json")
    log = MemoryTaskLog()
    task = IndexMembersSyncTask(
        source=source,
        history=history,
        store=store,
        task_log=log,
        index_codes=("000300", "000905"),
    )
    return task, history, store, log


def _source(**kwargs: object) -> _MemberSource:
    return _MemberSource(
        members={"000300": ("600000", "000001"), "000905": ("600000", "300750")},
        calendar=weekday_calendar(date(2026, 1, 1), date(2026, 12, 31)),
        **kwargs,
    )


def test_成分快照按抓取日登记并持久化(tmp_path) -> None:
    task, history, store, _ = _member_task(tmp_path, _source())

    result = task.run(on=AS_OF)

    assert result.ok
    assert result.added == {"000300": 2, "000905": 2}
    assert history.resolve("000300", AS_OF) == ("600000", "000001")
    assert store.load().resolve("000905", AS_OF) == ("600000", "300750")


def test_当日重复运行幂等跳过不重复请求(tmp_path) -> None:
    source = _source()
    task, _, _, log = _member_task(tmp_path, source)
    task.run(on=AS_OF)
    source.calls.clear()

    result = task.run(on=AS_OF)

    assert result.skipped == ("000300", "000905")
    assert source.calls == [], "当日已有快照就不该再打接口"
    assert not result.added
    assert log.records[-1].status is TaskStatus.SKIPPED


def test_早于抓取日的查询仍不可用(tmp_path) -> None:
    task, history, _, _ = _member_task(tmp_path, _source())
    task.run(on=AS_OF)

    with pytest.raises(DataNotReadyError):
        select_universe(
            history,
            as_of=date(2026, 10, 13),
            stock_info=_stock_info(),
            universe_filter=UniverseFilter(index_code="000300"),
        )


def test_空成分与失败分别记录(tmp_path) -> None:
    source = _source(fail={"000905"})
    source.members["000300"] = ()
    task, _, _, log = _member_task(tmp_path, source)

    result = task.run(on=AS_OF)

    assert result.empty == ("000300",)
    assert [item.code for item in result.failed] == ["000905"]
    assert not result.ok
    assert log.records[-1].status is TaskStatus.SUCCESS  # 单指数失败不整体中断
