"""AkShare 联网冒烟测试（默认不执行）。

运行方式::

    pytest -m network tests/test_akshare_live.py

为什么单独隔离：外网接口会失败、会改字段、会被限流，让它污染日常 `pytest`
会让人逐渐习惯"红的是网络问题"，最终所有红灯都被忽略——那比没有测试更危险。
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import pytest

from engine.datasource.akshare_source import AkShareClient, AkShareDataSource

pytestmark = pytest.mark.network

#: 用浦发银行做冒烟：老牌主板股，停牌/退市风险极低，历史数据稳定
SMOKE_CODE = "600000"

#: akshare 免费接口对连续请求比较敏感（实测会出现 RemoteDisconnected），
#: 联网用例统一放慢节奏：间隔 1.5s、退避 1.0s 起。
_QUIET_INTERVAL = 1.5


@pytest.fixture
def source() -> AkShareDataSource:
    return AkShareDataSource(request_interval=_QUIET_INTERVAL, max_retries=3, retry_base_delay=1.0)


def test_能取到交易日历(source: AkShareDataSource) -> None:
    calendar = source.trading_calendar()
    assert calendar.count() > 4000, "A 股交易日历应覆盖 2005 年以来四千多个交易日"
    assert calendar.is_trade_day(date(2026, 10, 14))


def test_能取到日线且字段与单位正确(source: AkShareDataSource) -> None:
    end = date.today()
    start = end - timedelta(days=30)
    frame = source.daily([SMOKE_CODE], start, end)

    assert not frame.empty
    assert set(frame["code"]) == {SMOKE_CODE}
    assert frame["close"].min() > 0
    # 成交量应为人股（手 × 100），量级远大于「手」
    assert frame["volume"].max() > 100_000


def test_能取到股票基础信息(source: AkShareDataSource) -> None:
    info = source.stock_info()
    assert len(info) > 4000, "A 股标的数应在 4000 以上"
    assert info["code"].str.len().eq(6).all()


def test_成交量换算与akshare原始数据严格一致(source: AkShareDataSource) -> None:
    """交叉校验单位换算：我们输出的 volume 必须等于 akshare 原始「成交量（手）」× 100。

    这类错误不会报异常，只会让所有量价因子和成交约束静默失准，因此必须与原始接口对账。
    """
    end = date.today()
    start = end - timedelta(days=20)
    ours = source.daily([SMOKE_CODE], start, end, adj="none")

    time.sleep(_QUIET_INTERVAL)  # 直连 client 不受 source 的限速器约束，这里手动让路
    raw = AkShareClient().daily(symbol=SMOKE_CODE, start=start, end=end, adjust="")
    raw["date"] = pd.to_datetime(raw["日期"]).dt.normalize()
    merged = ours.merge(raw[["date", "成交量"]], on="date", how="inner")

    assert not merged.empty, "对账窗口内应有重叠交易日"
    assert (merged["volume"].to_numpy() == (merged["成交量"] * 100).to_numpy()).all()
