"""回测服务（P1-B6）：从"本地数据 + 策略目录"到"报告文件"的完整用例。

这一层是回测的编排边界：
- 数据侧：校验本地数据就绪 → 读取区间行情 → 构建回测数据源；
- 策略侧：加载策略目录（config.yaml + CLI override 合并）；
- 输出侧：metrics.json / nav.csv / trades.csv 落盘到 ``data/backtests/<run>/``。

报告必须落盘而不是只在控制台打印：回测结论要能被翻旧账，
"上次那个策略跑出来多少"必须能找到文件。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine.backtest.data import DataFrameBarProvider
from engine.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from engine.backtest.metrics import PerformanceMetrics
from engine.datahub.store import ParquetStore
from engine.datasource.local_source import LocalDataSource
from engine.errors import ValidationError
from engine.strategy.loader import load_strategy

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    strategy_dir: Path
    start: date
    end: date
    initial_cash: str = "1000000"
    codes: tuple[str, ...] = ()
    params_override: Mapping[str, Any] = field(default_factory=dict)
    slippage_bps: float = 5.0
    max_vol_ratio: float = 0.10
    max_delay_days: int = 2
    risk_free_rate: float = 0.02
    dataset: str = "daily"

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValidationError(
                "回测区间非法：start 晚于 end",
                detail={"start": self.start.isoformat(), "end": self.end.isoformat()},
            )


@dataclass
class BacktestRunReport:
    strategy_id: str
    params: dict[str, Any]
    output_dir: Path
    result: BacktestResult
    start: date
    end: date

    @property
    def metrics(self) -> PerformanceMetrics:
        return self.result.metrics

    def summary(self) -> str:
        metrics = self.result.metrics
        return (
            f"[{self.strategy_id}] {self.start}~{self.end}（{self.result.trading_days} 个交易日）"
            f" 总收益 {metrics.total_return:+.2%}，年化 {metrics.annual_return:+.2%}，"
            f"最大回撤 {metrics.max_drawdown:.2%}，夏普 {metrics.sharpe_ratio:.2f}，"
            f"成交 {len(self.result.trades)} 笔"
            + (f"，⚠ 策略停仓：{self.result.halted}" if self.result.halted else "")
        )


class BacktestService:
    """回测编排服务（只依赖 Settings，测试可注入独立数据目录）。"""

    def __init__(self, settings: Any) -> None:  # Settings（避免循环依赖，字段级使用）
        self.settings = settings

    def run(self, request: BacktestRequest) -> BacktestRunReport:
        store = ParquetStore(self.settings.parquet_dir)
        store.require_data(request.dataset)  # 数据门禁：没同步过数据直接拒绝，而不是跑出空回测

        local = LocalDataSource(store, dataset=request.dataset)
        frame = local.daily(
            list(request.codes) if request.codes else self._all_codes(local, request.end),
            request.start,
            request.end,
            adj="post",
        )
        if frame.empty:
            raise ValidationError(
                "回测区间内没有行情数据",
                detail={
                    "dataset": request.dataset,
                    "start": request.start.isoformat(),
                    "end": request.end.isoformat(),
                    "hint": "检查区间与本地数据覆盖范围（aimoney-data coverage）",
                },
            )

        loaded = load_strategy(request.strategy_dir, params_override=request.params_override)
        config = BacktestConfig(
            start=request.start,
            end=request.end,
            initial_cash=request.initial_cash,
            slippage_bps=request.slippage_bps,
            max_vol_ratio=request.max_vol_ratio,
            max_delay_days=request.max_delay_days,
            risk_free_rate=request.risk_free_rate,
        )
        engine = BacktestEngine(config, provider=DataFrameBarProvider(frame), strategy=loaded.strategy)
        result = engine.run()

        output_dir = self._write_report(
            result,
            strategy_id=loaded.strategy_id,
            params=loaded.params,
            request=request,
        )
        _logger.info("回测完成：%s，报告目录 %s", loaded.strategy_id, output_dir)
        return BacktestRunReport(
            strategy_id=loaded.strategy_id,
            params=loaded.params,
            output_dir=output_dir,
            result=result,
            start=request.start,
            end=request.end,
        )

    # ============================================================ 内部
    def _all_codes(self, local: LocalDataSource, end: date) -> list[str]:
        """全部在市标的（退市标的没有新行情，排除）。"""
        info = local.stock_info()
        codes = info["code"].astype(str)
        if "delist_date" in info.columns:
            alive = info["delist_date"].isna() | (pd.to_datetime(info["delist_date"]) > pd.Timestamp(end))
            codes = codes[alive]
        return sorted(set(codes))

    def _write_report(
        self,
        result: BacktestResult,
        *,
        strategy_id: str,
        params: dict[str, Any],
        request: BacktestRequest,
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.settings.data.root_dir) / "backtests" / f"{strategy_id}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "strategy_id": strategy_id,
            "params": params,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "trading_days": result.trading_days,
            "total_signals": result.total_signals,
            "skipped_signals": result.skipped_signals,
            "halted": result.halted,
            "config": {
                "initial_cash": str(request.initial_cash),
                "slippage_bps": request.slippage_bps,
                "max_vol_ratio": request.max_vol_ratio,
                "max_delay_days": request.max_delay_days,
                "risk_free_rate": request.risk_free_rate,
                "codes": list(request.codes) or "universe",
            },
            "metrics": result.metrics.to_jsonable(),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        nav_frame = result.nav.rename("nav").to_frame()
        nav_frame.index.name = "date"
        nav_frame.to_csv(output_dir / "nav.csv", encoding="utf-8")
        result.trades_frame().to_csv(output_dir / "trades.csv", index=False, encoding="utf-8")
        return output_dir


__all__ = ["BacktestRequest", "BacktestRunReport", "BacktestService"]
