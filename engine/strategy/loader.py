"""用户策略加载器（P1-B5）。

约定目录结构（见需求文档附录 B）::

    strategies/<name>/
        config.yaml    # 可选：{strategy_id: ..., params: {...}}
        strategy.py    # 必须定义 create_strategy(params: Mapping) -> Strategy

加载不是"执行任意代码那么简单"，三条底线：
1. 模块加载 / 工厂调用的一切异常包装为 ``StrategyError``（不泄漏堆栈到调用方之外）；
2. ``create_strategy`` 缺失或不可调用 → 明确报错，而不是 import 时静默;
3. ``config.yaml`` 的 params 与外部 override 合并（override 优先），策略收到的参数永远可追溯。
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from engine.errors import StrategyError
from engine.ports import Strategy

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedStrategy:
    strategy: Strategy
    strategy_id: str
    params: dict[str, Any]
    path: Path


def load_strategy(
    directory: Path | str,
    *,
    params_override: Mapping[str, Any] | None = None,
) -> LoadedStrategy:
    """从策略目录加载策略实例。"""
    path = Path(directory)
    strategy_file = path / "strategy.py"
    if not path.is_dir():
        raise StrategyError("策略目录不存在", detail={"directory": str(path)})
    if not strategy_file.is_file():
        raise StrategyError("策略目录缺少 strategy.py", detail={"directory": str(path)})

    module = _import_strategy_module(strategy_file)
    factory = getattr(module, "create_strategy", None)
    if not callable(factory):
        raise StrategyError(
            "策略模块必须定义 create_strategy(params: Mapping) -> Strategy",
            detail={"file": str(strategy_file)},
        )

    params = _load_params(path / "config.yaml")
    params.update(dict(params_override or {}))

    try:
        strategy = factory(params)
    except Exception as exc:
        raise StrategyError(
            f"create_strategy 执行失败：{type(exc).__name__}: {exc}", detail={"file": str(strategy_file)}
        ) from exc

    strategy_id = str(getattr(strategy, "strategy_id", "") or path.name)
    if not hasattr(strategy, "on_bar"):
        raise StrategyError(
            "create_strategy 返回的对象不是策略（缺少 on_bar）",
            detail={"file": str(strategy_file), "type": type(strategy).__name__},
        )
    _logger.info("策略已加载：%s（params=%s）", strategy_id, sorted(params))
    return LoadedStrategy(strategy=strategy, strategy_id=strategy_id, params=params, path=path)


def _import_strategy_module(strategy_file: Path) -> Any:
    module_name = f"aimoney_strategy_{abs(hash(str(strategy_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    if spec is None or spec.loader is None:  # pragma: no cover —— 常规路径不会发生
        raise StrategyError("无法构建策略模块加载器", detail={"file": str(strategy_file)})
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise StrategyError(
            f"策略模块加载失败：{type(exc).__name__}: {exc}", detail={"file": str(strategy_file)}
        ) from exc
    return module


def _load_params(config_file: Path) -> dict[str, Any]:
    if not config_file.is_file():
        return {}
    try:
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise StrategyError(f"策略配置解析失败：{config_file}", detail={"error": str(exc)}) from exc
    if not isinstance(payload, dict):
        raise StrategyError("策略配置必须是映射", detail={"file": str(config_file)})
    return dict(payload)


__all__ = ["LoadedStrategy", "load_strategy"]
