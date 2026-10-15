"""分层配置加载（T-02）。

优先级（低 -> 高）：字段默认值 < conf/aimoney.yaml < 环境变量 < 代码传参。

环境变量约定：前缀 ``AIMONEY_``，嵌套用 ``__`` 分隔
    AIMONEY_LOG__LEVEL=DEBUG
    AIMONEY_DATA__MAX_RETRIES=5

设计约束：
- **fail-fast**：配置非法（缺失/类型错/多余字段）直接拒绝启动，绝不带着半懂配置运行；
- 路径类字段默认按项目根目录推导，换机器部署不需要改配置；
- 敏感信息不放在本文件加载的 yaml 中，走 conf/secrets.yaml + AES-GCM 解密。
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from engine.errors import ConfigError

# <项目根>/app/core/config.py -> parents[2] == <项目根>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "conf" / "aimoney.yaml"

_ENV_PREFIX = "AIMONEY_"
#: 指定配置文件路径的环境变量，便于多环境/测试切换配置
CONFIG_FILE_ENV: Final[str] = f"{_ENV_PREFIX}CONFIG_FILE"

_logger = logging.getLogger(__name__)


class _Section(BaseModel):
    """配置片段基类：统一禁止未知字段，避免拼错字段名却静默失效。"""

    model_config = ConfigDict(extra="forbid")


class AppSection(_Section):
    name: str = "aimoney"
    env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = True


class LogSection(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    #: True 输出 JSON（生产采集用），False 输出人类可读文本（开发用）
    # 注意：字段名不能叫 json，会与 pydantic BaseModel.json() 方法冲突
    json_output: bool = False
    dir: Path | None = None  # 默认 <data.root>/logs


class RiskSection(_Section):
    """风控阈值（事前限额 + 事中触线）。

    ``enabled=False`` 只应出现在研究/调试场景：实盘链路必须保证风控在岗，
    设计文档 D6「风控不可绕过」不是可选项。
    """

    enabled: bool = True
    max_weight: float = Field(default=0.2, gt=0.0, le=1.0)
    max_order_value: Decimal = Field(default=Decimal("200000"), gt=0)
    max_orders_per_day: int = Field(default=100, gt=0)
    max_drawdown: float = Field(default=0.1, gt=0.0, le=1.0)
    max_daily_loss: float = Field(default=0.03, gt=0.0, le=1.0)
    liquidation_drawdown: float = Field(default=0.2, gt=0.0, le=1.0)


class QualitySection(_Section):
    """数据质量门禁阈值。超出即判定 FAIL —— 宁可当日空仓，不拿脏数据下单。

    注意这里**没有**"统一涨跌幅上限"这类阈值：主板 10%、创业板 20%、北交所 30%、
    ST 5%，一刀切必然要么误报正常波动、要么漏掉真异常。涨跌幅异常改由按板块自适应的
    连续性探针判定（``engine.datahub.adjust``），此处只配置它的容差。
    """

    max_missing_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    continuity_tolerance: float = Field(default=0.02, ge=0.0, le=0.1)
    max_volume_jump_ratio: float = Field(default=10.0, gt=0.0)
    trade_gate_on_fail: bool = True


class DataSection(_Section):
    root: Path | None = None
    parquet_dir: Path | None = None
    #: 可选：持久化分析库。默认查询直接读 Parquet（见 DuckDBReader），
    #: 不依赖该文件存在，避免"库文件损坏/并发锁"变成启动故障。
    duckdb_path: Path | None = None
    quality_dir: Path | None = None
    gate_file: Path | None = None
    kill_switch_file: Path | None = None
    secrets_file: Path | None = None
    #: 数据源请求最小间隔（秒），防限流封禁
    request_interval: float = Field(default=0.3, ge=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff: float = Field(default=1.5, ge=1.0)
    quality: QualitySection = QualitySection()

    @model_validator(mode="after")
    def _derive_paths(self) -> DataSection:
        if self.root is None:
            self.root = PROJECT_ROOT / "data"
        if self.parquet_dir is None:
            self.parquet_dir = self.root / "parquet"
        if self.duckdb_path is None:
            self.duckdb_path = self.root / "duckdb" / "aimoney.duckdb"
        if self.quality_dir is None:
            self.quality_dir = self.root / "quality"
        if self.gate_file is None:
            self.gate_file = self.root / "state" / "trade_gate.json"
        if self.kill_switch_file is None:
            self.kill_switch_file = self.root / "state" / "kill_switch.json"
        if self.secrets_file is None:
            self.secrets_file = PROJECT_ROOT / "conf" / "secrets.yaml"
        return self

    @property
    def root_dir(self) -> Path:
        """数据根目录（校验阶段已填充，此处断言仅用于类型收窄）。"""
        assert self.root is not None
        return self.root


class Settings(BaseSettings):
    """全局配置对象。通过 ``get_settings()`` 获取单例。"""

    model_config = SettingsConfigDict(
        env_prefix=_ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
        case_sensitive=False,
    )

    app: AppSection = AppSection()
    log: LogSection = LogSection()
    data: DataSection = DataSection()
    risk: RiskSection = RiskSection()

    # -------------------------------------------------- 配置源装配
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """按优先级装配配置源（先者优先）。

        注意（踩过的坑）：pydantic-settings 仅在 model_config 里写 ``yaml_file``
        是**不会生效且不报错**的，必须在此显式加入 ``YamlConfigSettingsSource``。
        为防止这种"配了但没加载"的静默故障，本方法对缺失的显式配置文件直接抛错。
        """
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        config_file = _resolve_config_file()
        if config_file is not None:
            sources.append(
                YamlConfigSettingsSource(settings_cls, yaml_file=config_file, yaml_file_encoding="utf-8")
            )
        sources.append(file_secret_settings)
        return tuple(sources)

    @model_validator(mode="after")
    def _derive_log_dir(self) -> Settings:
        if self.log.dir is None:
            self.log.dir = self.data.root_dir / "logs"
        return self

    # -------------------------------------------------- 便捷属性
    @property
    def is_prod(self) -> bool:
        return self.app.env == "prod"

    @property
    def parquet_dir(self) -> Path:
        """已保证非 None（模型校验阶段填充）。"""
        assert self.data.parquet_dir is not None
        return self.data.parquet_dir

    @property
    def duckdb_path(self) -> Path:
        assert self.data.duckdb_path is not None
        return self.data.duckdb_path

    @property
    def log_dir(self) -> Path:
        assert self.log.dir is not None
        return self.log.dir

    @property
    def secrets_file(self) -> Path:
        assert self.data.secrets_file is not None
        return self.data.secrets_file

    @property
    def quality_dir(self) -> Path:
        assert self.data.quality_dir is not None
        return self.data.quality_dir

    @property
    def gate_file(self) -> Path:
        assert self.data.gate_file is not None
        return self.data.gate_file

    @property
    def kill_switch_file(self) -> Path:
        assert self.data.kill_switch_file is not None
        return self.data.kill_switch_file

    def ensure_dirs(self) -> list[Path]:
        """创建运行所需目录，返回创建/确认过的目录列表（幂等）。"""
        targets: list[Path] = [
            self.data.root_dir,
            self.parquet_dir,
            self.duckdb_path.parent,
            self.log_dir,
            self.quality_dir,
            self.gate_file.parent,
        ]
        for path in targets:
            path.mkdir(parents=True, exist_ok=True)
        return targets


def _resolve_config_file() -> Path | None:
    """定位配置文件：环境变量显式指定则必须存在；否则用默认文件，缺失时跳过 yaml 源。"""
    raw = os.environ.get(CONFIG_FILE_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"配置文件不存在：{path}",
                detail={"env_var": CONFIG_FILE_ENV, "path": str(path)},
            )
        return path
    if DEFAULT_CONFIG_FILE.is_file():
        return DEFAULT_CONFIG_FILE
    _logger.warning("默认配置文件不存在，将仅使用默认值与环境变量：%s", DEFAULT_CONFIG_FILE)
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例。首次调用即校验，配置非法会在此抛 pydantic ValidationError。"""
    return Settings()


def reload_settings() -> Settings:
    """清缓存重新加载（测试与配置热更新用）。"""
    get_settings.cache_clear()
    return get_settings()


__all__ = [
    "CONFIG_FILE_ENV",
    "DEFAULT_CONFIG_FILE",
    "PROJECT_ROOT",
    "AppSection",
    "DataSection",
    "LogSection",
    "QualitySection",
    "Settings",
    "get_settings",
    "reload_settings",
]
