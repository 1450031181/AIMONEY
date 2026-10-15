"""T-02 配置加载测试：分层优先级、fail-fast、目录初始化。

回归重点：pydantic-settings 仅在 model_config 中写 ``yaml_file`` 不会生效也**不会报错**，
因此必须有测试证明 yaml 真的被加载了（否则整个配置文件会静默失效）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import CONFIG_FILE_ENV, PROJECT_ROOT, Settings
from engine.errors import ConfigError


def _write_config(tmp_path: Path, text: str) -> Path:
    config = tmp_path / "aimoney.yaml"
    config.write_text(text, encoding="utf-8")
    return config


def test_默认路径按项目根目录推导() -> None:
    settings = Settings()
    assert settings.data.root == PROJECT_ROOT / "data"
    assert settings.parquet_dir == PROJECT_ROOT / "data" / "parquet"
    assert settings.duckdb_path == PROJECT_ROOT / "data" / "duckdb" / "aimoney.duckdb"
    assert settings.log_dir == PROJECT_ROOT / "data" / "logs"
    assert settings.secrets_file == PROJECT_ROOT / "conf" / "secrets.yaml"


def test_yaml配置文件真实生效(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, "log:\n  level: WARNING\ndata:\n  max_retries: 7\n")
    monkeypatch.setenv(CONFIG_FILE_ENV, str(config))

    settings = Settings()
    assert settings.log.level == "WARNING", "yaml 未生效：YamlConfigSettingsSource 可能未装配"
    assert settings.data.max_retries == 7


def test_环境变量优先级高于yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, "log:\n  level: WARNING\n")
    monkeypatch.setenv(CONFIG_FILE_ENV, str(config))
    monkeypatch.setenv("AIMONEY_LOG__LEVEL", "ERROR")

    assert Settings().log.level == "ERROR"


def test_传参优先级高于环境变量(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, "log:\n  level: WARNING\n")
    monkeypatch.setenv(CONFIG_FILE_ENV, str(config))
    monkeypatch.setenv("AIMONEY_LOG__LEVEL", "ERROR")

    assert Settings(log={"level": "DEBUG"}).log.level == "DEBUG"


def test_指定的配置文件不存在时直接报错(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(tmp_path / "不存在.yaml"))
    with pytest.raises(ConfigError):
        Settings()


def test_非法取值直接拒绝启动() -> None:
    with pytest.raises(PydanticValidationError):
        Settings(log={"level": "NOT_A_LEVEL"})


def test_未知字段直接拒绝启动() -> None:
    """拼错字段名必须报错，而不是静默忽略。"""
    with pytest.raises(PydanticValidationError):
        Settings(log={"levle": "DEBUG"})


def test_质量阈值越界被拒() -> None:
    with pytest.raises(PydanticValidationError):
        Settings(data={"quality": {"max_missing_ratio": 1.5}})


def test_请求间隔不能为负() -> None:
    with pytest.raises(PydanticValidationError):
        Settings(data={"request_interval": -1})


def test_ensure_dirs幂等创建目录(tmp_path: Path) -> None:
    settings = Settings(data={"root": str(tmp_path / "data")})
    first = settings.ensure_dirs()
    second = settings.ensure_dirs()  # 重复调用不应抛异常
    assert all(path.is_dir() for path in first)
    assert all(path.is_dir() for path in second)
    assert (tmp_path / "data" / "parquet").is_dir()
    assert (tmp_path / "data" / "duckdb").is_dir()
    assert (tmp_path / "data" / "logs").is_dir()
