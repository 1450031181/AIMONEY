"""T-02 敏感配置加密测试：往返、篡改检测、密钥缺失 fail-fast、日志打码。"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.core.crypto import (
    ENC_PREFIX,
    MASTER_KEY_ENV,
    decrypt,
    encrypt,
    generate_key,
    is_encrypted,
    load_master_key,
    load_secrets,
    mask,
)
from engine.errors import ConfigError, CryptoError


@pytest.fixture
def master_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = generate_key()
    monkeypatch.setenv(MASTER_KEY_ENV, key)
    return key


def test_生成的密钥为32字节(master_key: str) -> None:
    assert len(load_master_key()) == 32


def test_加解密往返(master_key: str) -> None:
    plaintext = "国金证券-8880123456"
    token = encrypt(plaintext)
    assert is_encrypted(token)
    assert token.startswith(ENC_PREFIX)
    assert plaintext not in token
    assert decrypt(token) == plaintext


def test_相同明文每次密文不同() -> None:
    """随机 nonce 保证同一明文两次密文不同，避免密文可比对泄漏信息。"""
    key = base64.urlsafe_b64decode(generate_key() + "==")
    first = encrypt("同一明文", key)
    second = encrypt("同一明文", key)
    assert first != second
    assert decrypt(first, key) == decrypt(second, key) == "同一明文"


def test_明文原样返回无需密钥(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MASTER_KEY_ENV, raising=False)
    assert decrypt("simulated") == "simulated"


def test_未设置主密钥时加密失败(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MASTER_KEY_ENV, raising=False)
    with pytest.raises(ConfigError):
        encrypt("明文")


def test_主密钥格式非法(master_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "这不是合法的base64!!!")
    with pytest.raises(ConfigError):
        load_master_key()


def test_主密钥长度不足被拒(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MASTER_KEY_ENV, "YWJj")  # "abc" 只有 3 字节
    with pytest.raises(ConfigError):
        load_master_key()


def test_密钥不匹配解密失败(master_key: str) -> None:
    token = encrypt("明文")
    other_key = base64.urlsafe_b64decode(generate_key() + "==")
    with pytest.raises(CryptoError):
        decrypt(token, other_key)


def test_密文被篡改可检出(master_key: str) -> None:
    token = encrypt("明文")
    # 篡改密文最后一个字符，GCM 认证标签应当校验失败
    tampered = token[:-2] + ("A" if token[-2] != "A" else "B") + token[-1]
    with pytest.raises(CryptoError):
        decrypt(tampered)


def test_密文格式非法(master_key: str) -> None:
    with pytest.raises(CryptoError):
        decrypt(f"{ENC_PREFIX}没有分隔符")


def test_日志打码() -> None:
    assert mask("8880123456") == "88******56"
    assert mask("") == ""
    assert mask("abc") == "***"


def test_加载敏感配置递归解密(master_key: str, tmp_path: Path) -> None:
    token = encrypt("secret-token")
    secret_file = tmp_path / "secrets.yaml"
    secret_file.write_text(
        f"data_source:\n  tushare_token: {token}\nbroker:\n  mode: simulated\n  qmt:\n    account_id: {token}\n",
        encoding="utf-8",
    )

    secrets = load_secrets(secret_file)
    assert secrets["data_source"]["tushare_token"] == "secret-token"
    assert secrets["broker"]["qmt"]["account_id"] == "secret-token"
    assert secrets["broker"]["mode"] == "simulated"


def test_密钥文件不存在返回空字典() -> None:
    assert load_secrets(Path("不存在的目录/secrets.yaml")) == {}
