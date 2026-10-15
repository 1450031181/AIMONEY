"""敏感配置加解密（T-02）。

算法：AES-256-GCM（带认证标签，能检出密文被篡改）。
密文格式：``enc:v1:<base64url(nonce)>:<base64url(ciphertext+tag)>``
主密钥来源：环境变量 ``AIMONEY_MASTER_KEY``（**不落盘、不入库、不进 Git**）。

明文值（不带 ``enc:v1:`` 前缀）在解密时原样返回，方便本地用模拟盘开发；
但生产环境的券商账号/行情 Token 必须是密文。

命令行：
    python -m app.core.crypto keygen
    python -m app.core.crypto encrypt "明文"
    python -m app.core.crypto decrypt "enc:v1:..."
"""

from __future__ import annotations

import base64
import binascii
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import yaml
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from engine.errors import ConfigError, CryptoError

ENC_PREFIX: Final[str] = "enc:v1:"
MASTER_KEY_ENV: Final[str] = "AIMONEY_MASTER_KEY"
_KEY_BYTES: Final[int] = 32
_NONCE_BYTES: Final[int] = 12


# ============================================================ 主密钥
def generate_key() -> str:
    """生成一个新的 base64url 编码的 256 位主密钥。"""
    return _b64e(os.urandom(_KEY_BYTES))


def load_master_key(env_var: str = MASTER_KEY_ENV) -> bytes:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        raise ConfigError(
            f"未设置主密钥环境变量 {env_var}，无法加解密敏感配置",
            detail={"env_var": env_var, "hint": "python -m app.core.crypto keygen"},
        )
    key = _decode_key(raw)
    if len(key) != _KEY_BYTES:
        raise ConfigError(
            f"主密钥长度非法：期望 {_KEY_BYTES} 字节，实际 {len(key)} 字节",
            detail={"env_var": env_var},
        )
    return key


def _decode_key(raw: str) -> bytes:
    for decode in _KEY_DECODERS:
        try:
            return decode(raw)
        except (binascii.Error, ValueError, TypeError):
            continue
    raise ConfigError("主密钥格式非法：应为 base64 / base64url 或 hex 编码的 32 字节随机值")


# ============================================================ 加解密
def is_encrypted(value: str) -> bool:
    return isinstance(value, str) and value.startswith(ENC_PREFIX)


def encrypt(plaintext: str, key: bytes | None = None) -> str:
    secret = key if key is not None else load_master_key()
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(secret).encrypt(nonce, plaintext.encode("utf-8"), None)
    return f"{ENC_PREFIX}{_b64e(nonce)}:{_b64e(ciphertext)}"


def decrypt(token: str, key: bytes | None = None) -> str:
    """解密；非密文（无 enc:v1: 前缀）原样返回。"""
    if not is_encrypted(token):
        return token
    body = token[len(ENC_PREFIX) :]
    nonce_b64, sep, ciphertext_b64 = body.partition(":")
    if not sep:
        raise CryptoError("密文格式非法：期望 enc:v1:<nonce>:<ciphertext>", detail={"token_len": len(token)})
    secret = key if key is not None else load_master_key()
    try:
        nonce = _b64d(nonce_b64)
        ciphertext = _b64d(ciphertext_b64)
        return AESGCM(secret).decrypt(nonce, ciphertext, None).decode("utf-8")
    except InvalidTag as exc:
        raise CryptoError("解密失败：主密钥不匹配或密文已被篡改") from exc
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise CryptoError("解密失败：密文编码非法") from exc


def mask(value: str, *, head: int = 2, tail: int = 2) -> str:
    """日志打码：保留首尾少量字符，中间用 * 替代。"""
    if not value:
        return ""
    if len(value) <= head + tail:
        return "*" * len(value)
    return f"{value[:head]}{'*' * (len(value) - head - tail)}{value[-tail:]}"


# ============================================================ 密钥文件加载
def load_secrets(path: Path | str | None = None, key: bytes | None = None) -> dict[str, Any]:
    """加载敏感配置 yaml，并递归解密所有 ``enc:v1:`` 值。

    文件不存在时返回空 dict（本地开发用模拟盘，可以没有密钥文件）。
    """
    if path is None:
        return {}
    secret_file = Path(path)
    if not secret_file.exists():
        return {}
    try:
        raw = yaml.safe_load(secret_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"敏感配置文件解析失败：{secret_file}", detail={"error": str(exc)}) from exc
    if not isinstance(raw, dict):
        raise ConfigError("敏感配置文件根节点必须是映射（key: value）", detail={"file": str(secret_file)})
    decrypted = _decrypt_tree(raw, key)
    if not isinstance(decrypted, dict):
        raise ConfigError("敏感配置解密结果不是映射", detail={"file": str(secret_file)})
    return decrypted


def _decrypt_tree(node: Any, key: bytes | None) -> Any:
    if isinstance(node, dict):
        return {k: _decrypt_tree(v, key) for k, v in node.items()}
    if isinstance(node, list):
        return [_decrypt_tree(item, key) for item in node]
    if isinstance(node, str) and is_encrypted(node):
        return decrypt(node, key)
    return node


# ============================================================ base64 工具
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded)


#: 主密钥允许的编码形式（依次尝试）。显式标注类型，避免 mypy 因 lambda 推断不出签名而报 untyped-call
_KEY_DECODERS: Final[tuple[Callable[[str], bytes], ...]] = (_b64d, bytes.fromhex)


# ============================================================ CLI
_USAGE = """AIMoney 密钥工具

用法：
  python -m app.core.crypto keygen             生成主密钥（写入环境变量 AIMONEY_MASTER_KEY）
  python -m app.core.crypto encrypt <明文>     加密明文，输出 enc:v1:...
  python -m app.core.crypto decrypt <密文>     解密密文
  python -m app.core.crypto mask <明文>        打码查看
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_USAGE)
        return 0

    command, rest = args[0], args[1:]
    try:
        if command == "keygen":
            print(generate_key())
            return 0
        if command == "mask":
            if not rest:
                print("缺少参数：<明文>", file=sys.stderr)
                return 2
            print(mask(rest[0]))
            return 0
        if command in {"encrypt", "decrypt"}:
            if not rest:
                print(f"缺少参数：<{'明文' if command == 'encrypt' else '密文'}>", file=sys.stderr)
                return 2
            print(encrypt(rest[0]) if command == "encrypt" else decrypt(rest[0]))
            return 0
    except (ConfigError, CryptoError) as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 1

    print(f"未知命令：{command}\n", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
