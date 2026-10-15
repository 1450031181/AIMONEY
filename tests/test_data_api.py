"""T-12 数据接口测试：覆盖率 / 门禁 / 质量报告，以及异常到 HTTP 的映射。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from engine.datahub.quality import TradeGate
from engine.datahub.store import ParquetStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(data={"root": str(tmp_path / "data")})
    settings.ensure_dirs()
    store = ParquetStore(settings.parquet_dir)
    store.write(
        "daily",
        pd.DataFrame(
            [
                {
                    "code": "600000",
                    "date": "2026-10-13",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2,
                    "volume": 1000,
                    "adj_factor": 1.0,
                },
                {
                    "code": "000001",
                    "date": "2026-10-14",
                    "open": 20.0,
                    "high": 20.5,
                    "low": 19.9,
                    "close": 20.2,
                    "volume": 2000,
                    "adj_factor": 1.0,
                },
            ]
        ),
        data_version="20261014.1",
    )
    return settings


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_健康检查(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["gate"]["open"] is False, "未跑质量校验前门禁必须关闭"


def test_覆盖率接口(client: TestClient) -> None:
    response = client.get("/api/v1/data/coverage")

    assert response.status_code == 200
    body = response.json()
    dataset = body["dataset"]
    assert dataset["dataset"] == "daily"
    assert dataset["row_count"] == 2
    assert dataset["code_count"] == 2
    assert dataset["start"] == "2026-10-13"
    assert dataset["end"] == "2026-10-14"
    assert dataset["data_version"] == "20261014.1"
    assert body["gate"]["open"] is False


def test_数据集名非法返回422(client: TestClient) -> None:
    response = client.get("/api/v1/data/coverage", params={"dataset": "bad/name"})

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_门禁接口反映状态(client: TestClient, settings: Settings) -> None:
    TradeGate(settings.gate_file).open(as_of=date(2026, 10, 14), note="质量 PASS")

    body = client.get("/api/v1/data/gate").json()

    assert body["open"] is True
    assert body["as_of"] == "2026-10-14"


def test_质量报告不存在返回404(client: TestClient) -> None:
    response = client.get("/api/v1/data/quality/2026-10-14")

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


def test_质量报告存在时返回内容(client: TestClient, settings: Settings) -> None:
    report = {"dataset": "daily", "status": "PASS", "failed_rules": []}
    path = settings.quality_dir / "quality_20261014.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    response = client.get("/api/v1/data/quality/2026-10-14")

    assert response.status_code == 200
    assert response.json()["status"] == "PASS"


def test_日期参数非法返回422(client: TestClient) -> None:
    response = client.get("/api/v1/data/quality/not-a-date")

    assert response.status_code == 422
