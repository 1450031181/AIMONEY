"""参考数据存储（T-13）：低频、不按日期分区的基础信息。

为什么单独存：``stock_info``（股票列表 / ST / 上市日）没有 ``date`` 主键，
塞进 ``ParquetStore`` 的 ``year=/month=`` 分区布局只会让"数据集"概念失真
（``list`` 会发现它、DuckDB 层会当成分区数据集去扫）。

放在 ``data/reference/`` 下，与行情分区数据彻底分开；同步任务每次顺带刷新一次，
之后 ``check --index`` 之类的离线场景就不再依赖网络。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.datahub.schema import STOCK_INFO_REQUIRED_COLUMNS, ensure_columns
from engine.errors import DataNotReadyError

STOCK_INFO = "stock_info"


class ReferenceStore:
    """参考数据的原子读写（单文件 parquet）。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def save(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            frame.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self.path(name)
        if not path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_stock_info(self) -> pd.DataFrame:
        """读取股票基础信息；缺失即报错（调用方应提示先执行 sync）。"""
        frame = self.load(STOCK_INFO)
        if frame.empty:
            raise DataNotReadyError(
                "本地尚无股票基础信息，无法进行股票池筛选",
                detail={"path": str(self.path(STOCK_INFO)), "hint": "先执行 aimoney-data sync"},
            )
        ensure_columns(frame, STOCK_INFO_REQUIRED_COLUMNS, dataset=STOCK_INFO)
        return frame


__all__ = ["STOCK_INFO", "ReferenceStore"]
