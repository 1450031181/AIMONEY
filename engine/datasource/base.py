"""数据源适配器公共设施（T-06）：限速与重试。

**限速**不是可选项：免费行情源有 QPS 限制，全 A 股 5000 只标的的批量拉取一旦
不做限速，轻则被限流、重则被封 IP，同步任务再重试也救不回来。

**重试**必须带退避（backoff）：立即重试在数据源抖动时只会放大压力。
最终失败统一包装成 ``DataSourceError``，让上层（同步任务）能用一种异常类型决定
"是否禁止当日交易"。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from engine.errors import DataSourceError

T = TypeVar("T")


@dataclass
class RateLimiter:
    """线程安全的最小间隔限速器。

    用法：在**每一次**外部请求前调用 ``acquire()``。
    """

    min_interval: float = 0.3
    #: 可注入的睡眠函数（测试传零等待替身，避免用例变慢）
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _last_call: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self.sleep(remaining)
            self._last_call = time.monotonic()


def retry_call(
    operation: Callable[[], T],
    *,
    description: str,
    retries: int = 3,
    backoff: float = 1.5,
    base_delay: float = 0.5,
    retry_on: Sequence[type[BaseException]] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """带指数退避的重试；耗尽后抛 ``DataSourceError``（保留原始异常为 cause）。

    ``sleep`` 可注入，便于测试时零等待。
    """
    attempt = 0
    while True:
        try:
            return operation()
        # 异常类型来自参数（合法调用场景），不是裸 except
        except tuple(retry_on) as exc:
            attempt += 1
            if attempt > retries:
                raise DataSourceError(
                    f"{description} 失败，已重试 {retries} 次",
                    detail={
                        "description": description,
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ) from exc
            sleep(base_delay * (backoff ** (attempt - 1)))


__all__ = ["RateLimiter", "retry_call"]
