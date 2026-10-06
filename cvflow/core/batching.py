"""动态批处理执行器：把并发到来的请求合并成一批，一次执行。

典型用法是深度学习推理：多路相机的图像分别到达，在一个很短的等待窗口里凑成一个批次，
一次推理完成，再把结果按来源拆回去。等待窗口到期就按当前已有的数量执行，不会无限等。

调度采用"领队-跟随"模型，不额外开线程：
  * 任何一个提交者发现当前没有领队，就自己当领队；
  * 领队在等待窗口内收集队列，然后执行一批，并唤醒该批里的所有人；
  * 其余提交者作为跟随者等待自己的结果，领队让位后它们中的一个会接着当领队。

``wait_s`` 为 0 时完全不等待，直接执行当前队列里已有的请求。同一次提交的多个条目
（例如一个节点的多个输入图像）天然在同一批里，不受等待窗口影响。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Sequence

__all__ = ["BatchExecutor", "BatchTimeout"]


class BatchTimeout(TimeoutError):
    """等待批处理结果超时。"""


class _Request:
    __slots__ = ("items", "results", "error", "done")

    def __init__(self, items: Sequence[Any]) -> None:
        self.items = list(items)
        self.results: list[Any] = []
        self.error: BaseException | None = None
        self.done = False


class BatchExecutor:
    """把并发请求合并成批执行。

    ``fn`` 接收一个条目列表，返回等长、一一对应的结果列表。``fn`` 由领队线程调用，
    同一时刻只有一个线程在调用它，因此 ``fn`` 内部不需要自己加锁。
    """

    def __init__(self, fn: Callable[[list], Sequence[Any]], max_batch: int = 8,
                 wait_s: float = 0.0, name: str = "", chunk: bool = True) -> None:
        self._fn = fn
        self.max_batch = max(1, int(max_batch))
        # chunk=False 表示 fn 自己负责把超过 max_batch 的提交切块执行。需要"整批只选一个领队"
        # 之类的全局决定时必须这样，否则 fn 只看到自己那一块，没法做跨块的决定。
        self.chunk = bool(chunk)
        self.wait_s = max(0.0, float(wait_s))
        self.name = name
        self._cv = threading.Condition()
        self._queue: deque[_Request] = deque()
        self._leader = False
        self._closed = False
        self.stats = {"submits": 0, "items": 0, "batches": 0, "largest_batch": 0, "waited_s": 0.0}

    # ---- 对外 ----
    def submit(self, items: Sequence[Any], timeout: float = 30.0) -> list[Any]:
        """提交一组条目，返回对应的结果。多个线程并发调用时会被合并成一批执行。"""
        if self._closed:
            raise RuntimeError(f"批处理执行器 {self.name!r} 已关闭")
        items = list(items)
        if not items:
            return []
        req = _Request(items)
        end = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            self._queue.append(req)
            self.stats["submits"] += 1
            self.stats["items"] += len(items)
            self._cv.notify_all()          # 可能让正在等窗口的领队提前凑满
        while True:
            with self._cv:
                if req.done:
                    break
                if self._leader:           # 已有领队，当跟随者等结果
                    left = end - time.monotonic()
                    if left <= 0 or not self._cv.wait(left):
                        if not req.done:
                            self._abandon(req)
                            raise BatchTimeout(f"批处理 {self.name!r} 等待结果超时（{timeout} 秒）")
                    continue
                self._leader = True        # 没有领队，自己来
            try:
                self._collect_and_run()
            finally:
                with self._cv:
                    self._leader = False
                    self._cv.notify_all()
        if req.error is not None:
            raise req.error
        return req.results

    def pending(self) -> int:
        with self._cv:
            return sum(len(r.items) for r in self._queue)

    def close(self) -> None:
        """拒绝新请求并唤醒所有等待者。"""
        with self._cv:
            self._closed = True
            for r in self._queue:
                if not r.done:
                    r.error = RuntimeError(f"批处理执行器 {self.name!r} 已关闭")
                    r.done = True
            self._queue.clear()
            self._cv.notify_all()

    # ---- 内部 ----
    def _abandon(self, req: _Request) -> None:
        """超时离开的请求要从队列里摘掉，否则会被后来的领队当成有效请求执行。"""
        try:
            self._queue.remove(req)
        except ValueError:
            pass

    def _collect_and_run(self) -> None:
        batch = self._collect()
        if not batch:
            return
        flat = [it for req in batch for it in req.items]
        try:
            results = self._run_in_chunks(flat)
            if len(results) != len(flat):
                raise RuntimeError(f"批处理 {self.name!r}：期望 {len(flat)} 个结果，实际 {len(results)} 个")
            at = 0
            for req in batch:
                req.results = results[at:at + len(req.items)]
                at += len(req.items)
        except BaseException as e:         # 整批失败，每个提交者都拿到同一个异常
            for req in batch:
                req.error = e
        finally:
            with self._cv:
                self.stats["batches"] += 1
                self.stats["largest_batch"] = max(self.stats["largest_batch"], len(flat))
                for req in batch:
                    req.done = True
                self._cv.notify_all()

    def _collect(self) -> list[_Request]:
        """在等待窗口内收集请求，返回本批要执行的请求组。"""
        t0 = time.monotonic()
        deadline = t0 + self.wait_s
        with self._cv:
            while self.wait_s > 0:
                queued = sum(len(r.items) for r in self._queue)
                if queued >= self.max_batch or not self._queue:
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)
            self.stats["waited_s"] += time.monotonic() - t0
            batch: list[_Request] = []
            total = 0
            while self._queue:
                head = self._queue[0]
                if batch and total + len(head.items) > self.max_batch:
                    break                  # 放不下了，留给下一批；但第一组再大也先收下
                self._queue.popleft()
                batch.append(head)
                total += len(head.items)
            return batch

    def _run_in_chunks(self, flat: list) -> list:
        """单次提交的条目数可能超过 max_batch，按上限切块执行。"""
        if not self.chunk or len(flat) <= self.max_batch:
            return list(self._fn(flat))
        out: list[Any] = []
        for i in range(0, len(flat), self.max_batch):
            out.extend(self._fn(flat[i:i + self.max_batch]))
        return out
