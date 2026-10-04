"""动态批处理执行器的黑盒测试：只通过 submit 观察行为。"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cvflow.core import BatchExecutor, BatchTimeout


def recording_fn(log: list, delay: float = 0.0):
    def fn(items):
        if delay:
            time.sleep(delay)
        log.append(list(items))
        return [x * 10 for x in items]
    return fn


def test_batching_single_submit_runs_immediately():
    log = []
    ex = BatchExecutor(recording_fn(log), max_batch=8, wait_s=0.0)
    assert ex.submit([1, 2, 3]) == [10, 20, 30]
    assert log == [[1, 2, 3]]                       # 同一次提交天然在同一批里
    assert ex.stats["batches"] == 1 and ex.stats["largest_batch"] == 3


def test_batching_zero_wait_does_not_delay():
    ex = BatchExecutor(recording_fn([]), max_batch=8, wait_s=0.0)
    t0 = time.perf_counter()
    for _ in range(20):
        ex.submit([1])
    assert (time.perf_counter() - t0) < 0.2         # 不等待窗口，20 次调用应当很快
    assert ex.stats["batches"] == 20


def test_batching_merges_concurrent_submits_within_window():
    log = []
    ex = BatchExecutor(recording_fn(log), max_batch=4, wait_s=0.5)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: ex.submit([i])[0], [1, 2, 3, 4]))
    assert sorted(results) == [10, 20, 30, 40]      # 每个提交者都拿回自己的结果
    assert len(log) == 1 and sorted(log[0]) == [1, 2, 3, 4]   # 四路合成了一批


def test_batching_window_expires_and_runs_partial_batch():
    log = []
    ex = BatchExecutor(recording_fn(log), max_batch=8, wait_s=0.15)
    t0 = time.perf_counter()
    assert ex.submit([7]) == [70]                   # 凑不满，等到窗口到期就执行
    waited = time.perf_counter() - t0
    assert 0.1 < waited < 1.0 and log == [[7]]


def test_batching_full_batch_does_not_wait_for_window():
    log = []
    ex = BatchExecutor(recording_fn(log), max_batch=2, wait_s=5.0)
    t0 = time.perf_counter()
    assert ex.submit([1, 2]) == [10, 20]            # 一次就凑满 max_batch，立刻执行
    assert (time.perf_counter() - t0) < 1.0 and log == [[1, 2]]


def test_batching_respects_max_batch_and_splits_oversized_submit():
    log = []
    ex = BatchExecutor(recording_fn(log), max_batch=2, wait_s=0.0)
    assert ex.submit([1, 2, 3, 4, 5]) == [10, 20, 30, 40, 50]
    assert log == [[1, 2], [3, 4], [5]]             # 超出上限的提交按块执行，结果顺序不变


def test_batching_concurrent_groups_never_exceed_max_batch():
    log = []
    ex = BatchExecutor(recording_fn(log, delay=0.02), max_batch=3, wait_s=0.2)
    with ThreadPoolExecutor(6) as pool:
        out = list(pool.map(lambda i: ex.submit([i, i + 100]), range(6)))
    assert out == [[i * 10, (i + 100) * 10] for i in range(6)]
    assert all(len(b) <= 3 for b in log), log
    assert sum(len(b) for b in log) == 12


def test_batching_error_reaches_every_member_of_the_batch():
    def boom(items):
        raise ValueError(f"坏了 {len(items)}")
    ex = BatchExecutor(boom, max_batch=4, wait_s=0.3)
    errors = []

    def go(i):
        try:
            ex.submit([i])
        except ValueError as e:
            errors.append(str(e))
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(go, range(3)))
    assert len(errors) == 3 and all("坏了" in e for e in errors)
    assert ex.submit.__self__ is ex                 # 出错后执行器仍可用
    ex2 = BatchExecutor(recording_fn([]), max_batch=2, wait_s=0.0)
    assert ex2.submit([5]) == [50]


def test_batching_wrong_result_count_is_reported():
    ex = BatchExecutor(lambda items: [1], max_batch=4, wait_s=0.0)
    with pytest.raises(RuntimeError, match="期望 2 个结果"):
        ex.submit([1, 2])


def test_batching_timeout_abandons_request_without_corrupting_others():
    started = threading.Event()
    release = threading.Event()

    def slow(items):
        started.set()
        release.wait(5)
        return [x * 10 for x in items]
    ex = BatchExecutor(slow, max_batch=1, wait_s=0.0)
    blocker = threading.Thread(target=lambda: ex.submit([1]), daemon=True)
    blocker.start()
    assert started.wait(3)
    with pytest.raises(BatchTimeout):
        ex.submit([2], timeout=0.2)                 # 领队被占住，跟随者超时离开
    release.set()
    blocker.join(5)
    assert ex.pending() == 0                        # 超时的请求已从队列摘除
    assert ex.submit([3]) == [30]                   # 执行器未被污染


def test_batching_close_releases_waiters():
    ex = BatchExecutor(recording_fn([]), max_batch=4, wait_s=0.0)
    ex.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        ex.submit([1])


def test_batching_high_concurrency_every_caller_gets_own_result():
    ex = BatchExecutor(lambda items: [(x, len(items)) for x in items], max_batch=8, wait_s=0.05)
    with ThreadPoolExecutor(16) as pool:
        out = list(pool.map(lambda i: ex.submit([i])[0], range(200)))
    assert [o[0] for o in out] == list(range(200))  # 顺序与归属都正确
    assert max(o[1] for o in out) > 1               # 确实发生了合批
    assert ex.stats["items"] == 200
