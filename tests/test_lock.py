"""Single-instance locking for live trading.

Two runners against one account do not degrade gracefully -- they fight. An
orphaned runner holding a 25-symbol universe ran alongside a new 110-symbol
one; their target books differed, so each cycle one bought what the other had
just sold. Seven BTC and six SOL round trips in eight minutes, -0.72% of the
account, no position change. Nothing in the strategy can detect this: each
runner sees a balance disagreeing with its model and dutifully corrects it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from nbtrend.live.lock import AlreadyRunning, SingleInstanceLock


def test_second_acquire_in_process_is_refused(tmp_path):
    path = tmp_path / "live.lock"
    first = SingleInstanceLock(path)
    first.acquire()
    try:
        with pytest.raises(AlreadyRunning):
            SingleInstanceLock(path).acquire()
    finally:
        first.release()


def test_lock_is_released_for_the_next_holder(tmp_path):
    path = tmp_path / "live.lock"
    with SingleInstanceLock(path):
        pass
    second = SingleInstanceLock(path)
    second.acquire()
    second.release()


def test_a_separate_process_is_refused(tmp_path):
    path = tmp_path / "live.lock"
    held = SingleInstanceLock(path)
    held.acquire()
    try:
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(pathlib_src())!r})
            from nbtrend.live.lock import SingleInstanceLock, AlreadyRunning
            try:
                SingleInstanceLock({str(path)!r}).acquire()
                print("ACQUIRED")
            except AlreadyRunning:
                print("REFUSED")
        """)
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert "REFUSED" in out.stdout, out.stdout + out.stderr
    finally:
        held.release()


def test_the_lock_records_the_holding_pid(tmp_path):
    path = tmp_path / "live.lock"
    lock = SingleInstanceLock(path)
    lock.acquire()
    try:
        assert path.read_text().strip() == str(os.getpid())
    finally:
        lock.release()


def test_a_dead_holder_does_not_block_forever(tmp_path):
    """flock is used rather than a bare PID file precisely so the kernel
    releases it when the holder dies -- including a SIGKILLed orphan, which is
    how the colliding runner survived a supervisor stop."""
    path = tmp_path / "live.lock"
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(pathlib_src())!r})
        from nbtrend.live.lock import SingleInstanceLock
        SingleInstanceLock({str(path)!r}).acquire()
        print("HELD", flush=True)
        time.sleep(60)
    """)
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "HELD"
        with pytest.raises(AlreadyRunning):
            SingleInstanceLock(path).acquire()
    finally:
        child.kill()
        child.wait(timeout=10)

    # The kernel dropped it when the process died.
    survivor = SingleInstanceLock(path)
    survivor.acquire()
    survivor.release()


def test_runner_acquires_the_lock_before_trading():
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "nbtrend" / "live" / "runner.py"
    ).read_text()
    assert "self._lock.acquire()" in source
    assert "self._lock.release()" in source
    # Namespaced by mode so a paper run cannot block a live one.
    assert 'f"data/state/{cfg.mode}.lock"' in source


def pathlib_src() -> str:
    import pathlib

    return str(pathlib.Path(__file__).resolve().parents[1] / "src")
