"""Single-instance lock for live trading.

Two runners against one account is not a race that degrades gracefully -- it
is two strategies fighting. Observed: an orphaned runner holding a 25-symbol
universe ran alongside a new 110-symbol one. Their selections differed, so
each cycle one bought what the other had just sold, paying the spread and two
fees on every leg. Seven BTC round trips and six SOL round trips in eight
minutes, for -0.72% of the account and no position change.

Nothing in the strategy detects this: each runner sees a balance that
disagrees with its own model and dutifully "corrects" it.

`flock` is used rather than a bare PID file because the kernel releases it
when the holder dies, however it dies -- SIGKILL, crash, or a supervisor that
reaps the wrapper but not the child (which is exactly how the orphan above
survived).
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from types import TracebackType

DEFAULT_LOCK_PATH = Path("data/state/live.lock")


class AlreadyRunning(RuntimeError):
    """Another process already holds the live-trading lock."""


class SingleInstanceLock:
    """Exclusive, non-blocking lock held for the lifetime of the process."""

    def __init__(self, path: Path | str = DEFAULT_LOCK_PATH, label: str = "live"):
        self.path = Path(path)
        self.label = label
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = _read_holder(self.path)
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise AlreadyRunning(
                    f"another {self.label} runner already holds {self.path}"
                    + (f" (pid {holder})" if holder else "")
                    + ". Two runners on one account fight each other: each sees the "
                    "other's fills as drift and reverses them, paying spread and fees "
                    "both ways. Stop it first -- and check for an orphaned child if a "
                    "supervisor only reaped the wrapper."
                ) from exc
            raise

        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def _read_holder(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None
