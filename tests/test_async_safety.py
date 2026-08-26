"""The event loop must never be blocked while the websocket is live.

Centrifugo drops a client that does not answer its ping within 25s. The
router's fill polling uses a blocking `time.sleep` and can hold for
`repost_after_s` (45s) per attempt, and the Nobitex REST client is
synchronous httpx. Running either of those directly on the event loop starves
the pong handler and the connection dies mid-rebalance -- observed in a live
shadow test as `ConnectionClosedError: no pong`.
"""

from __future__ import annotations

import asyncio
import pathlib
import time

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "nbtrend"


async def test_event_loop_stays_responsive_during_blocking_work():
    """`to_thread` keeps a heartbeat alive; a direct call would not."""
    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    before = beats

    await asyncio.to_thread(time.sleep, 0.3)

    task.cancel()
    assert beats - before > 5, "event loop was blocked while the thread ran"


async def test_direct_blocking_call_would_starve_the_loop():
    """Control case: this is what the bug looked like."""
    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    before = beats

    time.sleep(0.3)          # blocking, on the loop -- the bug
    await asyncio.sleep(0)

    task.cancel()
    assert beats - before <= 1, "expected the loop to be starved"


def test_runner_offloads_order_execution():
    source = (SRC / "live" / "runner.py").read_text()
    assert "await asyncio.to_thread(self._apply_target" in source, (
        "_apply_target drives the router, which blocks on time.sleep; it must "
        "run off the event loop"
    )
    assert "await asyncio.to_thread(self.equity_rial)" in source


def test_feed_offloads_synchronous_rest():
    source = (SRC / "data" / "feed.py").read_text()
    assert source.count("await asyncio.to_thread(self.fetch_local") == 2, (
        "both the local and FX candle fetches are synchronous httpx and must "
        "run off the event loop"
    )
