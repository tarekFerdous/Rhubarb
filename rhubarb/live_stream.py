"""In-memory per-session event buffer with live fan-out, backing the SSE
stream endpoint. Not persisted -- a session interrupted by a server restart
is recovered through the existing Retry action instead of stream replay.
"""

import asyncio

_buffers: dict[int, list[dict]] = {}
_subscribers: dict[int, list[asyncio.Queue]] = {}
_loop: asyncio.AbstractEventLoop | None = None
_last_usage: dict | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def subscribe(card_id: int) -> tuple[list[dict], asyncio.Queue]:
    """Register a new subscriber and return (history-so-far, queue) together,
    so nothing published in between is missed or duplicated."""
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(card_id, []).append(queue)
    return list(_buffers.get(card_id, [])), queue


def unsubscribe(card_id: int, queue: asyncio.Queue) -> None:
    subs = _subscribers.get(card_id)
    if subs and queue in subs:
        subs.remove(queue)


def publish(card_id: int, event: dict) -> None:
    """Append to history and fan out to subscribers. Safe to call from a
    worker thread -- queue delivery is marshalled onto the event loop."""
    global _last_usage
    if event.get("type") == "usage":
        _last_usage = event

    _buffers.setdefault(card_id, []).append(event)

    for queue in list(_subscribers.get(card_id, [])):
        if _loop is not None:
            _loop.call_soon_threadsafe(queue.put_nowait, event)
        else:
            queue.put_nowait(event)


def last_usage() -> dict | None:
    return _last_usage
