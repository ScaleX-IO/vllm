from __future__ import annotations

from collections import defaultdict


class StoreCompletionTracker:
    """Track async stores independently of request-finish timing."""

    def __init__(self) -> None:
        self._pending: dict[str, int] = defaultdict(int)
        self._seen: set[str] = set()
        self._finished_waiting: set[str] = set()

    def started(self, request_id: str) -> None:
        self._seen.add(request_id)
        self._pending[request_id] += 1

    def completed(self, request_id: str) -> None:
        pending = self._pending.get(request_id, 0)
        if pending <= 0:
            raise RuntimeError(
                f"store completion has no matching start for {request_id}"
            )
        if pending == 1:
            del self._pending[request_id]
        else:
            self._pending[request_id] = pending - 1

    def mark_requests_finished(self, request_ids: set[str]) -> None:
        self._finished_waiting.update(request_ids & self._seen)

    def take_ready(self) -> set[str]:
        ready = {
            request_id
            for request_id in self._finished_waiting
            if request_id not in self._pending
        }
        self._finished_waiting.difference_update(ready)
        self._seen.difference_update(ready)
        return ready

    def pending_count(self, request_id: str) -> int:
        return self._pending.get(request_id, 0)
