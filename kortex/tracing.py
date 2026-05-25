"""
Structured tracing primitives for Kortex Core execution.

The trace layer records deterministic lifecycle events without depending on a
specific storage backend. Callers can keep events in memory for tests, serialize
them to JSON, or forward them into the episodic memory pipeline.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TraceEvent:
    """A single structured event emitted during an agent run."""

    run_id: str
    stage: str
    message: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class TraceRecorder:
    """In-memory trace recorder used by the agent runtime and tests."""

    def __init__(self) -> None:
        """Initialize an empty trace event buffer."""
        self.events: list[TraceEvent] = []

    def emit(
        self,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> TraceEvent:
        """Append and return a trace event."""
        event = TraceEvent(
            run_id=run_id or str(uuid4()),
            stage=stage,
            message=message,
            payload=payload or {},
        )
        self.events.append(event)
        return event

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return trace events as JSON-serializable dictionaries."""
        return [
            {
                "event_id": event.event_id,
                "run_id": event.run_id,
                "stage": event.stage,
                "message": event.message,
                "payload": event.payload,
                "timestamp": event.timestamp,
            }
            for event in self.events
        ]
