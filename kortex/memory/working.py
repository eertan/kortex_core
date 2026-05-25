"""Unified working-memory state for active Kortex runs."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from kortex.memory.records import (
    MemoryRecord,
    MemoryType,
    PlannerFactPayload,
)


class WorkingMemoryState(BaseModel):
    """Typed active cognitive state shared by planning, execution, and memory."""

    working_memory_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str | None = None
    active_goal: dict[str, Any] | None = None
    goal_stack: list[dict[str, Any]] = Field(default_factory=list)
    active_entities: list[str] = Field(default_factory=list)
    current_facts: list[PlannerFactPayload] = Field(default_factory=list)
    current_bindings: dict[str, Any] = Field(default_factory=dict)
    active_task: str | None = None
    selected_method: str | None = None
    planner_tier: str | None = None
    pending_clarifications: list[dict[str, Any]] = Field(default_factory=list)
    hitl_state: dict[str, Any] | None = None
    retrieved_memory_records: list[str] = Field(default_factory=list)
    trace_event_ids: list[str] = Field(default_factory=list)

    def remember_retrieved_record(self, record: MemoryRecord) -> None:
        """Track a retrieved memory record by id without promoting its payload."""
        if record.record_id not in self.retrieved_memory_records:
            self.retrieved_memory_records.append(record.record_id)

    def hydrate_planner_fact(self, record: MemoryRecord) -> None:
        """Promote an allowed planner fact record into active working state."""
        if record.memory_type != MemoryType.PLANNER_FACT:
            raise ValueError(
                f"Only planner fact records can hydrate planner facts; received "
                f"'{record.memory_type}'."
            )
        if not record.can_hydrate_planner():
            raise ValueError(
                f"Planner fact record '{record.record_id}' is not validated or promoted."
            )
        if not isinstance(record.payload, PlannerFactPayload):
            raise TypeError("Planner fact record has an invalid payload type.")

        self._upsert_current_fact(record.payload)
        self.remember_retrieved_record(record)

    def add_trace_event(self, event_id: str) -> None:
        """Attach a trace event reference to this working-memory state."""
        if event_id not in self.trace_event_ids:
            self.trace_event_ids.append(event_id)

    def _upsert_current_fact(self, fact: PlannerFactPayload) -> None:
        """Insert or replace a current fact by fluent and argument identity."""
        for index, existing in enumerate(self.current_facts):
            if existing.fluent == fact.fluent and existing.args == fact.args:
                self.current_facts[index] = fact
                return
        self.current_facts.append(fact)

    def to_bootstrapper_initial_state(self) -> list[dict[str, Any]]:
        """Convert active planner facts into bootstrapper-compatible fact dicts."""
        return [
            {
                "fluent": fact.fluent,
                "args": fact.args,
                "value": fact.value,
            }
            for fact in self.current_facts
        ]

