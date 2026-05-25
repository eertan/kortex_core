"""Typed memory record envelopes for Kortex cognitive memory layers."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, confloat, model_validator


class MemoryType(StrEnum):
    """Supported long-term memory record categories."""

    CONVERSATION = "conversation"
    VALIDATED_TRACE = "validated_trace"
    PLANNER_FACT = "planner_fact"
    PROCEDURAL_SKILL = "procedural_skill"
    SEMANTIC_ENTITY = "semantic_entity"
    EXTERNAL_KNOWLEDGE = "external_knowledge"


class MemoryScope(StrEnum):
    """Visibility scope for a memory record."""

    SESSION = "session"
    USER = "user"
    ENTITY = "entity"
    DOMAIN = "domain"
    GLOBAL = "global"


class MemoryLifecycleState(StrEnum):
    """Governance state for memory records."""

    DRAFT = "draft"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class MemorySource(BaseModel):
    """Origin metadata for a memory record."""

    system: str = Field(description="Subsystem or external endpoint that produced the record.")
    reference: str | None = Field(
        default=None,
        description="Optional source-specific id, path, query id, run id, or trace id.",
    )


class ConversationMemoryPayload(BaseModel):
    """Narrative interaction memory used for continuity and recovery."""

    payload_type: Literal[MemoryType.CONVERSATION] = MemoryType.CONVERSATION
    role: str
    content: str
    turn_id: str | None = None


class PlannerFactPayload(BaseModel):
    """Planner-consumable boolean fact payload."""

    payload_type: Literal[MemoryType.PLANNER_FACT] = MemoryType.PLANNER_FACT
    fluent: str
    args: list[str] = Field(default_factory=list)
    value: bool = True


class ValidatedTracePayload(BaseModel):
    """Normalized deterministic execution trace eligible for reflection."""

    payload_type: Literal[MemoryType.VALIDATED_TRACE] = MemoryType.VALIDATED_TRACE
    root_task: str | None = None
    state_goals: list[dict[str, Any]] = Field(default_factory=list)
    planner_tier: str
    primitive_actions: list[dict[str, Any]] = Field(default_factory=list)
    result: str
    hitl_decisions: list[dict[str, Any]] = Field(default_factory=list)
    final_facts: list[PlannerFactPayload] = Field(default_factory=list)
    validation_passed: bool = False


class ProceduralSkillPayload(BaseModel):
    """Procedural-memory representation of a condition-based learned skill."""

    payload_type: Literal[MemoryType.PROCEDURAL_SKILL] = MemoryType.PROCEDURAL_SKILL
    target_task: str
    parameters: dict[str, str] = Field(default_factory=dict)
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    effects: list[dict[str, Any]] = Field(default_factory=list)
    ordered_subtasks: list[list[str]] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    utility: float | None = None


class SemanticEntityPayload(BaseModel):
    """Semantic-memory payload for stable entity or concept knowledge."""

    payload_type: Literal[MemoryType.SEMANTIC_ENTITY] = MemoryType.SEMANTIC_ENTITY
    entity_id: str
    labels: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExternalKnowledgePayload(BaseModel):
    """Scoped evidence returned by an external knowledge endpoint."""

    payload_type: Literal[MemoryType.EXTERNAL_KNOWLEDGE] = MemoryType.EXTERNAL_KNOWLEDGE
    endpoint_id: str
    query: dict[str, Any]
    result: dict[str, Any]
    may_hydrate_planner: bool = False


MemoryPayload = (
    ConversationMemoryPayload
    | PlannerFactPayload
    | ValidatedTracePayload
    | ProceduralSkillPayload
    | SemanticEntityPayload
    | ExternalKnowledgePayload
)


class MemoryRecord(BaseModel):
    """Uniform governance envelope around specialized memory payloads."""

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    memory_type: MemoryType
    scope: MemoryScope
    subject_ids: list[str] = Field(default_factory=list)
    source: MemorySource
    payload: MemoryPayload
    provenance: dict[str, Any] = Field(default_factory=dict)
    confidence: confloat(ge=0.0, le=1.0) = 1.0
    authority: str | None = None
    lifecycle_state: MemoryLifecycleState = MemoryLifecycleState.DRAFT
    observed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    valid_at: str | None = None
    expires_at: str | None = None

    @model_validator(mode="after")
    def payload_matches_type(self) -> MemoryRecord:
        """Ensure the envelope type matches the specialized payload type."""
        if self.payload.payload_type != self.memory_type:
            raise ValueError(
                f"Memory payload type '{self.payload.payload_type}' does not match "
                f"record type '{self.memory_type}'."
            )
        return self

    def can_hydrate_planner(self) -> bool:
        """Return whether this record is allowed to hydrate planner truth."""
        return (
            self.memory_type == MemoryType.PLANNER_FACT
            and self.lifecycle_state
            in {MemoryLifecycleState.VALIDATED, MemoryLifecycleState.PROMOTED}
        )

