"""Conversation-facing interaction layer for Kortex Core."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Protocol
from uuid import uuid4

import instructor
from google import genai
from pydantic import BaseModel, Field

from kortex.agent import AgentDomainContext, AgentRunResult, KortexAgent
from kortex.memory.records import (
    ConversationMemoryPayload,
    MemoryLifecycleState,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from kortex.memory.working import WorkingMemoryState


class InteractionMemorySink(Protocol):
    """Storage boundary for typed interaction memory records."""

    def hook_memory_record(self, record: MemoryRecord) -> None:
        """Append a typed memory record."""
        ...


class InteractionInterpreter(Protocol):
    """Optional LLM-backed interpreter for conversational turn understanding."""

    def interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
    ) -> "InteractionInterpretation":
        """Return a structured interpretation of one user turn."""
        ...


class InteractionInterpretation(BaseModel):
    """Structured output schema for the conversation interpreter."""

    turn_type: str = Field(
        description="One of: conversation, task, clarification_answer.",
    )
    task_prompt: str | None = Field(
        default=None,
        description="Canonical task request to pass to the deterministic task agent.",
    )
    response_text: str | None = Field(
        default=None,
        description="Friendly response for conversation-only turns.",
    )
    candidate_entities: list[str] = Field(default_factory=list)
    candidate_directives: list[str] = Field(default_factory=list)
    memory_notes: list[str] = Field(default_factory=list)


class GeminiInteractionInterpreter:
    """Gemini-backed structured interpreter for the interaction layer."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the Gemini interpreter with strict structured outputs."""
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via GOOGLE_AI_API_KEY, "
                "GOOGLE_API_KEY, or GEMINI_API_KEY."
            )
        self.model_name = model_name or os.environ.get(
            "KORTEX_GEMINI_MODEL",
            "gemini-3.1-pro-preview",
        )
        base_client = genai.Client(api_key=self.api_key)
        self.client = instructor.from_genai(
            client=base_client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        )

    def interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
    ) -> InteractionInterpretation:
        """Interpret a user turn without granting execution authority."""
        system_prompt = (
            "You are the Kortex interaction interpreter. You do not execute tools, "
            "approve actions, mutate planner facts, or decide safety. Your job is "
            "only to classify the user's turn and return structured candidates. "
            "Use turn_type='conversation' for normal chat, 'task' for an actionable "
            "request, and 'clarification_answer' only when the working memory has "
            "pending clarifications. If task-like, produce a concise task_prompt "
            "that preserves the user's named concepts. Never claim execution."
        )
        working_context = {
            "session_id": working_memory.session_id,
            "user_id": working_memory.user_id,
            "active_task": working_memory.active_task,
            "active_goal": working_memory.active_goal,
            "active_entities": working_memory.active_entities,
            "current_bindings": working_memory.current_bindings,
            "pending_clarifications": working_memory.pending_clarifications,
        }
        return self.client.models.generate_content(
            model=self.model_name,
            contents=[
                system_prompt,
                f"Working memory context: {working_context}",
                f"User turn: {user_text}",
            ],
            response_model=InteractionInterpretation,
        )


@dataclass(frozen=True)
class InteractionTurnResult:
    """Result returned after handling one user-facing turn."""

    session_id: str
    user_id: str | None
    response_text: str
    status: str
    working_memory: WorkingMemoryState
    agent_result: AgentRunResult | None = None
    blocked_reason: str | None = None


@dataclass
class InteractionPolicy:
    """Deterministic gatekeeper for conversation and task execution."""

    blocked_directives: tuple[str, ...] = (
        "ignore previous instructions",
        "bypass approval",
        "skip hitl",
        "disable safety",
    )
    task_directive_prefixes: tuple[str, ...] = (
        "run",
        "execute",
        "move",
        "deliver",
        "wipe",
        "check",
        "clear",
        "build",
        "create",
        "plan",
    )

    def classify_turn(self, user_text: str) -> tuple[str, str | None]:
        """Classify a user turn as task, conversation, or blocked."""
        normalized = user_text.strip().lower()
        if not normalized:
            return "conversation", None

        for directive in self.blocked_directives:
            if directive in normalized:
                return "blocked", f"Blocked unsafe directive: {directive}"

        if normalized.endswith("?"):
            return "conversation", None

        first_word = normalized.split(maxsplit=1)[0]
        if first_word in self.task_directive_prefixes:
            return "task", None

        return "conversation", None


@dataclass
class PreResponseGuard:
    """Deterministic guard that prevents overclaiming in assistant responses."""

    def validate(
        self,
        proposed_response: str,
        agent_result: AgentRunResult | None,
    ) -> tuple[bool, str | None]:
        """Return whether a proposed response is allowed for the observed state."""
        normalized = proposed_response.lower()
        if agent_result is None:
            if "executed" in normalized or "completed" in normalized:
                return False, "Response claims execution without an agent result."
            return True, None

        if agent_result.status != "success" and (
            "executed" in normalized or "completed" in normalized
        ):
            return (
                False,
                f"Response claims execution but agent status is '{agent_result.status}'.",
            )
        return True, None


@dataclass
class InteractionSession:
    """Conversation/session layer above the deterministic Kortex agent."""

    agent: KortexAgent
    context: AgentDomainContext
    memory_sink: InteractionMemorySink | None = None
    interpreter: InteractionInterpreter | None = None
    policy: InteractionPolicy = field(default_factory=InteractionPolicy)
    response_guard: PreResponseGuard = field(default_factory=PreResponseGuard)
    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    working_memory: WorkingMemoryState | None = None
    pending_agent_result: AgentRunResult | None = None
    pending_prompt: str | None = None

    async def handle_turn(self, user_text: str) -> InteractionTurnResult:
        """Handle one user turn with policy, memory writeback, and optional execution."""
        self._write_conversation_record(role="user", content=user_text)
        if self.working_memory is None:
            self.working_memory = WorkingMemoryState(
                session_id=self.session_id,
                user_id=self.user_id,
            )

        interpretation = self._interpret_turn(user_text)

        if self.pending_agent_result is not None:
            self.working_memory.pending_clarifications.append(
                {
                    "answer": user_text,
                    "resumes_result": self.pending_agent_result.status,
                }
            )
            resumed_prompt = self._build_resumed_prompt(user_text, interpretation)
            agent_result = await self.agent.run(resumed_prompt, self.context)
            self.pending_agent_result = None
            self.pending_prompt = None
            if agent_result.working_memory is not None:
                agent_result.working_memory.session_id = self.session_id
                agent_result.working_memory.user_id = self.user_id
                self.working_memory = agent_result.working_memory

            response = self._response_for_agent_result(agent_result)
            allowed, reason = self.response_guard.validate(response, agent_result)
            if not allowed:
                response = self._guard_fallback(reason)
            self._write_conversation_record(role="assistant", content=response)
            return InteractionTurnResult(
                session_id=self.session_id,
                user_id=self.user_id,
                response_text=response,
                status=agent_result.status,
                working_memory=self.working_memory,
                agent_result=agent_result,
            )

        classification, blocked_reason = self.policy.classify_turn(user_text)
        if classification != "blocked" and interpretation is not None:
            classification = self._merge_interpreted_classification(
                policy_classification=classification,
                interpretation=interpretation,
            )
        if classification == "blocked":
            response = "I cannot follow that directive."
            self._write_conversation_record(role="assistant", content=response)
            return InteractionTurnResult(
                session_id=self.session_id,
                user_id=self.user_id,
                response_text=response,
                status="blocked",
                working_memory=self.working_memory,
                blocked_reason=blocked_reason,
            )

        if classification == "conversation":
            response = (
                interpretation.response_text
                if interpretation is not None and interpretation.response_text
                else "I noted that."
            )
            allowed, reason = self.response_guard.validate(response, None)
            if not allowed:
                response = self._guard_fallback(reason)
            self._write_conversation_record(role="assistant", content=response)
            return InteractionTurnResult(
                session_id=self.session_id,
                user_id=self.user_id,
                response_text=response,
                status="conversation",
                working_memory=self.working_memory,
            )

        task_prompt = (
            interpretation.task_prompt
            if interpretation is not None and interpretation.task_prompt
            else user_text
        )
        agent_result = await self.agent.run(task_prompt, self.context)
        self.pending_agent_result = (
            agent_result if agent_result.status == "clarification_required" else None
        )
        self.pending_prompt = task_prompt if self.pending_agent_result is not None else None
        if agent_result.working_memory is not None:
            agent_result.working_memory.session_id = self.session_id
            agent_result.working_memory.user_id = self.user_id
            self.working_memory = agent_result.working_memory

        response = self._response_for_agent_result(agent_result)
        allowed, reason = self.response_guard.validate(response, agent_result)
        if not allowed:
            response = self._guard_fallback(reason)
        self._write_conversation_record(role="assistant", content=response)
        return InteractionTurnResult(
            session_id=self.session_id,
            user_id=self.user_id,
            response_text=response,
            status=agent_result.status,
            working_memory=self.working_memory,
            agent_result=agent_result,
        )

    def _response_for_agent_result(self, result: AgentRunResult) -> str:
        """Create a conservative user-facing response from an agent result."""
        if result.status == "success":
            return "Completed the requested task."
        if result.status == "clarification_required" and result.clarification is not None:
            return result.clarification.question
        if result.status == "impasse":
            return "I could not find a valid plan for that request."
        return f"The request ended with status: {result.status}."

    def _interpret_turn(self, user_text: str) -> InteractionInterpretation | None:
        """Interpret a turn with the optional LLM interpreter."""
        if self.interpreter is None or self.working_memory is None:
            return None
        return self.interpreter.interpret_turn(user_text, self.working_memory)

    def _merge_interpreted_classification(
        self,
        policy_classification: str,
        interpretation: InteractionInterpretation,
    ) -> str:
        """Merge deterministic policy with non-authoritative LLM classification."""
        if interpretation.turn_type == "task":
            return "task"
        if interpretation.turn_type == "conversation":
            return "conversation"
        return policy_classification

    def _build_resumed_prompt(
        self,
        user_text: str,
        interpretation: InteractionInterpretation | None,
    ) -> str:
        """Build a resumed task prompt from the pending request and clarification."""
        answer = (
            interpretation.task_prompt
            if interpretation is not None and interpretation.task_prompt
            else user_text
        )
        if self.pending_prompt is None:
            return answer
        return (
            f"{self.pending_prompt}\n"
            f"Clarification answer: {answer}"
        )

    def _guard_fallback(self, reason: str | None) -> str:
        """Return a conservative response when pre-response validation fails."""
        if reason:
            return f"I need to be more precise: {reason}"
        return "I need to be more precise about what happened."

    def _write_conversation_record(self, role: str, content: str) -> None:
        """Persist one conversation turn as a typed memory record."""
        if self.memory_sink is None:
            return

        turn_id = str(uuid4())
        record = MemoryRecord(
            memory_type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            subject_ids=[self.user_id] if self.user_id is not None else [],
            source=MemorySource(system="interaction_session", reference=self.session_id),
            lifecycle_state=MemoryLifecycleState.VALIDATED,
            payload=ConversationMemoryPayload(
                role=role,
                content=content,
                turn_id=turn_id,
            ),
        )
        self.memory_sink.hook_memory_record(record)
