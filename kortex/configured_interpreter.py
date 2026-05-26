"""LLM-backed interpreter for config-aware interaction sessions."""

from __future__ import annotations

from typing import Any, Protocol
import os

import instructor
from google import genai
from pydantic import BaseModel, Field, field_validator

from kortex.configured_interaction import ConfiguredTurnInterpretation
from kortex.domain_package import DomainPackage, IntentCatalog
from kortex.intent_runtime import IntentClarification
from kortex.memory.working import WorkingMemoryState


class ConfiguredLLMClient(Protocol):
    """Minimal structured generation client expected by the interpreter."""

    class Models(Protocol):
        """Structured model generation namespace."""

        def generate_content(
            self,
            *,
            model: str,
            contents: list[str],
            response_model: type[BaseModel],
        ) -> BaseModel:
            """Generate a structured response."""
            ...

    models: Models


class ConfiguredInterpreterOutput(BaseModel):
    """LLM output schema for config-aware turn interpretation."""

    turn_type: str = Field(
        description="One of: conversation, task, clarification_answer.",
    )
    intent_name: str | None = Field(
        default=None,
        description="Configured intent name for task or clarification turns.",
    )
    slots: list["ConfiguredSlotValue"] = Field(
        default_factory=list,
        description="Raw slot values as configured slot-name/value pairs.",
    )
    response_text: str | None = Field(
        default=None,
        description="Short response for conversation-only turns.",
    )
    candidate_entities: list[str] = Field(default_factory=list)
    candidate_directives: list[str] = Field(default_factory=list)
    memory_notes: list[str] = Field(default_factory=list)

    @field_validator("turn_type")
    @classmethod
    def validate_turn_type(cls, value: str) -> str:
        """Validate the coarse turn type."""
        allowed = {"conversation", "task", "clarification_answer"}
        if value not in allowed:
            raise ValueError(f"turn_type must be one of {sorted(allowed)}.")
        return value


class ConfiguredSlotValue(BaseModel):
    """One extracted configured slot value."""

    slot_name: str = Field(description="Configured slot name.")
    value: str = Field(description="Raw user-provided slot value.")


class GeminiConfiguredTurnInterpreter:
    """Gemini-backed configured turn interpreter.

    The model is only allowed to classify the turn and extract raw configured
    slot values. Intent validation, slot filtering, planning, approval, memory
    promotion, and execution remain deterministic outside this class.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: ConfiguredLLMClient | None = None,
    ) -> None:
        """Initialize a Gemini structured-output interpreter."""
        self.model_name = model_name or os.environ.get(
            "KORTEX_GEMINI_MODEL",
            "gemini-3.1-pro-preview",
        )
        if client is not None:
            self.client = client
            return

        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
        if not resolved_key:
            raise ValueError(
                "API key must be provided or set via GOOGLE_AI_API_KEY, "
                "GOOGLE_API_KEY, or GEMINI_API_KEY."
            )
        base_client = genai.Client(api_key=resolved_key)
        self.client = instructor.from_genai(
            client=base_client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        )

    def interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
        package: DomainPackage,
        pending_clarification: IntentClarification | None,
    ) -> ConfiguredTurnInterpretation:
        """Interpret one turn using config-derived schema context."""
        if package.intents is None:
            raise ValueError("Configured LLM interpreter requires intents.yaml.")
        contents = [
            self._system_prompt(package.intents),
            self._runtime_context(working_memory, pending_clarification),
            f"User turn: {user_text}",
        ]
        output = self._generate_structured(
            contents=contents,
            response_model=ConfiguredInterpreterOutput,
        )
        if not isinstance(output, ConfiguredInterpreterOutput):
            output = ConfiguredInterpreterOutput.model_validate(output)
        return self._validated_interpretation(output, package.intents)

    def _generate_structured(
        self,
        contents: list[str],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Generate structured output with current or legacy client surfaces."""
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": contents[0]},
                    {
                        "role": "user",
                        "content": "\n\n".join(contents[1:]),
                    },
                ],
                response_model=response_model,
            )
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            response_model=response_model,
        )

    def _system_prompt(self, catalog: IntentCatalog) -> str:
        """Build a strict interpreter prompt from interaction config."""
        return (
            "You are the Kortex config-aware interaction interpreter. You do "
            "not plan, approve actions, execute tools, mutate memory, or claim "
            "completion. Return only structured output matching the response "
            "model. Use the configured intents and slots below. Extract raw "
            "slot values exactly enough for deterministic normalization. If the "
            "user is only greeting or chatting, use turn_type='conversation'. "
            "If there is a pending clarification, use "
            "turn_type='clarification_answer' and include any supplied missing "
            "slots. Do not invent unknown slot keys or unknown intent names.\n"
            f"Interaction config: {self._catalog_schema(catalog)}"
        )

    def _catalog_schema(self, catalog: IntentCatalog) -> dict[str, Any]:
        """Return compact schema data derived from intents.yaml."""
        return {
            "domain": catalog.domain,
            "allowed_topics": catalog.scope.allowed_topics,
            "refusal_response": catalog.scope.refusal_response,
            "intents": {
                intent_name: {
                    "description": intent.description,
                    "examples": intent.examples,
                    "planner_binding": intent.planner_binding,
                    "slots": {
                        slot_name: {
                            "slot_type": slot.slot_type,
                            "required": slot.required,
                            "default": slot.default,
                            "values": slot.values,
                            "clarification": slot.clarification,
                        }
                        for slot_name, slot in intent.slots.items()
                    },
                }
                for intent_name, intent in catalog.intents.items()
            },
        }

    def _runtime_context(
        self,
        working_memory: WorkingMemoryState,
        pending_clarification: IntentClarification | None,
    ) -> str:
        """Build compact runtime context for interpretation."""
        return (
            "Runtime context: "
            + str(
                {
                    "session_id": working_memory.session_id,
                    "active_task": working_memory.active_task,
                    "current_bindings": working_memory.current_bindings,
                    "pending_clarification": (
                        pending_clarification.model_dump()
                        if pending_clarification is not None
                        else None
                    ),
                }
            )
        )

    def _validated_interpretation(
        self,
        output: ConfiguredInterpreterOutput,
        catalog: IntentCatalog,
    ) -> ConfiguredTurnInterpretation:
        """Filter model output against configured intents and slots."""
        intent_name = output.intent_name
        if output.turn_type == "conversation":
            return ConfiguredTurnInterpretation(
                turn_type="conversation",
                response_text=output.response_text,
                candidate_entities=output.candidate_entities,
                candidate_directives=output.candidate_directives,
                memory_notes=output.memory_notes,
            )

        if intent_name not in catalog.intents:
            return ConfiguredTurnInterpretation(
                turn_type="conversation",
                response_text="I need a configured intent before I can continue.",
            )

        allowed_slots = set(catalog.intents[intent_name].slots)
        raw_slots = {
            slot.slot_name: slot.value
            for slot in output.slots
            if slot.slot_name in allowed_slots and slot.value != ""
        }
        return ConfiguredTurnInterpretation(
            turn_type=output.turn_type,  # type: ignore[arg-type]
            intent_name=intent_name,
            raw_slots=raw_slots,
            response_text=output.response_text,
            candidate_entities=output.candidate_entities,
            candidate_directives=output.candidate_directives,
            memory_notes=output.memory_notes,
        )
