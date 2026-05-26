"""Config-aware interaction session for domain-packaged Kortex agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from kortex.config.bootstrapper import DomainBootstrapper
from kortex.domain_package import DomainPackage
from kortex.intent_runtime import IntentClarification, IntentFrame, IntentFrameBuilder
from kortex.interaction import InteractionMemorySink, InteractionPolicy
from kortex.memory.adapters import (
    planner_fact_record_from_dict,
    planner_fact_records_from_action_effects,
)
from kortex.memory.records import (
    ConversationMemoryPayload,
    MemoryLifecycleState,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
    OptimizationDecisionPayload,
)
from kortex.memory.working import WorkingMemoryState
from kortex.optimization import OptimizationExecutionOutput
from kortex.plugins.registry import PluginRegistry, registry as default_registry
from kortex.responses import ResponseFrame, ResponseRenderer
from kortex.spine.planner import HTNMethodTieImpasse, KortexPlanner
from kortex.tracing import TraceEvent, TraceRecorder


ConfiguredTurnType = Literal[
    "conversation",
    "task",
    "clarification_answer",
    "approval_response",
    "correction",
    "change_request",
    "cancel",
]


class ConfiguredTurnInterpretation(BaseModel):
    """Structured, non-authoritative interpretation of a user-facing turn."""

    turn_type: ConfiguredTurnType
    intent_name: str | None = None
    raw_slots: dict[str, Any] = Field(default_factory=dict)
    response_text: str | None = None
    candidate_entities: list[str] = Field(default_factory=list)
    candidate_directives: list[str] = Field(default_factory=list)
    memory_notes: list[str] = Field(default_factory=list)


class ConfiguredTurnInterpreter(Protocol):
    """Boundary for LLM or deterministic turn interpreters.

    Implementations may classify turns and extract raw slots, but they do not
    plan, mutate planner facts, approve HITL gates, or claim execution.
    """

    def interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
        package: DomainPackage,
        pending_clarification: IntentClarification | None,
    ) -> ConfiguredTurnInterpretation:
        """Return a structured interpretation for one configured interaction turn."""
        ...


@dataclass(frozen=True)
class ConfiguredExecutionResult:
    """Deterministic execution result for a completed configured intent frame."""

    status: str
    working_memory: WorkingMemoryState
    trace: list[TraceEvent]
    plan_actions: list[str] = field(default_factory=list)
    execution: list[Any] = field(default_factory=list)
    selected_method: str | None = None
    approval_request: dict[str, Any] | None = None
    rendered_responses: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ConfiguredInteractionTurnResult:
    """Result returned after one config-aware interaction turn."""

    session_id: str
    user_id: str | None
    response_text: str
    status: str
    working_memory: WorkingMemoryState
    interpretation: ConfiguredTurnInterpretation | None = None
    intent_frame: IntentFrame | None = None
    clarification: IntentClarification | None = None
    execution_result: ConfiguredExecutionResult | None = None
    blocked_reason: str | None = None


@dataclass
class ConfiguredInteractionSession:
    """Multi-turn interaction layer driven by a loaded domain package.

    This layer owns conversation policy, domain scope checks, configured intent
    frame construction, pending clarification state, and response guarding. It
    delegates logical ordering and execution to the deterministic planner path.
    """

    package: DomainPackage
    objects: dict[str, str]
    initial_state: list[dict[str, Any]] = field(default_factory=list)
    registry: PluginRegistry | None = None
    interpreter: ConfiguredTurnInterpreter | None = None
    renderer: ResponseRenderer = field(default_factory=ResponseRenderer)
    memory_sink: InteractionMemorySink | None = None
    policy: InteractionPolicy = field(default_factory=InteractionPolicy)
    interactive_execution: bool = True
    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    working_memory: WorkingMemoryState | None = None
    pending_intent_name: str | None = None
    pending_slots: dict[str, Any] = field(default_factory=dict)
    pending_clarification: IntentClarification | None = None
    pending_plan: Any | None = None
    pending_bootstrapper: DomainBootstrapper | None = None
    pending_run_id: str | None = None
    pending_next_action_index: int | None = None
    pending_execution_results: list[Any] = field(default_factory=list)
    pending_rendered_responses: list[str] = field(default_factory=list)
    pending_plan_actions: list[str] = field(default_factory=list)
    pending_selected_method: str | None = None
    trace_recorder: TraceRecorder = field(default_factory=TraceRecorder)

    async def handle_turn(self, user_text: str) -> ConfiguredInteractionTurnResult:
        """Handle one configured conversation turn."""
        working_memory = self._ensure_working_memory()
        self._write_conversation_record(role="user", content=user_text)

        blocked_status, blocked_reason = self.policy.classify_turn(user_text)
        if blocked_status == "blocked":
            return self._conversation_result(
                response_text="I cannot follow that directive.",
                status="blocked",
                blocked_reason=blocked_reason,
            )

        if self.pending_plan is not None:
            # Fast-path: check for simple, unambiguous yes/no first to bypass the interpreter
            normalized = user_text.strip().lower()
            if normalized in {
                "y",
                "yes",
                "yes please",
                "approve",
                "approved",
                "approve it",
                "go ahead",
                "proceed",
                "n",
                "no",
                "no thanks",
                "deny",
                "denied",
                "stop",
                "cancel",
            }:
                approval_result = self._handle_approval_turn(user_text)
                if approval_result is not None:
                    self.working_memory = approval_result.working_memory
                    return self._conversation_result(
                        response_text=self._response_for_execution(approval_result),
                        status=approval_result.status,
                        execution_result=approval_result,
                    )

        interpretation = self._interpret_turn(user_text, working_memory)

        if self.pending_plan is not None:
            if interpretation.turn_type == "cancel":
                approval_result = self._deny_pending_approval(user_text)
                self.working_memory = approval_result.working_memory
                return self._conversation_result(
                    response_text="I cancelled the pending holds.",
                    status=approval_result.status,
                    execution_result=approval_result,
                    interpretation=interpretation,
                )
            elif interpretation.turn_type in {"correction", "change_request"}:
                current_slots = {}
                if working_memory.active_goal is not None and isinstance(working_memory.active_goal, dict):
                    current_slots = dict(working_memory.active_goal.get("slots", {}))
                current_slots.update(interpretation.raw_slots)
                intent_name = (
                    interpretation.intent_name
                    or (working_memory.active_goal.get("intent_name") if working_memory.active_goal else None)
                    or self.pending_intent_name
                )
                self._clear_pending_approval()
                self.pending_plan = None
                self.pending_bootstrapper = None
                self.pending_run_id = None
                self.pending_next_action_index = None
                self.pending_plan_actions = []
                self.pending_selected_method = None
                self.pending_execution_results = []
                self.pending_rendered_responses = []
                self.pending_slots = current_slots
                self.pending_intent_name = intent_name

                self._trace(
                    str(uuid4()),
                    working_memory,
                    "interaction.correction",
                    "User requested a correction mid-flight; cancelling pending plan and replanning",
                    {"corrected_slots": current_slots},
                )
            else:
                approval_result = self._handle_approval_turn(user_text)
                if approval_result is not None:
                    self.working_memory = approval_result.working_memory
                    return self._conversation_result(
                        response_text=self._response_for_execution(approval_result),
                        status=approval_result.status,
                        execution_result=approval_result,
                        interpretation=interpretation,
                    )

        if interpretation.turn_type == "conversation":
            response = interpretation.response_text or self._conversation_fallback()
            return self._conversation_result(
                response_text=response,
                status="conversation",
                interpretation=interpretation,
            )

        if interpretation.turn_type == "task" and not self._is_in_scope(
            user_text,
            interpretation,
        ):
            response = self._render_refusal()
            return self._conversation_result(
                response_text=response,
                status="out_of_domain",
                interpretation=interpretation,
            )

        intent_name = interpretation.intent_name or self.pending_intent_name
        if intent_name is None:
            response = "I need a configured intent before I can continue."
            return self._conversation_result(
                response_text=response,
                status="clarification_required",
                interpretation=interpretation,
            )

        raw_slots = self._merged_slots(interpretation)
        frame_or_clarification = self._intent_builder().build(intent_name, raw_slots, self.objects)
        if isinstance(frame_or_clarification, IntentClarification):
            self.pending_intent_name = intent_name
            self.pending_slots = {
                slot_name: value
                for slot_name, value in raw_slots.items()
                if slot_name not in frame_or_clarification.missing_slots
            }
            self.pending_clarification = frame_or_clarification
            working_memory.pending_clarifications.append(
                frame_or_clarification.model_dump()
            )

            # Use conversational elicitation if narrator supports it
            response_text = frame_or_clarification.question
            if self.renderer is not None and getattr(self.renderer, "narrator", None) is not None:
                narrator = self.renderer.narrator
                if hasattr(narrator, "narrate_elicitation"):
                    intent_spec = self.package.intents.intents[intent_name]
                    slot_clarifications = {
                        slot_name: intent_spec.slots[slot_name].clarification or f"What is your {slot_name}?"
                        for slot_name in frame_or_clarification.missing_slots
                    }
                    try:
                        response_text = narrator.narrate_elicitation(
                            intent_name=intent_name,
                            missing_slots=frame_or_clarification.missing_slots,
                            slot_clarifications=slot_clarifications,
                        )
                    except Exception:
                        response_text = frame_or_clarification.question

            return self._conversation_result(
                response_text=response_text,
                status="clarification_required",
                interpretation=interpretation,
                clarification=frame_or_clarification,
            )

        self.pending_intent_name = None
        self.pending_slots = {}
        self.pending_clarification = None
        working_memory.current_bindings.update(frame_or_clarification.normalized_parameters)
        working_memory.active_task = frame_or_clarification.planner_binding
        working_memory.active_goal = frame_or_clarification.model_dump()
        working_memory.active_entities = [
            str(value)
            for value in frame_or_clarification.normalized_parameters.values()
            if isinstance(value, str)
        ]

        # Dynamically register any extracted slot parameter into self.objects
        if self.package.intents is not None and intent_name in self.package.intents.intents:
            intent_spec = self.package.intents.intents[intent_name]
            for slot_name, slot_spec in intent_spec.slots.items():
                norm_val = frame_or_clarification.normalized_parameters.get(slot_name)
                if norm_val and norm_val not in self.objects:
                    if slot_name == "duration_days":
                        self.objects[norm_val] = "TripDuration"
                    elif slot_name == "budget":
                        self.objects[norm_val] = "Budget"

        execution_result = self._execute_intent_frame(frame_or_clarification)
        self.working_memory = execution_result.working_memory
        response = self._response_for_execution(execution_result)
        return self._conversation_result(
            response_text=response,
            status=execution_result.status,
            interpretation=interpretation,
            intent_frame=frame_or_clarification,
            execution_result=execution_result,
        )

    def _execute_intent_frame(
        self,
        intent_frame: IntentFrame,
    ) -> ConfiguredExecutionResult:
        """Plan and execute a complete intent frame through deterministic runtime."""
        run_id = str(uuid4())
        working_memory = self._ensure_working_memory()

        self._trace(
            run_id,
            working_memory,
            "interaction.intent_frame",
            "Configured interaction produced canonical planner parameters",
            intent_frame.model_dump(),
        )
        for fact in self.initial_state:
            working_memory.hydrate_planner_fact(
                planner_fact_record_from_dict(
                    fact,
                    source_system="configured_interaction_initial_state",
                    source_reference=run_id,
                )
            )

        planner = KortexPlanner("configured_interaction")
        registry = self.registry or default_registry
        bootstrapper = DomainBootstrapper(planner, registry=registry)
        bootstrapper.load_domain(str(self.package.domain_path))
        bootstrapper.load_problem_state(self.objects, self.initial_state)
        self._trace(
            run_id,
            working_memory,
            "planning.bootstrap",
            "Loaded configured domain package and problem state",
            {"domain_path": str(self.package.domain_path), "objects": self.objects},
        )

        binding = bootstrapper.intent_bindings.get(intent_frame.planner_binding)
        if binding is None:
            error = f"No manifest intent binding for '{intent_frame.planner_binding}'."
            self._trace(run_id, working_memory, "planning.binding_error", error)
            return self._execution_result("impasse", working_memory, run_id, error=error)

        missing = [
            param_name
            for param_name in binding.get("required_parameters", [])
            if param_name not in intent_frame.normalized_parameters
        ]
        if missing:
            error = f"Intent frame is missing bound parameters: {missing}."
            self._trace(run_id, working_memory, "planning.binding_error", error)
            return self._execution_result("impasse", working_memory, run_id, error=error)

        if binding["type"] != "task":
            error = "Configured interaction currently supports task intent bindings."
            self._trace(
                run_id,
                working_memory,
                "planning.binding_error",
                error,
                {"binding_type": binding["type"]},
            )
            return self._execution_result("impasse", working_memory, run_id, error=error)

        goal_args = [
            str(intent_frame.normalized_parameters[param_name])
            for param_name in binding.get("args", [])
        ]
        bootstrapper.create_goal(
            {
                "task": binding["task"],
                "args": goal_args,
                "selection_preferences": intent_frame.preference_tokens,
            }
        )
        working_memory.planner_tier = "htn"
        self._trace(
            run_id,
            working_memory,
            "planning.goal",
            "Created HTN goal from configured intent binding",
            {
                "task": binding["task"],
                "args": goal_args,
                "selection_preferences": intent_frame.preference_tokens,
            },
        )

        try:
            plan = planner.execute_plan()
        except HTNMethodTieImpasse as exc:
            working_memory.planner_tier = "tie_impasse"
            self._trace(
                run_id,
                working_memory,
                "planning.tie_impasse",
                "Multiple applicable HTN methods remained equally preferred",
                {
                    "task_name": exc.task_name,
                    "candidate_methods": exc.candidate_methods,
                },
            )
            return self._execution_result("tie_impasse", working_memory, run_id)

        if plan is None:
            working_memory.planner_tier = "impasse"
            self._trace(run_id, working_memory, "planning.impasse", "Planner returned no plan")
            return self._execution_result("impasse", working_memory, run_id)

        if planner.last_method_selection is not None:
            working_memory.selected_method = planner.last_method_selection.selected_method
        plan_actions = [
            action_instance.action.name
            for action_instance in plan.actions
        ]
        self._trace(
            run_id,
            working_memory,
            "planning.plan",
            "Planner produced executable plan",
            {
                "actions": plan_actions,
                "method_selection": (
                    planner.last_method_selection.__dict__
                    if planner.last_method_selection is not None
                    else None
                ),
            },
        )

        self.pending_execution_results = []
        self.pending_rendered_responses = []
        return self._execute_plan_from_index(
            plan=plan,
            bootstrapper=bootstrapper,
            registry=registry,
            run_id=run_id,
            start_index=0,
            plan_actions=plan_actions,
            selected_method=working_memory.selected_method,
            approve_first_action=False,
        )

    def _execution_result(
        self,
        status: str,
        working_memory: WorkingMemoryState,
        run_id: str,
        plan_actions: list[str] | None = None,
        execution: list[Any] | None = None,
        selected_method: str | None = None,
        approval_request: dict[str, Any] | None = None,
        rendered_responses: list[str] | None = None,
        error: str | None = None,
    ) -> ConfiguredExecutionResult:
        """Build an execution result from recorded trace events."""
        return ConfiguredExecutionResult(
            status=status,
            working_memory=working_memory,
            trace=[
                event
                for event in self.trace_recorder.events
                if event.run_id == run_id
            ],
            plan_actions=plan_actions or [],
            execution=execution or [],
            selected_method=selected_method,
            approval_request=approval_request,
            rendered_responses=rendered_responses or [],
            error=error,
        )

    def _execute_plan_from_index(
        self,
        plan: Any,
        bootstrapper: DomainBootstrapper,
        registry: PluginRegistry,
        run_id: str,
        start_index: int,
        plan_actions: list[str],
        selected_method: str | None,
        approve_first_action: bool,
    ) -> ConfiguredExecutionResult:
        """Execute a plan from an index, pausing before approval-gated actions."""
        working_memory = self._ensure_working_memory()
        approval_granted_for_current_action = approve_first_action
        for index in range(start_index, len(plan.actions)):
            action_instance = plan.actions[index]
            action_name = action_instance.action.name
            kwargs = self._action_kwargs(action_instance)
            self._trace(
                run_id,
                working_memory,
                "execution.action.prepare",
                "Preparing physical action",
                {"action": action_name, "parameters": kwargs},
            )
            plugin_meta = registry.get_plugin(action_name)
            if plugin_meta.get("requires_approval", False):
                if not approval_granted_for_current_action:
                    approval_request = {
                        "action": action_name,
                        "parameters": kwargs,
                        "plan_action_index": index,
                    }
                    self._trace(
                        run_id,
                        working_memory,
                        "hitl.approval.required",
                        "Action requires human approval",
                        approval_request,
                    )
                    self._store_pending_approval(
                        plan=plan,
                        bootstrapper=bootstrapper,
                        run_id=run_id,
                        next_action_index=index,
                        plan_actions=plan_actions,
                        selected_method=selected_method,
                        approval_request=approval_request,
                    )
                    return self._execution_result(
                        "approval_required",
                        working_memory,
                        run_id,
                        plan_actions=plan_actions,
                        execution=list(self.pending_execution_results),
                        selected_method=selected_method,
                        approval_request=approval_request,
                        rendered_responses=list(self.pending_rendered_responses),
                    )
                self._trace(
                    run_id,
                    working_memory,
                    "hitl.approval.granted",
                    "Human approved action execution",
                    {"action": action_name, "parameters": kwargs},
                )
                approval_granted_for_current_action = False

            try:
                result = registry.execute_plugin(action_name, **kwargs)
            except Exception as exc:
                self._trace(
                    run_id,
                    working_memory,
                    "execution.action.failure",
                    "Physical action failed",
                    {
                        "action": action_name,
                        "parameters": kwargs,
                        "error": str(exc),
                    },
                )
                self._clear_pending_approval()
                return self._execution_result(
                    "execution_error",
                    working_memory,
                    run_id,
                    plan_actions=plan_actions,
                    execution=list(self.pending_execution_results),
                    selected_method=selected_method,
                    rendered_responses=list(self.pending_rendered_responses),
                    error=str(exc),
                )

            normalized_result = self._handle_plugin_result(
                result=result,
                action_name=action_name,
                run_id=run_id,
            )
            self.pending_execution_results.append(result)
            self._apply_action_effects(
                action_instance=action_instance,
                bootstrapper=bootstrapper,
                working_memory=working_memory,
                run_id=run_id,
            )
            self._trace(
                run_id,
                working_memory,
                "execution.action.success",
                "Physical action completed",
                {
                    "action": action_name,
                    "parameters": kwargs,
                    "result": normalized_result,
                },
            )

        execution = list(self.pending_execution_results)
        rendered_responses = list(self.pending_rendered_responses)
        working_memory.hitl_state = {"status": "completed"}
        self._clear_pending_approval()
        self._trace(
            run_id,
            working_memory,
            "execution.complete",
            "Executed configured plan through physical driver",
            {"results": execution},
        )
        return self._execution_result(
            "success",
            working_memory,
            run_id,
            plan_actions=plan_actions,
            execution=execution,
            selected_method=selected_method,
            rendered_responses=rendered_responses,
        )

    def _handle_plugin_result(
        self,
        result: Any,
        action_name: str,
        run_id: str,
    ) -> Any:
        """Handle structured plugin outputs and return trace-safe result data."""
        if isinstance(result, OptimizationExecutionOutput):
            self._record_optimization_output(
                output=result,
                action_name=action_name,
                run_id=run_id,
            )
            return result.model_dump()
        if isinstance(result, dict):
            try:
                output = OptimizationExecutionOutput.model_validate(result)
            except Exception:
                return result
            self._record_optimization_output(
                output=output,
                action_name=action_name,
                run_id=run_id,
            )
            return output.model_dump()
        return result

    def _record_optimization_output(
        self,
        output: OptimizationExecutionOutput,
        action_name: str,
        run_id: str,
    ) -> None:
        """Persist and summarize a structured optimization plugin output."""
        working_memory = self._ensure_working_memory()
        decision_payload = output.result.model_dump()
        self._trace(
            run_id,
            working_memory,
            "optimization.decision",
            "Optimizer selected a candidate bundle under constraints",
            decision_payload,
        )
        record = MemoryRecord(
            memory_type=MemoryType.OPTIMIZATION_DECISION,
            scope=MemoryScope.SESSION,
            subject_ids=output.subject_ids,
            source=MemorySource(
                system="configured_interaction",
                reference=f"{run_id}:{action_name}",
            ),
            lifecycle_state=MemoryLifecycleState.VALIDATED,
            payload=OptimizationDecisionPayload(**decision_payload),
        )
        if self.memory_sink is not None:
            self.memory_sink.hook_memory_record(record)
        self._trace(
            run_id,
            working_memory,
            "memory.optimization_decision",
            "Optimizer decision captured as a typed memory record",
            {
                "record_id": record.record_id,
                "memory_type": record.memory_type,
                "selected_candidate_ids": record.payload.selected_candidate_ids,
            },
        )
        rendered = self._render_optimizer_response(output)
        if rendered is None:
            return
        self.pending_rendered_responses.append(rendered)
        self._trace(
            run_id,
            working_memory,
            f"response.{output.response_type}",
            "Rendered guarded optimizer summary before HITL approval",
            {"text": rendered},
        )

    def _render_optimizer_response(
        self,
        output: OptimizationExecutionOutput,
    ) -> str | None:
        """Render a configured response for a structured optimization result."""
        if self.package.responses is None:
            return None
        policy = self.package.responses.responses.get(output.response_type)
        if policy is None:
            return None
        facts = dict(output.result.selected_attributes)
        facts.update(output.response_facts)
        result = self.renderer.render(
            ResponseFrame(
                response_type=output.response_type,
                facts=self._nested_response_facts(facts),
                required_claims=policy.required_fields,
                forbidden_claims=policy.forbidden_terms,
            ),
            policy,
        )
        return result.text

    def _nested_response_facts(self, facts: dict[str, Any]) -> dict[str, Any]:
        """Convert flat dotted fact keys into nested response facts."""
        nested: dict[str, Any] = {}
        for key, value in facts.items():
            current = nested
            parts = key.split(".")
            for part in parts[:-1]:
                next_value = current.setdefault(part, {})
                if not isinstance(next_value, dict):
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested

    def _handle_approval_turn(
        self,
        user_text: str,
    ) -> ConfiguredExecutionResult | None:
        """Resume or deny a pending approval-gated action from user text."""
        normalized = user_text.strip().lower()
        if normalized in {
            "y",
            "yes",
            "yes please",
            "approve",
            "approved",
            "approve it",
            "go ahead",
            "proceed",
        }:
            if (
                self.pending_plan is None
                or self.pending_bootstrapper is None
                or self.pending_run_id is None
                or self.pending_next_action_index is None
            ):
                return None
            return self._execute_plan_from_index(
                plan=self.pending_plan,
                bootstrapper=self.pending_bootstrapper,
                registry=self.registry or default_registry,
                run_id=self.pending_run_id,
                start_index=self.pending_next_action_index,
                plan_actions=self.pending_plan_actions,
                selected_method=self.pending_selected_method,
                approve_first_action=True,
            )
        if (
            normalized in {
                "n",
                "no",
                "no thanks",
                "deny",
                "denied",
                "stop",
                "cancel",
            }
            or any(
                phrase in normalized
                for phrase in [
                    "changed my mind",
                    "change my mind",
                    "do not approve",
                    "don't approve",
                ]
            )
        ):
            return self._deny_pending_approval(user_text)
        return self._repeat_pending_approval_request()

    def _deny_pending_approval(self, user_text: str) -> ConfiguredExecutionResult:
        """Deny and clear a pending approval-gated action."""
        del user_text
        working_memory = self._ensure_working_memory()
        run_id = self.pending_run_id or str(uuid4())
        approval_request = self._pending_approval_request()
        self._trace(
            run_id,
            working_memory,
            "hitl.approval.denied",
            "Human denied action execution",
            approval_request or {},
        )
        working_memory.hitl_state = {
            "status": "denied",
            "approval_request": approval_request,
        }
        execution = list(self.pending_execution_results)
        rendered_responses = list(self.pending_rendered_responses)
        plan_actions = list(self.pending_plan_actions)
        selected_method = self.pending_selected_method
        self._clear_pending_approval()
        return self._execution_result(
            "approval_denied",
            working_memory,
            run_id,
            plan_actions=plan_actions,
            execution=execution,
            selected_method=selected_method,
            approval_request=approval_request,
            rendered_responses=rendered_responses,
        )

    def _repeat_pending_approval_request(self) -> ConfiguredExecutionResult:
        """Return the current approval request without advancing execution."""
        working_memory = self._ensure_working_memory()
        run_id = self.pending_run_id or str(uuid4())
        approval_request = self._pending_approval_request()
        return self._execution_result(
            "approval_required",
            working_memory,
            run_id,
            plan_actions=list(self.pending_plan_actions),
            execution=list(self.pending_execution_results),
            selected_method=self.pending_selected_method,
            approval_request=approval_request,
            rendered_responses=list(self.pending_rendered_responses),
        )

    def _store_pending_approval(
        self,
        plan: Any,
        bootstrapper: DomainBootstrapper,
        run_id: str,
        next_action_index: int,
        plan_actions: list[str],
        selected_method: str | None,
        approval_request: dict[str, Any],
    ) -> None:
        """Store enough execution state to resume after HITL approval."""
        self.pending_plan = plan
        self.pending_bootstrapper = bootstrapper
        self.pending_run_id = run_id
        self.pending_next_action_index = next_action_index
        self.pending_plan_actions = list(plan_actions)
        self.pending_selected_method = selected_method
        self._ensure_working_memory().hitl_state = {
            "status": "approval_required",
            "approval_request": approval_request,
        }

    def _clear_pending_approval(self) -> None:
        """Clear pending HITL execution state."""
        self.pending_plan = None
        self.pending_bootstrapper = None
        self.pending_run_id = None
        self.pending_next_action_index = None
        self.pending_execution_results = []
        self.pending_rendered_responses = []
        self.pending_plan_actions = []
        self.pending_selected_method = None

    def _pending_approval_request(self) -> dict[str, Any] | None:
        """Return the active approval request from working memory."""
        working_memory = self._ensure_working_memory()
        if working_memory.hitl_state is None:
            return None
        request = working_memory.hitl_state.get("approval_request")
        return request if isinstance(request, dict) else None

    def _apply_action_effects(
        self,
        action_instance: Any,
        bootstrapper: DomainBootstrapper,
        working_memory: WorkingMemoryState,
        run_id: str,
    ) -> None:
        """Apply declared action effects to working memory after success."""
        for record in planner_fact_records_from_action_effects(
            action_instance,
            bootstrapper.action_specs,
            source_system="configured_interaction_execution",
            source_reference=run_id,
        ):
            working_memory.hydrate_planner_fact(record)

    def _action_kwargs(self, action_instance: Any) -> dict[str, str]:
        """Extract concrete action parameters from a UPF action instance."""
        kwargs: dict[str, str] = {}
        for param, actual_val in zip(
            action_instance.action.parameters,
            action_instance.actual_parameters,
            strict=True,
        ):
            kwargs[param.name] = actual_val.object().name
        return kwargs

    def _trace(
        self,
        run_id: str,
        working_memory: WorkingMemoryState,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Record a configured interaction trace event."""
        event = self.trace_recorder.emit(
            run_id=run_id,
            stage=stage,
            message=message,
            payload=payload,
        )
        working_memory.add_trace_event(event.event_id)
        return event

    def _interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
    ) -> ConfiguredTurnInterpretation:
        """Interpret a turn with an optional interpreter or deterministic fallback."""
        if self.interpreter is not None:
            return self.interpreter.interpret_turn(
                user_text=user_text,
                working_memory=working_memory,
                package=self.package,
                pending_clarification=self.pending_clarification,
            )

        if self.pending_clarification is not None:
            return ConfiguredTurnInterpretation(
                turn_type="clarification_answer",
                intent_name=self.pending_clarification.intent_name,
                raw_slots=self._fallback_clarification_slots(user_text),
            )

        classification, _reason = self.policy.classify_turn(user_text)
        if classification == "task":
            return ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name=self._default_intent_name(),
            )
        return ConfiguredTurnInterpretation(
            turn_type="conversation",
            response_text=self._conversation_fallback(),
        )

    def _fallback_clarification_slots(self, user_text: str) -> dict[str, Any]:
        """Map a bare clarification answer into the first missing slot."""
        if self.pending_clarification is None:
            return {}
        if len(self.pending_clarification.missing_slots) != 1:
            return {}
        return {self.pending_clarification.missing_slots[0]: user_text.strip()}

    def _merged_slots(
        self,
        interpretation: ConfiguredTurnInterpretation,
    ) -> dict[str, Any]:
        """Merge pending slots with newly extracted raw slots."""
        slots = dict(self.pending_slots)
        slots.update(interpretation.raw_slots)
        return slots

    def _grounding_clarification(
        self,
        intent_name: str,
        raw_slots: dict[str, Any],
        intent_frame: IntentFrame,
    ) -> IntentClarification | None:
        """Return a clarification if a planner object cannot be grounded."""
        if self.package.intents is None:
            return None
        intent_spec = self.package.intents.intents[intent_name]
        for slot_name, normalized_value in intent_frame.normalized_parameters.items():
            slot_spec = intent_spec.slots.get(slot_name)
            if slot_spec is None:
                continue
            slot_type = slot_spec.slot_type
            if slot_type in {"integer", "money", "enum"}:
                continue
            if not isinstance(normalized_value, str):
                continue
            if self.objects.get(normalized_value) == slot_type:
                continue
            raw_value = raw_slots.get(slot_name, normalized_value)
            return IntentClarification(
                intent_name=intent_name,
                missing_slots=[slot_name],
                question=self._grounding_question(
                    slot_name=slot_name,
                    slot_type=slot_type,
                    raw_value=raw_value,
                    fallback_question=slot_spec.clarification,
                ),
            )
        return None

    def _grounding_question(
        self,
        slot_name: str,
        slot_type: str,
        raw_value: Any,
        fallback_question: str | None,
    ) -> str:
        """Build a user-facing grounding clarification question."""
        if slot_name == "destination" and slot_type == "City":
            return (
                f"I couldn't ground '{raw_value}' to a specific city. "
                "What city do you want to visit?"
            )
        if slot_name == "origin" and slot_type == "City":
            return (
                f"I couldn't ground '{raw_value}' to a supported departure city. "
                "What city are you departing from?"
            )
        if fallback_question:
            return f"I couldn't ground '{raw_value}'. {fallback_question}"
        return f"I couldn't ground '{raw_value}' for {slot_name}. Can you clarify?"

    def _is_in_scope(
        self,
        user_text: str,
        interpretation: ConfiguredTurnInterpretation,
    ) -> bool:
        """Return whether a user request belongs to the configured domain."""
        if self.package.intents is None:
            return True
        if self._intent_builder().in_scope(user_text):
            return True
        if (
            interpretation.intent_name in self.package.intents.intents
            and bool(interpretation.raw_slots)
        ):
            return True
        return any(
            intent_name in self.package.intents.intents
            for intent_name in [self.pending_intent_name]
            if intent_name is not None
        )

    def _render_refusal(self) -> str:
        """Render the configured out-of-domain refusal response."""
        if self.package.intents is None or self.package.responses is None:
            return "I can't handle that in this domain."
        response_name = self.package.intents.scope.refusal_response
        policy = self.package.responses.responses.get(response_name)
        if policy is None:
            return "I can't handle that in this domain."
        return self.renderer.render(
            ResponseFrame(response_type=response_name),
            policy,
        ).text

    def _response_for_execution(self, result: ConfiguredExecutionResult) -> str:
        """Return a conservative user-facing response for execution state."""
        prefix = "\n\n".join(result.rendered_responses)
        if prefix:
            self.pending_rendered_responses = []
        if result.status == "success":
            response = "Completed the configured request."
            return f"{prefix}\n\n{response}" if prefix else response
        if result.status == "approval_required":
            response = (
                "I reached an approval-gated action and need authorization "
                "before continuing."
            )
            return f"{prefix}\n\n{response}" if prefix else response
        if result.status == "approval_denied":
            response = "I stopped before placing holds. What would you like to change: budget, dates, duration, hotel preference, or destination?"
            return f"{prefix}\n\n{response}" if prefix else response
        if result.status == "tie_impasse":
            return (
                "I found multiple equally preferred planning methods and need "
                "a preference to choose one."
            )
        if result.status == "impasse":
            return "I could not find a valid plan for that request."
        if result.error is not None:
            return f"The request ended with status {result.status}: {result.error}"
        return f"The request ended with status {result.status}."

    def _conversation_result(
        self,
        response_text: str,
        status: str,
        interpretation: ConfiguredTurnInterpretation | None = None,
        intent_frame: IntentFrame | None = None,
        clarification: IntentClarification | None = None,
        execution_result: ConfiguredExecutionResult | None = None,
        blocked_reason: str | None = None,
    ) -> ConfiguredInteractionTurnResult:
        """Build and record a turn result."""
        self._write_conversation_record(role="assistant", content=response_text)
        return ConfiguredInteractionTurnResult(
            session_id=self.session_id,
            user_id=self.user_id,
            response_text=response_text,
            status=status,
            working_memory=self._ensure_working_memory(),
            interpretation=interpretation,
            intent_frame=intent_frame,
            clarification=clarification,
            execution_result=execution_result,
            blocked_reason=blocked_reason,
        )

    def _conversation_fallback(self) -> str:
        """Return a conservative conversation-only response."""
        domain = (
            self.package.intents.domain
            if self.package.intents is not None
            else "this domain"
        )
        return f"Hello. I can help with {domain} requests."

    def _intent_builder(self) -> IntentFrameBuilder:
        """Return an intent frame builder for the loaded domain package."""
        if self.package.intents is None:
            raise ValueError("Configured interaction requires intents.yaml.")
        return IntentFrameBuilder(self.package.intents)

    def _default_intent_name(self) -> str | None:
        """Return the only configured intent name when a fallback can infer it."""
        if self.package.intents is None or len(self.package.intents.intents) != 1:
            return None
        return next(iter(self.package.intents.intents))

    def _ensure_working_memory(self) -> WorkingMemoryState:
        """Create or return session-scoped working memory."""
        if self.working_memory is None:
            self.working_memory = WorkingMemoryState(
                session_id=self.session_id,
                user_id=self.user_id,
            )
        return self.working_memory

    def _write_conversation_record(self, role: str, content: str) -> None:
        """Persist one conversation turn as a typed memory record."""
        if self.memory_sink is None:
            return

        record = MemoryRecord(
            memory_type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            subject_ids=[self.user_id] if self.user_id is not None else [],
            source=MemorySource(
                system="configured_interaction_session",
                reference=self.session_id,
            ),
            lifecycle_state=MemoryLifecycleState.VALIDATED,
            payload=ConversationMemoryPayload(
                role=role,
                content=content,
                turn_id=str(uuid4()),
            ),
        )
        self.memory_sink.hook_memory_record(record)
