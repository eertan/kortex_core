"""Travel concierge demo scenario built on the generic scenario harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from unified_planning.shortcuts import get_environment

from kortex.domain_package import DomainPackage, DomainPackageLoader
from kortex.intent_runtime import IntentFrame, IntentFrameBuilder
from kortex.memory.records import (
    ExternalKnowledgePayload,
    MemoryLifecycleState,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
    OptimizationDecisionPayload,
)
from kortex.memory.working import WorkingMemoryState
from kortex.optimization import (
    KortexOptimizer,
    OptimizationCandidate,
    OptimizationCandidateSet,
    OptimizationExecutionOutput,
    OptimizationPolicy,
    OptimizationResult,
)
from kortex.plugins.registry import PluginRegistry
from kortex.responses import (
    ResponseFrame,
    ResponseNarrator,
    ResponsePolicy,
    ResponseRenderer,
)
from scenarios.harness import (
    DemoLogger,
    build_working_memory,
    execute_plan_with_logging,
    setup_planner,
)


TRAVEL_PACKAGE_PATH = Path(__file__).parent / "domains" / "travel_concierge"
TRAVEL_DECISIONS: dict[str, OptimizationResult] = {}


class TravelDemoNarrator:
    """Deterministic narrator that exercises the guarded response path."""

    def narrate(self, frame: ResponseFrame, policy: ResponsePolicy) -> str:
        """Return a natural response using only frame facts."""
        del policy
        facts = frame.facts
        return (
            f"I found {facts['flight_count']} flight options and "
            f"{facts['hotel_count']} hotel options. The best fit is "
            f"{facts['flight']['name']} with {facts['hotel']['name']} in "
            f"{facts['hotel']['neighborhood']}. The estimated bundle total is "
            f"${facts['total_cost']}, and it stays within your budget. I can "
            "ask for approval before placing refundable holds."
        )

FLIGHT_OPTIONS: list[dict[str, object]] = [
    {
        "candidate_id": "flight_refundable_balanced",
        "name": "Pacific Arc 221",
        "carrier": "Pacific Arc",
        "route": "BOS -> HND",
        "departure": "2026-06-12 09:35",
        "arrival": "2026-06-13 14:10",
        "price": 780,
        "duration_hours": 16,
        "stops": 1,
        "tags": ["refundable", "one_stop"],
        "preference_fit": 0.8,
    },
    {
        "candidate_id": "flight_cheap_overnight",
        "name": "Northstar Saver 88",
        "carrier": "Northstar",
        "route": "BOS -> NRT",
        "departure": "2026-06-12 22:10",
        "arrival": "2026-06-14 06:40",
        "price": 620,
        "duration_hours": 24,
        "stops": 2,
        "tags": ["overnight_layover"],
        "preference_fit": 0.2,
    },
    {
        "candidate_id": "flight_premium_direct",
        "name": "Sakura Direct 17",
        "carrier": "Sakura Air",
        "route": "BOS -> HND",
        "departure": "2026-06-12 13:00",
        "arrival": "2026-06-13 16:20",
        "price": 1420,
        "duration_hours": 14,
        "stops": 0,
        "tags": ["refundable", "direct", "premium"],
        "preference_fit": 0.75,
    },
]

HOTEL_OPTIONS: list[dict[str, object]] = [
    {
        "candidate_id": "hotel_boutique_quiet",
        "name": "Yanaka Atelier Stay",
        "neighborhood": "Yanaka",
        "location_note": "quiet older Tokyo neighborhood near food streets",
        "nightly_rate": 260,
        "stars": 4,
        "tags": ["boutique", "quiet", "food_markets"],
        "preference_fit": 0.9,
    },
    {
        "candidate_id": "hotel_budget_chain",
        "name": "Shinjuku Central Inn",
        "neighborhood": "Shinjuku",
        "location_note": "central, efficient, busier area",
        "nightly_rate": 180,
        "stars": 3,
        "tags": ["chain", "central"],
        "preference_fit": 0.3,
    },
    {
        "candidate_id": "hotel_luxury_boutique",
        "name": "Aoyama Garden House",
        "neighborhood": "Aoyama",
        "location_note": "design-focused boutique stay near galleries",
        "nightly_rate": 410,
        "stars": 5,
        "tags": ["boutique", "quiet", "premium"],
        "preference_fit": 0.85,
    },
]


def build_registry(
    package: DomainPackage,
    logger: DemoLogger | None = None,
    working_memory: WorkingMemoryState | None = None,
    narrator: ResponseNarrator | None = None,
) -> PluginRegistry:
    """Create an isolated plugin registry for the travel demo."""
    registry = PluginRegistry()

    @registry.register_action("search_flights")
    def search_flights(
        origin: str,
        destination: str,
        travel_window: str,
        budget: str,
    ) -> str:
        """Return deterministic fake flight options."""
        _record_option_memory(
            logger=logger,
            working_memory=working_memory,
            endpoint_id="mock_flight_search",
            query={
                "origin": origin,
                "destination": destination,
                "travel_window": travel_window,
                "budget": budget,
            },
            options=FLIGHT_OPTIONS,
        )
        return (
            f"Found {len(FLIGHT_OPTIONS)} flight options from {origin} to "
            f"{destination}; lowest refundable fare is "
            f"${_lowest_refundable_flight_price()}."
        )

    @registry.register_action("search_hotels")
    def search_hotels(destination: str, budget: str, style: str) -> str:
        """Return deterministic fake hotel options."""
        _record_option_memory(
            logger=logger,
            working_memory=working_memory,
            endpoint_id="mock_hotel_search",
            query={"destination": destination, "budget": budget, "style": style},
            options=HOTEL_OPTIONS,
        )
        return (
            f"Found {len(HOTEL_OPTIONS)} hotel options in {destination}; best "
            "boutique match is Yanaka Atelier Stay."
        )

    @registry.register_action("search_local_experiences")
    def search_local_experiences(destination: str, style: str) -> str:
        """Return deterministic fake local experience options."""
        return (
            f"Found {style} local experiences in {destination}: food markets, "
            "neighborhood walks, and low-density evening options."
        )

    @registry.register_action("build_itinerary")
    def build_itinerary(
        destination: str,
        duration_days: str,
        style: str,
    ) -> str:
        """Return deterministic fake itinerary output."""
        return (
            f"Built a {style} {duration_days} itinerary for {destination} with "
            "light daily pacing."
        )

    @registry.register_action("optimize_travel_bundle")
    def optimize_travel_bundle(
        origin: str,
        destination: str,
        duration_days: str,
        budget: str,
    ) -> str:
        """Select a travel bundle with the generic optimizer."""
        del origin
        duration_count = int(duration_days.removeprefix("duration_").removesuffix("_days"))
        budget_limit = int(budget.removeprefix("budget_"))
        policy = _travel_bundle_policy(package, destination, budget_limit)
        result = KortexOptimizer().optimize(
            policy=policy,
            candidate_sets=[
                OptimizationCandidateSet(
                    set_id="flight_options",
                    candidates=[
                        _candidate_from_option("flight", option)
                        for option in FLIGHT_OPTIONS
                    ],
                ),
                OptimizationCandidateSet(
                    set_id="hotel_options",
                    candidates=[
                        _candidate_from_option("hotel", option)
                        for option in HOTEL_OPTIONS
                    ],
                ),
            ],
            context={"duration_days": duration_count},
        )
        TRAVEL_DECISIONS[result.decision_id] = result
        if logger is not None:
            decision_payload = result.model_dump()
            logger.event(
                "optimization.decision",
                "Generic optimizer selected a candidate bundle under constraints",
                decision_payload,
            )
            record = MemoryRecord(
                memory_type=MemoryType.OPTIMIZATION_DECISION,
                scope=MemoryScope.SESSION,
                subject_ids=[destination],
                source=MemorySource(
                    system="travel_concierge_demo",
                    reference="travel_concierge",
                ),
                lifecycle_state=MemoryLifecycleState.VALIDATED,
                payload=OptimizationDecisionPayload(**decision_payload),
            )
            logger.event(
                "memory.optimization_decision",
                "Optimizer decision captured as a typed memory record",
                {
                    "record_id": record.record_id,
                    "memory_type": record.memory_type,
                    "selected_candidate_ids": record.payload.selected_candidate_ids,
                },
            )
            response_result = _render_optimizer_summary(
                package=package,
                result=result,
                narrator=narrator or TravelDemoNarrator(),
            )
            logger.event(
                "response.optimizer_summary",
                "Rendered guarded optimizer summary before HITL approval",
                {
                    "mode_used": response_result.mode_used,
                    "guard_reason": response_result.guard_reason,
                    "text": response_result.text,
                },
            )
        selected_flight = _selected_option_from_ids("flight", result.selected_candidate_ids)
        selected_hotel = _selected_option_from_ids("hotel", result.selected_candidate_ids)
        message = (
            "Selected reservation group "
            f"{_selected_bundle_summary(result.selected_candidate_ids)} with "
            f"total cost ${result.selected_attributes.get('total_cost')}."
        )
        return OptimizationExecutionOutput(
            message=message,
            result=result,
            response_type="optimizer_summary",
            response_facts={
                "flight": selected_flight,
                "hotel": selected_hotel,
                "total_cost": result.selected_attributes["total_cost"],
                "flight_count": len(FLIGHT_OPTIONS),
                "hotel_count": len(HOTEL_OPTIONS),
            },
            subject_ids=[destination],
        )

    @registry.register_action("reserve_flight_hold", requires_approval=True)
    def reserve_flight_hold(origin: str, destination: str) -> str:
        """Create a fake refundable flight hold."""
        flight = _selected_option("flight")
        return (
            f"Placed refundable flight hold for {flight.get('name')} "
            f"from {origin} to {destination}."
        )

    @registry.register_action("reserve_hotel_hold", requires_approval=True)
    def reserve_hotel_hold(destination: str) -> str:
        """Create a fake refundable hotel hold."""
        hotel = _selected_option("hotel")
        return (
            f"Placed refundable hotel hold for {hotel.get('name')} "
            f"in {destination}."
        )

    @registry.register_action("finalize_trip_plan")
    def finalize_trip_plan(destination: str) -> str:
        """Finalize the fake trip plan after prerequisites are complete."""
        return f"Finalized travel plan for {destination}."

    return registry


def _candidate_from_option(
    candidate_type: str,
    option: dict[str, object],
) -> OptimizationCandidate:
    """Convert one mock option dictionary into an optimizer candidate."""
    return OptimizationCandidate(
        candidate_id=str(option["candidate_id"]),
        candidate_type=candidate_type,
        attributes={
            key: value
            for key, value in option.items()
            if key != "candidate_id"
        },
    )


def _record_option_memory(
    logger: DemoLogger | None,
    working_memory: WorkingMemoryState | None,
    endpoint_id: str,
    query: dict[str, object],
    options: list[dict[str, object]],
) -> None:
    """Record mock option data as external knowledge referenced by working memory."""
    record = MemoryRecord(
        memory_type=MemoryType.EXTERNAL_KNOWLEDGE,
        scope=MemoryScope.SESSION,
        subject_ids=[str(value) for value in query.values()],
        source=MemorySource(system="travel_concierge_demo", reference=endpoint_id),
        lifecycle_state=MemoryLifecycleState.VALIDATED,
        payload=ExternalKnowledgePayload(
            endpoint_id=endpoint_id,
            query=query,
            result={"options": options},
            may_hydrate_planner=False,
        ),
    )
    if working_memory is not None:
        working_memory.remember_retrieved_record(record)
    if logger is not None:
        logger.event(
            "memory.option_hydration",
            "Hydrated working memory with external option candidates",
            {
                "record_id": record.record_id,
                "endpoint_id": endpoint_id,
                "may_hydrate_planner": False,
                "option_count": len(options),
                "options": options,
            },
        )


def _lowest_refundable_flight_price() -> int:
    """Return the lowest refundable mock flight price."""
    refundable = [
        int(option["price"])
        for option in FLIGHT_OPTIONS
        if "refundable" in option.get("tags", [])
    ]
    return min(refundable)


def _selected_bundle_summary(selected_candidate_ids: list[str]) -> str:
    """Return readable selected flight/hotel names for demo output."""
    options = {
        str(option["candidate_id"]): option
        for option in [*FLIGHT_OPTIONS, *HOTEL_OPTIONS]
    }
    return " + ".join(
        str(options[candidate_id]["name"])
        for candidate_id in selected_candidate_ids
    )


def _selected_option(candidate_type: str) -> dict[str, object]:
    """Return the selected option details for a candidate type."""
    latest_decision = next(reversed(TRAVEL_DECISIONS.values()))
    selected_ids = latest_decision.selected_candidate_ids
    options = FLIGHT_OPTIONS if candidate_type == "flight" else HOTEL_OPTIONS
    for option in options:
        if option["candidate_id"] in selected_ids:
            return option
    raise KeyError(f"No selected {candidate_type} option is available.")


def _render_optimizer_summary(
    package: DomainPackage,
    result: OptimizationResult,
    narrator: ResponseNarrator,
) -> object:
    """Render a guarded natural summary of the optimizer decision."""
    selected_flight = _selected_option_from_ids("flight", result.selected_candidate_ids)
    selected_hotel = _selected_option_from_ids("hotel", result.selected_candidate_ids)
    frame = ResponseFrame(
        response_type="optimizer_summary",
        facts={
            "flight": selected_flight,
            "hotel": selected_hotel,
            "total_cost": result.selected_attributes["total_cost"],
            "flight_count": len(FLIGHT_OPTIONS),
            "hotel_count": len(HOTEL_OPTIONS),
        },
        required_claims=["flight.name", "hotel.name"],
        forbidden_claims=["booking confirmed", "payment processed"],
    )
    if package.responses is None:
        raise ValueError("Travel package is missing responses.yaml.")
    policy = package.responses.responses["optimizer_summary"]
    return ResponseRenderer(narrator=narrator).render(frame, policy)


def _travel_bundle_policy(
    package: DomainPackage,
    destination: str,
    budget_limit: int,
) -> OptimizationPolicy:
    """Load and specialize the travel bundle optimizer policy from config."""
    if package.decisions is None:
        raise ValueError("Travel package is missing decisions.yaml.")
    policy = package.decisions.decisions["choose_travel_bundle"].policy.model_copy(
        deep=True
    )
    policy.decision_id = f"{policy.decision_id}:{destination}"
    for constraint in policy.constraints:
        if constraint.field == "total_cost" and constraint.operator == "<=":
            constraint.value = budget_limit
    return policy


def _selected_option_from_ids(
    candidate_type: str,
    selected_candidate_ids: list[str],
) -> dict[str, object]:
    """Return selected option details from an explicit candidate-id list."""
    options = FLIGHT_OPTIONS if candidate_type == "flight" else HOTEL_OPTIONS
    for option in options:
        if option["candidate_id"] in selected_candidate_ids:
            return option
    raise KeyError(f"No selected {candidate_type} option is available.")


def run_travel_demo(log_path: Path, approval: str) -> None:
    """Run the travel concierge scenario and write structured logs."""
    get_environment().credits_stream = None
    TRAVEL_DECISIONS.clear()
    package = DomainPackageLoader().load(TRAVEL_PACKAGE_PATH)
    logger = DemoLogger()
    objects = {
        "boston": "City",
        "tokyo": "City",
        "next_month": "TravelWindow",
        "budget_2500": "Budget",
        "duration_3_days": "TripDuration",
        "relaxed": "TravelStyle",
    }
    initial_state: list[dict[str, object]] = []
    logger.start(
        "travel_concierge",
        "relaxed trip planning with preference-selected HTN and PDDL ordering",
        package.domain_path,
    )
    logger.note(
        "User request: Plan a relaxed three-day trip from Boston to Tokyo next "
        "month under $2500, and place refundable holds if I approve."
    )
    logger.note(
        "Retrieved memory note: user prefers boutique hotels, food markets, "
        "and lighter schedules."
    )
    if package.intents is None:
        raise ValueError("Travel package is missing intents.yaml.")
    intent_result = IntentFrameBuilder(package.intents).build(
        "plan_trip",
        {
            "origin": "boston",
            "destination": "tokyo",
            "duration_days": 3,
            "travel_window": "next_month",
            "budget": "$2500",
            "style": "relaxed",
        },
    )
    if not isinstance(intent_result, IntentFrame):
        logger.event(
            "interaction.clarification",
            "Intent config requested clarification before planning",
            intent_result.model_dump(),
        )
        logger.write_json(log_path)
        return
    logger.event(
        "interaction.intent_frame",
        "Intent config produced canonical planner parameters",
        intent_result.model_dump(),
    )

    working_memory = build_working_memory("travel_concierge", initial_state)
    registry = build_registry(package, logger, working_memory)
    planner, bootstrapper = setup_planner(
        domain_path=package.domain_path,
        registry=registry,
        objects=objects,
        initial_state=initial_state,
        planner_name="travel_concierge_demo",
    )
    candidate_methods = [
        {
            "name": method_spec["name"],
            "target_task": "build_trip_plan",
            "preconditions": method_spec.get("preconditions", []),
            "preference_matches": method_spec.get("preference_matches", []),
            "selection_priority": method_spec.get("selection_priority", 0),
            "subtask_mode": (
                "unordered_classical_planning"
                if method_spec.get("subtasks")
                else "ordered_expansion"
            ),
            "declared_subtasks": method_spec.get("subtasks")
            or method_spec.get("ordered_subtasks", []),
        }
        for method_spec in planner._htn_methods["build_trip_plan"]
    ]
    logger.event(
        "planning.method_candidates",
        "Planner loaded competing HTN methods for build_trip_plan",
        {"candidate_methods": candidate_methods},
    )
    logger.event(
        "planning.preference_input",
        "Planner will score applicable methods with extracted preference tokens",
        {
            "selection_preferences": [
                "relaxed",
                "style:relaxed",
                "boutique",
                "food_markets",
            ]
        },
    )
    bootstrapper.create_goal(
        {
            "task": bootstrapper.intent_bindings[intent_result.planner_binding]["task"],
            "args": [
                str(intent_result.normalized_parameters[param_name])
                for param_name in bootstrapper.intent_bindings[
                    intent_result.planner_binding
                ]["args"]
            ],
            "selection_preferences": intent_result.preference_tokens,
        }
    )
    plan = planner.execute_plan()
    logger.record_plan(plan)
    if planner.last_method_selection is not None:
        selected_spec = next(
            method_spec
            for method_spec in planner._htn_methods["build_trip_plan"]
            if method_spec["name"] == planner.last_method_selection.selected_method
        )
        logger.event(
            "planning.method_selection",
            "HTN control layer selected one applicable method",
            planner.last_method_selection.__dict__,
        )
        if selected_spec.get("subtasks"):
            logger.event(
                "planning.classical_subtask_ordering",
                (
                    "Selected method declared unordered subtasks; Pyperplan "
                    "ordered primitive actions from preconditions/effects"
                ),
                {
                    "selected_method": selected_spec["name"],
                    "declared_subtask_order": [
                        subtask[0]
                        for subtask in selected_spec["subtasks"]
                    ],
                    "planned_action_order": [
                        action_instance.action.name
                        for action_instance in plan.actions
                    ] if plan is not None else [],
                },
            )
        logger.note(
            "Selected method: "
            + json.dumps(planner.last_method_selection.__dict__, sort_keys=True)
        )
    if plan is None:
        logger.write_json(log_path)
        return

    with patch("builtins.input", return_value=approval):
        try:
            execute_plan_with_logging(
                plan=plan,
                registry=registry,
                logger=logger,
                bootstrapper=bootstrapper,
                working_memory=working_memory,
                interactive=True,
            )
        except PermissionError as error:
            logger.note(f"Execution stopped at HITL gate: {error}")

    logger.write_json(log_path)


def parse_args() -> argparse.Namespace:
    """Parse travel demo CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Kortex travel concierge demo scenario."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("demo_logs/travel_concierge_latest.json"),
        help="Path for the structured JSON log.",
    )
    parser.add_argument(
        "--approval",
        choices=["y", "n"],
        default="y",
        help="Scripted response for HITL approval prompts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the travel concierge demo from the CLI."""
    args = parse_args()
    run_travel_demo(log_path=args.log_path, approval=args.approval)


if __name__ == "__main__":
    main()
