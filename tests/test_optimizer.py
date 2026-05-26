"""Tests for the generic Kortex optimizer."""

from __future__ import annotations

from kortex.memory.records import (
    MemoryLifecycleState,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
    OptimizationDecisionPayload,
)
from kortex.optimization import (
    KortexOptimizer,
    OptimizationCandidate,
    OptimizationCandidateSet,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationPolicy,
)


def test_optimizer_filters_constraints_and_scores_candidates() -> None:
    """The optimizer should reject infeasible candidates and score feasible ones."""
    optimizer = KortexOptimizer()
    policy = OptimizationPolicy(
        decision_id="choose_flight",
        constraints=[
            OptimizationConstraint(field="price", operator="<=", value=900),
            OptimizationConstraint(field="tags", operator="contains", value="refundable"),
        ],
        objectives=[
            OptimizationObjective(field="price", direction="minimize", weight=0.6),
            OptimizationObjective(field="comfort", direction="maximize", weight=0.4),
        ],
    )
    candidates = [
        OptimizationCandidate(
            candidate_id="f1",
            candidate_type="flight",
            attributes={"price": 780, "comfort": 0.7, "tags": ["refundable"]},
        ),
        OptimizationCandidate(
            candidate_id="f2",
            candidate_type="flight",
            attributes={"price": 1100, "comfort": 1.0, "tags": ["refundable"]},
        ),
        OptimizationCandidate(
            candidate_id="f3",
            candidate_type="flight",
            attributes={"price": 700, "comfort": 0.2, "tags": ["overnight"]},
        ),
    ]

    result = optimizer.optimize(
        policy,
        [OptimizationCandidateSet(set_id="flights", candidates=candidates)],
    )

    assert result.selected_candidate_ids == ["f1"]
    assert result.feasible_candidate_count == 1
    assert {tuple(rejection.candidate_ids) for rejection in result.rejected} == {
        ("f2",),
        ("f3",),
    }


def test_optimizer_selects_best_bundle_under_budget() -> None:
    """The optimizer should enumerate candidate bundles across required sets."""
    optimizer = KortexOptimizer()
    policy = OptimizationPolicy(
        decision_id="choose_travel_bundle",
        aggregate_fields={
            "hotel_total": "product:hotel.nightly_rate*duration_days",
            "total_cost": "sum:flight.price+hotel_total",
            "bundle_preference_fit": "sum:flight.preference_fit+hotel.preference_fit",
        },
        constraints=[
            OptimizationConstraint(field="total_cost", operator="<=", value=2500),
            OptimizationConstraint(field="flight.tags", operator="contains", value="refundable"),
        ],
        objectives=[
            OptimizationObjective(field="total_cost", direction="minimize", weight=0.2),
            OptimizationObjective(field="bundle_preference_fit", direction="maximize", weight=0.4),
            OptimizationObjective(field="hotel.tags", direction="match", target="boutique", weight=0.5),
        ],
    )
    flights = OptimizationCandidateSet(
        set_id="flights",
        candidates=[
            OptimizationCandidate(
                candidate_id="f1",
                candidate_type="flight",
                attributes={"price": 780, "tags": ["refundable"], "preference_fit": 0.8},
            ),
            OptimizationCandidate(
                candidate_id="f2",
                candidate_type="flight",
                attributes={"price": 620, "tags": ["overnight"], "preference_fit": 0.2},
            ),
        ],
    )
    hotels = OptimizationCandidateSet(
        set_id="hotels",
        candidates=[
            OptimizationCandidate(
                candidate_id="h1",
                candidate_type="hotel",
                attributes={
                    "nightly_rate": 260,
                    "tags": ["boutique", "quiet"],
                    "preference_fit": 0.9,
                },
            ),
            OptimizationCandidate(
                candidate_id="h2",
                candidate_type="hotel",
                attributes={
                    "nightly_rate": 180,
                    "tags": ["chain"],
                    "preference_fit": 0.3,
                },
            ),
        ],
    )

    result = optimizer.optimize(
        policy,
        [flights, hotels],
        context={"duration_days": 3},
    )

    assert result.selected_candidate_ids == ["f1", "h1"]
    assert result.selected_attributes["total_cost"] == 1560.0
    assert result.selected_attributes["hotel_total"] == 780.0
    assert result.feasible_candidate_count == 2
    assert any(
        rejection.candidate_ids == ["f2", "h1"]
        for rejection in result.rejected
    )


def test_optimizer_result_can_be_stored_as_memory_record() -> None:
    """Optimizer decisions should use the standard memory record envelope."""
    result = KortexOptimizer().optimize(
        OptimizationPolicy(decision_id="choose_option"),
        [
            OptimizationCandidateSet(
                set_id="options",
                candidates=[
                    OptimizationCandidate(
                        candidate_id="o1",
                        candidate_type="option",
                        attributes={"score": 1.0},
                    )
                ],
            )
        ],
    )

    record = MemoryRecord(
        memory_type=MemoryType.OPTIMIZATION_DECISION,
        scope=MemoryScope.SESSION,
        source=MemorySource(system="optimizer", reference="run-1"),
        lifecycle_state=MemoryLifecycleState.VALIDATED,
        payload=OptimizationDecisionPayload(**result.model_dump()),
    )

    assert record.payload.decision_id == "choose_option"
    assert record.payload.selected_candidate_ids == ["o1"]
