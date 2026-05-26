"""Deterministic candidate optimization for Kortex decisions.

The optimizer is a generic decision layer between symbolic planning and
physical execution. It selects one candidate or candidate bundle from typed
options using hard constraints and weighted scoring, while returning an
auditable score breakdown.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

from pydantic import BaseModel, Field


class OptimizationConstraint(BaseModel):
    """A hard constraint over one candidate or candidate bundle aggregate."""

    field: str
    operator: Literal["<=", ">=", "==", "!=", "<", ">", "contains", "not_contains"]
    value: Any


class OptimizationObjective(BaseModel):
    """One weighted soft objective used to score feasible candidates."""

    field: str
    direction: Literal["minimize", "maximize", "match"]
    weight: float = 1.0
    target: Any | None = None


class OptimizationCandidate(BaseModel):
    """A typed option considered by the optimizer."""

    candidate_id: str
    candidate_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class OptimizationCandidateSet(BaseModel):
    """A named candidate collection, usually produced by one search action."""

    set_id: str
    candidates: list[OptimizationCandidate]
    required: bool = True


class OptimizationPolicy(BaseModel):
    """Declarative policy for one deterministic optimization decision."""

    decision_id: str
    constraints: list[OptimizationConstraint] = Field(default_factory=list)
    objectives: list[OptimizationObjective] = Field(default_factory=list)
    aggregate_fields: dict[str, str] = Field(default_factory=dict)


class OptimizationRejection(BaseModel):
    """Reason a candidate or bundle was rejected."""

    candidate_ids: list[str]
    reason: str


class OptimizationScoreBreakdown(BaseModel):
    """Per-objective score contribution for a feasible candidate or bundle."""

    objective: str
    raw_value: Any
    normalized_score: float
    weighted_score: float


class OptimizationResult(BaseModel):
    """Auditable output from deterministic candidate optimization."""

    decision_id: str
    selected_candidate_ids: list[str]
    selected_attributes: dict[str, Any] = Field(default_factory=dict)
    score: float
    score_breakdown: list[OptimizationScoreBreakdown] = Field(default_factory=list)
    rejected: list[OptimizationRejection] = Field(default_factory=list)
    feasible_candidate_count: int = 0
    policy_version: str = "weighted_scorer_v1"


class OptimizationExecutionOutput(BaseModel):
    """Structured plugin output for optimizer-backed primitive actions."""

    message: str
    result: OptimizationResult
    response_type: str = "optimizer_summary"
    response_facts: dict[str, Any] = Field(default_factory=dict)
    subject_ids: list[str] = Field(default_factory=list)


class KortexOptimizer:
    """Deterministic optimizer for candidate choices and bundles."""

    def optimize(
        self,
        policy: OptimizationPolicy,
        candidate_sets: list[OptimizationCandidateSet],
        context: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Select the highest-scoring feasible candidate or bundle."""
        context_values = context or {}
        bundles = self._candidate_bundles(candidate_sets)
        feasible: list[tuple[list[OptimizationCandidate], dict[str, Any]]] = []
        rejected: list[OptimizationRejection] = []

        for bundle in bundles:
            attributes = self._aggregate_bundle(
                bundle,
                policy.aggregate_fields,
                context_values,
            )
            failed_constraint = self._first_failed_constraint(policy.constraints, attributes)
            candidate_ids = [candidate.candidate_id for candidate in bundle]
            if failed_constraint is not None:
                rejected.append(
                    OptimizationRejection(
                        candidate_ids=candidate_ids,
                        reason=failed_constraint,
                    )
                )
                continue
            feasible.append((bundle, attributes))

        if not feasible:
            return OptimizationResult(
                decision_id=policy.decision_id,
                selected_candidate_ids=[],
                score=0.0,
                rejected=rejected,
            )

        scored = [
            self._score_bundle(policy, bundle, attributes, feasible)
            for bundle, attributes in feasible
        ]
        selected_bundle, selected_attributes, score, breakdown = max(
            scored,
            key=lambda item: item[2],
        )
        return OptimizationResult(
            decision_id=policy.decision_id,
            selected_candidate_ids=[
                candidate.candidate_id
                for candidate in selected_bundle
            ],
            selected_attributes=selected_attributes,
            score=score,
            score_breakdown=breakdown,
            rejected=rejected,
            feasible_candidate_count=len(feasible),
        )

    def _candidate_bundles(
        self,
        candidate_sets: list[OptimizationCandidateSet],
    ) -> list[list[OptimizationCandidate]]:
        """Return candidate combinations across all required sets."""
        required_sets = [
            candidate_set
            for candidate_set in candidate_sets
            if candidate_set.required
        ]
        if not required_sets:
            return [[]]
        return [
            list(bundle)
            for bundle in product(*[candidate_set.candidates for candidate_set in required_sets])
        ]

    def _aggregate_bundle(
        self,
        bundle: list[OptimizationCandidate],
        aggregate_fields: dict[str, str],
        context_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Aggregate candidate attributes into one bundle attribute map."""
        attributes: dict[str, Any] = {
            "candidate_ids": [candidate.candidate_id for candidate in bundle],
            "candidate_types": [candidate.candidate_type for candidate in bundle],
        }
        attributes.update(context_values or {})
        for candidate in bundle:
            for key, value in candidate.attributes.items():
                typed_key = f"{candidate.candidate_type}.{key}"
                attributes[typed_key] = value
                attributes.setdefault(key, value)

        for output_field, expression in aggregate_fields.items():
            if expression.startswith("sum:"):
                source_fields = [
                    field.strip()
                    for field in expression.removeprefix("sum:").split("+")
                    if field.strip()
                ]
                attributes[output_field] = sum(
                    self._numeric(attributes.get(field, 0))
                    for field in source_fields
                )
            elif expression.startswith("product:"):
                factors = [
                    factor.strip()
                    for factor in expression.removeprefix("product:").split("*")
                    if factor.strip()
                ]
                product_value = 1.0
                for factor in factors:
                    product_value *= self._numeric(attributes.get(factor, factor))
                attributes[output_field] = product_value
        return attributes

    def _first_failed_constraint(
        self,
        constraints: list[OptimizationConstraint],
        attributes: dict[str, Any],
    ) -> str | None:
        """Return a rejection reason for the first failed hard constraint."""
        for constraint in constraints:
            actual = attributes.get(constraint.field)
            if not self._constraint_holds(actual, constraint.operator, constraint.value):
                return (
                    f"{constraint.field} {constraint.operator} "
                    f"{constraint.value} failed with {actual}"
                )
        return None

    def _constraint_holds(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate one hard constraint."""
        if operator == "<=":
            return self._numeric(actual) <= self._numeric(expected)
        if operator == ">=":
            return self._numeric(actual) >= self._numeric(expected)
        if operator == "<":
            return self._numeric(actual) < self._numeric(expected)
        if operator == ">":
            return self._numeric(actual) > self._numeric(expected)
        if operator == "==":
            return actual == expected
        if operator == "!=":
            return actual != expected
        if operator == "contains":
            return expected in self._as_collection(actual)
        if operator == "not_contains":
            return expected not in self._as_collection(actual)
        raise ValueError(f"Unsupported constraint operator: {operator}")

    def _score_bundle(
        self,
        policy: OptimizationPolicy,
        bundle: list[OptimizationCandidate],
        attributes: dict[str, Any],
        feasible: list[tuple[list[OptimizationCandidate], dict[str, Any]]],
    ) -> tuple[
        list[OptimizationCandidate],
        dict[str, Any],
        float,
        list[OptimizationScoreBreakdown],
    ]:
        """Score one feasible bundle against soft objectives."""
        breakdown: list[OptimizationScoreBreakdown] = []
        total = 0.0
        for objective in policy.objectives:
            raw_value = attributes.get(objective.field)
            normalized = self._normalized_objective_score(objective, raw_value, feasible)
            weighted = normalized * objective.weight
            breakdown.append(
                OptimizationScoreBreakdown(
                    objective=objective.field,
                    raw_value=raw_value,
                    normalized_score=normalized,
                    weighted_score=weighted,
                )
            )
            total += weighted
        return bundle, attributes, total, breakdown

    def _normalized_objective_score(
        self,
        objective: OptimizationObjective,
        raw_value: Any,
        feasible: list[tuple[list[OptimizationCandidate], dict[str, Any]]],
    ) -> float:
        """Return a 0..1 objective score for one raw value."""
        if objective.direction == "match":
            if objective.target in self._as_collection(raw_value):
                return 1.0
            return 1.0 if raw_value == objective.target else 0.0

        numeric_values = [
            self._numeric(attributes.get(objective.field))
            for _bundle, attributes in feasible
        ]
        value = self._numeric(raw_value)
        minimum = min(numeric_values)
        maximum = max(numeric_values)
        if maximum == minimum:
            return 1.0
        if objective.direction == "minimize":
            return (maximum - value) / (maximum - minimum)
        return (value - minimum) / (maximum - minimum)

    def _numeric(self, value: Any) -> float:
        """Coerce common numeric values to float for scoring."""
        if isinstance(value, int | float):
            return float(value)
        return float(str(value))

    def _as_collection(self, value: Any) -> list[Any]:
        """Treat scalars and missing values uniformly for membership checks."""
        if value is None:
            return []
        if isinstance(value, list | tuple | set):
            return list(value)
        return [value]
