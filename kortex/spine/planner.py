from dataclasses import dataclass, field
from typing import Any
from unified_planning.shortcuts import (
    InstantaneousAction,
    OneshotPlanner as OnEnvSolver,
    Fluent,
    Problem,
)
from unified_planning.model.htn import HierarchicalProblem, Task, Method
from unified_planning.model import Object
from unified_planning.model.fnode import FNode
from unified_planning.plans import ActionInstance, Plan, SequentialPlan


class HTNMethodTieImpasse(ValueError):
    """Raised when multiple applicable HTN methods remain equally preferred."""

    def __init__(self, task_name: str, candidate_methods: list[str]) -> None:
        """Initialize the tie impasse with the unresolved task and candidates."""
        self.task_name = task_name
        self.candidate_methods = candidate_methods
        super().__init__(
            f"HTN task '{task_name}' has tied applicable methods: {candidate_methods}."
        )


@dataclass(frozen=True)
class HTNMethodSelection:
    """Auditable record of deterministic HTN method selection."""

    task_name: str
    selected_method: str
    applicable_methods: list[str]
    preference_matches: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class KortexPlanner:
    """
    KortexPlanner handles the deterministic execution layer via Hierarchical Task Networks (HTN).
    This module adheres to the requirement of keeping LLM extraction fully decoupled from the
    deterministic HTN execution layer, relying on standard PDDL and HTN definitions.
    """

    def __init__(self, name: str = "kortex_spine_problem") -> None:
        """Initialize a Unified Planning HierarchicalProblem (HTN + PDDL)."""
        self.problem = HierarchicalProblem(name)
        self.classical_problem = Problem(f"{name}_classical")
        self._htn_methods: dict[str, list[dict[str, Any]]] = {}
        self._htn_goals: list[tuple[str, list[str], list[str]]] = []
        self._initial_values: dict[str, bool] = {}
        self._initial_expressions: dict[str, FNode] = {}
        self.last_method_selection: HTNMethodSelection | None = None

    def register_fluent(self, fluent: Fluent) -> None:
        """Register a state fluent."""
        self.problem.add_fluent(fluent, default_initial_value=False)
        self.classical_problem.add_fluent(fluent, default_initial_value=False)

    def register_action(self, action: InstantaneousAction) -> None:
        """Register a PDDL primitive action."""
        self.problem.add_action(action)
        self.classical_problem.add_action(action)

    def register_object(self, obj: Object) -> None:
        """Register a concrete planning object in both planning views."""
        self.problem.add_object(obj)
        self.classical_problem.add_object(obj)

    def set_initial_value(self, fluent_expression: FNode, value: bool) -> None:
        """Set a world-state fact in both planning views."""
        self.problem.set_initial_value(fluent_expression, value)
        self.classical_problem.set_initial_value(fluent_expression, value)
        self._initial_values[str(fluent_expression)] = value
        self._initial_expressions[str(fluent_expression)] = fluent_expression

    def add_goal(self, goal_expression: FNode) -> None:
        """Register a classical state goal for Tier 2 planning."""
        self.problem.add_goal(goal_expression)
        self.classical_problem.add_goal(goal_expression)

    def register_task(self, task: Task) -> None:
        """Register an HTN abstract task."""
        self.problem.add_task(task)

    def register_method(self, method: Method) -> None:
        """Register an HTN method."""
        self.problem.add_method(method)

    def register_method_spec(
        self,
        method_name: str,
        target_task: str,
        parameter_names: list[str],
        ordered_subtasks: list[list[str]],
        subtasks: list[list[str]] | None = None,
        subtask_effects: dict[str, list[dict[str, Any]]] | None = None,
        preconditions: list[dict[str, Any]] | None = None,
        preference_matches: list[str] | None = None,
        selection_priority: int | float = 0,
    ) -> None:
        """Register deterministic YAML method metadata used by the local HTN expander."""
        method_specs = self._htn_methods.setdefault(target_task, [])
        method_specs.append({
            "name": method_name,
            "parameter_names": parameter_names,
            "ordered_subtasks": ordered_subtasks,
            "subtasks": subtasks or [],
            "subtask_effects": subtask_effects or {},
            "preconditions": preconditions or [],
            "preference_matches": [
                self._normalize_preference_token(token)
                for token in preference_matches or []
            ],
            "selection_priority": float(selection_priority),
        })

    def add_htn_goal(
        self,
        task_name: str,
        args: list[str],
        selection_preferences: list[str] | None = None,
    ) -> None:
        """Queue an abstract HTN task for deterministic expansion."""
        self._htn_goals.append((
            task_name,
            args,
            [
                self._normalize_preference_token(token)
                for token in selection_preferences or []
            ],
        ))

    def _expand_htn_goal(
        self,
        task_name: str,
        args: list[str],
        selection_preferences: list[str],
    ) -> list[ActionInstance]:
        """Expand a declared YAML HTN method into primitive UPF action instances."""
        if task_name not in self._htn_methods:
            raise KeyError(f"No HTN method registered for task '{task_name}'.")

        method_spec = self._select_htn_method(task_name, args, selection_preferences)
        parameter_names = method_spec["parameter_names"]
        if len(args) != len(parameter_names):
            raise ValueError(
                f"Task '{task_name}' expected {len(parameter_names)} arguments "
                f"({parameter_names}) but received {len(args)}."
            )

        bindings = dict(zip(parameter_names, args))
        self._validate_htn_preconditions(task_name, method_spec["preconditions"], bindings)
        expanded: list[ActionInstance] = []
        if method_spec["subtasks"]:
            return self._plan_unordered_subtasks(method_spec, bindings)

        for subtask in method_spec["ordered_subtasks"]:
            expanded.append(self._bound_action_instance(subtask, bindings))

        return expanded

    def _plan_unordered_subtasks(
        self,
        method_spec: dict[str, Any],
        bindings: dict[str, str],
    ) -> list[ActionInstance]:
        """Use classical planning to order one method's unordered primitive subtasks."""
        subtask_specs = method_spec["subtasks"]
        allowed_action_names = {str(subtask[0]) for subtask in subtask_specs}
        local_problem = Problem(f"{self.classical_problem.name}_{method_spec['name']}_subtasks")

        for fluent in self.classical_problem.fluents:
            local_problem.add_fluent(fluent, default_initial_value=False)
        bound_object_names = set(bindings.values())
        for obj in self.classical_problem.all_objects:
            if obj.name in bound_object_names:
                local_problem.add_object(obj)
        for action_name in sorted(allowed_action_names):
            local_problem.add_action(self.classical_problem.action(action_name))
        for expression in self._initial_expressions.values():
            local_problem.set_initial_value(expression, self._initial_values[str(expression)])

        for subtask in subtask_specs:
            for goal_expression in self._subtask_goal_expressions(
                subtask,
                bindings,
                method_spec["subtask_effects"],
            ):
                local_problem.add_goal(goal_expression)

        with OnEnvSolver(name="pyperplan") as planner:
            result = planner.solve(local_problem)
            if result.status.name not in ("SOLVED_SATISFICING", "SOLVED_OPTIMALLY"):
                raise ValueError(
                    f"Unordered subtasks for method '{method_spec['name']}' "
                    "could not be ordered into a valid primitive plan."
                )
            return list(result.plan.actions)

    def _subtask_goal_expressions(
        self,
        subtask: list[str],
        bindings: dict[str, str],
        subtask_effects: dict[str, list[dict[str, Any]]],
    ) -> list[FNode]:
        """Convert one unordered primitive subtask's positive effects into goals."""
        action_name = str(subtask[0])
        action = self.problem.action(action_name)
        symbolic_args = subtask[1:] or [parameter.name for parameter in action.parameters]
        action_bindings = {
            parameter.name: bindings.get(str(symbolic_arg), str(symbolic_arg))
            for parameter, symbolic_arg in zip(action.parameters, symbolic_args, strict=True)
        }
        goal_expressions: list[FNode] = []
        for effect in subtask_effects.get(action_name, []):
            if not effect.get("value", True):
                continue
            fluent = self.problem.fluent(effect["fluent"])
            objects = [
                self.problem.object(action_bindings[str(arg)])
                for arg in effect.get("args", [])
            ]
            goal_expressions.append(fluent(*objects))
        return goal_expressions

    def _bound_action_instance(
        self,
        subtask: list[str],
        bindings: dict[str, str],
    ) -> ActionInstance:
        """Create a concrete action instance from symbolic method arguments."""
        action_name = subtask[0]
        action = self.problem.action(action_name)
        symbolic_args = subtask[1:] or [parameter.name for parameter in action.parameters]
        if len(symbolic_args) != len(action.parameters):
            raise ValueError(
                f"Subtask '{action_name}' expected {len(action.parameters)} arguments "
                f"but method provided {len(symbolic_args)}."
            )

        actual_objects = []
        for symbolic_arg in symbolic_args:
            object_name = bindings.get(str(symbolic_arg), str(symbolic_arg))
            actual_objects.append(self.problem.object(object_name))

        return ActionInstance(action, tuple(actual_objects))

    def _select_htn_method(
        self,
        task_name: str,
        args: list[str],
        selection_preferences: list[str],
    ) -> dict[str, Any]:
        """Select one applicable HTN method using deterministic preference scores."""
        candidate_specs = self._htn_methods[task_name]
        applicable_specs: list[tuple[dict[str, Any], dict[str, str]]] = []

        for method_spec in candidate_specs:
            parameter_names = method_spec["parameter_names"]
            if len(args) != len(parameter_names):
                raise ValueError(
                    f"Task '{task_name}' expected {len(parameter_names)} arguments "
                    f"({parameter_names}) but received {len(args)}."
                )
            bindings = dict(zip(parameter_names, args))
            if self._htn_preconditions_hold(method_spec["preconditions"], bindings):
                applicable_specs.append((method_spec, bindings))

        if not applicable_specs:
            if len(candidate_specs) == 1:
                method_spec = candidate_specs[0]
                bindings = dict(zip(method_spec["parameter_names"], args))
                self._validate_htn_preconditions(task_name, method_spec["preconditions"], bindings)
            raise ValueError(f"No applicable HTN method found for task '{task_name}'.")

        scored: list[tuple[dict[str, Any], float, list[str]]] = []
        preference_set = set(selection_preferences)
        for method_spec, _bindings in applicable_specs:
            method_preferences = set(method_spec.get("preference_matches", []))
            matches = sorted(preference_set & method_preferences)
            score = float(method_spec.get("selection_priority", 0)) + float(len(matches))
            scored.append((method_spec, score, matches))

        top_score = max(score for _method_spec, score, _matches in scored)
        top_methods = [
            method_spec
            for method_spec, score, _matches in scored
            if score == top_score
        ]
        scores = {
            str(method_spec["name"]): score
            for method_spec, score, _matches in scored
        }

        if len(top_methods) > 1:
            raise HTNMethodTieImpasse(
                task_name=task_name,
                candidate_methods=[str(method_spec["name"]) for method_spec in top_methods],
            )

        selected = top_methods[0]
        selected_matches = next(
            matches
            for method_spec, _score, matches in scored
            if method_spec is selected
        )
        self.last_method_selection = HTNMethodSelection(
            task_name=task_name,
            selected_method=str(selected["name"]),
            applicable_methods=[
                str(method_spec["name"])
                for method_spec, _bindings in applicable_specs
            ],
            preference_matches=selected_matches,
            scores=scores,
        )
        return selected

    def _htn_preconditions_hold(
        self,
        preconditions: list[dict[str, Any]],
        bindings: dict[str, str],
    ) -> bool:
        """Return whether symbolic method preconditions hold in current state."""
        for precondition in preconditions:
            fluent_name = precondition["fluent"]
            bound_args = [
                bindings.get(str(arg), str(arg))
                for arg in precondition.get("args", [])
            ]
            fluent = self.problem.fluent(fluent_name)
            objects = [self.problem.object(arg) for arg in bound_args]
            expression = fluent(*objects)
            expected = precondition.get("value", True)
            actual = self._initial_values.get(str(expression), False)
            if actual != expected:
                return False
        return True

    def _validate_htn_preconditions(
        self,
        task_name: str,
        preconditions: list[dict[str, Any]],
        bindings: dict[str, str],
    ) -> None:
        """Ensure a learned HTN method's symbolic preconditions hold before expansion."""
        for precondition in preconditions:
            fluent_name = precondition["fluent"]
            bound_args = [
                bindings.get(str(arg), str(arg))
                for arg in precondition.get("args", [])
            ]
            fluent = self.problem.fluent(fluent_name)
            objects = [self.problem.object(arg) for arg in bound_args]
            expression = fluent(*objects)
            expected = precondition.get("value", True)
            actual = self._initial_values.get(str(expression), False)
            if actual != expected:
                raise ValueError(
                    f"HTN method for task '{task_name}' requires "
                    f"{expression}={expected}, but current state has {actual}."
                )

    def execute_plan(self, initial_state: dict[Any, Any] = None, goal_task: str = None) -> Plan | None:
        """
        Execute the plan using pyperplan via UPF.
        
        Args:
            initial_state: Dictionary containing initial state mappings (optional if set by bootstrapper).
            goal_task: The name of the goal task to achieve (if running HTN).
            
        Returns:
            The resulting execution Plan if found, otherwise None.
        """
        if initial_state:
            for fluent, value in initial_state.items():
                self.set_initial_value(fluent, value)

        if goal_task:
            self.add_htn_goal(goal_task, [])

        if self._htn_goals:
            actions: list[ActionInstance] = []
            for task_name, task_args, selection_preferences in self._htn_goals:
                actions.extend(self._expand_htn_goal(
                    task_name,
                    task_args,
                    selection_preferences,
                ))
            return SequentialPlan(actions)

        # Uses pyperplan solver via standard UPF planner interface
        with OnEnvSolver(name="pyperplan") as planner:
            result = planner.solve(self.classical_problem)
            if result.status.name in ("SOLVED_SATISFICING", "SOLVED_OPTIMALLY"):
                return result.plan
            return None

    def _normalize_preference_token(self, token: Any) -> str:
        """Normalize a selection preference token for deterministic matching."""
        return str(token).strip().lower().replace(" ", "_").replace("-", "_")
