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
        self._htn_methods: dict[str, dict[str, Any]] = {}
        self._htn_goals: list[tuple[str, list[str]]] = []
        self._initial_values: dict[str, bool] = {}

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
        target_task: str,
        parameter_names: list[str],
        ordered_subtasks: list[list[str]],
        preconditions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register deterministic YAML method metadata used by the local HTN expander."""
        self._htn_methods[target_task] = {
            "parameter_names": parameter_names,
            "ordered_subtasks": ordered_subtasks,
            "preconditions": preconditions or [],
        }

    def add_htn_goal(self, task_name: str, args: list[str]) -> None:
        """Queue an abstract HTN task for deterministic expansion."""
        self._htn_goals.append((task_name, args))

    def _expand_htn_goal(self, task_name: str, args: list[str]) -> list[ActionInstance]:
        """Expand a declared YAML HTN method into primitive UPF action instances."""
        if task_name not in self._htn_methods:
            raise KeyError(f"No HTN method registered for task '{task_name}'.")

        method_spec = self._htn_methods[task_name]
        parameter_names = method_spec["parameter_names"]
        if len(args) != len(parameter_names):
            raise ValueError(
                f"Task '{task_name}' expected {len(parameter_names)} arguments "
                f"({parameter_names}) but received {len(args)}."
            )

        bindings = dict(zip(parameter_names, args))
        self._validate_htn_preconditions(task_name, method_spec["preconditions"], bindings)
        expanded: list[ActionInstance] = []

        for subtask in method_spec["ordered_subtasks"]:
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
                object_name = bindings.get(symbolic_arg, symbolic_arg)
                actual_objects.append(self.problem.object(object_name))

            expanded.append(ActionInstance(action, tuple(actual_objects)))

        return expanded

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
            for task_name, task_args in self._htn_goals:
                actions.extend(self._expand_htn_goal(task_name, task_args))
            return SequentialPlan(actions)

        # Uses pyperplan solver via standard UPF planner interface
        with OnEnvSolver(name="pyperplan") as planner:
            result = planner.solve(self.classical_problem)
            if result.status.name in ("SOLVED_SATISFICING", "SOLVED_OPTIMALLY"):
                return result.plan
            return None
