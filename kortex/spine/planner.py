from typing import Any
from unified_planning.shortcuts import (
    InstantaneousAction,
    OneshotPlanner as OnEnvSolver,
    Fluent
)
from unified_planning.model.htn import HierarchicalProblem, Task, Method
from unified_planning.plans import Plan


class KortexPlanner:
    """
    KortexPlanner handles the deterministic execution layer via Hierarchical Task Networks (HTN).
    This module adheres to the requirement of keeping LLM extraction fully decoupled from the
    deterministic HTN execution layer, relying on standard PDDL and HTN definitions.
    """

    def __init__(self, name: str = "kortex_spine_problem") -> None:
        """Initialize a Unified Planning HierarchicalProblem (HTN + PDDL)."""
        self.problem = HierarchicalProblem(name)

    def register_fluent(self, fluent: Fluent) -> None:
        """Register a state fluent."""
        self.problem.add_fluent(fluent)

    def register_action(self, action: InstantaneousAction) -> None:
        """Register a PDDL primitive action."""
        self.problem.add_action(action)

    def register_task(self, task: Task) -> None:
        """Register an HTN abstract task."""
        self.problem.add_task(task)

    def register_method(self, method: Method) -> None:
        """Register an HTN method."""
        self.problem.add_method(method)

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
                self.problem.set_initial_value(fluent, value)

        if goal_task:
            task = self.problem.get_task(goal_task)
            self.problem.task_network.add_subtask(task)

        # Uses pyperplan solver via standard UPF planner interface
        with OnEnvSolver(name="pyperplan") as planner:
            result = planner.solve(self.problem)
            if result.status.name in ("SOLVED_SATISFICING", "SOLVED_OPTIMALLY"):
                return result.plan
            return None
