import concurrent.futures
import threading
from typing import Dict, Any, List
from kortex.spine.planner import KortexPlanner
from kortex.spine.driver import ExecutionDriver
from kortex.memory.adapters import (
    planner_fact_record_from_dict,
    planner_fact_records_from_action_effects,
)
from kortex.memory.working import WorkingMemoryState

class Orchestrator:
    """
    Manages the multi-goal dispatching for Kortex Core.
    It takes a list of goals, checks if they are independent (no shared mutable fluents),
    and executes them concurrently if possible.
    """
    
    def __init__(self, bootstrapper, driver: ExecutionDriver | None = None):
        """Initialize the orchestrator with shared bootstrapper and driver."""
        self.bootstrapper = bootstrapper
        self.driver = driver or ExecutionDriver(registry=bootstrapper.registry)
        self.last_working_memory: WorkingMemoryState | None = None
        self._working_memory_lock = threading.Lock()

    def _are_goals_independent(self, goal_list: List[Dict[str, Any]]) -> bool:
        """
        Heuristic check: If goals target different fluent args (e.g. different locations or objects),
        we consider them independent enough for async execution in the MVP.
        In a full implementation, you'd analyze the PDDL interference graph.
        """
        # Collect all targeted args for each goal
        targeted_objects = set()
        for goal in goal_list:
            args = tuple(goal.get('args', []))
            if args in targeted_objects:
                return False # Collision detected (two goals targeting the same object)
            targeted_objects.add(args)
            
        return True

    def _execute_single_goal(
        self,
        goal: Dict[str, Any],
        initial_state: dict,
        working_memory: WorkingMemoryState | None = None,
    ) -> List[Any]:
        """Creates a localized planner instance for a single goal and runs it."""
        print(f"[Orchestrator] Spawning localized planner for goal: {goal['fluent']}({goal.get('args')})")
        
        # We create a fresh planner for this specific isolated goal
        local_planner = KortexPlanner(name=f"kortex_async_{goal['fluent']}")
        
        # Re-inject the parsed domain from the bootstrapper into this local planner
        # (Types, Fluents, Actions)
        for fluent in self.bootstrapper.fluents.values():
            local_planner.register_fluent(fluent)
        for action in self.bootstrapper.planner.problem.actions:
            local_planner.register_action(action)
        for obj in self.bootstrapper.objects.values():
            local_planner.register_object(obj)
            
        # Set State
        for fluent, value in initial_state.items():
             local_planner.set_initial_value(fluent, value)
             
        # Add Goal
        fl = self.bootstrapper.fluents[goal['fluent']]
        args = [self.bootstrapper.objects[arg] for arg in goal.get('args', [])]
        if goal.get('value', True):
            local_planner.add_goal(fl(*args))
        else:
            local_planner.add_goal(~fl(*args))
            
        # Plan and Execute
        plan = local_planner.execute_plan()
        if not plan:
            raise RuntimeError(f"Impasse reached for goal {goal}. No plan found.")

        execution = self.driver.execute_plan(plan)
        if working_memory is not None:
            self._apply_plan_effects_to_working_memory(plan, working_memory)
        return execution

    def dispatch(
        self,
        goals: List[Dict[str, Any]],
        master_initial_state: dict,
        working_memory: WorkingMemoryState | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a list of goals and dispatches them either sequentially or concurrently.
        """
        results = {}
        active_memory = working_memory or WorkingMemoryState(session_id="orchestrator")
        self.last_working_memory = active_memory
        active_memory.goal_stack.extend(goals)
        active_memory.planner_tier = "classical"
        self._seed_working_memory_from_master_state(active_memory, master_initial_state)
        
        if self._are_goals_independent(goals):
            print("[Orchestrator] Goals are independent. Executing concurrently.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all goals to the thread pool
                future_to_goal = {
                    executor.submit(
                        self._execute_single_goal,
                        g,
                        master_initial_state,
                        active_memory,
                    ): g['fluent']
                    for g in goals
                }
                
                for future in concurrent.futures.as_completed(future_to_goal):
                    g_name = future_to_goal[future]
                    try:
                        res = future.result()
                        results[g_name] = {"status": "success", "execution": res}
                    except Exception as exc:
                        results[g_name] = {"status": "failed", "error": str(exc)}
        else:
            print("[Orchestrator] Goals have dependencies. Executing sequentially.")
            # For the MVP, sequential execution just iterates. 
            # In production, the planner handles multi-goal interference internally.
            for g in goals:
                try:
                    res = self._execute_single_goal(g, master_initial_state, active_memory)
                    results[g['fluent']] = {"status": "success", "execution": res}
                    # Update master state here based on effects if necessary
                except Exception as exc:
                    results[g['fluent']] = {"status": "failed", "error": str(exc)}
                    
        return results

    def _seed_working_memory_from_master_state(
        self,
        working_memory: WorkingMemoryState,
        master_initial_state: dict,
    ) -> None:
        """Best-effort projection of master UPF state into working memory."""
        for expression, value in master_initial_state.items():
            fact = self._fact_dict_from_expression(expression, bool(value))
            if fact is None:
                continue
            record = planner_fact_record_from_dict(
                fact,
                source_system="orchestrator_initial_state",
                source_reference=working_memory.session_id,
            )
            with self._working_memory_lock:
                working_memory.hydrate_planner_fact(record)

    def _fact_dict_from_expression(
        self,
        expression: Any,
        value: bool,
    ) -> dict[str, Any] | None:
        """Convert a simple UPF fluent expression into a fact dict."""
        fluent_name = None
        arg_names: list[str] = []
        if hasattr(expression, "fluent") and expression.fluent() is not None:
            fluent_name = expression.fluent().name
            arg_names = [arg.object().name for arg in expression.args]
        else:
            rendered = str(expression)
            if "(" not in rendered or not rendered.endswith(")"):
                return None
            fluent_name, raw_args = rendered[:-1].split("(", maxsplit=1)
            arg_names = [arg.strip() for arg in raw_args.split(",") if arg.strip()]

        return {"fluent": fluent_name, "args": arg_names, "value": value}

    def _apply_plan_effects_to_working_memory(
        self,
        plan,
        working_memory: WorkingMemoryState,
    ) -> None:
        """Apply declared action effects to the orchestrator working memory."""
        for action_instance in plan.actions:
            for record in planner_fact_records_from_action_effects(
                action_instance,
                self.bootstrapper.action_specs,
                source_system="orchestrator_execution_effects",
                source_reference=working_memory.session_id,
            ):
                with self._working_memory_lock:
                    working_memory.hydrate_planner_fact(record)
