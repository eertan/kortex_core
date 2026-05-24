import concurrent.futures
from typing import Dict, Any, List
from kortex.spine.planner import KortexPlanner
from kortex.spine.driver import ExecutionDriver

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

    def _execute_single_goal(self, goal: Dict[str, Any], initial_state: dict) -> List[Any]:
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
            
        return self.driver.execute_plan(plan)

    def dispatch(self, goals: List[Dict[str, Any]], master_initial_state: dict) -> Dict[str, Any]:
        """
        Evaluates a list of goals and dispatches them either sequentially or concurrently.
        """
        results = {}
        
        if self._are_goals_independent(goals):
            print("[Orchestrator] Goals are independent. Executing concurrently.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all goals to the thread pool
                future_to_goal = {
                    executor.submit(self._execute_single_goal, g, master_initial_state): g['fluent'] 
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
                    res = self._execute_single_goal(g, master_initial_state)
                    results[g['fluent']] = {"status": "success", "execution": res}
                    # Update master state here based on effects if necessary
                except Exception as exc:
                    results[g['fluent']] = {"status": "failed", "error": str(exc)}
                    
        return results
