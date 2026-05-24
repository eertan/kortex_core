from typing import Any, List
from unified_planning.plans import Plan
from kortex.plugins.registry import registry

class ExecutionDriver:
    """
    Executes a UPF generated Plan by mapping the symbolic actions to real Python plugins.
    """

    def execute_plan(self, plan: Plan) -> List[Any]:
        """
        Iterate through the UPF symbolic plan and invoke the bound python functions.
        """
        results = []
        for action_instance in plan.actions:
            name = action_instance.action.name
            
            # UPF actual parameters are expressions (FNodes). We extract the string name of the Object.
            kwargs = {}
            for param, actual_val in zip(action_instance.action.parameters, action_instance.actual_parameters):
                # actual_val.object() returns the UPF Object representation
                kwargs[param.name] = actual_val.object().name
                
            print(f"[Driver] Executing physical action: {name}(**{kwargs})")
            
            try:
                res = registry.execute_plugin(name, **kwargs)
                results.append(res)
                print(f"[Driver] Action '{name}' succeeded -> {res}")
            except Exception as e:
                print(f"[Driver] Action '{name}' failed! Impasse detected: {e}")
                raise e
                
        return results
