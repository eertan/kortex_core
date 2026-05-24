from typing import Any, Callable, List
from unified_planning.plans import Plan
from kortex.plugins.registry import PluginRegistry, registry as default_registry

TraceCallback = Callable[[str, str, dict[str, Any]], None]

class ExecutionDriver:
    """
    Executes a UPF generated Plan by mapping the symbolic actions to real Python plugins.
    Enforces Human-In-The-Loop (HITL) authorization for sensitive operations.
    """

    def __init__(
        self,
        interactive: bool = True,
        trace_callback: TraceCallback | None = None,
        registry: PluginRegistry | None = None,
    ):
        """Initialize the execution driver."""
        self.interactive = interactive
        self.trace_callback = trace_callback
        self.registry = registry or default_registry

    def _trace(self, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
        """Emit an execution trace event when a callback is configured."""
        if self.trace_callback is not None:
            self.trace_callback(stage, message, payload or {})

    def _request_human_approval(self, action_name: str, kwargs: dict) -> bool:
        """Prompts the user for authorization before executing a critical action."""
        self._trace(
            "hitl.approval.required",
            "Action requires human approval",
            {"action": action_name, "parameters": kwargs},
        )
        if not self.interactive:
            # If running in non-interactive (e.g., CI/CD), default to block for safety.
            print(f"[Driver-HITL] Action '{action_name}' requires approval, but running non-interactively. Blocked.")
            self._trace(
                "hitl.approval.denied",
                "Approval denied because driver is non-interactive",
                {"action": action_name, "parameters": kwargs},
            )
            return False
            
        print(f"\n[Driver-HITL] ⚠️  SECURITY AUTHORIZATION REQUIRED ⚠️")
        print(f"The planner intends to execute a sensitive operation:")
        print(f" -> Action: {action_name}")
        print(f" -> Parameters: {kwargs}")
        
        while True:
            response = input("Do you authorize this execution? [y/N]: ").strip().lower()
            if response in ('y', 'yes'):
                print("[Driver-HITL] Execution Authorized.")
                self._trace(
                    "hitl.approval.granted",
                    "Human approved action execution",
                    {"action": action_name, "parameters": kwargs},
                )
                return True
            elif response in ('n', 'no', ''):
                print("[Driver-HITL] Execution Denied.")
                self._trace(
                    "hitl.approval.denied",
                    "Human denied action execution",
                    {"action": action_name, "parameters": kwargs},
                )
                return False
            else:
                print("Please answer 'y' or 'n'.")

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
                
            print(f"[Driver] Preparing physical action: {name}(**{kwargs})")
            self._trace(
                "execution.action.prepare",
                "Preparing physical action",
                {"action": name, "parameters": kwargs},
            )
            
            try:
                plugin_meta = self.registry.get_plugin(name)
                
                # Check HITL Security Authorization
                if plugin_meta.get("requires_approval", False):
                    approved = self._request_human_approval(name, kwargs)
                    if not approved:
                        raise PermissionError(f"Human denied execution of '{name}'.")
                        
                res = self.registry.execute_plugin(name, **kwargs)
                results.append(res)
                print(f"[Driver] Action '{name}' succeeded -> {res}")
                self._trace(
                    "execution.action.success",
                    "Physical action completed",
                    {"action": name, "parameters": kwargs, "result": res},
                )
            except Exception as e:
                print(f"[Driver] Action '{name}' failed! Impasse detected: {e}")
                self._trace(
                    "execution.action.failure",
                    "Physical action failed",
                    {"action": name, "parameters": kwargs, "error": str(e)},
                )
                raise e
                
        return results
