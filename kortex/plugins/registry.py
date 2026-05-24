from typing import Callable, Dict, Any, List
import inspect

class PluginRegistry:
    """
    Registry for dynamic Python plugins that act as HTN/PDDL primitive actions.
    """
    
    def __init__(self):
        self.plugins: Dict[str, Dict[str, Any]] = {}
        
    def register_action(self, name: str, preconditions: Dict[str, Any] = None, effects: Dict[str, Any] = None):
        """
        Decorator to register a Python function as a UPF primitive action.
        
        Args:
            name: The name of the primitive action matching the HTN spec.
            preconditions: A dictionary of state variables and expected values.
            effects: A dictionary of state variables mutated by this action.
        """
        def decorator(func: Callable):
            sig = inspect.signature(func)
            
            self.plugins[name] = {
                "func": func,
                "name": name,
                "signature": sig,
                "preconditions": preconditions or {},
                "effects": effects or {}
            }
            return func
        return decorator

    def get_plugin(self, name: str) -> Dict[str, Any]:
        """Retrieve a registered plugin by name."""
        if name not in self.plugins:
            raise KeyError(f"Plugin action '{name}' not found in registry.")
        return self.plugins[name]

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self.plugins.keys())

    def execute_plugin(self, name: str, **kwargs) -> Any:
        """Execute a registered plugin with the given arguments."""
        plugin = self.get_plugin(name)
        # Type checking/coercion could be added here based on signature
        return plugin["func"](**kwargs)

# Global default registry instance
registry = PluginRegistry()
action = registry.register_action
