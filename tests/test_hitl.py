import pytest
from unittest.mock import patch, MagicMock
from unified_planning.plans import Plan, SequentialPlan
from unified_planning.shortcuts import InstantaneousAction
from kortex.spine.driver import ExecutionDriver
from kortex.plugins.registry import PluginRegistry, registry

hitl_registry = PluginRegistry()

# Register a sensitive plugin for testing
@hitl_registry.register_action("drop_table", requires_approval=True)
def mock_drop_table(table_name: str) -> str:
    return f"Dropped {table_name}"

def test_execution_driver_hitl_approval_granted():
    # 1. Create a fake UPF plan with our action
    action = InstantaneousAction("drop_table")
    
    # We need a dummy ActionInstance. For testing, we mock it.
    mock_action_instance = MagicMock()
    mock_action_instance.action.name = "drop_table"
    mock_param = MagicMock()
    mock_param.name = "table_name"
    mock_action_instance.action.parameters = [mock_param]
    
    # UPF arguments return an FNode with an object() representation
    mock_arg = MagicMock()
    mock_arg.object().name = "users_db"
    mock_action_instance.actual_parameters = [mock_arg]
    
    plan = SequentialPlan([mock_action_instance])
    
    driver = ExecutionDriver(interactive=True, registry=hitl_registry)
    
    # 2. Patch the built-in input() to simulate user typing 'y'
    with patch('builtins.input', return_value='y'):
        results = driver.execute_plan(plan)
        
    assert len(results) == 1
    assert results[0] == "Dropped users_db"

def test_execution_driver_hitl_approval_denied():
    action = InstantaneousAction("drop_table")
    
    mock_action_instance = MagicMock()
    mock_action_instance.action.name = "drop_table"
    mock_param = MagicMock()
    mock_param.name = "table_name"
    mock_action_instance.action.parameters = [mock_param]
    
    mock_arg = MagicMock()
    mock_arg.object().name = "users_db"
    mock_action_instance.actual_parameters = [mock_arg]
    
    plan = SequentialPlan([mock_action_instance])
    
    driver = ExecutionDriver(interactive=True, registry=hitl_registry)
    
    # 2. Patch the built-in input() to simulate user typing 'n'
    with patch('builtins.input', return_value='n'):
        with pytest.raises(PermissionError) as exc_info:
            driver.execute_plan(plan)
            
    assert "Human denied execution" in str(exc_info.value)

def test_execution_driver_emits_hitl_trace_events():
    events = []
    mock_action_instance = MagicMock()
    mock_action_instance.action.name = "drop_table"
    mock_param = MagicMock()
    mock_param.name = "table_name"
    mock_action_instance.action.parameters = [mock_param]
    mock_arg = MagicMock()
    mock_arg.object().name = "users_db"
    mock_action_instance.actual_parameters = [mock_arg]
    plan = SequentialPlan([mock_action_instance])
    driver = ExecutionDriver(
        interactive=True,
        trace_callback=lambda stage, message, payload: events.append(
            (stage, message, payload)
        ),
        registry=hitl_registry,
    )

    with patch('builtins.input', return_value='y'):
        driver.execute_plan(plan)

    assert [event[0] for event in events] == [
        "execution.action.prepare",
        "hitl.approval.required",
        "hitl.approval.granted",
        "execution.action.success",
    ]
    assert events[1][2]["action"] == "drop_table"
    assert events[1][2]["parameters"] == {"table_name": "users_db"}


def test_execution_driver_defaults_to_global_registry_for_compatibility():
    @registry.register_action("global_compat_action")
    def global_compat_action() -> str:
        """Global registry compatibility action."""
        return "global ok"

    mock_action_instance = MagicMock()
    mock_action_instance.action.name = "global_compat_action"
    mock_action_instance.action.parameters = []
    mock_action_instance.actual_parameters = []
    plan = SequentialPlan([mock_action_instance])

    driver = ExecutionDriver(interactive=False)

    assert driver.execute_plan(plan) == ["global ok"]
