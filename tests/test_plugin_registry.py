import pytest
from kortex.plugins.registry import PluginRegistry

def test_plugin_registration_and_execution():
    registry = PluginRegistry()
    
    # 1. Register a mock plugin
    @registry.register_action(
        name="mock_move", 
        preconditions={"robot_at": "start"}, 
        effects={"robot_at": "end"}
    )
    def mock_move_action(target: str) -> str:
        return f"Moved to {target}"
        
    # 2. Check metadata
    plugin_meta = registry.get_plugin("mock_move")
    assert plugin_meta["name"] == "mock_move"
    assert plugin_meta["preconditions"] == {"robot_at": "start"}
    assert plugin_meta["effects"] == {"robot_at": "end"}
    
    # 3. Test execution
    result = registry.execute_plugin("mock_move", target="vault")
    assert result == "Moved to vault"
    
def test_plugin_not_found():
    registry = PluginRegistry()
    with pytest.raises(KeyError):
        registry.execute_plugin("non_existent_action")
