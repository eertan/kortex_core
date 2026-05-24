import os
import yaml
import pytest
from kortex.spine.planner import KortexPlanner
from kortex.config.bootstrapper import DomainBootstrapper
from kortex.spine.driver import ExecutionDriver
from kortex.plugins.registry import PluginRegistry

e2e_registry = PluginRegistry()

# Mock YAML Domain
YAML_DOMAIN = """
domain_name: "test_robot_domain"
types:
  - Location
fluents:
  robot_at:
    signature: { loc: Location }
  door_unlocked:
    signature: { loc: Location }
actions:
  - name: move
    parameters: { frm: Location, to: Location }
    preconditions:
      - fluent: robot_at
        args: [frm]
        value: true
    effects:
      - fluent: robot_at
        args: [frm]
        value: false
      - fluent: robot_at
        args: [to]
        value: true
  - name: unlock
    parameters: { loc: Location }
    preconditions:
      - fluent: robot_at
        args: [loc]
        value: true
    effects:
      - fluent: door_unlocked
        args: [loc]
        value: true
"""

# Define actual python executions matching the actions
@e2e_registry.register_action("move")
def move_plugin(frm: str, to: str) -> str:
    return f"Robot drove from {frm} to {to}"

@e2e_registry.register_action("unlock")
def unlock_plugin(loc: str) -> str:
    return f"Robot used badge to unlock {loc}"


def test_end_to_end_yaml_to_execution(tmp_path):
    # 1. Create temporary YAML domain file
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(YAML_DOMAIN)
    
    # 2. Bootstrapper Phase
    planner = KortexPlanner("test_spine")
    bootstrapper = DomainBootstrapper(planner, registry=e2e_registry)
    bootstrapper.load_domain(str(domain_file))
    
    # Check parsed data
    assert "Location" in bootstrapper.types
    assert "robot_at" in bootstrapper.fluents
    
    # 3. Feed State from "Extraction/Memory" phase
    # Simulated initial state (Robot is in lobby, vault is locked)
    objects = {"lobby": "Location", "hallway": "Location", "vault": "Location"}
    initial_state = [
        {"fluent": "robot_at", "args": ["lobby"], "value": True},
        {"fluent": "door_unlocked", "args": ["lobby"], "value": True}
    ]
    bootstrapper.load_problem_state(objects, initial_state)
    
    # 4. Set the Goal (Extracted from "I need to get into the vault")
    # Goal: robot_at(vault) AND NOT door_locked(vault)
    # Pyperplan only supports positive goals (STRIPS limitation).
    # So we'll track "door_unlocked" instead of "door_locked(False)".
    bootstrapper.create_goal({"fluent": "robot_at", "args": ["vault"], "value": True})
    bootstrapper.create_goal({"fluent": "door_unlocked", "args": ["vault"], "value": True})
    
    # 5. Plan Generation (Tier 2 PDDL search)
    plan = planner.execute_plan()
    assert plan is not None, "Planner failed to find a state-space path!"
    
    # Extract planned steps
    action_names = [a.action.name for a in plan.actions]
    # Expectation: move to vault, unlock vault
    # Note: Depending on pyperplan search, it might be `move(lobby, vault) -> unlock(vault)`
    assert "move" in action_names
    assert "unlock" in action_names
    
    # 6. Physical Execution Phase
    driver = ExecutionDriver(registry=e2e_registry)
    results = driver.execute_plan(plan)
    
    # Verify the python functions actually ran
    assert len(results) == 2
    assert "Robot drove from lobby to vault" in results
    assert "Robot used badge to unlock vault" in results
