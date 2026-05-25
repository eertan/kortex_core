import os
import pytest
import yaml
from unittest.mock import MagicMock, patch

from kortex.spine.planner import KortexPlanner
from kortex.config.bootstrapper import DomainBootstrapper
from kortex.spine.driver import ExecutionDriver
from kortex.plugins.registry import PluginRegistry
from kortex.sandbox.chunker import IntraDomainLearner
from kortex.sandbox.novelty import NoveltyBranch
from kortex.memory.reflector import SleepReflector, SynthesizedMetaTask

scenario_registry = PluginRegistry()

# === SCENARIO PLUGINS (PHYSICAL DRIVERS) ===
@scenario_registry.register_action("move")
def move(frm: str, to: str) -> str:
    return f"Moved from {frm} to {to}"

@scenario_registry.register_action("unlock")
def unlock(loc: str) -> str:
    return f"Unlocked {loc}"

@scenario_registry.register_action("wipe_server", requires_approval=True)
def wipe_server(srv: str) -> str:
    return f"WIPED DATA ON {srv}"

@scenario_registry.register_action("check_disk_space")
def check_disk_space(srv: str) -> str:
    return f"Checked disk space on {srv}"

@scenario_registry.register_action("clear_cache")
def clear_cache(srv: str) -> str:
    return f"Cleared cache on {srv}"

# === BASE DOMAIN YAML ===
BASE_YAML = """
domain_name: "scenario_domain"
types:
  - Location
  - Server
fluents:
  robot_at:
    signature: { loc: Location }
  door_unlocked:
    signature: { loc: Location }
  server_unreachable:
    signature: { srv: Server }
  storage_checked:
    signature: { srv: Server }
  cache_cleared:
    signature: { srv: Server }
  audit_completed:
    signature: { srv: Server }
actions:
  - name: move
    parameters: { frm: Location, to: Location }
    preconditions:
      - fluent: robot_at
        args: [frm]
    effects:
      - fluent: robot_at
        args: [frm]
        value: false
      - fluent: robot_at
        args: [to]
  - name: unlock
    parameters: { loc: Location }
    preconditions:
      - fluent: robot_at
        args: [loc]
    effects:
      - fluent: door_unlocked
        args: [loc]
  - name: wipe_server
    parameters: { srv: Server }
    preconditions: []
    effects:
      - fluent: server_unreachable
        args: [srv]
        value: true
  - name: check_disk_space
    parameters: { srv: Server }
    preconditions: []
    effects:
      - fluent: storage_checked
        args: [srv]
        value: true
  - name: clear_cache
    parameters: { srv: Server }
    preconditions: []
    effects:
      - fluent: cache_cleared
        args: [srv]
        value: true
htn_methods:
  # Pre-compiled (Perfectly Specified) Task for Scenario 1
  - name: m_deliver_straight
    target_task: "deliver_straight"
    ordered_subtasks:
      - ["move", "frm", "to"]
  - name: m_wipe
    target_task: "wipe_server_task"
    ordered_subtasks:
      - ["wipe_server", "srv"]
"""

@pytest.fixture
def base_domain(tmp_path):
    d_path = tmp_path / "domain.yaml"
    d_path.write_text(BASE_YAML)
    return d_path

def setup_planner(domain_path, objects, initial_state):
    planner = KortexPlanner("test_spine")
    bootstrapper = DomainBootstrapper(planner, registry=scenario_registry)
    bootstrapper.load_domain(str(domain_path))
    bootstrapper.load_problem_state(objects, initial_state)
    return planner, bootstrapper


def test_scenario_1_perfect_htn(base_domain):
    """
    Scenario 1: Perfectly specified task in HTN, straightforward execution.
    The agent receives an explicit root task ('deliver_straight') which maps directly to a method.
    """
    objects = {"lobby": "Location", "hallway": "Location"}
    initial_state = [{"fluent": "robot_at", "args": ["lobby"]}]
    
    planner, bootstrapper = setup_planner(base_domain, objects, initial_state)
    
    # We set the goal as the HTN task directly
    bootstrapper.create_goal({"task": "deliver_straight", "args": ["lobby", "hallway"]})
    
    plan = planner.execute_plan()
    assert plan is not None
    
    driver = ExecutionDriver(interactive=False, registry=scenario_registry)
    results = driver.execute_plan(plan)
    
    # It just executed the pre-compiled ['move'] task
    assert "lobby" in results[0]
    assert "hallway" in results[0]


def test_scenario_2_vague_gap(base_domain):
    """
    Scenario 2: HTN has the goal and primitives but needs a planner to decompose the task.
    (Vague gap: Needs to get into the vault, but it's locked. Pyperplan searches for a solution).
    """
    objects = {"lobby": "Location", "vault": "Location"}
    initial_state = [
        {"fluent": "robot_at", "args": ["lobby"]},
        {"fluent": "door_unlocked", "args": ["lobby"]} # Vault is locked
    ]
    
    planner, bootstrapper = setup_planner(base_domain, objects, initial_state)
    
    # Set a vague PDDL state goal instead of a strict HTN task
    bootstrapper.create_goal({"fluent": "robot_at", "args": ["vault"]})
    bootstrapper.create_goal({"fluent": "door_unlocked", "args": ["vault"]})
    
    # Planner does state-space search to bridge the gap
    plan = planner.execute_plan()
    assert plan is not None
    
    action_names = [a.action.name for a in plan.actions]
    assert "move" in action_names
    assert "unlock" in action_names
    
    driver = ExecutionDriver(interactive=False, registry=scenario_registry)
    results = driver.execute_plan(plan)
    assert len(results) == 2


def test_scenario_3_learned_chunk(base_domain):
    """
    Scenario 3: Same as Scenario 2, but we simulate that the Macro-Operator Chunker 
    already learned this sequence. The agent executes the newly learned HTN method instantly.
    """
    # 1. Manually trigger the chunker to learn the gap from Scenario 2
    chunker = IntraDomainLearner(str(base_domain))
    chunker.chunk_successful_plan(
        failed_task_name="access_secure_vault",
        preconditions={}, # Simplified
        plan_actions=[["move", "frm", "to"], ["unlock", "to"]]
    )
    
    # 2. Reload the domain, which now contains the new learned method
    objects = {"lobby": "Location", "vault": "Location"}
    initial_state = [{"fluent": "robot_at", "args": ["lobby"]}]
    
    planner, bootstrapper = setup_planner(base_domain, objects, initial_state)
    
    # 3. Call the newly learned abstract task directly!
    bootstrapper.create_goal({"task": "access_secure_vault", "args": ["lobby", "vault"]})
    
    plan = planner.execute_plan()
    assert plan is not None
    
    # It executes the chunked plan exactly as searched last time
    action_names = [a.action.name for a in plan.actions]
    assert action_names == ["move", "unlock"]


def test_learned_chunk_requires_inferred_preconditions(base_domain):
    """
    A learned HTN method must preserve the condition contract discovered during
    planning. The access_secure_vault chunk requires robot_at(frm), so it should
    not expand if the robot is not at the source location.
    """
    chunker = IntraDomainLearner(str(base_domain))
    chunker.chunk_successful_plan(
        failed_task_name="access_secure_vault",
        preconditions={},
        plan_actions=[["move", "frm", "to"], ["unlock", "to"]],
    )

    objects = {"lobby": "Location", "vault": "Location"}
    initial_state = []

    planner, bootstrapper = setup_planner(base_domain, objects, initial_state)
    bootstrapper.create_goal({"task": "access_secure_vault", "args": ["lobby", "vault"]})

    with pytest.raises(ValueError, match="requires robot_at\\(lobby\\)=True"):
        planner.execute_plan()


def test_scenario_4_hitl_approval(base_domain):
    """
    Scenario 4: Like scenario 1 or 2, but requires approval.
    We try to execute 'wipe_server', which the Plugin Registry has flagged as requires_approval=True.
    """
    objects = {"prod_db": "Server"}
    initial_state = [{"fluent": "server_unreachable", "args": ["prod_db"], "value": False}]
    
    planner, bootstrapper = setup_planner(base_domain, objects, initial_state)
    
    # Vague goal: Make the server unreachable (which forces 'wipe_server')
    bootstrapper.create_goal({"fluent": "server_unreachable", "args": ["prod_db"]})
    
    plan = planner.execute_plan()
    assert plan is not None
    
    # Execution Driver MUST halt and ask for approval
    driver = ExecutionDriver(interactive=True, registry=scenario_registry)
    
    # Simulate human typing 'n' (Deny)
    with patch('builtins.input', return_value='n'):
        with pytest.raises(PermissionError) as exc:
            driver.execute_plan(plan)
            
    assert "Human denied execution" in str(exc.value)
    
    # Simulate human typing 'y' (Approve)
    with patch('builtins.input', return_value='y'):
        results = driver.execute_plan(plan)
        
    assert "WIPED DATA ON prod_db" in results[0]


def test_scenario_5_total_impasse_routes_to_novelty(base_domain):
    """
    Scenario 5: Complete impasse.
    The goal is valid in the domain, but no primitive action can achieve it. The
    planner returns no plan, so the novelty branch receives the failed context.
    """
    objects = {"prod_db": "Server"}
    initial_state = []

    planner, bootstrapper = setup_planner(base_domain, objects, initial_state)
    failed_goal = {"fluent": "audit_completed", "args": ["prod_db"], "value": True}
    bootstrapper.create_goal(failed_goal)

    plan = planner.execute_plan()
    assert plan is None

    novelty = NoveltyBranch(workspace_path=str(base_domain.parent))
    resolved = novelty.resolve_impasse(
        failed_goal=failed_goal,
        current_state={},
        available_actions=scenario_registry.list_plugins(),
    )

    assert resolved is True


def test_scenario_6_sleep_reflection_creates_executable_meta_task(base_domain):
    """
    Scenario 6: Sleep reflection detects a recurring operational sequence,
    injects a semantic HTN meta-task, and that learned task executes directly.
    """
    os.environ["GEMINI_API_KEY"] = "fake-key"
    reflector = SleepReflector(str(base_domain))
    reflector.client.models.generate_content = MagicMock(
        return_value=SynthesizedMetaTask(
            meta_task_name="optimize_local_storage",
            preconditions={"server_reachable": True},
            ordered_subtasks=[
                ["check_disk_space", "srv"],
                ["clear_cache", "srv"],
            ],
        )
    )

    reflected = reflector.reflect_and_synthesize(
        [
            ["ssh_login", "check_disk_space", "clear_cache", "reindex_table"],
            ["check_disk_space", "clear_cache", "restart_service"],
            ["download_logs", "check_disk_space", "clear_cache"],
        ]
    )
    assert reflected is True

    planner, bootstrapper = setup_planner(
        base_domain,
        {"prod_db": "Server"},
        [],
    )
    bootstrapper.create_goal(
        {"task": "optimize_local_storage", "args": ["prod_db"]}
    )

    plan = planner.execute_plan()
    assert plan is not None
    assert [action.action.name for action in plan.actions] == [
        "check_disk_space",
        "clear_cache",
    ]

    driver = ExecutionDriver(interactive=False, registry=scenario_registry)
    results = driver.execute_plan(plan)

    assert results == [
        "Checked disk space on prod_db",
        "Cleared cache on prod_db",
    ]
