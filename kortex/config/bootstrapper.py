from typing import Any
import yaml
from unified_planning.shortcuts import (
    UserType, Fluent, BoolType, InstantaneousAction, Object
)
from unified_planning.model.htn import Task, Method
from kortex.config.validation import DomainManifestValidator
from kortex.plugins.registry import registry
from kortex.spine.planner import KortexPlanner

class DomainBootstrapper:
    """
    Parses a declarative YAML domain manifest and injects it into the UPF Planner.
    This fulfills Phase 3 (Zero-Config Bootstrapper).
    """

    def __init__(self, planner: KortexPlanner):
        self.planner = planner
        self.types: dict[str, UserType] = {}
        self.fluents: dict[str, Fluent] = {}
        self.objects: dict[str, Object] = {}

    def load_domain(self, filepath: str) -> None:
        """Load domain structures (types, fluents, actions) from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        validator = DomainManifestValidator()
        validator.validate(data)
        validator.validate_plugin_bindings(data, registry)

        # 1. Parse Types
        for t_name in data.get('types', []):
            self.types[t_name] = UserType(t_name)

        # 2. Parse State Fluents (Variables)
        for f_name, f_def in data.get('fluents', {}).items():
            sig = {k: self.types[v] for k, v in f_def.get('signature', {}).items()}
            fl = Fluent(f_name, BoolType(), **sig)
            self.fluents[f_name] = fl
            self.planner.register_fluent(fl)

        # 3. Parse Primitive Actions
        for act_def in data.get('actions', []):
            a_name = act_def['name']
            params = {k: self.types[v] for k, v in act_def.get('parameters', {}).items()}
            action = InstantaneousAction(a_name, **params)
            
            # Preconditions
            for pre in act_def.get('preconditions', []):
                fl = self.fluents[pre['fluent']]
                args = [action.parameter(p) for p in pre['args']]
                if pre.get('value', True):
                    action.add_precondition(fl(*args))
                else:
                    action.add_precondition(~fl(*args))
                    
            # Effects
            for eff in act_def.get('effects', []):
                fl = self.fluents[eff['fluent']]
                args = [action.parameter(p) for p in eff['args']]
                action.add_effect(fl(*args), eff.get('value', True))
                
            self.planner.register_action(action)
            
        # 4. Parse HTN Tasks and Methods (from Chunking/Manifest)
        for method_def in data.get('htn_methods', []):
            task_name = method_def['target_task']
            
            # In UPF, method parameters must be declared at instantiation.
            # For this MVP chunking, we extract all required parameters from the primitive subtasks.
            method_params = {}
            subtasks_to_add = []
            
            for sub_name in method_def.get('ordered_subtasks', []):
                action = self.planner.problem.action(sub_name[0])
                subtask_args = [str(arg) for arg in sub_name[1:]]
                subtasks_to_add.append((action, subtask_args))
                if subtask_args:
                    for action_param, method_param_name in zip(action.parameters, subtask_args):
                        if method_param_name not in method_params:
                            method_params[method_param_name] = action_param.type
                else:
                    for p in action.parameters:
                        if p.name not in method_params:
                            method_params[p.name] = p.type
                        
            # Create the abstract task if it doesn't exist
            if not self.planner.problem.has_task(task_name):
                task = Task(task_name, **method_params)
                self.planner.register_task(task)
            else:
                task = self.planner.problem.get_task(task_name)
            
            method = Method(method_def['name'], **method_params)
            
            # Now we must map the Method parameters back to the Task parameters
            # For this prototype, we'll just bind them all sequentially
            task_args = []
            for p_name in task.parameters:
                task_args.append(method.parameter(p_name.name))
                
            method.set_task(task, *task_args)
            
            # Map subtasks
            for action, subtask_args in subtasks_to_add:
                symbolic_args = subtask_args or [p.name for p in action.parameters]
                action_args = [method.parameter(p_name) for p_name in symbolic_args]
                method.add_subtask(action, *action_args)
                
            self.planner.register_method(method)
            self.planner.register_method_spec(
                target_task=task_name,
                parameter_names=list(method_params.keys()),
                ordered_subtasks=method_def.get('ordered_subtasks', []),
            )

    def load_problem_state(self, objects: dict[str, str], initial_state: list[dict[str, Any]]):
        """Register specific instances and starting facts for the world."""
        # Add objects
        for obj_name, type_name in objects.items():
            obj = Object(obj_name, self.types[type_name])
            self.objects[obj_name] = obj
            self.planner.register_object(obj)

        # Apply initial state
        for fact in initial_state:
            fl = self.fluents[fact['fluent']]
            args = [self.objects[arg] for arg in fact.get('args', [])]
            self.planner.set_initial_value(fl(*args), fact.get('value', True))

    def create_goal(self, goal_def: dict[str, Any]):
        """Convert a goal dictionary into a UPF goal expression."""
        if 'fluent' in goal_def:
            fl = self.fluents[goal_def['fluent']]
            args = [self.objects[arg] for arg in goal_def.get('args', [])]
            if goal_def.get('value', True):
                self.planner.add_goal(fl(*args))
            else:
                self.planner.add_goal(~fl(*args))
        elif 'task' in goal_def:
            # We also allow setting an HTN task directly as the goal
            args = [str(arg) for arg in goal_def.get('args', [])]
            self.planner.add_htn_goal(goal_def['task'], args)
