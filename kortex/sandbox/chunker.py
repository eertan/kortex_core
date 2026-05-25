import os
from typing import Any, Sequence

import yaml

class IntraDomainLearner:
    """
    Implements Macro-Operator Chunking (Tier 2 Learning).
    When the PDDL state-space solver bridges a gap using primitive actions,
    this module extracts that trace and compiles it into a static HTN Method.
    This bypasses the LLM entirely and speeds up future execution via symbolic compilation.
    """
    def __init__(self, manifest_path: str = "domain_manifest.yaml"):
        self.manifest_path = manifest_path

    def chunk_successful_plan(
        self,
        failed_task_name: str,
        preconditions: dict,
        plan_actions: Sequence[str | Sequence[str]],
    ) -> None:
        """
        Takes a sequence of successful primitive actions and writes them as an HTN method.
        """
        print(f"[IntraDomainLearner] Extracting macro-operator for task '{failed_task_name}'")
        
        # 1. Format the action trace into a clean subtask sequence
        subtask_list: list[list[str]] = []
        for action_spec in plan_actions:
            if isinstance(action_spec, str):
                subtask_list.append([action_spec])
            else:
                subtask_list.append([str(part) for part in action_spec])

        manifest = self._load_manifest()
        compiled_contract = self._infer_symbolic_contract(
            manifest=manifest,
            ordered_subtasks=subtask_list,
        )
            
        # 2. Structure the new HTN Method "Chunk"
        new_method = {
            "name": f"m_compiled_{failed_task_name}_{len(subtask_list)}steps",
            "target_task": failed_task_name,
            "preconditions": compiled_contract["preconditions"] or preconditions,
            "effects": compiled_contract["effects"],
            "ordered_subtasks": subtask_list,
            "provenance": {
                "source": "intra_domain_chunking",
                "source_trace": subtask_list,
            },
        }
        if compiled_contract["parameters"]:
            new_method["parameters"] = compiled_contract["parameters"]
        
        # 3. Append the compiled reflex back to the declarative file
        if "htn_methods" not in manifest:
            manifest["htn_methods"] = []
            
        manifest["htn_methods"].append(new_method)
        
        with open(self.manifest_path, 'w') as f:
            yaml.dump(manifest, f, sort_keys=False)
            
        print(f"[IntraDomainLearner] Macro-Operator Chunked: Compiled '{failed_task_name}' into static HTN method.")

    def _load_manifest(self) -> dict[str, Any]:
        """Load the domain manifest used as the source of action semantics."""
        if not os.path.exists(self.manifest_path):
            print(f"[IntraDomainLearner] Manifest {self.manifest_path} not found. Creating new.")
            return {"htn_methods": []}

        with open(self.manifest_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _infer_symbolic_contract(
        self,
        manifest: dict[str, Any],
        ordered_subtasks: list[list[str]],
    ) -> dict[str, Any]:
        """Infer method parameters, external preconditions, and net effects."""
        action_defs = {
            action_def["name"]: action_def
            for action_def in manifest.get("actions", [])
        }
        if not action_defs:
            return {"parameters": {}, "preconditions": [], "effects": []}

        method_parameters: dict[str, str] = {}
        required_facts: list[dict[str, Any]] = []
        achieved_facts: dict[tuple[str, tuple[str, ...]], bool] = {}

        for subtask in ordered_subtasks:
            action_name = subtask[0]
            action_def = action_defs.get(action_name)
            if action_def is None:
                continue

            action_params = list(action_def.get("parameters", {}).items())
            symbolic_args = subtask[1:] or [name for name, _ in action_params]
            bindings = {
                action_param_name: method_param_name
                for (action_param_name, _), method_param_name
                in zip(action_params, symbolic_args)
            }

            for action_param_name, type_name in action_params:
                method_param_name = bindings.get(action_param_name, action_param_name)
                if method_param_name not in method_parameters:
                    method_parameters[method_param_name] = type_name

            for precondition in action_def.get("preconditions", []):
                normalized = self._bind_fact(precondition, bindings)
                key = self._fact_key(normalized)
                expected_value = normalized.get("value", True)
                if achieved_facts.get(key) != expected_value:
                    self._append_unique_fact(required_facts, normalized)

            for effect in action_def.get("effects", []):
                normalized = self._bind_fact(effect, bindings)
                achieved_facts[self._fact_key(normalized)] = normalized.get("value", True)

        net_effects = [
            {
                "fluent": fluent,
                "args": list(args),
                "value": value,
            }
            for (fluent, args), value in achieved_facts.items()
        ]
        return {
            "parameters": method_parameters,
            "preconditions": required_facts,
            "effects": net_effects,
        }

    def _bind_fact(
        self,
        fact_def: dict[str, Any],
        bindings: dict[str, str],
    ) -> dict[str, Any]:
        """Replace action-local parameter names with learned-method parameter names."""
        normalized = {
            "fluent": fact_def["fluent"],
            "args": [bindings.get(str(arg), str(arg)) for arg in fact_def.get("args", [])],
        }
        if "value" in fact_def:
            normalized["value"] = fact_def["value"]
        return normalized

    def _fact_key(self, fact_def: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
        """Return the identity key for a symbolic fluent fact."""
        return (
            str(fact_def["fluent"]),
            tuple(str(arg) for arg in fact_def.get("args", [])),
        )

    def _append_unique_fact(
        self,
        facts: list[dict[str, Any]],
        fact_def: dict[str, Any],
    ) -> None:
        """Append a fact if an equivalent required condition is not already present."""
        key = self._fact_key(fact_def)
        value = fact_def.get("value", True)
        for existing in facts:
            if self._fact_key(existing) == key and existing.get("value", True) == value:
                return
        facts.append(fact_def)
