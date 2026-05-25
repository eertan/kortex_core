"""
Validation helpers for declarative Kortex domain manifests.

The bootstrapper should fail before planning when a manifest references missing
types, fluents, actions, parameters, or incompatible plugin signatures.
"""

from collections.abc import Mapping
from typing import Any

from kortex.plugins.registry import PluginRegistry


class DomainManifestError(ValueError):
    """Raised when a domain manifest is structurally invalid."""


class DomainManifestValidator:
    """Validates declarative YAML domain manifests before bootstrap."""

    def validate(self, manifest: Mapping[str, Any]) -> None:
        """Validate manifest structure and internal references."""
        types = self._validate_types(manifest)
        fluents = self._validate_fluents(manifest, types)
        actions = self._validate_actions(manifest, types, fluents)
        self._validate_methods(manifest, actions)

    def validate_plugin_bindings(
        self,
        manifest: Mapping[str, Any],
        registry: PluginRegistry,
    ) -> None:
        """Validate that registered plugins can accept declared action parameters."""
        for action_def in manifest.get("actions", []):
            action_name = action_def["name"]
            if action_name not in registry.plugins:
                continue

            expected_params = set(action_def.get("parameters", {}).keys())
            plugin = registry.get_plugin(action_name)
            signature = plugin["signature"]
            accepted_params = set(signature.parameters.keys())
            has_var_kwargs = any(
                param.kind.name == "VAR_KEYWORD"
                for param in signature.parameters.values()
            )

            if not has_var_kwargs and not expected_params.issubset(accepted_params):
                missing = sorted(expected_params - accepted_params)
                raise DomainManifestError(
                    f"Plugin '{action_name}' is missing parameters declared by "
                    f"the action spec: {missing}."
                )

    def _validate_types(self, manifest: Mapping[str, Any]) -> set[str]:
        """Validate and return declared type names."""
        types = manifest.get("types", [])
        if not isinstance(types, list) or not all(isinstance(item, str) for item in types):
            raise DomainManifestError("'types' must be a list of type names.")
        return set(types)

    def _validate_fluents(
        self,
        manifest: Mapping[str, Any],
        types: set[str],
    ) -> dict[str, set[str]]:
        """Validate fluent signatures and return fluent argument names."""
        fluents = manifest.get("fluents", {})
        if not isinstance(fluents, Mapping):
            raise DomainManifestError("'fluents' must be a mapping.")

        fluent_args: dict[str, set[str]] = {}
        for fluent_name, fluent_def in fluents.items():
            signature = fluent_def.get("signature", {})
            if not isinstance(signature, Mapping):
                raise DomainManifestError(f"Fluent '{fluent_name}' signature must be a mapping.")
            for arg_name, type_name in signature.items():
                if type_name not in types:
                    raise DomainManifestError(
                        f"Fluent '{fluent_name}' argument '{arg_name}' references "
                        f"unknown type '{type_name}'."
                    )
            fluent_args[str(fluent_name)] = {str(arg) for arg in signature.keys()}

        return fluent_args

    def _validate_actions(
        self,
        manifest: Mapping[str, Any],
        types: set[str],
        fluents: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """Validate action parameter, precondition, and effect references."""
        actions = manifest.get("actions", [])
        if not isinstance(actions, list):
            raise DomainManifestError("'actions' must be a list.")

        action_params: dict[str, set[str]] = {}
        for action_def in actions:
            action_name = action_def.get("name")
            if not isinstance(action_name, str):
                raise DomainManifestError("Every action must declare a string 'name'.")

            params = action_def.get("parameters", {})
            if not isinstance(params, Mapping):
                raise DomainManifestError(f"Action '{action_name}' parameters must be a mapping.")
            for param_name, type_name in params.items():
                if type_name not in types:
                    raise DomainManifestError(
                        f"Action '{action_name}' parameter '{param_name}' references "
                        f"unknown type '{type_name}'."
                    )

            known_params = {str(param_name) for param_name in params.keys()}
            action_params[action_name] = known_params
            for section in ("preconditions", "effects"):
                for fact_def in action_def.get(section, []):
                    self._validate_action_fact(action_name, section, fact_def, fluents, known_params)

        return action_params

    def _validate_action_fact(
        self,
        action_name: str,
        section: str,
        fact_def: Mapping[str, Any],
        fluents: dict[str, set[str]],
        known_params: set[str],
    ) -> None:
        """Validate one precondition/effect fact reference."""
        fluent_name = fact_def.get("fluent")
        if fluent_name not in fluents:
            raise DomainManifestError(
                f"Action '{action_name}' {section} references unknown fluent '{fluent_name}'."
            )

        for arg_name in fact_def.get("args", []):
            if arg_name not in known_params:
                raise DomainManifestError(
                    f"Action '{action_name}' {section} references unknown "
                    f"parameter '{arg_name}'."
                )

    def _validate_methods(
        self,
        manifest: Mapping[str, Any],
        actions: dict[str, set[str]],
    ) -> None:
        """Validate HTN method primitive subtask references."""
        methods = manifest.get("htn_methods", [])
        if not isinstance(methods, list):
            raise DomainManifestError("'htn_methods' must be a list when provided.")

        fluents = manifest.get("fluents", {})

        for method_def in methods:
            method_name = method_def.get("name", "<unnamed>")
            method_params = set(method_def.get("parameters", {}).keys())
            for subtask in method_def.get("ordered_subtasks", []):
                if not isinstance(subtask, list) or not subtask:
                    raise DomainManifestError(
                        f"Method '{method_name}' contains an invalid subtask entry."
                    )
                action_name = subtask[0]
                if action_name not in actions:
                    raise DomainManifestError(
                        f"Method '{method_name}' references unknown action '{action_name}'."
                    )
                if len(subtask) > 1 and len(subtask[1:]) != len(actions[action_name]):
                    raise DomainManifestError(
                        f"Method '{method_name}' subtask '{action_name}' provides "
                        f"{len(subtask[1:])} args, but action expects "
                        f"{len(actions[action_name])}."
                    )
                method_params.update(str(arg) for arg in subtask[1:])

            for section in ("preconditions", "effects"):
                facts = method_def.get(section, [])
                if isinstance(facts, Mapping):
                    continue
                if not isinstance(facts, list):
                    raise DomainManifestError(
                        f"Method '{method_name}' {section} must be a list when provided."
                    )
                for fact_def in facts:
                    self._validate_method_fact(
                        method_name=method_name,
                        section=section,
                        fact_def=fact_def,
                        fluents=fluents,
                        known_params=method_params,
                    )

    def _validate_method_fact(
        self,
        method_name: str,
        section: str,
        fact_def: Mapping[str, Any],
        fluents: Mapping[str, Any],
        known_params: set[str],
    ) -> None:
        """Validate one HTN method precondition/effect fact reference."""
        fluent_name = fact_def.get("fluent")
        if fluent_name not in fluents:
            raise DomainManifestError(
                f"Method '{method_name}' {section} references unknown fluent '{fluent_name}'."
            )

        for arg_name in fact_def.get("args", []):
            if arg_name not in known_params:
                raise DomainManifestError(
                    f"Method '{method_name}' {section} references unknown "
                    f"parameter '{arg_name}'."
                )
