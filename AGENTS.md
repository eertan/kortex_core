# Kortex Core - AI Assistant Rules & Guidelines

Welcome to the `kortex_core` project. Any AI coding assistant working on this repository must strictly adhere to the following rules to ensure high-quality, production-ready code and a smooth workflow.

## 1. Code Quality & Architecture
- **Production-Level Practices Only:** Do not implement amateurish, hacky, or one-off solutions. All code must be written with production in mind.
- **Modularity & Generalizability:** Design systems to be highly modular and generalizable. Keep concerns separated (e.g., keep the LLM extraction layer completely decoupled from the deterministic HTN execution layer).
- **Strict Typing:** Python 3.14+ type hints are mandatory everywhere.
- **Documentation:** Maintain clear docstrings for all classes and functions explaining their deterministic purpose.

## 2. Agent Workflow & Permissions
- **Git Operations:** **Do not execute git operations yourself.** Your role is to write code and remind the user when a logical milestone is reached so the user can review and commit the changes.
- **Command Execution:** 
  - **No Permission Needed:** You may freely run commands for reading files (`cat`, `ls`, `grep`), checking project structure, or reading documentation.
  - **Permission Required:** You **must ask for permission** before running potentially long-running commands, large-scale tests, installing new dependencies, or making sweeping refactors across many files.

## 3. Project Context (Kortex Core)
- **Architecture Philosophy:** The system uses an LLM (e.g., Hermes) strictly as an **Intent and Parameter Extractor**, returning structured JSON. It **must not** perform logical reasoning or planning.
- **Deterministic Execution:** All planning and execution are handled by a Hierarchical Task Network (HTN) engine. The LLM feeds the root task and parameters to the HTN, which then executes deterministically based on hardcoded specs.
- **Pydantic Validation:** Rely heavily on Pydantic to enforce the schema between the LLM layer and the HTN layer.
