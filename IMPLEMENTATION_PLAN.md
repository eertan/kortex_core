# Kortex Core - Implementation Plan & Task List

Based on the detailed architectural discussion, here is the structured implementation plan to build **Kortex Core** (a deterministic, neuro-symbolic open-world agent kernel).

## Core Architecture Layers
1. **Extractor (LLM):** Pydantic-constrained natural language to JSON intent parser.
2. **Orchestration Spine (HTN + PDDL):** Unified Planning Framework (UPF) handling strict recipes (Tier 1) and state-space search (Tier 2).
3. **Memory & Ingestion:** Graphiti (Neo4j) handling temporal episodic/semantic graphs, isolated from external world KGs.
4. **Sub-Cognition (Soar):** Encapsulated Soar VMs for fast, rule-based diagnostic logic.
5. **Novelty Branch (Tier 3):** Sandboxed LLM coding agent for handling total impasses by generating new primitive skills.

---

## Phase 1: Project Scaffolding & Dependencies
**Goal:** Set up the Python project with strict typing, linting, and core dependencies.
- [x] Add dependencies to `pyproject.toml` (e.g., `unified-planning`, `pyperplan`, `pydantic`, `instructor`, `google-genai`, `pyyaml`, `networkx`, `kuzu`, `graphiti`).
- [x] Create the core module structure:
  - `/kortex` (Main package)
  - `/kortex/extractor` (LLM Interface & Pydantic schemas)
  - `/kortex/spine` (UPF HTN & PDDL wrappers, Orchestrator)
  - `/kortex/subagents` (Generic Tool Enclosures, isolating external binaries)
  - `/kortex/memory` (Graphiti/Kuzu connection)
  - `/kortex/sandbox` (Novelty branch using pi SDK, AST guards)
  - `/kortex/plugins` (Task-specific Python operator registry)
  - `/kortex/config` (Domain manifest YAML parsers)

## Phase 2: Cognitive Orchestration Spine (Tier 1 & Tier 2)
**Goal:** Implement the deterministic planner using the Unified Planning Framework (UPF).
- [x] Initialize the `HierarchicalProblem` (HTN + PDDL hybrid) environment.
- [x] Implement the `execute_compiled_plan` loop that steps through UPF-generated plans.
- [x] Create testing harness: Write a simple mock domain with HTN methods and primitive actions to prove UPF can solve top-down and fill in PDDL gaps (`pyperplan`).
- [x] Implement **Macro-Operator Chunking**: When Tier 2 (PDDL state-space search) successfully resolves a vague goal or gap, extract the primitive action sequence, compile it into a reusable HTN Method, and persist it to the `domain_manifest.yaml` (Intra-Domain Learner).

## Phase 3: Declarative Configuration & Bootstrapper
**Goal:** Implement the "Zero-Config" Open-World Kernel loader.
- [x] Define the schema for `domain_manifest.yaml`.
- [x] Implement `bootstrap_domain_from_manifest` in the `OpenWorldAgentKernel` to parse YAML.
- [x] Dynamically register UPF types, fluents (state variables), and primitive actions from the parsed YAML.
- [x] Implement **Plugin Registry Module**: Develop a dynamic loader inside `/kortex/plugins` allowing task-specific Python operator functions to be mapped instantly as executable UPF primitive actions.

## Phase 4: Model Symbol Mapping Interface (Extractor)
**Goal:** Hook the LLM strictly as an intent and parameter extractor.
- [x] Create Pydantic definitions for `HTNLaunchPad` (root task and initial state arguments).
- [x] Implement the extraction logic using `instructor` or a structured generation wrapper to force local LLMs (Hermes/Gemini) to output pure JSON.
- [x] Write logic to bind the extracted JSON intent into the UPF initial state and goal definitions.

## Phase 5: Sub-Cognition Enclosures (External Agent Tools)
**Goal:** Wrap external cognitive engines (Soar, custom scripts) inside an opaque "black-box" primitive for the UPF planner.
- [x] Implement the `SubagentEnclosure` wrapper with strict Read-Isolate-Write boundaries.
- [x] Implement state ingress mapping: filter and copy specified master states to the subagent via JSON stdin.
- [x] Implement state egress filtering: extract validated mutations from stdout JSON and update the UPF planner state.

## Phase 6: Federated Memory Engine
**Goal:** Implement the isolated Graphiti interaction ledger using local Kùzu.
- [x] Connect Graphiti engine to local `kuzu://` database.
- [x] Set up the dual-phase memory cadence (fast-append synchronous hook vs. async sleep-phase digestion).
- [x] Implement the Agent Memory ledger (storing agent-user interactions via `add_episode`).

## Phase 7: Metacognitive Novelty Branch (Tier 3)
**Goal:** Safely handle total domain impasses with a sandboxed coding subagent using the pi SDK.
- [x] Detect planning failure from UPF (Impasse) and construct the impasse prompt (current state, available operators, failed goal).
- [x] Invoke the **pi Coding Agent SDK** (`pi run worker`) to act as the subagent, delegating the task of resolving the impasse.
- [x] Build the `sandbox` validation pipeline:
  - Run the `pi` agent's generated code through AST analyzer (ban `exec`, `eval`, `os`, `sys`, etc.).
  - Run isolated `pytest` assertions on the generated primitive.
- [ ] On success, the `pi` agent natively writes/edits the new YAML/Python primitive into the plugin registry or domain manifest, triggering a hot-reload.

## Phase 8: Human-in-the-Loop (HITL) & Full Agent Tracing
**Goal:** Ensure the system can be audited, safely interrupted, and can ask for clarification when faced with ambiguity.
- [ ] **Full Tracing:** Implement a structured tracing logger that records the entire lifecycle (NL Intent -> PDDL State -> Planner Search -> Physical Execution).
- [ ] **Security Authorization:** Add an authorization layer in the `ExecutionDriver`. High-risk primitive actions must pause execution and request human approval.
- [ ] **Cognitive Clarification (HITL):** 
  - *Pre-Planning:* Update the Extractor so if an intent is missing mandatory schema parameters (e.g., target entity for a churn model is ambiguous), it returns a `ClarificationRequired` payload instead of guessing.
  - *Mid-Execution:* Give primitive actions and subagents a standard way to yield to the user (`ask_human`) if they discover ambiguity in the data while running.

## Phase 9: Memory Integration & Sleep-Phase Metacognition
**Goal:** Implement the "Synergy" reflection loop and active RAG.
- [ ] **Episodic Context Injection:** Modify the Extractor and Bootstrapper so they actively query the Graphiti/Kùzu memory for past context *before* planning (resolving vague entities using past knowledge).
- [ ] **Meta-Task Reflection (Sleep Phase):** Implement the asynchronous sleep-phase LLM routine that scans multiple distinct episodic traces in Graphiti, finds common structural intersections, and synthesizes abstract "Meta-Tasks" (HTN Grammar Learning).

## Phase 10: End-to-End Complex Scenarios
**Goal:** Prove the architecture works holistically with concrete scenarios.
- [ ] **Scenario 1 - The Vague Gap:** (Tier 2 chunking) A robot is asked to deliver a package but the room is locked. It must use autonomous PDDL search to realize it needs to fetch a badge, execute, and chunk the macro-operator.
- [ ] **Scenario 2 - The Complete Impasse:** (Tier 3 Novelty) A completely unknown request is made. The planner fails, spawns the Pi subagent to write a new Python plugin, dynamically registers it, and completes the task.
- [ ] **Scenario 3 - Memory Synergy:** (Sleep Phase) Run 3 distinct tasks that share a hidden sub-pattern. Trigger the sleep-phase reflection and verify it generates a new abstract HTN method.

---

Does this capture our discussion accurately? Let me know which Phase we should start with!