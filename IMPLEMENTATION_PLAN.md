# Kortex Core Implementation Plan

This document tracks the actual implementation state of Kortex Core. The goal
is a deterministic agent kernel where LLMs extract structured intent and handle
novelty outside the normal execution spine, while planning and execution remain
auditable and schema-bound.

## Current Architecture

```text
Natural language request
  -> Extractor returns HTNLaunchPad or ClarificationRequired
  -> Optional explicit memory fact hydration
  -> YAML domain bootstrap and validation
  -> Tier 1 deterministic HTN method expansion
  -> Tier 2 classical Pyperplan fallback for state goals
  -> Plugin-backed execution with approval HITL
  -> Structured trace + memory writeback
  -> Tier 3 provider-neutral novelty branch on total impasse
```

## Completed

### 1. Project Scaffold

- [x] Python package structure under `kortex/`.
- [x] Dependencies in `pyproject.toml` for UPF, Pyperplan, Pydantic,
  Instructor, Google GenAI, Kuzu, Graphiti, and YAML support.
- [x] Smoke and scenario tests under `tests/` and `scenarios/`.

### 2. Extractor Boundary

- [x] `HTNLaunchPad` for extracted task/goal payloads.
- [x] `ClarificationRequired` for pre-planning HITL.
- [x] Gemini extractor wrapper using structured output.
- [x] Tests for normal extraction and ambiguity handling.

### 3. Planning Spine

- [x] `KortexPlanner` with separate planning views:
  - hierarchical `HierarchicalProblem` for declared HTN metadata
  - classical UPF `Problem` for Pyperplan fallback
- [x] Deterministic expansion of YAML-declared HTN methods.
- [x] Classical state-goal fallback through Pyperplan.
- [x] Multi-goal orchestrator for basic independent goal dispatch.

### 4. Domain Bootstrap and Validation

- [x] YAML domain loader for types, fluents, actions, HTN methods, objects,
  initial state, and goals.
- [x] Manifest validation for unknown types, fluents, action references, method
  subtasks, and bad parameter references.
- [x] Plugin/action signature validation.

### 5. Plugin Registry and Driver

- [x] `PluginRegistry` for primitive Python operators.
- [x] Explicit registry injection through bootstrapper, driver, agent, and
  orchestrator.
- [x] Global registry retained only as a backwards-compatible default.
- [x] `ExecutionDriver` for physical action execution.
- [x] Approval HITL with `requires_approval=True`.

### 6. Tracing and Agent Loop

- [x] `TraceRecorder` and structured `TraceEvent`.
- [x] Top-level `KortexAgent` loop:
  extraction, clarification, hydration, bootstrap, planning, execution, trace,
  memory writeback.
- [x] Driver-level traces for action preparation, success, failure, and HITL
  approval decisions.

### 7. Memory

- [x] Explicit `MemoryFact` model for planner-consumable state.
- [x] `InMemoryFactStore` for deterministic tests.
- [x] `KuzuFactStore` for local persistence.
- [x] `StateHydrator` queries explicit facts and returns bootstrapper-compatible
  state facts.
- [x] Graphiti-backed `MemoryManager` scaffold for episodic digestion.
- [x] `SleepReflector` detects repeated action subsequences and injects a
  semantic HTN meta-task.

### 8. Learning and Novelty

- [x] `IntraDomainLearner` chunks successful Tier 2 action traces into HTN
  methods.
- [x] `SecurityValidator` checks generated Python plugin code.
- [x] Provider-neutral novelty interface:
  - `NoveltyRequest`
  - `NoveltyResult`
  - `NoveltyAgent`
  - `NoveltyBranch`
- [x] `PiNoveltyAgent` default backend in dry-run/testable mode.

### 9. Subagent Enclosure

- [x] Read-isolate-write wrapper for external subprocess/subagent tools.
- [x] Tests for controlled state ingress/egress behavior.

### 10. Scenario Coverage

Implemented scenarios:

- [x] Scenario 1: perfectly specified HTN task, direct execution.
- [x] Scenario 2: goal and primitives exist, classical planner decomposes.
- [x] Scenario 3: same as Scenario 2, but learned chunk executes directly.
- [x] Scenario 4: execution requires human approval.
- [x] Scenario 5: total impasse routes to provider-neutral novelty branch.
- [x] Scenario 6: sleep reflection creates an executable meta-task.

Latest verified test state: `33 passed`.

## Remaining Work

### A. Production Memory Bridge

- [ ] Define how Graphiti-derived episodes become explicit `MemoryFact` records.
- [ ] Add conflict resolution for stale facts.
- [ ] Add temporal scoping: latest fact by entity/fluent, invalidation, and
  confidence/provenance fields.
- [ ] Decide whether Kuzu stores planner facts only, Graphiti data only, or both
  in separate schemas.
- [ ] Add an external knowledge endpoint abstraction for task-specific access to
  outside KGs without merging them directly into agent memory.

#### Memory Architecture Notes

Memory should be split into three explicit layers. These layers may share a
backend, but they should not share semantics.

1. **Conversation Memory**
   - Purpose: context continuity and recovery.
   - Stores raw or lightly structured interaction history:
     - user requests
     - assistant/system responses
     - clarifications asked and answered
     - references to recent entities/tasks
     - conversation state after interruptions
   - This layer is allowed to be narrative and incomplete. It should help the
     agent recover context such as "the thing mentioned two turns ago."
   - It must not directly become planner truth.

2. **Validated Trace Memory**
   - Purpose: evidence set for sleep reflection and metacognitive learning.
   - Stores normalized records of deterministic runs:
     - extracted root task or state goal
     - initial explicit facts
     - planner tier used
     - primitive action sequence
     - action inputs and outputs
     - HITL approvals/denials
     - execution result
     - final explicit facts
     - validation/test status
   - Sleep reflection should learn from this layer, not directly from raw
     conversation memory.
   - Only successful, validated, non-denied traces should be eligible for
     automatic HTN chunking or meta-task synthesis.
   - Implementation note: `SleepReflector` currently accepts raw
     `list[list[str]]` traces. Before deeper Graphiti integration, this should
     be replaced with a `ValidatedTraceMemory` interface so reflection cannot
     accidentally learn from arbitrary prose episodes or failed/denied runs.

3. **Planner Fact Memory**
   - Purpose: current explicit world state for planning.
   - Stores boolean or typed facts that the deterministic planner can consume.
   - Requires provenance, freshness, invalidation, and conflict handling.
   - This is the only memory layer that should hydrate UPF initial state.

Episodic memory is therefore the broad evidence/history substrate, but the
sleep reflector should operate over validated traces derived from episodes.
Planner facts should be promoted from memory only through explicit validation
or deterministic extraction rules.

#### External Knowledge Graph Notes

External knowledge graphs should not be connected directly into agent memory as
if they are the same substrate. They should be isolated knowledge endpoints with
typed query contracts.

Recommended pattern:

```text
Planner/operator needs external knowledge
  -> query KnowledgeEndpoint
  -> validate/transform KnowledgeResult
  -> use result for the current task
  -> optionally cache with provenance, scope, and TTL
  -> promote to Planner Fact Memory only through an explicit adapter
```

External KG facts can inform planning, but they should not automatically become
agent memory or planner truth. Each result should preserve:

- endpoint/source id
- query payload
- timestamp
- provenance
- confidence or authority level
- permitted scopes
- TTL/freshness policy
- whether it may hydrate planner state
- whether it may be cached

This implies another optional memory-adjacent layer:

```text
ExternalKnowledgeEndpoint(s)
ExternalKnowledgeCache
```

The cache is not the same as conversation memory, validated trace memory, or
planner fact memory. It is a scoped, provenance-preserved record of outside
knowledge used by tasks.

### B. Novelty Hardening

- [ ] Add Codex, LangChain DeepAgents, or OpenAI Agents SDK novelty providers
  behind the `NoveltyAgent` protocol.
- [ ] Run `SecurityValidator` automatically on changed/generated plugin files.
- [ ] Run `DomainManifestValidator` automatically after novelty edits.
- [ ] Define hot-reload semantics for newly generated manifests/plugins.
- [ ] Capture changed files in `NoveltyResult`.

### C. HITL Clarification During Execution

- [ ] Define a typed `ClarificationYield` result for primitives/subagents.
- [ ] Teach `ExecutionDriver` how to pause, ask, resume, or abort.
- [ ] Add scenarios for data-discovered ambiguity, such as household vs.
  individual churn modeling after inspecting a dataset.

### D. Stronger Planning Semantics

- [ ] Decide whether true HTN search is required or deterministic method
  expansion is enough for the core product.
- [ ] If true HTN search is required, evaluate a planner that natively supports
  the desired HDDL/HTN features.
- [ ] Keep Pyperplan fallback deliberately simple, or replace it when domains
  need richer PDDL features.

### E. Runtime and Developer Experience

- [ ] Add a CLI or service entrypoint around `KortexAgent`.
- [ ] Add example domain manifests outside test files.
- [ ] Add structured trace export to JSONL.
- [ ] Add strict type-checking and lint commands.
- [ ] Add dependency/update policy for Graphiti, Kuzu, Google GenAI, and UPF.

## Architectural Non-Negotiables

- The LLM extraction layer must stay decoupled from deterministic execution.
- Normal execution operators must not call LLMs for planning/reasoning.
- Novelty providers may write deterministic knowledge, but generated artifacts
  must pass validation before entering the runtime.
- Plugin registries should be injected per runtime/domain. The global registry
  is a convenience default only.
