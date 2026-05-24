# Kortex Core

Kortex Core is an experimental deterministic agent kernel. The central design
constraint is that an LLM may extract intent and parameters, but it must not be
the normal execution planner. Planning and execution are handled by explicit
domain manifests, deterministic planning code, registered Python primitives,
and validation.

## Architecture

The current runtime path is:

```text
user request
  -> Pydantic intent extraction
  -> optional clarification HITL
  -> optional memory fact hydration
  -> domain manifest bootstrap
  -> deterministic planning
  -> plugin-backed execution
  -> structured trace + memory writeback
```

Core modules:

- `kortex.extractor`: Pydantic schemas and Gemini extractor. The extractor
  returns either `HTNLaunchPad` or `ClarificationRequired`.
- `kortex.spine`: deterministic planning and execution spine.
  - HTN methods declared in YAML are expanded locally.
  - Classical state goals are sent to Pyperplan through a plain UPF `Problem`.
  - The hierarchical and classical planning views are intentionally separate.
- `kortex.config`: YAML domain bootstrapper and manifest/plugin validation.
- `kortex.plugins`: explicit `PluginRegistry` for primitive Python actions.
  The global registry remains as a convenience default, but production paths
  should inject a registry per runtime/domain.
- `kortex.agent`: top-level request loop that composes extraction, memory,
  bootstrap, planning, execution, tracing, and memory writeback.
- `kortex.tracing`: structured in-memory trace events.
- `kortex.memory`: explicit planner facts, Kuzu-backed fact store, state
  hydration, Graphiti episode manager, and sleep reflection.
- `kortex.sandbox`: macro chunking, generated-code validation, and
  provider-neutral novelty treatment.
- `kortex.subagents`: black-box subagent enclosure for read-isolate-write
  external tools.

## Planning Tiers

Kortex currently uses three resolution tiers:

1. **Tier 1: Direct HTN method expansion**
   Declared `htn_methods` in the domain manifest expand into primitive action
   plans without involving a search planner.

2. **Tier 2: Classical fallback planning**
   If the request is a state goal, Kortex uses Pyperplan over a plain classical
   UPF problem. Successful traces can be chunked into reusable HTN methods with
   `IntraDomainLearner`.

3. **Tier 3: Novelty branch**
   If deterministic planning cannot solve a goal, Kortex builds a
   provider-neutral `NoveltyRequest`. The default backend is `PiNoveltyAgent`,
   but the interface is designed to support Codex, LangChain DeepAgents,
   OpenAI Agents SDK, or local coding agents.

Novelty providers are instructed to prefer new deterministic HTN methods using
existing primitives. New Python plugins are a last resort when a physical
primitive capability is genuinely missing.

## HITL and Tracing

Kortex supports two HITL categories:

- **Clarification HITL:** the extractor returns `ClarificationRequired` when
  required task parameters are ambiguous or missing.
- **Approval HITL:** plugins can be registered with `requires_approval=True`.
  The `ExecutionDriver` pauses before running those actions.

Trace events cover request receipt, extraction, clarification pauses, memory
hydration, domain bootstrap, goal creation, plan creation, per-action execution,
approval decisions, failures, and completion.

## Memory

The memory system has two different responsibilities:

- `MemoryFact` / `FactStore`: explicit boolean facts that can safely hydrate
  deterministic planner state.
- `MemoryManager` / Graphiti: episodic text digestion for longer-term memory.

`StateHydrator` consumes explicit facts only. It does not ask an LLM to infer
planner state at runtime.

The intended memory design has three layers:

- **Conversation memory:** raw interaction continuity and recovery.
- **Validated trace memory:** normalized successful execution traces used as
  evidence for sleep reflection and meta-task learning.
- **Planner fact memory:** explicit current world facts used to hydrate planner
  state.

Sleep reflection should learn from validated traces, not directly from raw
conversation memory. Planner state should be hydrated only from explicit facts.

Current caveat: `SleepReflector` still accepts raw action-sequence lists. Before
Graphiti is wired more deeply into reflection, this should become a
`ValidatedTraceMemory` interface so procedural learning cannot occur from raw
episodes, failed executions, denied actions, or unvalidated prose.

External knowledge graphs should be accessed as isolated typed endpoints, not
merged directly into agent memory. Endpoint results may inform a task and may be
cached with provenance, scope, and TTL, but they should become planner facts
only through an explicit validation/promotion step.

## Validation

The bootstrapper validates manifests before planning:

- unknown types
- unknown fluents
- bad action parameter references
- bad HTN subtask references
- plugin/action signature mismatches

Generated plugin code can be checked with `SecurityValidator`, which rejects
unsafe constructs such as `eval`, `exec`, and dangerous imports.

## Current Test Coverage

The test suite currently covers:

- extractor schema outputs and clarification
- plugin registry behavior
- domain bootstrap and execution
- planner fallback and orchestration
- HITL approval and trace events
- macro chunking
- memory fact hydration and Kuzu fact persistence
- sleep reflection manifest injection
- subagent enclosure
- novelty provider abstraction
- six end-to-end scenarios

Run:

```bash
.venv/bin/pytest -q
```

Latest verified state: `33 passed`.

## Known Limitations

- HTN support is deterministic method expansion, not full HTN search.
- Pyperplan is a simple STRIPS fallback planner; expressive PDDL features are
  intentionally limited.
- Graphiti integration exists, but the production-grade bridge from Graphiti
  episodes into explicit `MemoryFact` updates is not complete.
- Novelty treatment is provider-neutral and testable, but generated code is not
  yet hot-loaded after validation.
- Mid-execution clarification is not yet a first-class typed yield mechanism.
- There is no CLI or service wrapper around `KortexAgent` yet.
- Type hints exist, but the repo does not yet run a strict type checker in CI.
