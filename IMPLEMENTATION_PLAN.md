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
- [x] Manifest-level `intent_bindings` map named extractor parameters into HTN
  task invocations or classical fluent goals, so runtime planning no longer
  depends on positional `args` for configured domains.
- [x] `InteractionSession` shell above `KortexAgent`:
  - persists user/assistant turns as `ConversationMemoryPayload` records
  - applies deterministic interaction policy before task execution
  - routes allowed task turns into the agent loop
  - re-runs pending tasks with the original prompt plus clarification answers
  - runs a conservative pre-response guard to prevent execution overclaims
- [x] Optional `GeminiInteractionInterpreter`:
  - strict `InteractionInterpretation` schema
  - defaults to `gemini-3.1-pro-preview`
  - reads `GOOGLE_AI_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY`
  - interprets/classifies turns but has no execution, approval, or planner-fact
    authority
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
  methods with inferred typed parameters, external preconditions, net effects,
  and provenance.
- [x] Competing HTN method selection:
  - multiple methods may target the same abstract task
  - hard preconditions filter applicability
  - extracted preference tokens and `selection_priority` score candidates
  - selected method is recorded in working memory and trace payloads
  - unresolved equal candidates return a typed tie impasse
- [x] Unordered HTN subtask planning:
  - `ordered_subtasks` remains fixed procedural expansion
  - `subtasks` declares unordered primitive milestones
  - the selected HTN method creates a bounded classical planning problem using
    only its declared primitive actions
  - Pyperplan orders those actions from preconditions/effects
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
- [x] Scenario 2: goal and primitives exist, classical planner decomposes and
  saves a condition-based learned skill.
- [x] Scenario 3: same request as Scenario 2, but learned chunk executes
  directly when its inferred preconditions hold.
- [x] Scenario 4: execution requires human approval.
- [x] Scenario 5: total impasse routes to provider-neutral novelty branch.
- [x] Scenario 6: sleep reflection creates an executable meta-task.
- [x] Travel concierge demo package:
  - shared scenario harness
  - multi-file travel domain package loaded by `DomainPackageLoader`
  - `domain.yaml` planner/HTN manifest
  - `intents.yaml` interaction intent catalog
  - `decisions.yaml` optimizer policy catalog
  - `responses.yaml` guarded response policy catalog
  - fake travel plugin registry
  - preference-selected HTN method
  - unordered subtask ordering through Pyperplan
  - generic optimizer step for candidate bundle selection
  - typed optimization-decision memory record
  - guarded response rendering for optimizer summaries before HITL approval
  - HITL approval for refundable holds
  - structured JSON log output

Latest full-suite verified test state before current increment: `33 passed`.
Latest focused verification after domain-package loader integration:
`10 passed` for `tests/test_domain_package.py`, `tests/test_responses.py`,
`tests/test_optimizer.py`, and `tests/test_travel_concierge_demo.py`. Legacy
runner smoke command also passed after harness extraction:
`.venv/bin/python -m scenarios.run_demo --scenario 1 --log-path demo_logs/scenario_demo_smoke.json`.

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

Representation rule: use a uniform `MemoryRecord` envelope for governance,
retrieval, provenance, confidence, authority, lifecycle, and audit metadata,
while keeping specialized typed payloads for conversation turns, validated
traces, planner facts, procedural skills, semantic entities, and external
knowledge results.

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

#### Cognitive Architecture Notes

Kortex already has cognitive-architecture-shaped subsystems, but they need to
be made first-class and governed.

1. **Procedural Memory**
   - Current representation:
     - domain HTN manifests
     - primitive action specs
     - learned chunks from `IntraDomainLearner`
     - intrinsic orchestration flow in `KortexAgent`
   - Procedural memory should be treated as durable, inspectable knowledge, not
     just YAML loaded by the bootstrapper.
   - Learned procedures should retain:
     - typed parameters
     - preconditions
     - effects
     - provenance/source traces
     - success/failure history
     - confidence or utility
     - promotion/deprecation status
     - approval requirements for risky procedures
   - Current status: learned chunks now preserve inferred parameters,
     preconditions, effects, and provenance. Remaining work is lifecycle,
     utility/confidence, negative examples, and competing-method selection.

2. **Working Memory**
   - Current gap: working state is spread across extractor output, planner
     objects/facts/goals, trace recorder, HITL state, memory hydration, and
     orchestration locals.
   - Next architecture step: define a typed unified working-memory model for:
     - active user/entity/session
     - current goal stack
     - active task and bindings
     - explicit planner facts
     - assumptions and unresolved variables
     - selected planner tier
     - HITL/clarification state
     - trace/event references
   - This should become the shared cognitive state that retrieval, planning,
     execution, reflection, and novelty operate against.
   - Current status: `WorkingMemoryState` exists with explicit planner-fact
     hydration rules. `KortexAgent.run` now creates one per request, records
     extraction/task/entity context, promotes validated hydrated facts into
     active planner state, tracks planner tier, records trace references,
     applies declared action effects after execution, emits a typed
     `ValidatedTracePayload` memory record, and returns it on `AgentRunResult`.
     The orchestrator and scenario demo also project declared action effects
     into working memory. Remaining work is persistent working-memory snapshots,
     broader runtime APIs, and Graphiti persistence for typed records.

3. **Metacognition**
   - Current offline form: sleep reflection over traces to synthesize reusable
     meta-tasks.
   - Current online forms: planning impasse routing, HITL gating, and
     clarification stops.
   - Remaining work:
     - explicit policy for when to ask, search, use memory, execute, reflect,
       route to novelty, or refuse
     - confidence/utility checks before using learned skills
     - validation before promoting reflected skills
     - background sleep/reflection over `ValidatedTraceMemory`
     - audit records for all self-modification proposals

The target framing is:

```text
conversation/semantic/episodic memory provide context and evidence
working memory holds the current cognitive state
procedural memory proposes condition-based operators and HTN methods
planner/executor applies them deterministically
metacognition regulates uncertainty, learning, novelty, and promotion
```

#### Reinforcement Learning Expansion Notes

RL should be treated as control knowledge, not operator knowledge. It may rank
or prioritize choices, but it must not bypass symbolic preconditions,
validation, HITL gates, or deterministic execution.

Promising integration points:

1. **HTN Method Selection**
   - When multiple methods can satisfy the same abstract task, use a learned
     policy or contextual bandit to rank applicable methods.
   - Input features can include:
     - active working-memory facts
     - task name and bindings
     - user/entity context
     - method provenance
     - prior success/failure history
     - cost/latency/risk metadata
   - Output is only a ranking or preference. The planner still verifies method
     preconditions before execution.

2. **Planner Tier Selection**
   - Learn when to prefer direct HTN expansion, classical planning, memory
     lookup, clarification, HITL escalation, novelty routing, or refusal.
   - This would make the metacognitive control policy adaptive while keeping
     each tier deterministic and auditable.
   - Reward signals may include successful completion, avoided unnecessary
     clarification, user correction, HITL denial, execution failure, latency,
     and novelty validation outcome.

3. **Learned Skill Utility and Confidence**
   - Add utility/confidence fields to learned procedural memory:
     - success count
     - failure count
     - average cost
     - average latency
     - last validated timestamp
     - confidence score
     - utility score
   - Update these values from validated execution outcomes.
   - Use them to choose among competing learned skills and to decide whether a
     skill should stay draft, become promoted, require revalidation, or be
     deprecated.

Recommended first implementation is not full deep RL. Start with transparent
utility updates or contextual bandits over validated traces, because those fit
the current HTN/planner architecture and are easier to audit.

#### Domain Package and Intent Config Notes

For a general-purpose Kortex harness, domains should eventually be packaged as
multiple coordinated config files rather than one overloaded manifest.

Recommended domain package shape:

```text
domains/<domain_name>/domain.yaml
domains/<domain_name>/intents.yaml
domains/<domain_name>/memory_policy.yaml
domains/<domain_name>/plugins.py
```

Responsibilities:

1. **Task/domain manifest**
   - types
   - fluents
   - primitive actions
   - HTN methods
   - plugin/action bindings
   - approval requirements

2. **Intent/interface manifest**
   - user-facing intents
   - descriptions and examples
   - required and optional slots
   - slot types and normalization rules
   - clarification prompts
   - named parameter mappings into HTN tasks or fluent goals
   - synonyms/aliases
   - out-of-domain boundaries

3. **Memory policy manifest**
   - which memory layers to query for each intent
   - retrieval hints, such as user preferences or semantic domain facts
   - promotion rules from retrieved context to planner facts
   - cache/provenance/freshness rules

The extractor/interpreter should be configured from this package. It should not
guess task schemas from unstructured prose. The LLM receives available intents,
slot schemas, clarification rules, examples, and domain boundaries, then returns
validated structured output. Deterministic policy, intent bindings, and the
planner remain authoritative.

Current status: `intent_bindings` live inside the domain YAML for MVP
simplicity. Future work should split them into an explicit `intents.yaml` while
keeping backward compatibility.

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

- [x] Add first-class competing-method selection for deterministic HTN
  expansion, including applicability filtering, preference scoring, selected
  method traceability, and tie impasse reporting.
- [ ] Decide whether true HTN search is required beyond deterministic
  competing-method selection.
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

### F. Config-Aware Interaction Session

Status: initial implementation exists in `kortex/configured_interaction.py`
with focused tests in `tests/test_configured_interaction.py`. Existing
`InteractionSession` remains available, while `ConfiguredInteractionSession`
is the package-driven runtime surface.

Implemented responsibilities:

1. **Package Loading**
   - Accept a `DomainPackage` from `DomainPackageLoader`.
   - Use:
     - `domain.yaml` for planner/HTN bindings
     - `intents.yaml` for scope, slots, clarification, normalization, and
       preference tokens
     - `decisions.yaml` for optimizer policies
     - `responses.yaml` for deterministic templates and guarded narration

2. **Turn Classification**
   - Handle conversation-only turns without planner execution.
   - Enforce domain scope from `intents.yaml`.
   - Return configured out-of-domain responses via `responses.yaml`.
   - Keep blocked unsafe directives deterministic.

3. **Intent Frame Construction**
   - Use `IntentFrameBuilder` to build canonical intent frames.
   - If required slots are missing, return `IntentClarification` and store
     pending clarification state.
   - On clarification answer, merge new slot values with the pending frame and
     rebuild the frame.
   - Normalize slot values into planner object names before planning.

4. **Planning/Execution Bridge**
   - Convert complete `IntentFrame` objects into planner goals through
     `domain.yaml` `intent_bindings`.
   - Pass `IntentFrame.preference_tokens` into HTN method selection.
   - Preserve deterministic separation: LLMs may only interpret/extract, never
     choose planner actions, approve actions, or mutate planner truth.

5. **Response Rendering**
   - Configured out-of-domain responses are rendered through
     `ResponseRenderer`.
   - Optimizer-backed plugins can return `OptimizationExecutionOutput`; the
     configured session records the decision and renders the configured
     optimizer summary before HITL approval.
   - Execution responses are conservative deterministic strings for now.
   - Remaining work: build richer `ResponseFrame` objects for clarification,
     planner impasse, HITL approval/denial, and successful completion.

6. **Memory and Trace**
   - Persist conversation turns as `ConversationMemoryPayload`.
   - Preserve trace stages for intent frame construction, planning, execution,
     and HITL pause states.
   - HITL approval state is now session-owned: configured execution pauses
     before an approval-gated primitive, stores the pending plan position, and
     resumes or denies from a later user turn without replanning.
   - Structured optimizer outputs are captured as
     `OptimizationDecisionPayload` memory records.
   - Remaining work: make external option-memory writeback fully generic
     instead of travel-demo-specific.

Initial travel interaction tests cover:

- greeting/conversation-only turn
- out-of-domain refusal
- missing origin/budget clarification
- clarification answer and resumption
- travel planning run through planner/executor until HITL approval is required
- approval branch resumes the pending plan and can complete after multiple
  gated actions
- denial branch stops before the gated primitive
- optimizer summary before HITL through the generic configured session

Open implementation choice: whether to adapt the existing `InteractionSession`
after this layer stabilizes, or keep `ConfiguredInteractionSession` as the main
public runtime surface for domain packages.

## Architectural Non-Negotiables

- The LLM extraction layer must stay decoupled from deterministic execution.
- Normal execution operators must not call LLMs for planning/reasoning.
- Novelty providers may write deterministic knowledge, but generated artifacts
  must pass validation before entering the runtime.
- Plugin registries should be injected per runtime/domain. The global registry
  is a convenience default only.
