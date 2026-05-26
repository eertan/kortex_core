# Progress

## Current Status

Kortex Core has a working MVP skeleton for deterministic agent execution:

- structured extractor schemas
- split HTN/classical planning spine
- YAML domain bootstrap and validation
- named intent bindings for order-independent task/goal parameters
- interaction session shell with deterministic policy, Gemini structured
  interpreter, clarification resumption, and pre-response guard
- explicit plugin registries
- execution driver with approval HITL
- top-level agent loop
- structured tracing
- explicit memory fact hydration with Kuzu support
- uniform memory record envelope, runtime working-memory state wiring, and
  validated trace record writeback
- macro chunking with inferred skill preconditions/effects
- provider-neutral novelty branch
- sleep reflection scaffold
- subagent enclosure
- multi-file domain package loader for `domain.yaml`, `intents.yaml`,
  `decisions.yaml`, and `responses.yaml`
- config-driven intent frame construction for slot validation, clarification,
  normalization, and preference-token extraction
- config-aware multi-turn interaction session for domain packages, including
  conversation turns, out-of-domain refusal, configured clarification,
  clarification resumption, planner execution, HITL pause reporting, and
  resumable approval/denial branches
- generic optimizer for candidate/bundle decisions with typed
  optimization-decision memory payloads
- structured optimizer plugin outputs that the configured interaction session
  records, summarizes, and surfaces before HITL approval
- guarded response renderer with template and constrained narration modes
- travel interaction CLI for scripted and interactive
  `ConfiguredInteractionSession` demos
- config-derived Gemini turn interpreter for domain packages. The LLM-facing
  output model uses Gemini-compatible slot/value pairs and is validated back
  into internal `ConfiguredTurnInterpretation` objects before planning.

## Latest Verification

```bash
.venv/bin/pytest -q
```

Latest full-suite verified result: `33 passed`.

Latest targeted verification after Gemini configured interpreter work:
`27 passed` for the configured Gemini interpreter, travel interaction CLI,
configured interaction, domain package loading, intent runtime, travel
interaction config, travel demo, response rendering, and optimizer tests.

Live Gemini check passed with:

```text
I want to visit Tokyo from Boston for 3 days next week under 2000 dollars
```

The interpreter produced a `plan_trip` task and normalized planner parameters:
`origin=boston`, `destination=tokyo`, `duration_days=duration_3_days`,
`travel_window=next_week`, `budget=budget_2000`, `style=relaxed`.

## Important Caveat

This is not yet a production agent. It is a tested architectural prototype.
The remaining hard work is memory semantics, real novelty provider execution,
hot-loading generated artifacts, mid-execution clarification, richer
config-driven response rendering, and a production runtime entrypoint.

Next planned step is to decide whether the next demo surface should be a tiny
HTTP service/UI or whether to harden the CLI transcript, response policies, and
memory record display first.
