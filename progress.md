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
- generic optimizer for candidate/bundle decisions with typed
  optimization-decision memory payloads
- guarded response renderer with template and constrained narration modes

## Latest Verification

```bash
.venv/bin/pytest -q
```

Latest full-suite verified result: `33 passed`.

Latest targeted verification after travel package/config work:
`14 passed` for domain package loading, intent runtime, travel interaction
config, travel demo, response rendering, and optimizer tests.

## Important Caveat

This is not yet a production agent. It is a tested architectural prototype.
The remaining hard work is memory semantics, real novelty provider execution,
hot-loading generated artifacts, mid-execution clarification, and a real runtime
entrypoint.

Next planned step is a config-aware interaction session, likely
`kortex/configured_interaction.py`, that consumes loaded domain packages and
drives multi-turn interaction: greeting, out-of-domain refusal, configured
clarification, clarification resumption, planner execution, optimizer summary,
HITL approval/denial, and guarded final responses.
