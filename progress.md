# Progress

## Current Status

Kortex Core has a working MVP skeleton for deterministic agent execution:

- structured extractor schemas
- split HTN/classical planning spine
- YAML domain bootstrap and validation
- explicit plugin registries
- execution driver with approval HITL
- top-level agent loop
- structured tracing
- explicit memory fact hydration with Kuzu support
- macro chunking with inferred skill preconditions/effects
- provider-neutral novelty branch
- sleep reflection scaffold
- subagent enclosure

## Latest Verification

```bash
.venv/bin/pytest -q
```

Latest full-suite verified result: `33 passed`.

Latest targeted verification after skill-contract learning update:
`9 passed` for chunking plus scenario tests.

## Important Caveat

This is not yet a production agent. It is a tested architectural prototype.
The remaining hard work is memory semantics, real novelty provider execution,
hot-loading generated artifacts, mid-execution clarification, and a real runtime
entrypoint.
