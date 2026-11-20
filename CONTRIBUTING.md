# Contributing research changes

1. Read `AGENTS.md`; keep this ML repository independent of Module Lab. Preserve
   existing uncommitted work and user weights.
2. State a falsifiable hypothesis, its baseline, budget, seeds, split and primary
   metrics before an experiment. Separate hard constraints, decoder heuristics,
   auxiliary supervision and learned policy behaviour.
3. Use a new output directory and versioned config. Retain failures and complete
   traces. Report setup/training/evaluation time separately and disclose hardware.
4. Do not equate a single learning curve, best episode, attractive render or a
   nonzero gradient with convergence. Do not compare raw scores across different
   task rewards as though they were the same metric.
5. Run locked-environment unit/geometry tests, JS tests and the synthetic pipeline.
   Add independent numerical references or invariant tests for meaningful changes.
6. Commit only reviewable source, configs, documentation and compact evidence.
   `runs/`, `.venv/`, caches and private weights stay local. Publish external data
   only with its provenance/licensing accounted for. The project's own license
   still requires an explicit owner decision.

See `docs/EXPERIMENT_PROTOCOL.md` for artifacts, resume guarantees and known gaps.
