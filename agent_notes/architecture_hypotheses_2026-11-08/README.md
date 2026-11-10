# Architecture hypotheses — 2026-11-08

**Local 3D features show a useful synthetic signal, but none of these changes is
established as a better production policy.** The simple uniform-mask control still
beats all twelve trained configurations on average held-out synthetic proxy
quality. The live model implementation and original checkpoints were not changed.

## Findings

| Hypothesis | Result | Interpretation |
| --- | --- | --- |
| One operation-conditioned output reduces wasted work and improves learning | One output is computed. After packing conditioning up front, forward speed differs from dual output by only a few percent in most cases. Final quality varies by seed. | No meaningful speed case for changing the head alone; quality benefit remains unproven. |
| Explicit facade inputs improve decisions | Exact current eligibility channels are supplied and become trainable/used. Paired quality effects change sign across seeds. | Mechanism works; this short run does not establish better quality or completion. |
| Cross-floor 3D features help | Adjacent-floor influence is verified. Compared with 2D, final synthetic proxy quality improves by +0.0209 on average, positive in all three seed-level effects. Additional trained-minus-initial improvement is +0.0163. | Promising on synthetic probes. Added learning benefit on Barcelona is approximately -0.0008 and mixed across seeds; generalization is not established. |
| Multiscale context improves on local 3D | Eight-cell-distant influence is verified. Final synthetic difference is +0.0029 on average, but signs vary across seeds; Barcelona effects are also mixed. | Wider context works mechanically, but its extra compute/capacity is not justified by this run. |
| Combining changes produces a reliable advantage | The single-head/facade interaction changes sign across seeds. The complete multiscale combination has 29/30 FAR completions; every other configuration has 30/30. | No consistent synergy was established. |

These are descriptive effects from three seeds, not statistical significance or
convergence results. H1 combines head sharing with explicit operation conditioning;
those two learning effects are not separately identified. Model widths are fixed,
not parameter counts: 11,267 baseline parameters, 29,987 local 3D, 61,139 multiscale.

## Study size and controls

The full combination matrix contains twelve configurations × three seeds. Each
gets exactly 512 attempted training actions: 64 for each plot/operation pair,
18,432 total. Models train from scratch, using the same A2C algorithm, facade
projection and existing objective. The 176-update imported production checkpoint
is not used as a teacher or experiment initialization.

There are 720 initial/trained model evaluations and 60 control evaluations across
held-out synthetic plots and one cached Barcelona site, split between addition
and subtraction. Trained models reach FAR in 359/360 evaluations. Quality is
compared at the same target cell count and only on common successful cases; pair
counts are retained. The one failed result is never counted as FAR success.

The uniform control averages **0.6710 synthetic quality**, compared with
**0.6148–0.6448** for the twelve trained configurations. On Barcelona, uniform is
0.1261 and the sky heuristic is 0.1346; model averages range 0.1183–0.1401, conditional
on completion. Three paired seeds are too few, and one real site too narrow, to
claim a general advantage. These are uncalibrated environmental proxies.

See [final quality including controls](QUALITY.md), [learning deltas and complete
mask latency](TABLE.md), and [per-seed learning gains](learning-gains.png). A larger
trained-minus-initial delta can reflect a worse random initial model; it is not
by itself evidence of a better final model.

## Runtime and the conditioning follow-up

All 48 L/XL CPU/MPS benchmark cases completed, including actual backward execution,
with MPS fallback disabled. CPU geometry remains CPU geometry. For fixed XL inputs
with facade channels, dual forward medians are approximately 16.93 / 59.35 / 82.36 ms
on CPU for 2D / local 3D / multiscale, and 2.83 / 12.33 / 14.51 ms on MPS.
Complete mask generation also includes transfers, decoding and projection; those
raw medians and p95 values are reported separately in TABLE.md/raw JSON.

The first single-head implementation creates an operation tensor inside every
forward. On MPS it measured 6–43% slower than dual output. A posthoc follow-up
packs the operation with FAR before forwarding, with numerically equivalent
single-head outputs. This removes most of that slowdown. The packed single-head
speed differences relative to dual range from about -0.1% to +1.8% on CPU, and
-0.9% to +6.4% on MPS in that follow-up. These are forward-only measurements,
excluding input packing, not end-to-end gains. See [paired head timings](HEAD_LATENCY.md).

Memory is retained in raw results: cumulative process RSS high-water ranged
approximately 522–873 MB, and MPS current allocations 52–227 MB. These include the
resident environment; cumulative RSS cannot identify each model's isolated peak.
Setup, inputs, projection, application, backward, raw timings and source hashes
are preserved. No browser rendering is timed. No field engine or shadow method
was changed; existing geometry/field regression tests still pass.

## Failure and feature-use diagnostics

The failed `multiscale_single_facade`, seed 456, stopped at 4,353 occupied cells
on Barcelona, target 4,352. Replaying its checkpoint reproduces the failure after
86 actions. A valid final single-cell edit exists, but all eight neural
sampling/projection attempts were rejected. This is a shared generation/constraint
interaction, not proof that the FAR is geometrically unreachable. No fallback or
candidate-mask policy was inserted. See [failure replay](failure-replay.json).

Frozen-checkpoint diagnostics show facade-channel ablation changes eligible-cell
probabilities (median absolute change 0.000132) and flipping explicit operation
conditioning also changes them (median 0.0000106). This establishes sensitivity,
not useful learning; effects remain small after the short budget. Exact values
are in the feature-sensitivity JSON.

## Verification and limits

All **48 Python tests pass**. They include exact baseline equivalence, single
computed output, trainable operation/facade inputs, cross-floor and distant
information paths, equal training exposure, all twelve variants' mask/FAR
invariants, and the existing physics/checkpoint regressions.

This is a bounded architecture screen: small learning fixtures, 512 training
actions per seed, three seeds, one real site, fixed reward/decoder, no
parameter-matched controls or long training curves. The original experiments
used unequal per-condition exposure despite equal total budgets; that pilot was
stopped, preserved, and replaced with the balanced run. Supplemental conditioning,
control replication, feature-use and failure diagnostics are explicitly posthoc.
See [the recorded plan and amendments](PLAN.md) and [full method](METHOD.md).

The next justified research direction is a longer, capacity-controlled local-3D
comparison across more cached sites, retaining simple controls and resolving
residual-mask rejection. The results do not justify automatically replacing the
current production model with the largest combination.

## Artifacts and reproduction

- [Mechanistic input-gradient checks](mechanistic.json) and [test log](tests.txt)
- [Raw factorial results](../../results/architecture-hypotheses-2026-11-08.json)
- [Paired effects and summaries](../../results/architecture-hypotheses-summary-2026-11-08.json)
- [Original interleaved head timings](../../results/architecture-paired-head-latency-2026-11-08.json)
- [Packed-conditioning follow-up](../../results/architecture-packed-condition-latency-2026-11-08.json)
- [Replicated controls](../../results/architecture-controls-replicated-2026-11-08.json)
- [Feature-use diagnostics](../../results/architecture-feature-sensitivity-2026-11-08.json)
- [Preserved preliminary run](../../results/architecture-hypotheses-pilot-2026-11-08.json)
- Final experimental weights: `results/checkpoints/architecture-hypotheses-balanced-2026-11-08/`.
  Preliminary weights are retained separately. These are not production V5 checkpoints.

Run inside VoxelForm-RL with the sibling Python environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m tools.architecture_hypotheses
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m tools.paired_head_latency
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m tools.packed_condition_latency
../Module-Lab/.venv/bin/python -m tools.replicate_architecture_controls
../Module-Lab/.venv/bin/python -m tools.architecture_sensitivity
../Module-Lab/.venv/bin/python -m tools.summarize_architecture_hypotheses
../Module-Lab/.venv/bin/python -m tools.plot_architecture_hypotheses
../Module-Lab/.venv/bin/python -m unittest discover -s tests
```

Uniform-control terminology: probabilities were uniform across eligible cells. Greedy inference used deterministic top-k tie-breaking, with stochastic retries on rejection; this was not a uniformly random-mask baseline. The index-order bias limits interpretation of the earlier comparison.
