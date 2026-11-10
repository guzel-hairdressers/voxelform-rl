# Architecture hypotheses — preregistered before results

Live code/policy unchanged. Experimental code uses the production facade constraint,
mask sampler/decoder, reward, A2C optimizer, 32-action rollout and eight-action audit.
No checkpoint migration or replacement in the live server.

Full factorial: output {dual, operation-conditioned single} × explicit facade
inputs {off,on} × encoder {original 2D, local 3D, multiscale 3D}: 12 configurations.
The multiscale comparison is against local 3D, separating cross-floor information
from wider spatial context. All use 24 feature channels; parameter counts differ
and are recorded (this is not a parameter-matched architecture experiment).

H1: One conditioned head reduces output computation without harming held-out
completion/quality. The shared trunk dominates, so a large speedup is not assumed.
H2: Explicit removal/addition eligibility channels improve learning efficiency,
completion or constraint acceptance. Hard constraints remain identical.
H3: 3D features enable cross-floor dependence (mechanistic test), and may improve
held-out quality; merely enabling dependence is not evidence of better learning.
H4: Multiscale features enable longer-range dependence and may improve held-out
quality. They cost extra compute/capacity, explicitly reported.
Interactions: test whether single-head+facade input benefits depend on 3D/context.

Three paired training seeds (123,456,789), 512 attempted actions each, CPU one
thread. All variants train from scratch; common baseline weights copied where
shapes permit. No imported 176-update policy is used. Balanced removal/growth,
four plot shapes and two environmental contexts. Training environment dimensions
(4,12,16) are small learning probes, not throughput evidence. Eight-action audits,
32-action optimizer rollouts, nonterminal bootstrap. Time/episode caps are failures
or truncations, never FAR successes. Equal actions, record actual updates/time.

Evaluate the same frozen initial and final models, deterministic decoding (with
production stochastic retry if needed), on held-out synthetic shapes/seeds and
cached real Barcelona geometry. Record operation-separated completion, final
quality at matched FAR only, edits, rejection/projection, latency and paired
trained-minus-initial differences. Retain every failed/capped episode. Simple
uniform-probability and fixed-confidence controls contextualize the trained results.

Independently benchmark L (4,48,64; eight suns) and XL (8,64,96; twelve suns), 16
context buildings, actual CPU and available MPS, with device synchronization,
setup, inputs, forward, complete generation/application, backward, parameter/memory
counts and raw timings. CPU topology stays on CPU; no GPU-residency claim.

Decision: isolated/interaction results are exploratory across three seeds, without
claims of convergence, significance or real-site generalization. Do not promote a
variant on speed or one favourable example alone. Publish null/negative results.

## Supplementary measurement added before hardware results

The end-to-end generation benchmark changes geometry between samples, so mask
selection/topology can confound the tiny final-layer cost. Also run 30 randomized,
interleaved dual/single forward pairs on identical frozen L/XL inputs for every
encoder/facade configuration, CPU and MPS. Exclude input preparation/topology from
these paired timings. Preserve raw samples; small percentage changes may be noise.

Implementation detail/deviation: training uses four distinct scene seeds, one per
plot shape, rather than two context seeds per shape. Addition/subtraction episodes
alternate; their valid action counts need not be equal. The total attempted-action
budget is identical, and per-operation episodes/updates are retained. All variants
share this exact schedule. Architecture widths are held fixed, not parameter counts.

## Balanced final run amendment

The preliminary run was stopped after 26/36 models; its raw data and source are
retained as `results/architecture-hypotheses-pilot-2026-11-08.json` and
`pilot-source.py.txt`. Equal overall budgets allowed different per-operation and
per-scene exposure because episode lengths differ. The final run allocates exactly
64 attempted actions to each of four plots × two operations (512 total), restarting
an episode within its condition if necessary and bootstrapping condition-boundary
truncations. Every condition's action count is saved. This change controls exposure;
it is not a change of objective or a selection of favourable results. Final and
pilot checkpoints use separate directories. Three-seed outcome comparisons use
only the final balanced run.
