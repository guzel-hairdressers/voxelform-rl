# Coherent masks and GPU planar depth — 2026-11-09

Implemented in the preview at http://127.0.0.1:8011. It retains Module Lab's actual
3D engine and fetched surroundings. The preview now uses **smoothed correlated
patches on MPS**, optional facade-plus-depth subtraction eligibility, and atomic
automatic checkpoint saves. Current live settings enable deep cuts, limit edits to
256 cells and four patches, and allow one new interior opening per floor with a
minimum footprint of 4 m². These are adjustable research constraints, not compliance.

## What generates an edit

1. One shared floor-CNN pass produces subtract/add logit fields and a critic value.
   The underlying five-channel CNN is retained to isolate the sampling change.
2. The selected field is pooled to a coarse grid: two floors × 4 m × 4 m strides.
   Its values are the means of a Gaussian latent distribution, with a learned
   per-operation standard deviation.
3. Draw that latent, interpolate it to the voxel grid, and Gaussian-smooth it.
   The Gaussian sigmas are two floors (6 m) vertically and 2 m horizontally.
   Deep-interior scores share their vertical signal to encourage open wells;
   ordinary facade scores retain variation between floors.
4. Multiply sigmoid scores by the frozen eligibility weights. For subtraction,
   `W = max(current_facade, clamp((horizontal_depth_metres-8)/8,0,1))` in deep mode.
   Facade-only mode remains available. Addition remains adjacent envelope growth.
5. The sampled field's weighted area determines an edit count, clipped to the
   remaining FAR budget and configured maximum. Threshold/rank that single field
   into a binary mask. There is no geometric patch library or ranked mask bank.
6. Prune unsafe generated bits and commit the complete retained mask in bulk.
   Recompute depth and facade eligibility before the next action. New facade
   layers are not peeled repeatedly within one action.

Neighbouring decisions therefore share a latent field. Smoothing probabilities
followed by independent Bernoulli sampling remains a control mode; it does not
provide the same spatial correlation. Inference samples the distribution too:
mean-field previews were a retained negative diagnostic, not the final behaviour.

The A2C score uses the **Gaussian latent sample's** log probability. Decoding,
area clipping, eligibility weighting and projection depend on that sample and
state. It does not reuse an independent-Bernoulli score for correlated masks.
When retries are needed, the loss includes exactly the consumed prefix of draws.
Tests verify mean/std gradients and prefix handling. The joint score is scaled by
`1/sqrt(latent_count)` for conditioning, following the existing normalized-score
A2C baseline; entropy regularizes the latent Gaussian, not binary-mask entropy. No convergence claim is made.

## Depth is strictly 2D per floor

For each occupied voxel, measure horizontal Euclidean distance from its centre to
the nearest exterior wall segment **on the same floor**. Roofs, undersides, nearby
floors and three-dimensional diagonal shortcuts never shorten it. The wall-adjacent
cell has centre distance 0.5 m. Badness is zero through 8 m, 0.5 at 12 m, and one
from 16 m onward. Empty cells do not get occupied-floor depth.

The GPU implementation builds a half-cell wall lattice and treats floors as an
independent batch. Within each plane, prefix scans find row-axis wall distances,
then a min-plus reduction supplies exact column-axis distances. All that arithmetic
runs in native torch operations on MPS. CPU SciPy EDT is the reference/CPU option;
MPS checks explicitly reject a CPU-depth fallback. Checked fixtures and the final
real-site snapshot matched the reference with zero error. Independent continuous
wall-segment tests cover concavities, corners, open courtyards and sealed voids.

A courtyard fixture reduces an unchanged voxel's depth from 11.5 m to 3.5 m. A
sealed void does not create daylight frontage; a roof opening only affects floors
where it exposes a lateral courtyard wall. The 3D correlation of **edits** is
separate from this 2D **depth measurement**.

The default quality objective remains unchanged. The depth field supplies the new
edit weights and diagnostics; optional depth reward/input variants remain research
options. This field is a geometric proxy, not measured daylight penetration.

## Connected openings that can later break through the facade

The user clarified: a new interior courtyard starts roof-open and can later join
the side facade. New interior subtraction must connect to exterior air after the
whole action. Sealed pockets are pruned/rejected. Existing courtyard walls enter
the next step's facade mask and can expand progressively toward the outside.

Opening-count checks operate per floor inside the permitted footprint, so several
separate roof holes do not become one opening merely because they share the sky.
The live limit is one interior opening per floor, with at least 4 m² for each new
opening. Existing imported openings are preserved. Merging a courtyard into an
outside bay is allowed and frees its interior-opening count; it is not forced to
remain enclosed forever. A regression test removes the intervening strip in
successive valid facade steps until the court joins the outer edge.

Ordinary retained edit regions have at least four cells and at most four
26-connected regions per action; physical mass and air use face connectivity.
The final quantized FAR residual can be smaller. Minimum footprint and opening
count are separate from smoothness; Gaussian filtering alone cannot guarantee
these properties. Projection only prunes selected edits, never synthesizes
unselected cells to make a well connect.

## Evidence and important negative results

The primary ablation trained **24 models**: four samplers × two eligibility modes
× three seeds, each with 256 attempted training actions, exactly 32 for each of
four plots × two operations (6,144 total). The same network, A2C loss, reward and
safety checks were shared. Small training plots have no >8 m interior. Barcelona
has maximum depth about 9.19 m and only 42 deep cells, so a separate wide plate
was necessary to exercise interior cuts.

- Fixed edit-count L probes: independent sampling averaged 37.8 regions, correlated
  sampling 8.2, and correlation plus Gaussian smoothing 4.6.
- XL probes: 104.3 → 20.9 → 10.7 regions. Smoothing a constant probability field
  followed by independent draws is identical to the independent control.
- Actual sampled held-out trajectories averaged **19.35 regions per edit for the
  independent control versus 3.26 for combined sampling**. Mean region size rose
  from 1.15 to 5.53 cells. All **240/240 sampled FAR evaluations** completed. These
  ablations used min1/unlimited region count, isolating sampling from hard guards.
- Greedy-first/mean-field trajectories did not preserve that advantage. Those
  initial/trained results remain in the primary JSON; one deep-independent trained
  evaluation failed near FAR. They were not discarded or represented as sampled
  inference results.
- A broad held-out 4×40×48 plate reaches 19.5 m depth. Sampled depth-enabled cuts
  create full-height interior void columns; strict facade mode cannot start those
  interior cuts. Stronger footprint/count guards completed all three wide tests
  and all six real-site addition/subtraction tests. Each wide floor had one opening;
  observed footprints ranged from 11 to 72 m². No wide-scene fine tuning was used.
- The final GPU preview reached FAR 3 in **97 masks / 4,438 changed voxels / 3.87 s
  active time** on cached `m_bcn_0093`. One independently checked action changed
  39 cells in four patches. Final depth matched an independent CPU calculation,
  and inference left all model weights unchanged.

These are structural/speed results and short learning probes, not evidence of
improved architectural quality, daylight compliance or generalization. Quality
scores and failures remain in [sampled results](SAMPLED_RESULTS.md). Larger patches
are not by themselves better buildings. See [the coherence figure](coherence.png),
[recorded plan/amendments](PLAN.md), and raw data below.

## What runs on the GPU

MPS handles the network, coarse latent sampling, interpolation, Gaussian kernels,
eligibility weighting, threshold/ranking, exact same-floor depth, and the existing
tensor environmental fields. Cached Gaussian kernels and stable 256-cell ranking
buffers avoid repeated setup. The fast path draws once; if rejected, the remaining
seven draws are decoded in one on-device batch. No per-draw confidence maps or
scalar voxel counts are transferred to the host.

CPU NumPy/SciPy still provides exact exterior-air reachability, mass connectivity,
ground anchors and per-floor opening safeguards. A small eligibility mirror and
selected masks cross the device boundary for those checks. This is deliberately
not claimed to be a fully GPU-resident simulator. There is no Python voxel-deletion
loop. Static scene import and UI serialization also remain host work.

Twenty warmed fixed-state measurements put adaptive MPS mask generation at
**15.32 ms (L) / 10.80 ms (XL)** in this run, with no failed draws and zero depth
reference disagreement. Apply/depth refresh, cold costs, backward updates and
memory are separate in [timings](TIMINGS.md). CPU is faster on the small fixture;
GPU execution is not automatically a latency win. A blind eight-draw batch was
slower when the first draw was already valid, so the final implementation uses
one draw plus a conditional seven-draw batch. Cold setup is not hidden in warm
medians. Rendering is not included in kernel timings.

Device selection now prefers available CUDA/MPS, otherwise CPU. The Mac preview
uses actual MPS with fallback disabled. `VOXELFORM_DEVICE=cpu` remains available.
Only CPU and MPS were measured here; no CUDA performance claim is made.

## Checkpoints, preservation and recovery

`voxelform.patch.v1` stores Gaussian parameters, sampling/eligibility/guard settings,
weights and optimizer. Older V1–V5 policies retain compatible CNN/head/critic
weights as initialization; optimizer semantics and the new sampling update count
reset. Imported historical updates are shown separately. Original files remain
under ignored `results/checkpoints/pre-coherent-2026-11-09-*` and other historical
backups; the demonstration's original geometry is also backed up.

The previous browser tab displayed 200 newer mask updates, but its backend was no
longer listening and no matching saved policy was found in the project/Downloads.
This preview resumes the available saved checkpoint (176 historical updates), not
those unavailable in-memory updates. That limitation is recorded rather than
presented as successful weight recovery.

Automatic saves now write policy/optimizer and geometry/settings to a new generation,
then atomically advance `results/checkpoints/autosave/latest.json`. Completed
optimizer updates, pauses and explicit edits/settings changes are saved. Normal
startup restores the latest pair, paused; explicit resume environment variables
still override it. Tests recover weights, Adam state, mass and settings, including
actual MPS optimizer save/load/update. Partial rollout graphs are preserved during
in-process pause, but are not serialized as autograd graphs across process restarts.

All **80 Python tests pass**, including actual MPS execution, no CPU-depth fallback,
2D floor independence, consumed-prefix policy gradients, sampling caps, opening
limits, courtyard-to-facade merging, checkpoint migration and autosave recovery.
The browser depth layer, patch controls, FAR completion and original graphic engine
were checked with no console errors. See [the preview](preview.png) and
[live verification](live-verification.json).

## Raw data and reproduction

- [Primary sampler ablation / greedy diagnostic](../../results/coherent-masks-2026-11-09.json)
- [Actual sampled evaluations](../../results/coherent-sampled-2026-11-09.json)
- [Wide plate / greedy diagnostic](../../results/coherent-wide-2026-11-09.json)
- [Opening footprint/count follow-up](../../results/coherent-guarded-2026-11-09.json)
- [Eight-draw batching profile](../../results/coherent-gpu-2026-11-09.json)
- [Final adaptive GPU profile](../../results/coherent-gpu-adaptive-2026-11-09.json)
- [Descriptive summary](../../results/coherent-summary-2026-11-09.json)
- [Test log](tests.txt), [final source manifest](final-source-manifest.json).

Measured earlier implementations are retained as source snapshots beside this
report. Primary hashes refer to that measurement phase; later optional safeguards,
GPU field computation and batching have their own measurements/hashes. Historical
ablation/field timing files are not silently relabelled as the final pipeline.

Run from VoxelForm-RL using the sibling Python environment:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m tools.coherent_masks
../Module-Lab/.venv/bin/python -m tools.coherent_wide
../Module-Lab/.venv/bin/python -m tools.coherent_sampled
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m tools.coherent_guarded
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m tools.coherent_gpu
../Module-Lab/.venv/bin/python -m tools.report_coherent_masks
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m unittest discover -s tests
PYTORCH_ENABLE_MPS_FALLBACK=0 ../Module-Lab/.venv/bin/python -m uvicorn voxelform.server:app --host 127.0.0.1 --port 8011 --no-access-log
```
