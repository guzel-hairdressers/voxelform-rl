# M4 training speed, rewards and facade light — 2026-11-09

The measured training throughput is feasible for short iterative experiments on
this **Apple M4 / 16 GB** machine. It is not evidence that architectural quality
has converged. The current reward can in principle improve environmental scores
across training episodes, but stronger weights/curriculum did not consistently
improve held-out quality in the one-minute pilots. Reaching FAR still ends the
current episode, so it cannot subsequently move floor area into taller towers.

See [measurements](MEASUREMENTS.md), [experiment plan](PLAN.md) and
[the new redistribution requirement](REDISTRIBUTION.md).

## Implemented exact speed changes

- `Environment.precompute_external()` prepares fixed surroundings visibility for
  all possible allowed receiver faces on the selected device. Scene/sun setup is
  immutable per environment. Existing cached queries remain valid across edits
  and episode resets. Self shadows, planar depth and diffuse sky fields remain
  dynamic. No simulation resolution or sky/sun audit fidelity was reduced.
- FAR-only retargeting reuses already-current environmental fields. If geometry
  has stale fields, it still audits. Depth is unchanged by retargeting.
- A headless, time-bounded trainer keeps a small pool of prepared plots, uses the
  same network/A2C/masks, checks held-out scenes, writes separate experimental
  checkpoints and can resume one arm. Minute checkpoints use atomic replacement.
  Existing result tags cannot be overwritten accidentally. It does not replace
  or reset the live preview model.

Warm real-Barcelona training episodes, including backward updates, fell from
**1.42 to 0.76 s on MPS** and **1.75 to 0.50 s on CPU**. Preparing the complete
external cache costs **18.2 s on MPS / 33.6 s on CPU**, in addition to scene setup,
and adds about 1.3 MB for this cache. Amortize it over repeated training episodes;
it is not appropriate to block every one-off preview with this preparation.
The original lazy cache also warms over longer runs. See raw timing breakdowns,
memory and cold costs before extrapolating.

## Diffuse light correction

The original `light` array is an **air-cell sky-diffusion field**; it is zero in
solid cells by construction. That is a storage convention, not evidence that all
facades receive zero/equal light. Where the grid contains air outside a facade,
the original local CNN can see some nearby context through those air values.
The earlier description that it was completely blind to surroundings was too
broad. A fully solid grid without air padding can indeed present identical
observations despite different boundary exposure; there is a regression fixture
for that narrower aliasing case.

New `facade_sky_access` samples the diffuse field on the **air side of each
lateral facade** and retains four oriented face fields plus a per-cell facade
mean. It uses the existing sky-openness boundary at grid edges. Nearby buildings
occlude the sky boundary, and open wells/crevices attenuate access through the air
field. Roofs are excluded from this facade channel. No additional ray casts are
needed per edit. It is still an approximate sky-access proxy, not lux.

Direct sunlight is separate: sampled solar directions, neighbour occlusion,
self-shadowing and incidence angles. The direct-sun feature in the earlier
eight-channel input trial does **not** substitute for diffuse light. Its separate
`voxelform.environment_patch.v1` checkpoints remain archived as that experiment.
The corrected nine-channel experimental actor (`quality-facade`, checkpoint
format `.v2`) adds diffuse facade sky access explicitly. Its inputs are occupancy,
air sky access, noise prior, allowed volume, height coordinate, planar depth
badness, edit eligibility, fixed direct-sun potential, and diffuse facade access.
The new input weights start at zero; other spatial weights are retained.

Tests confirm: unobstructed rectangular facades receive equal open-sky values;
a nearby wall reduces its facing-side value (~0.56 vs ~0.89 opposite in a fixture);
the bottom of an open well is darker than its top; changing solar sample count
does not change diffuse access; GPU/CPU facade sampling agrees. The corrected
actor executes an actual MPS optimizer update and saves/loads. It has not yet
undergone a quality-learning comparison, so no quality gain is claimed.

## Short learning experiments

Two seeds × 60 seconds per arm, with cached wide/irregular/Barcelona training
scenes and three held-out scenes:

| Arm | Actions/min | Held-out FAR successes |
|---|---:|---:|
| Current reward, mixed difficulty | 1,320 | 6/6 |
| Stronger quality reward, mixed difficulty | 1,186 | 6/6 |
| Stronger quality reward, curriculum | 1,112 | 5/6 |

Quality deltas were small and inconsistent. Curriculum has not earned preference
over mixed training. Those rates suggest roughly 8–10 minutes per 10,000 actions
on this particular training mix; all-XL training is slower and convergence time
remains unknown. The eight-channel/direct-sun-only follow-up ran at ~311 actions
per minute with no clear quality gain and has not been adopted. Its preparation
overlapped GPU correctness tests, so its setup times are not comparable; it needs
isolated profiling before interpreting the throughput difference.

No overnight/convergence claim follows from these pilots. Before spending more
training time, the action space must allow the requested redistribution: removal
and addition at the same FAR, a taller permitted envelope than initial occupancy,
and more than one grounded building. Optional whole-top-storey duplication is a
promising growth primitive. See the paired-edit experiment in REDISTRIBUTION.md.

## Reproduction and continuation

The first three commands reproduce timing/pilot structure using current code;
historical results retain their original hashes and source snapshots. Choose a
new `--tag` to preserve existing measurements.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.training_profile --tag fresh-baseline
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.training_profile --precompute --tag fresh-prepared
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.train_m4 --seconds 60 --tag fresh-pilot
# One bounded, corrected-input research arm; separate from the live preview:
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.train_m4 --arms quality-facade --seeds 123 --seconds 600 --tag facade-10min
```

`--resume PATH` resumes one selected arm/seed with matching edit settings. Extra
input checkpoints are deliberately experimental and cannot be silently loaded
into the production five-channel actor. Current production reward/action semantics
and saved preview weights were preserved. No new long training run was launched.

107 Python tests pass, including MPS execution with fallback disabled. Source
hashes, measured source versions, all raw failures, independent checkpoint replay,
separate per-metric evaluations and experimental checkpoints are retained. A
standalone profiler failure-path reward/audit ordering issue was corrected after
the timing run; raw historical data is preserved, and those medians are throughput
measurements rather than exact GUI-session behavioral reproductions.
