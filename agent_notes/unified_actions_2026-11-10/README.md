# Unified removal/growth action pilot — 2026-11-10

The new research policy has no correction-then-expansion phase. The recommended
`--net-guided` variant allows paired facade removal/roof growth from the first
step while directing net floor-area change toward the FAR band. The first fully
free-operation variant failed FAR; that failure and subsequent ablations are
retained below. This is separate from the
user's currently running live FAR-only policy; it has not replaced its weights,
checkpoint protocol or six-storey envelope.

## Action and learning contract

- One actor forward pass returns both dense spatial fields, the value estimate
  and four operation logits. A categorical operation is learned by A2C alongside
  the correlated Gaussian latent fields. Only consumed channels and rejection
  retries contribute their Gaussian scores; all operation draws are scored.
- The operation head starts with **learnable weights** favouring FAR correction
  (current/target normalized FAR weights ±8). This is initialization, not a
  hard operation switch: growth and exchange retain nonzero probability above
  FAR, and removal/exchange remain available below it. Finish is masked outside
  the accepted FAR band. There is no scheduled operation sequence.
- The ten observation channels are occupancy, air diffuse field, noise,
  allowed envelope, floor elevation, same-floor depth badness, removal weights,
  growth eligibility, immutable external direct-sun potential, and separate
  diffuse facade access. Goal and remaining action budget are also inputs.
- Roof growth is one frozen layer above current occupied cells plus ordinary
  lateral growth, bounded by a separate height ceiling. Multiple voxels can grow
  at once. Initial mass and ceiling are independent. No top-storey-copy action
  is required, and no whole-storey categorical action has been added here.
- Remove/grow can change FAR in either direction. Exchanges balance removed and
  added cells. Counts use a geometric/workload allocator; spatial masks and
  operation are neural. A fixed initial workload budget plus eight extra actions
  bounds exploration. Rejected draws consume action budget, preventing infinite
  retries. These allocator/budget choices remain explicit research assumptions.
- Whole-edit guards enforce frozen eligibility, at most two grounded connected
  masses, exterior-connected removed space, preservation of old exterior air,
  and per-floor courtyard constraints. Projection only prunes selected bits;
  it never invents bridges, connectors or repair voxels. This is not structural
  engineering validation.
- Reaching FAR does not force episode termination. A legal Finish action or the
  action limit ends it; final acceptance still requires FAR in range. Reward
  combines continuous FAR progress, environmental quality change, depth change,
  small action cost and a continuous residual FAR penalty at the deadline.
  The existing uncalibrated environmental proxies remain a limitation.
- Native MPS handles the actor, latent fields, ranking, depth and environmental
  fields. Topology checks and whole-mask projection still use CPU SciPy/NumPy.

The network retains a small floorwise CNN (7×7 local planar receptive field);
it does not add full 3D feature exchange. Correlated mask decoding smooths across
floors, but that is not equivalent to a full 3D actor. Operation choice uses
pooled features from all floors.

## Correctness checks

Seven dedicated tests verify growth above initial height on the first action
while above FAR, remove/exchange availability without phases, continuation at
accepted FAR, stale edit rejection, actor/operation gradients, checkpoint
round-trip/rejection of legacy format, and actual MPS backward without fallback.
These are capability/invariant checks, not proof of learned tower designs.

The first test pass caught non-contiguous tensor mask updates through a flattened
copy. Sampling now creates contiguous masks explicitly; the committed mask is
the selected mask, rather than an empty original tensor.

New checkpoints use `voxelform.unified.v1`. The timed study wraps protocol and
initial/ceiling metadata in `voxelform.unified-study.v1`, independent of live
`voxelform.patch.v1`. Experimental weights are never silently loaded into the UI.

## Timed comparison

`tools/unified_training.py`: two seeds (123,456), phased versus unified, 90 active
training seconds per arm/seed, actual MPS with fallback disabled. Both start with
six storeys in a ten-storey ceiling, target FAR 3±0.1, two facade layers and coherent
mask safeguards. Train on a synthetic wide site and Barcelona 0093, mixing FAR 5
and FAR 3 exercises. Held out: synthetic XL 10×64×96 (12 sun samples, 16 neighbours)
and Barcelona 0094 (original fetched context, eight sun samples). Both real sites
use Auto alignment. Initial/final evaluation seeds match within each arm.

The phased baseline uses the nine-channel environmental actor and quality reward
with four post-FAR exchanges. Unified uses ten channels, a learned operation head
and its explicit budget/reward schedule. This tests the complete action-space
bundle, not an isolated ablation of only the ordering restriction.

The user's live MPS training ran concurrently. Wall-time budgets match, but GPU
throughput/thermal state are not isolated; do not present this as a clean speed
benchmark. Fixed-scene preparation is reported separately. Raw failures,
interrupted rollouts, metric vectors, action traces and source hashes are retained
in `results/unified-training-2026-11-10.json`. Source snapshots are in `source/`.

All four timed runs finished. Total fixed-scene setup was 227.7 s under concurrent
live training. Individual active training budgets were 90.05–90.96 s.

| Original arm | Seed | Updates | Held-out FAR success |
|---|---:|---:|---:|
| Phased | 123 | 22 | 1/2 |
| Phased | 456 | 112 | 2/2 |
| Fully free unified | 123 | 13 | 0/2 |
| Fully free unified | 456 | 29 | 0/2 |

All initial phased evaluations succeeded; no initial free-unified evaluation
succeeded. The original unified policy did grow early (401/389 roof cells before
FAR in the two final XL trials; final height 7/8 storeys), but remained near FAR
5.33/5.11 on XL and 3.42/3.69 on Barcelona. Its larger raw quality scores are at
different FAR and must not be presented as better designs.

## Follow-up hypothesis tests

Paired controls use the same initialized actor weights/sampling seed, and repeat
with the two trained unified checkpoints. Three untrained seeds (123,456,789)
plus two trained checkpoints = five configurations per plot, not five independent
sites. These are decoding/control tests without additional training.

1. **Projection-retention feedback alone:** rejected. All five XL trials still
   failed; four ended further from target. Larger requests did not fix wholesale
   topology rejection. `results/unified-feedback-2026-11-10.json` retains results.
2. **Prune unsafe removal bits using the existing editor, allowing two grounded
   masses:** substantially improved progress. XL final FAR fell from roughly
   5.1–5.3 to 3.10–3.37, but all five still failed. No unselected cells were added.
   `results/unified-projection-2026-11-10.json`.
3. **Pruning coupled with retention feedback:** improved some residual errors,
   still 0/5 XL successes (final FAR 3.114–3.196).
   `results/unified-coupled-2026-11-10.json`.
4. **FAR-guided net area, preserving simultaneous removal/roof growth:** 5/5
   successes on XL and 5/5 on the held-out fetched Barcelona plot. This is the
   feasible variant selected for further training, not a quality winner.

In the last variant pure net growth above the FAR band and pure net removal below
it are unavailable. **Paired growth is available throughout**: above target the
policy can grow roof patches while removing a larger amount elsewhere; below
target the reverse applies; inside the band exchanges balance. FAR sets net
area, not an edit order. The operation head still chooses pure correction versus
mixed redistribution, and generates both spatial masks in one pass. A Finish
action remains learnable inside the band; there is no forced stop at first FAR
success. This is deliberately tighter than the failed unrestricted four-mode
prototype. All variant flags are persisted in checkpoints. Old checkpoints load
with their original decoding flags, not silently upgraded semantics.

| Paired control | XL successes | Barcelona successes |
|---|---:|---:|
| Pruning + feedback, unrestricted net operations | 0/5 | 2/5 |
| Same actors + net-area guidance | **5/5** | **5/5** |

XL accepted edits: 27–33; final heights 8–9 storeys; 330–1178 cells were added
above the six-storey initial mass **before** FAR was reached. Barcelona accepted
edits: 10–12; final heights 7–8; 52–123 early roof cells. Attempt budgets include
rejections and Finish (XL 28–33 attempts, Barcelona 11–13). This does not mean the
model learned two well-positioned towers. Most behavior here is feasibility,
constraint handling and initialized spatial fields, not demonstrated quality
learning. Proxy calibration and longer matched training remain necessary.

Raw paired results: `results/unified-net-2026-11-10.json` and
`results/unified-net-real-2026-11-10.json`. Barcelona masks/allowed arrays and the
original transformed context are saved separately under ignored
`results/checkpoints/unified-net-proposals/`. Each result records source hashes.
Initial source snapshots are in `source/`; current implementations are in
`source_final/`. Additional warm real-site setup was 6.61 s and XL 2.50 s; these
do not justify a cold-start speedup claim under changing live load/caches.

## Current use and verification

133 Python tests passed, including native MPS backward with net-area paired
editing, direct first-action roof growth while above FAR, and grounded splitting
with pruning confined to selected bits. Ten Node viewer/geometry tests pass.
JS syntax and whitespace checks pass. The live server/policy were not restarted,
paused or replaced. The user continued their own training throughout.

Start a separate experiment (using the project's Python environment):

```sh
python -m tools.unified_training --tag net-pilot --arms unified --net-guided --seconds 90
```

Resume a saved experimental network with `--resume PATH --seeds 123`. If decoding
options change, the runner retains the network and resets Adam state. This path
does not accept live five-channel weights as if they were the ten-channel actor.
No new quality-trained policy has been promoted to the live UI.
