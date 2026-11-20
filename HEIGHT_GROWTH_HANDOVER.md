# Height, growth and floor-area redistribution handover

**Written:** 14 November 2026, Europe/Rome.

**Code baseline:** `b913a15` on GitHub `main`; local branch `codex/voxel-prototype`.

**Repository:** `/Users/ruslan_faz/Desktop/Work/Thesis/VoxelForm-RL`.

**Purpose:** let a fresh agent continue this specific discussion without inheriting the whole chat. This records the user's intent, the approaches actually attempted, their evidence, and the restrictions still present. User requirements and agent-selected heuristics are distinguished below.

## 1. Read this first

The user wants a model that can **redistribute building floor area into better positions and heights**, including carving away poorly exposed volume and putting that area into taller, potentially separate buildings. Reaching FAR should not make those transformations impossible.

The active preview now supports carving and whole-top-floor copying throughout an episode. Copied cells remain linked to their source during subsequent cuts. It runs through `main.py` at **http://127.0.0.1:8011**. This establishes a usable experiment for feedback, **not a demonstrated policy for designing well-positioned, livable towers**.

The most important unresolved issues are:

- The active copy-only growth decoder cannot independently refill cuts on existing floors.
- Persistent source/copy links prevent independently deleting just one copied storey. There is no dedicated height-decrease or undo-copy action.
- Direct vertical support and permanent linking are agent-selected restrictions, not immutable user requirements.
- Copying a poor or tiny footprint can still produce poor floors. There are no room-width, circulation, core or residential-program checks.
- No convincing learned daylight-quality or convergence advantage has been established.

Continue from this implementation and the user's feedback. Do not silently restore the old carve-first/grow-later schedule, and do not mistake the current safety rules for the complete design objective.

## 2. What the user was trying to achieve

### The motivating example

The user showed a long, relatively low generated mass behind a tall neighbouring building. They drew a possible alternative with two taller masses. Their reasoning was:

1. Some currently occupied floor area is badly exposed because of surrounding buildings.
2. Remove volume from those locations, possibly cutting the original mass into separate buildings.
3. Recover the removed floor area by growing upward in better locations.
4. Continue evaluating the actual changed geometry, including mutual shading, rather than merely maximizing height.

Two towers were an example of a useful transformation, not a mandatory output template. Starting the initial volume close to the desired FAR would not solve the problem if the model still could not remove and replace area after reaching FAR.

The user separately insisted that **diffuse light/sky access and direct sunlight/shadows are different things**. Nearby context and crevices should affect diffuse facade exposure. A single rendered shadow is an illustration, not annual sunlight evidence or the diffuse-light calculation.

### The user's proposed growth ideas

The user suggested top-storey duplication: carve a useful footprint, then copy the top floor upward to recover FAR. Later they clarified that random vertical patches were creating apparently unlivable spaces, and proposed either:

- copying a top floor after it has been carved sufficiently well; or
- copying it and editing the copied floor together with its source.

They encouraged alternative approaches and testing. They did **not** require a hardcoded “perfect footprint first” phase or a fixed number of floors.

### Other constraints relevant to height

- Directly generated, smooth multi-voxel edit masks; no candidate-mask library and no one-voxel-per-action policy.
- Frozen facade eligibility per action, with optional deeper cuts. The user allowed testing two or three exterior layers rather than insisting on exactly three.
- Horizontal depth on the **same floor**: roof proximity must not make a deep floor plate appear shallow. Badness ramps from depth 8 to 16 m and is combined with facade eligibility.
- Roof-open courtyards may later connect to the outside through a side opening.
- FAR 3.0 can mean 2.9–3.1. Presets are low 1.5, medium 3.0 and high 4.5, plus custom.
- Prefer few productive edits, but do not let a harsh deadline reward overwhelm environmental quality. The user suggested estimating work from required change divided by initial editable facade capacity rather than a universal 5–15-step limit.
- Speed matters on their M4. Test changes as hypotheses; changes may be grouped when they interact. Do not launch expensive training merely to compensate for an action-space restriction.

## 3. How floor count and height are actually decided

FAR is occupied floor area divided by the **source parcel area**. With 1 × 1 × 3 m voxels, each occupied voxel contributes 1 m² of floor area. The 3 m storey height changes physical height, not that area contribution.

The original fetched-scene preview filled a fixed six-storey envelope. Its initial height and maximum permitted height were the same. There was no learned choice of initial floor count, and no headroom for meaningful vertical expansion.

The current active `UnifiedSession` uses:

| Quantity | Current default | Meaning |
|---|---:|---|
| Initial occupied storeys | 6 | Starting mass fills allowed footprint on these floors |
| Height ceiling | 10 | Computational envelope permits later growth up to this level |
| Storey height | 3 m | Six storeys are approximately 18 m; ten are 30 m |
| Maximum grounded masses | 2 | Whole-edit topology guard |

These are experiment defaults in `UnifiedSession.__init__`, not zoning, a FAR-derived height formula, or a learned height controller. The UI exposes the ceiling in its envelope label but does not currently provide dedicated initial/maximum-storey controls.

During an episode the model selects actions that can change occupied height. It does not output an integer “final floor count.” The permitted grid remains ten floors tall even if fewer floors are occupied. A ten-storey ceiling can itself prevent escaping the influence of a much taller neighbour; extra training cannot exceed that ceiling.

## 4. Approaches taken, in order

### A. Original FAR-only carving/addition

`Environment.apply` / the old `PatchLearner` session chose subtraction above FAR, addition below it, and stopped on acceptance. The envelope was already filled to its maximum height. The original connectivity rule also resisted increasing the number of solid components.

Consequences: no post-FAR redistribution, no roof headroom, and a connected building could not freely become two grounded buildings. More training could not remove those restrictions.

### B. Paired constant-area redistribution prototype — 9 November

`voxelform/redistribution.py` introduced one atomic edit containing a neural removal mask and an addition mask. Both spatial fields came from one actor pass. Counts were balanced to preserve area, with selected-bit pruning and whole-edit topology checks.

It allowed roof growth inside a separately specified ceiling and up to two grounded masses. A deterministic fixture demonstrated splitting a bridge into two grounded masses while adding roof volume at the same FAR. Eight successive generated exchanges were checked on CPU and MPS. Early versions stalled after 1–3 moves; pruning and rebalancing improved feasibility.

`duplicate_top` also existed as an optional whole-top-storey primitive, selected by the caller. It was not yet the live learned action protocol.

Facade-layer tests compared 1/2/3 layers. Two layers completed 12/12 small CPU/MPS cases, compared with 10/12 for one and 9/12 for three. This justified trying two layers; it did not establish architectural quality.

### C. Hardcoded FAR correction followed by refinement — 9 November

The experimental trainer in `tools/train_m4.py` first corrected FAR, then allowed a small number of paired exchanges, commonly four. Experiments used six initial floors inside a ten-floor ceiling.

**The user challenged this ordering, and the agent confirmed it was hardcoded.** The live app at that time still had no vertical growth. The phased experiment was not evidence that the model had learned to decide when to grow.

An equal-time study compared the earlier actor/reward with an enhanced environmental-input/quality-reward bundle. Changes in held-out quality were small and conflicting; useful exchanges were often rejected. It did not demonstrate learned tower placement. The staged training discussion also produced no established curriculum advantage.

### D. Unified learned remove/grow/exchange/finish policy — 10 November

`voxelform/unified.py` and `tools/unified_training.py` removed the phase schedule. The actor learned a four-way operation alongside spatial fields. Growth and paired redistribution could occur from the first action, with FAR acceptance enabling Finish rather than forcing immediate termination.

The first fully free-operation pilot failed FAR on both held-out scenes for both trained seeds after 90-second runs. It grew early, but failed to recover the required area target. Higher raw quality at the wrong FAR was not a valid design win.

Follow-up controls tried retention feedback, safe removal pruning, their combination, then **net-area guidance**. The last combination reached FAR in 5/5 configurations on synthetic XL and 5/5 on a fetched Barcelona plot. These were three initial seeds plus two trained checkpoints per plot, not ten independent sites.

Net guidance preserves simultaneous removal/growth but restricts the final net area outside the FAR band. It is a handcoded feasibility rule, not a carve-before-growth phase. The experimental roof decoder still selected arbitrary roof patches, which later caused the user's complaint about scattered, unusable-looking growth.

### E. Other agent's `devox` branch

The user identified `origin/devox`, commit `ec0c38f` (“Add unified roof-growth preview and 0-3 devoxelization lab”). It added `unified_server.py`, a static comparative lab and its builder. The branch was fast-forward integrated before further edits, preserving its history.

The imported preview exposed the growing policy but initially used the unsuccessful free-operation setup and could load an older experimental checkpoint automatically. Integration selected net guidance, removed that silent load, and promoted this server to the normal `main.py` entry point.

### F. Active whole-floor copying with persistent links — 14 November

To address random roof growth, the active policy now uses `floor_copy=True`:

1. Find each grounded mass's highest occupied floor.
2. Its complete footprint, including courtyard holes, defines a possible one-storey extension inside the ceiling.
3. A learned spatial field selects which existing mass to extend when there is more than one. This is an explicit floor-copy primitive, not a library of sampled geometric proposals.
4. In an exchange, source-floor cuts also remove the corresponding prospective copy cells. The new floor equals the resulting source footprint for that mass.
5. `env.copy_labels` links copied cells to their source cells for later cuts. Repeated copying extends those vertical groups.
6. Partial selections of a linked group are pruned. Projection does not invent extra cuts to make the group complete.
7. Every retained upper voxel must have an occupied voxel immediately below. Removal ranking also favours coherent top-down selections.

There is **no “almost perfect floor” detector**, no required delay before copying, and no automatic claim that the source plate is useful. An existing top plate can be copied on the first action.

These changes were tested as a coupled decoder. Their individual causal effects have not been isolated.

## 5. Current decision process: learned parts and explicit rules

| Part | Current mechanism |
|---|---|
| Remove / grow / exchange / finish | Learned categorical operation, with legality masking |
| Removal locations | Correlated Gaussian latent field from the actor; eligibility weighting and ranking |
| Growth location | Learned field selects an existing mass's complete top plate |
| Shape of copied addition | Determined by the current source footprint |
| Edit amount | Workload/retention controller and area-balancing projection |
| Support, linked cuts, courtyard validity | Explicit guards; invalid selected bits are pruned or a draw is rejected |
| Initial height and ceiling | Fixed session configuration |
| “Ready to duplicate” quality threshold | Not implemented |

One actor pass produces two dense fields, a value estimate and operation logits. The actor is a small **floorwise 2D CNN**, with a 7 × 7 planar receptive field for spatial output, plus global pooling for operation/value decisions. It uses Gaussian-latent A2C; this is not PPO. It has ten input channels: occupancy, air diffuse field, noise, allowed envelope, elevation, same-floor depth badness, removal eligibility, growth eligibility, external direct-sun potential and separate diffuse facade access. FAR and budget features also condition it.

There is no full 3D spatial feature exchange in the CNN. Cross-floor smoothing and the copy/support decoder provide coherence, which is different from learning general 3D coordination. With only one mass there is only one plate to copy, so the addition field has no choice between masses on that step.

In normal generation/training:

- Above the FAR band, a pure copy is unavailable; an exchange can still copy upward while removing more area elsewhere.
- Below the band, pure removal is unavailable. A whole copy may be used if it fits the upper FAR bound; exchanges can copy and offset part of the added area.
- Outside the band, accepted edits must make net progress toward it. This is still a strong feasibility priority and may limit temporarily worsening FAR to improve later geometry.
- Inside the band, exchanges must remain in it. Finish is legal; quality refinement is permitted but not guaranteed to happen before Finish.

The operation head starts with learnable FAR-correction-biased weights. That initialization is not an ordering schedule.

## 6. Height decrease and reversibility: important gaps

There is no separate “remove top storey” action. Removing upper cells can lower the occupied height through ordinary legal cuts, but removing area does not necessarily lower the highest occupied floor. One remaining high voxel still determines that height.

Persistent linking is especially restrictive: after copying a floor, a cut selecting only its upper copy is discarded because it omits the source. Deleting that copied storey independently is therefore unavailable under the present rule. The model must select the linked cells together, which can remove several levels at once or make the desired deletion impossible.

Similarly, **active growth adds only top-floor copies**. It cannot independently refill a notch on an existing floor, regrow a removed ground footprint, start a disconnected new building on empty ground, or grow laterally as the earlier roof/facade decoder could. Two buildings can emerge by splitting an existing grounded mass, subject to the two-component guard.

Direct vertical support enforces nested occupied floor plates and forbids cantilevers or a well-exposed upper volume above an empty lower position. This is an agent-chosen simplification in response to bad growth, not a requirement supplied by the user and not a structural-engineering test.

These restrictions should be discussed/tested when continuing. Do not silently describe the active version as an unrestricted, reversible subtraction/addition model.

## 7. The source/copy area-balancing bug and fix

A source-floor cut has two effects in a simultaneous copy action: it removes an existing cell and withholds the corresponding new cell. Linked stacks add further area costs. Treating a copy as a fixed added area while independently counting removal caused oversubtraction or stalled growth near FAR.

`balance_copy` now uses the exact final area and prunes a prefix of the **same neural cut ranking**, respecting support/link groups. This is deterministic quota projection, not generating/evaluating candidate buildings. It preserves the complete resulting source/copy footprint.

CPU fixture, seed 123, four initial floors, eight-floor ceiling:

| Target | Before balancing | After balancing |
|---|---|---|
| FAR 1.5 | Success, 1.5625, 7 attempts | Same result |
| FAR 4.5 | Failure, 4.36979, 13 attempts | Success, 4.40104, 2 attempts |
| FAR 7.5 | Failure, 7.00521, 36 attempts | Success, 7.40104, 5 attempts |

**Attempts include Finish and rejected draws.** The successful FAR 4.5 result is one accepted geometry edit plus Finish, not two separate building edits. Full before/after traces, including failures, are retained.

## 8. Budget, reward and speed

The user feared that deadline punishment would teach the model to strip as much facade as possible at the expense of quality. That tradeoff remains an evaluation concern; no complete multi-objective solution has been demonstrated.

The workload controller estimates required area change `M` and initial editable facade capacity `N`. Its historical productive estimate uses `ceil(M / (0.5*N))`, with margin/reserve and optional cell cap. The active unified deadline is `max(8, workload.limit + 8)` attempts. The extra eight are an agent-selected exploration allowance, not a user requirement or learned minimum. Retention feedback adjusts requested area after pruning/rejection.

Current live reward includes normalized FAR progress, depth improvement, a small action cost, environmental-quality change on audit, Finish reward, rejection cost and continuous residual FAR penalty on failure. This is still an explicitly weighted reward plus feasibility rules. Curriculum, separate critics, Pareto tradeoffs and constrained RL were discussed; they were not all implemented in this growing preview.

Scene setup and immutable neighbour visibility are cached. Actor/field work, ranking, planar depth and environmental tensors run on the selected device; connectivity, copied-plate bookkeeping and some projection remain CPU work. Devoxelization is kept out of training.

Integration also fixed the preview's missing attempt-counter increments, immediate episode termination on rejected draws, terminal value bootstrapping and FAR-retarget countdown reset. A 32-action optimizer rollout remains distinct from an episode. Environmental audits occur every eight accepted masks and at completion.

## 9. What the latest tests establish

Final matched-seed comparison: **two real sites × seeds 123/456**, six initial storeys, ten-floor ceiling, target FAR 3 ±0.1, two facade layers, untrained policies. Both arms succeeded in all four runs.

| Measure | Earlier roof patches | Linked whole-floor growth |
|---|---:|---:|
| Accepted geometry edits | 14–21 | 10–14 |
| Attempts including rejects/Finish | 15–21 | 11–16 |
| Upper cells without direct support | 428–480 | 0 |
| Median cached-scene generation | 1.25 s | 0.80 s |
| Observed generation range | 1.05–2.63 s | 0.57–1.61 s |

Setup was separate: approximately 7.33 s for NYC and 11.35 s for Barcelona, with 19/20 context buildings. Hardware/background UI activity was not isolated; one copy run was slower than its paired patch run. Do not claim a universal speedup.

Final occupied heights for the copy arm were 7, 10, 10 and 6 storeys. Thus reaching FAR does not guarantee height growth in every run. One Barcelona result had only **40 m² on each of its top two floors**. The current plate minimum can be as low as four cells. Complete copying and support alone do not establish a good residential floor.

Five-second MPS training probes made 2 and 4 optimizer updates with nonzero gradients in the final retest. The earlier pre-balancing pilot made 8 and 6; preserve those as different measurements. This proves that the decoder is trainable, not that longer training has learned quality or two towers.

At the implementation handoff, **154 Python and 10 JavaScript checks passed**, including actual MPS backward, linked cuts, frozen eligibility, first-action growth/exchange, courtyard copying, area regression and preview controls. These were the previous implementation checks; this documentation-only handover does not rerun training or reset a live session.

## 10. Active app, saved weights and devoxelization

- `main.py` now imports `unified_server.app`; default port 8011. `voxelform.server` is the historical FAR-only app.
- Active policy flags: `floor_copy=True`, `net_guided=True`, `amount_feedback=True`.
- The user explicitly said they did not care about preserving the old session weights and would save themselves. Promotion started a fresh policy; no long-trained, proven quality policy was promoted.
- Startup is fresh and paused. Manual policy export/import persists model, optimizer, configuration and decoder flags. It is not a full geometry/session snapshot; do not assume it restores occupancy or `copy_labels`.
- Legacy `voxelform.unified.v1` states without the new flags retain old decoding semantics in research code. The active loader requires whole-floor/net-guided policy files. Do not silently load an old roof-patch checkpoint as the new policy.
- **Generate one** runs a normal inference episode. **One mask → Copy top floor · probe** explicitly forces a geometry probe and can bypass FAR-direction guidance. Normal Train/Generate one do not use that forced probe.
- `UnifiedLearner()` alone still defaults to `floor_copy=False` for research compatibility. `tools/unified_training.py --net-guided` alone does not reproduce the active whole-floor decoder. The active server and `tools/check_floor_growth.py` explicitly enable it.

The other agent's devoxelization work is integrated: mode 0 raw, 1 correlated loft, 2 extruded slabs, 3 VSA planes. It is display-only, off by default and applied after a completed inference generation. It changes neither occupancy nor FAR/rewards. The `/devoxelize` comparison lab uses a fixed synthetic fixture, not the live building. Complex reconstructed meshes can retain seams/nonmanifold edges. A smooth-looking surface must not be used as evidence that fragmented floor area became livable.

Preserve the existing shared Module Lab graphic engine, fetched surroundings, 3D-only UI and blue palette. Do not edit the sibling Module-Lab project as part of height work.

## 11. Suggested continuation — proposals, not completed work

1. Obtain feedback on the active growth behaviour before another large architecture change.
2. Test **reversible height actions**: whole-storey removal/undo-copy, and explicit policies for which source/copy relationships can be released. Compare permanent links with temporary coupling; do not assume permanent locking is desirable.
3. Test restoring coherent existing-floor addition alongside copying, so a mistaken cut need not be compensated only by more height.
4. Evaluate copying eligibility/plate quality with explicit, cheap geometric proxies (width, area, fragmentation, depth). Keep those separate from actual rooms, cores or a livability guarantee. Do not introduce a mandatory “perfect first” phase without discussing it.
5. Test different initial heights and ceilings on the same plots. Track height per mass, floor-plate sizes, lateral footprint recovery, FAR, exposure/depth and edits separately.
6. Compare at equal final FAR and matched wall-clock budgets. Keep untrained controls, failed runs and individual environmental metrics. Separate setup cost, per-edit geometry/projection, audits and optimization.
7. Investigate learned quality only after the action space supports the requested transformations. No evidence currently warrants promising that more training alone will remove badly exposed volume and relocate it well.

## 12. Files and evidence for the next agent

Implementation:

- `unified_server.py`: active defaults, live episode loop, controls and policy import/export.
- `voxelform/unified.py`: `copy_plates`, `supported_removal`, `balance_copy`, `project_floor_copy`, `valid_move`, `commit`, `UnifiedLearner`.
- `voxelform/redistribution.py`: earlier paired roof/facade growth and optional `duplicate_top` primitive.
- `voxelform/patch_policy.py`, `voxelform/workload.py`: frozen eligibility, patch/topology projection and workload estimates.
- `tools/train_m4.py`: historical correction-then-refinement runner; do not confuse with the active protocol.
- `tools/unified_training.py`: unified research episode runner; old roof decoder unless explicitly enabled otherwise in policy construction.
- `tools/check_floor_growth.py`: actual roof-patch versus linked-copy comparison, including training probes.
- `tests/test_floor_copy.py`, `tests/test_unified.py`, `tests/test_unified_server.py`: current invariants and control regression checks.

Evidence and deeper history:

- [Current growing-preview implementation notes](agent_notes/growing_preview_2026-11-14/README.md).
- [Latest real-site comparison](results/floor-growth-preview-balanced-2026-11-14.json).
- [Source/copy gap regression, including failures](results/floor-copy-far-gap-regression-2026-11-14.json).
- [Initial copy comparison](results/floor-growth-preview-2026-11-14.json).
- [Unified-action failures and net-guidance controls](agent_notes/unified_actions_2026-11-10/README.md).
- [Original paired edits and optional top-storey duplication](agent_notes/redistribution_2026-11-09/README.md).
- [Phased refinement and matched quality study](agent_notes/grid_quality_2026-11-09/README.md).
- [Original redistribution request and blockers](agent_notes/m4_training_2026-11-09/REDISTRIBUTION.md).
- [M4 speed, rewards and diffuse-facade clarification](agent_notes/m4_training_2026-11-09/README.md).

Historical reports describe the state on their dates. Their claims that growth was isolated or the live policy remained unchanged were true then; the 14 November promotion supersedes them.

For local development, from this repository:

```sh
../Module-Lab/.venv/bin/python main.py
# http://127.0.0.1:8011

../Module-Lab/.venv/bin/python -m unittest discover -s tests -q
node --test tests/*.test.mjs

# Runs measurements/training probes; choose a NEW tag to retain prior evidence.
../Module-Lab/.venv/bin/python -m tools.check_floor_growth --tag new-unique-tag
```

Do not start a duplicate server or reset a current session merely to read this handover. Inspect the current branch/worktree and live state first. The code baseline was pushed as `b913a15`; this handover is a later documentation addition. Preserve concurrent work and saved policies.

## 13. Update: Multi-Tier Tower Extrusion, Reversible Unstacking, and Column-Consistent Redistribution (14 November, `v2-tower-growth`)

### Root Causes Addressed After `b913a15`

1. **`torch.cummin` order-statistic bias in top-down removal**: Independent per-floor values dropped expected lower-floor scores sharply after top-down `cummin`, so `topk` shaved upper perimeter rings (`z=4, 5`) into tapering pancakes or `40 m²` slivers while leaving bloated, dark lower storeys (`z=0..3`) intact.
2. **Whole-component top-floor-only `copy_plates`**: Labeling 3D components exposed only the highest level of each connected mass, preventing lower podium terraces of a stepped mass from growing, forcing 100% slab copies on wide plots, and allowing 4-cell (`40 m²`) slivers on upper levels.
3. **Permanent `copy_labels` coupling**: Partial upper-tier cuts remained locked to source cells with no reversible whole-tier unstacking rule or automatic release of orphaned `copy_labels`.

### Implemented Algorithmic Changes (`voxelform/unified.py`)

- **Minimum coherent plate & core check (`min_plate_area`, `coherent_plate`)**: On urban-scale plots (`plot_area >= 400`) and upper floors (`z >= 4`, or `z is None`), plates require `max(minimum_patch, min(96, max(16, round(0.06 * plot_area))))` cells and at least 4 interior cells surviving `2x2` binary erosion, rejecting `40 m²` chimney needles and 1-cell fins while preserving `4`-cell compatibility on small synthetic fixtures (`plot_area < 400`).
- **Multi-tier roof growth (`copy_plates`, `extract_tower_plate`)**: `copy_plates` now includes both complete highest floors of 3D masses and exposed 2D horizontal roof terraces across all levels `1..nz-1`. On urban plots (`plot_area >= 400`), when a chosen plate exceeds `max(min_plate_area, round(0.30 * plot_area))`, `extract_tower_plate` extracts the top-scoring connected 2D tower sub-plate inside the exposed roof footprint (hole-filled within support and `2x2`-opened), enabling podium + tower stepping without copying full plot slabs.
- **Reversible whole-tier unstacking (`unstack_cells`, `supported_removal`, `project_floor_copy`, `valid_move`, `commit`)**: Complete exposed 2D floor tiers (`unstack_cells`) with no remaining occupied cells directly above them can be removed in full cleanly even wider than `2 * facade_layers` and without deleting their supporting source floor `z-1`. After `commit`, `copy_labels` entries that no longer share a vertical neighbor above or below are cleared so future partial cuts on the lower tier remain unblocked.
- **Column-consistent & exposure-aware scoring (`UnifiedLearner.choose`)**: On urban plots (`plot_area >= 400`), removal and growth fields incorporate same-floor depth badness and external/diffuse facade light (`external_potential`, `facade_light`), and removal scores blend vertical occupied-column averages (`0.85 * col_mean + 0.15 * scores[channel]`) before `cummin` + tiny height tie-breaker, carving full-height shadowed/deep shafts rather than shaving only upper rims.

### Verified Checks & Benchmark Results (`results/floor-growth-preview-v2-tower-growth-2026-11-14.json`)

- **Unit & frontend tests**: **157 Python tests** (`tests/test_floor_copy.py` includes reversible whole-top-floor unstacking, stepped podium/tower multi-tier plates, and urban-scale sliver/fin rejection) and **10 JavaScript tests** passed.
- **Matched-seed real-site comparison (`tools/check_floor_growth.py --tag v2-tower-growth`)** on MPS (2 sites × seeds 123/456, 6 initial storeys, 10-floor ceiling, target FAR `3.0 ± 0.1`, `floor-copy-linked` arm vs prior `results/floor-growth-preview-2026-11-14.json` baseline):

| Site & Seed | Result / FAR | Edits / Attempts | Unsupported | Height & `floorAreas` (m²/floor) | Quality (vs prev) | Usable Area (vs prev) | Diffuse Facade / Direct Sun (vs prev) |
|---|---|---:|---:|---|---:|---:|---:|
| `m_nyc_0003` · `123` | `far_reached` (`2.958`) | 12 / 14 (`0.54s`) | 0 | 9F `[481, 481, 480, 480, 480, 480, 402, 402, 402, 0]` | **0.2066** (0.0511) | **2,257 m²** (129) | **0.0744 / 0.3256** (0.0115 / 0.1244) |
| `m_nyc_0003` · `456` | `far_reached` (`3.099`) | 10 / 14 (`0.79s`) | 0 | 8F `[580, 579, 579, 579, 577, 574, 408, 408, 0, 0]` | **0.1767** (0.0523) | **1,836 m²** (36) | **0.0591 / 0.2789** (0.0117 / 0.1321) |
| `m_bcn_0094` · `123` | `far_reached` (`3.100`) | 11 / 17 (`1.35s`) | 0 | 7F `[647, 646, 646, 646, 646, 646, 302, 0, 0, 0]` | **0.1656** (0.0958) | **1,380 m²** (611) | **0.0477 / 0.3557** (0.0285 / 0.2550) |
| `m_bcn_0094` · `456` | `far_reached` (`3.017`) | 11 / 12 (`0.49s`) | 0 | 8F `[604, 599, 599, 599, 599, 597, 389, 81, 0, 0]` | **0.1664** (0.0826) | **1,369 m²** (376) | **0.0642 / 0.3997** (0.0170 / 0.2118) |

- **5-second MPS training probes**: `m_nyc_0003` (`5.08s`) completed 6 optimizer updates (`5` finished `far_reached` episodes with heights `7–9F`, usable area `1,755–2,257 m²`, plus `1` deadline-interrupted mid-episode); `m_bcn_0094` (`5.22s`) completed 5 updates (`4` finished `far_reached` episodes with heights `7–10F`, usable area `1,380–2,072 m²`, plus `1` deadline-interrupted).
- **Default startup/demo site**: `unified_server.py` (`VOXELFORM_SITE_ID`), `voxelform/session.py`, and `public/app.js` default initial site selection to `m_bcn_0093` for demo startup.
- **Remaining caveats**: Gains above come from the improved geometric primitives, reversible unstacking, and exposure-guided column/tower scoring rather than long-run converged RL training. Existing-floor notch refilling outside upper roof growth remains separate, and environmental metrics remain uncalibrated research proxies rather than livability or daylight compliance.

## 14. Update: Continuous Multi-Episode Training, Outward-to-Inward Facade Peeling, and Vectorized Anti-Sliver Filtering (`voxelform/unified.py`, `unified_server.py`)

- **Continuous training loop (`unified_server.py`, `voxelform/unified.py`)**: Fixed `UnifiedSession.finish_episode` pausing training on `step_budget_exhausted` or zero-edit rejected episodes; training now continues across episodes until `pause()` unless `edits == 0 and unified_actions == 0`. `UnifiedLearner.choose` also masks `finish` on step 0 (`e.edits == 0`) during `training=True` when edit operations are legal, and caps `toward` removal `amount` by distance to the lower FAR band edge (`e.count - acceptable_counts[0]`) plus retention-compensated exchange sizing so near-band edits do not stall.
- **Vectorized anti-sliver filtering (`min_room_width`, `clean_floor_tensor`, `clean_floor_numpy`)**: On urban plots (`plot_area >= 400`), `min_room_width=3` runs vectorized 2D `F.avg_pool2d`/`F.max_pool2d` checks across all floors simultaneously: a `3×3` morphological opening, `1×3`/`3×1` directional chord checks (rejecting pinch necks), `n8 >= 3` & `n4 >= 2` neighbor counts, and a `2×2` opening. `supported_removal` (also fixing `np.bincount(..., minlength=max_lab)` for unstacked labels), `coherent_plate`, `extract_tower_plate`, `project_floor_copy`, and `valid_move` reject newly created slim remnants (`clean_floor_numpy(..., 3)`), while `UnifiedLearner.choose` absorbs eligible slim remnants upward via `cummax` before projection.
- **Outward-to-inward `remove_skin` ordering**: Surface light propagates inward across `facade_band` via `F.max_pool2d` and inner-layer removal scores are clamped below adjacent `remove_skin` scores (`0.999` decay per layer) while keeping deep interior wells (`> 2px` from `remove_skin`) unclamped, eliminating 1-voxel hollow outer walls.
- **Verified checks**: **160 Python tests** and **10 JavaScript tests** passed (`unittest` + `node --test`). On MPS real-site checks (`m_bcn_0093`, `m_bcn_0094`, `m_nyc_0003` × seeds `123`/`456`), **6/6** runs reached FAR (`3.021–3.099` in `8–14` accepted edits), with `0` unsupported cells and `0` slim remnants (`clean_floor_tensor(..., 3)`). Three-episode continuous training on `m_bcn_0093` completed 3 episodes (`34` ticks, `3` updates) without auto-pausing until `pause()`.

## 15. Update: Organic 3D Volumetric Mode, Two-Tier Vectorized Anti-Sliver Defense, Exact Disk Scene Cache, and RL Stabilization (18 November, `organic_3d`)

### User Feedback & Root Causes Addressed
1. **Stripped cantilever / vertical-coupling restrictions**: Removed the no-cantilever check (`changed[1:] & ~changed[:-1]`), persistent `copy_labels` vertical coupling, `0.85 * col_mean` / `cummin` column-collapse ranking, and hardcoded `extract_tower_plate` / heuristic score overrides from the active preview by introducing `organic_3d=True` (`Organic3DNetwork` + `project_organic_3d`) while preserving `floor_copy=True` strictly for legacy unit tests.
2. **Stale in-memory server (`main.py` without `--reload`) & auto-pause**: Confirmed `UnifiedSession.finish_episode` must never pause during `self.training == True` (`if not self.training: self.pause()`); note live browser sessions running `main.py` before code edits must be restarted to load updated modules.
3. **2-voxel and `< 6`-voxel slim strips without deadlocks**:
   - **Hard width floor (`min_width=3` on `plot_area >= 400`)**: Simplified `clean_floor_tensor` to a sync-free 2-pool morphological opening on GPU and replaced `clean_floor_numpy` with separable pure-NumPy bitwise slicing (`~0.15 ms`, 45× faster), eliminating all 1-voxel slits/walls and 2-voxel strips on both carving and growth (`slimVoxels = 0` across all 76 benchmark episodes).
   - **`< 6m` long narrow strip defense (`long_narrow_mask(..., width=6, stub=2)`)**: Vectorized 3-pool detector (`avg_pool2d(6)` -> `max_pool2d(6)` -> `max_pool2d(5)`) flags `< 6m` strips extending `> 2 m` outside any 6×6 core (exempting 2 m stubs on room facades). Without explicit carving deterrence, `< 6m` fractions reached `38.4%` in an unpenalized ablation; adding it as CNN channel 11, a 1-pass soft GPU carve pinch-guard (attempt < 3), a growth filter on floors with 6×6 cores, excluding it from `usableGoodArea`, and penalizing `-0.50 * ln6_frac` in `organic_objective` lowered it to `6.86%–8.80%` with `0` deadlocks.
4. **GPU sync elimination & `scene_cache_v2` speedup**: Eliminated intermediate GPU-CPU sync stalls in `_choose_organic_3d`, passed `validated=True` to `commit`, and added atomic disk caching for `ContextLibrary.load` / `fit_grid`, `sky_boundary`, `static_receivers`, and `SelfShadowMaps` DDA tables (`results/scene_cache/`).
5. **Monotonic RL credit assignment**: Replaced intermediate telescoping quality diffs (which suffered a negative drop cliff from `FAR 5.68 -> 3.0`) with complete-episode `mark_done()` GAE batches (`rollout_steps=32`, `lr=8e-4`), eligible-masked centered `1/sqrt(N_elig)` spatial log-probabilities (`dist.log_prob(latent) + dist.entropy().detach()`), and low-jitter eval (`logits * 2.5`, `0.10 * std` on attempt 0).

### Verified Checks & Benchmark Results (`results/organic-3d-rl-demo-2026-11-18.json`, MPS)
- **Full test suite**: **165 Python tests** + **10 JavaScript tests** passed (`test_unified.py` verifies `clean_floor_numpy == clean_floor_tensor` bit-exactly across widths `2, 3, 4, 6`, cantilevers, sliver rejection, `long_narrow_mask`, GAE backward, and checkpoint roundtrip; `test_unified_server.py` verifies `scene_cache_v2` bit-exact `sky_boundary`/`static_receivers`/`visible`/`quality`, continuous training, policy import/export, and cantilevered `devoxelize_live`).
- **Scene load speed (`scene_cache_v2` + `_load_context`)**: `m_bcn_0093` `37.594 s -> 0.401 s` (**93.75×**), `m_bcn_0094` `17.128 s -> 0.333 s`, `m_nyc_0003` `12.391 s -> 0.302 s`.
- **60-episode continuous MPS training on `m_bcn_0093`** (`103.55 s`, 16 GAE updates, `farSuccessRate = 1.0`, `maxSlimVoxels = 0`):
  - `first8Summary` -> `last12Summary`: `meanOrganicObjective` `0.2091 -> 0.2277`, `meanQuality` `0.1358 -> 0.1414`, `meanUsableGoodArea` `752.8 -> 786.1 m²`, `meanDirectSunProxy` `0.3825 -> 0.3973`, `meanDiffuseFacadeProxy` `0.0315 -> 0.0337`, `meanLongNarrowFraction` `8.36% -> 6.86%`.
- **5-ep eval (`trainedEval` vs `untrainedEval` on `m_bcn_0093`)**: speed `1.098 s/ep` (vs `1.549 s/ep`), `meanOrganicObjective` `0.2371` (vs `0.2215`), `meanQuality` `0.1533` (vs `0.1453`), `meanDirectSunProxy` `0.4338` (vs `0.3726`), `meanLongNarrowFraction` `8.80%` (vs `12.59%`), `meanCantileverVoxels` `650.6` (`maxSlimVoxels = 0`), reshaping massing from scattered 15-step cuts into carved lower courtyards (`z=0`: `418–477 m²`) and sunlit upper overhangs (`z=4,5`: `856–1053 m²`); `usableGoodArea` was `803.4 m²` vs `937.6 m²` (untrained baseline included two 15-step outlier draws of `1,209` and `1,230 m²`).
- **Zero-shot held-out eval**: `m_bcn_0094` `3/3` FAR (`0.874 s/ep`, `ln6 8.70%`), `m_nyc_0003` `3/3` FAR (`1.037 s/ep`, `ln6 4.52%`). Multi-site training on varied plots remains future work.

## 16. Update: Dense Spatial Credit Assignment, Coarse 2D+1D Network, End-to-End Speedups, and Live Dashboard (19 November, `coarse_2d1d_v2`)

### Root Causes Diagnosed & Addressed
1. **Why scalar episode REINFORCE struggled to clear the shadowed wing**: Scalar GAE assigned the same episode advantage across ~2,100 continuous Gaussian latents per step while `Organic3DNetwork` lacked `e.usable` as an explicit channel and used an expensive full-resolution `Conv3d` trunk (`224.2 ms` fwd+bwd on MPS).
2. **Why the live server felt slow**: `F.conv3d` in `gaussian_field`, 48× `boundary.clone()` allocations in `diffuse_access`, per-sun loops in `SelfShadowMaps.query`, duplicate `exterior_air`/`ndi.label` calls between `project_organic_3d` and `valid_move`, full `e.reset()` + double `evaluate()` in `next_episode()`, 1-step-per-lock ticks with `rollout_steps=32`, and uncached `1.19 MB` `/api/state` JSON polls every `700 ms` triggering Three.js `voxelContours` rebuilds mid-episode.

### Implemented Changes
- **Dense Spatial RL + Coarse 2D+1D CNN (`voxelform/unified.py`)**: Upgraded `Organic3DNetwork` (`coarse_2d1d_v2`) to 12 channels (adding `e.usable.float()`), strided 2D stem/down -> factored 2D+1D blocks + `1×1` coarse `local_skip` + `forward_coarse` + bounded `3.0 * torch.tanh(coarse / 3.0)` logits (`10.8 ms` fwd+bwd on MPS, **20.8× faster**). Inference runs purely through `forward_coarse` with zero hardcoded overrides. During training (`no_grad` mask decode), `UnifiedLearner.finish()` combines global multi-episode GAE (`rollout_steps=16`) with exact control-variate dense spatial REINFORCE (`(latent.sigmoid() - mean.detach().sigmoid()) * des_c`, weighted `1.2 / sqrt(N_elig)`) from `compute_spatial_desirability`.
- **Vectorized speedups (`voxelform/patch_policy.py`, `voxelform/fields.py`, `voxelform/shadows.py`, `voxelform/unified.py`)**: Separable 1D/2D `gaussian_field`, preallocated `padded` buffer in `diffuse_access`, batched `(U, V, S)` matmul in `SelfShadowMaps.query`, and deduplicated `_remaining_organic_valid` in `project_organic_3d`.
- **Fast session & Live Training Dashboard (`unified_server.py`, `public/index.html`, `public/style.css`, `public/app.js`)**: Cached initial-state snapshot in `next_episode()` (`1.34 ms`), 4-step training bursts in `loop()`, cached/view-aware `/api/state?view=solid` (`18.1 ms` / `211 KB` uncached, `0.02 ms` cached vs `73.2 ms` / `1,186 KB`), `Episode-sync 3D during training`, and an open-by-default **Training & Massing Dashboard** (6 KPI cards, learning-curve + FAR/speed canvases, and recent-episodes table).

### Verified Results (`results/organic-3d-rl-demo-2026-11-18.json`, M4 MPS)
- **Unit & frontend tests**: **168 Python tests** + **10 JavaScript tests** passed (`0` failures).
- **`m_bcn_0093` 35-ep training** (`53.85 s`, `1.539 s/ep`, `19` updates, `maxSlimVoxels = 0`): EP 30 reached `obj=0.7041, good=3,061 m², sun=47.8%, diff=9.9%`; EP 35 reached `obj=0.8052, good=2,893 m², sun=59.9%, diff=13.6%, ln6=5.9%`.
- **5-ep eval (`trainedEval` vs `untrainedEval` on `m_bcn_0093`)**: `meanOrganicObjective` `0.3260 -> 0.8224`, `meanUsableGoodArea` `995.0 -> 2,957.0 m²`, `meanDirectSunProxy` `38.3% -> 61.2%`, `meanDiffuseFacadeProxy` `4.0% -> 14.0%`, `meanLongNarrowFraction` `13.6% -> 5.5%`, `maxSlimVoxels = 0`, `deepFloorArea = 0.0 m²`; `farSuccessRate` was `3/5` (`meanFar = 3.1084`, 2 slight overshoots at `3.14` and `3.16` above `3.10`).
- **Zero-shot held-out eval (before -> after)**:
  - `m_bcn_0094`: `obj 0.1089 -> 0.6307`, `usableGoodArea 288.7 -> 1,973.3 m²`, `directSunProxy 25.9% -> 51.0%`, `diffuseFacadeProxy 2.4% -> 11.6%`, `ln6 20.6% -> 8.7%`, `slim=0`, `farSuccessRate 2/3` (`3.1004`).
  - `m_nyc_0003`: `obj 0.0515 -> 0.4490`, `usableGoodArea 130.0 -> 1,944.3 m²`, `directSunProxy 14.6% -> 22.8%`, `diffuseFacadeProxy 1.6% -> 6.5%`, `ln6 13.2% -> 4.4%`, `slim=0`, `farSuccessRate 3/3`, `0.761 s/ep`.

## 17. Update: Scalable Variable-Height Architecture, 5× Speedup, Reset Weights, and Deferred Proposals (19 November, `coarse_2d1d_v3`)

### Architectural & Computational Changes
- **Zero-memory dynamic self-shadows (`voxelform/shadows.py`: `FastVoxelShadows`)**: Replaced dense O(`Z·X·Y·S`) `SelfShadowMaps` DDA tables (`59.8 MB` at `10F`) with a `0 MB`, `0.0 s`-init GPU 2-bucket (`t_hi <= 16 m` short rays vs full interior rays) active-Z ray-marcher, filtered by `active_mask=outside` and skipping roof faces strictly at `z_max`; enabled on live sessions via `fast_self_shadows=True` (default `False` preserved for legacy bit-exact unit tests).
- **Static mesh Z-culling & slim cache (`voxelform/shadows.py`, `voxelform/environment.py`)**: `MeshShadows` precomputes `max_z` and `tri_max_z` for instant upward-sun clearance above surrounding context roofs (`face != 5 and pz >= max_z` skips ray queries in `precompute_external()`), while `scene_cache_v3_fast` omits `dynamic_maps` (shrinking disk cache from `~59 MB` to `1.92–3.82 MB`).
- **Active-Z diffusion & incremental surface tracking (`voxelform/fields.py`, `voxelform/environment.py`, `voxelform/actions.py`, `voxelform/patch_policy.py`)**: Added optional warm-start `init` to `diffuse_access(..., iterations=20)` over active slice `[:min(nz, z_top + 4)]` (`fast_diffuse=True` on audit edits), sliced `exterior_air(occupied)` to `[:min(nz, occ_z[-1] + 2)]`, and upgraded `PatchFacades.refresh()` to recompute `depth_tensor` / `_floor_voids` strictly on changed floors (`changed_z`).
- **No-permute 2D+1D network with exact halo-2 tiled vertical sliding 3D window (`voxelform/unified.py`: `Organic3DNetwork` `coarse_2d1d_v3`)**:
  - Replaced MPS-permuted `Conv2d(32, 32, (3, 1))` with zero-permute `Conv2d(32, 96, 1, bias=False)` + shared 32-ch `bias` initialized to exact `1/√(32·3)` variance (`_zmix` via shift-`cat`), allowing batched `finish()` backward (`groups` by active Z) with zero 5D padding warnings.
  - Added `forward_coarse_batch(..., window_floors=16)` tiling tall `Z > 16` stacks into overlapping vertical 3D windows (`halo = 2`, stitching interior slices and exact global floor sum/max pooling) plus quantized active-Z slicing (`z_act = min(e.nz, max(min(e.nz, 10), ((z_top + 4) // 4) * 4))`) and ceiling-independent normalized height (`rel_z = tanh(z_m / 36.0)`) and `_organic_goal()` scaling (`max(10.0, 3.0 * target_far)`).
  - Upgraded `project_organic_3d` (6 repair rounds + ground-stem / bridge-only connectivity repair when removals sever a narrow lower stem so valid upper cuts are preserved instead of full-mask rejection) and strengthened terminal FAR penalty in `unified_server.py` (`-(0.35 + 0.85 * err)`).
- **Context-derived storey envelopes, Reset weights, and UI (`unified_server.py`, `voxelform/context.py`, `public/*`)**:
  - `derive_site_storeys` derives initial & ceiling storeys from surrounding context (`m_bcn_0093` `9F -> 20F`, `m_bcn_0094` `7F -> 17F`, `m_nyc_0003` `12F -> 40F`), exposed alongside a collapsible `Height envelope` manual/Auto sidebar override (`/api/control` `command: 'envelope'`, `1..60` ceiling) and context height metrics in the 3D parcel tooltip.
  - Added `Policy -> Reset weights` (`#resetWeights`, `/api/control` `command: 'reset_weights'`) to re-initialize network weights/optimizer/session stats on demand.
  - Removed per-step `sync()` inside training `advance_one()` (`burst = 8` during training in `loop()`).

### Verified Benchmark & Test Results (`results/scalable-height-speedup-2026-11-19.json`, M4 MPS)

| Benchmark Arm | Envelope / Shape | Cold -> Warm Load (Cache / Shadow RAM) | Train Speed (vs `1.539 s/ep` Baseline) | Untrained -> Trained FAR Success | Untrained -> Trained `organicObjective` | Usable Good Area / Direct Sun / Diffuse / `ln6` / Slim |
|---|---|---|---:|---:|---:|---|
| **`matched10F` (`m_bcn_0093`, 35 ep)** | `6F -> 10F` `[10, 56, 58]` | `8.162 s -> 0.037 s` (`1.92 MB` / `0 MB`) | **`0.2836 s/ep`** (`9.926 s` total, **5.43× faster**) | `1.0 -> 1.0` (`5/5`) | `0.2792 -> 0.6714` | `2,640 m²` / `48.5%` / `8.9%` / `7.2%` / `0` |
| **`matched10F` (`m_bcn_0093`, 50 ep)** | `6F -> 10F` `[10, 56, 58]` | `8.162 s -> 0.037 s` (`1.92 MB` / `0 MB`) | **`0.3408 s/ep`** (`17.041 s` total, **3.16× faster wall-clock**) | `1.0 -> 1.0` (`5/5` vs old `3/5`) | `0.2792 ->` **`0.8277`** (vs old `0.8224`) | **`3,037 m²`** / **`60.1%`** / **`13.2%`** / **`4.0%`** / `0` |
| **`contextBcn20F` (`m_bcn_0093`, 25 ep)** | `9F -> 20F` `[20, 56, 58]` | `8.526 s -> 0.046 s` (`3.82 MB` / `0 MB`) | **`0.5791 s/ep`** (`14.478 s` total, `0.4232 s/ep` eval) | `1.0 -> 1.0` (`5/5`) | `0.2132 -> 0.6831` | `1,786 m²` / `66.1%` / `9.1%` / `3.0%` / `0` |
| **`contextNyc40F` (`m_nyc_0003`, 40 ep)** | `12F -> 40F` `[40, 47, 34]` | `4.571 s -> 0.041 s` (`3.75 MB` / `0 MB`) | **`0.7074 s/ep`** (`28.296 s` total, `0.3992 s/ep` eval) | `1.0 -> 1.0` (`5/5`) | `-0.0124 -> 0.6831` | `2,893 m²` / `39.3%` / `10.6%` / `12.2%` / `0` |

### Proposals for Further Development (Deferred Architectural Constraints)
The following constraints were explicitly scoped as follow-up work and are **not** implemented in this pass:
1. **Explicit Core Logic & Super-Linear Core Volume Scaling**: Require explicit dedicated vertical circulation/MEP core voxels whose floorplate core area scales upward with building height `H` (yielding slightly super-linear total core volume `area(H) × H`), alongside 3D path/egress distances from floorplate voxels to cores—naturally penalizing unchecked vertical pushing above neighboring shadow lines via rising core-to-GFA overhead.
2. **Envelope Efficiency**: Add a surface-to-GFA (`exterior skin / occupied floor area`) metric/term to penalize unnecessarily fragmented facades and reward compact daylight-effective envelopes.
3. **45° Overhang & Multi-Floor Truss Cantilever Rules**: Explicitly deferred—do **not** enforce 45° support cones or truss-depth overhang rules yet until core and envelope-efficiency terms are integrated.
