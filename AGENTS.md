# VoxelForm RL — Agent Instructions

## ML research contract (2026-11-20, takes precedence over older notes)
- This is an ML research project for structured/constrained 3D reinforcement
  learning. Urban geometry is a test domain and UI, not an architectural thesis.
- Read `docs/EXPERIMENT_PROTOCOL.md`, `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`
  and `docs/CORE_LAYOUTS.md` for current scientific assumptions and limitations.
- Canonical compact-core library: `voxelform/core_plans.py`, version
  `compact-core.v5`. T1 (1–5F) is stairs only, two flights/two landings, 3x5 m
  grid. T2 (6–10F) is one stair + one lift, 5x5 m. T4 (19–30F) and T5 (31–60F)
  share two interlocking scissor routes in one enclosure, separate +1.5 m
  landings, and four lifts, with a 6.55x8.40 m metric / 7x9 m grid footprint.
  `cores.py`, API/SVGs and exported plans share this source. Do not
  reinstate the prior 30/48/72/108/160 m² rectangles or compliance claims.
- Final core distances converge on an 8-neighbour occupied grid without blocked
  corner cutting, tested against independent Dijkstra. They are footprint-
  distance proxies, not actual door/corridor/egress solutions.
- Corrected organic `record` writes dictionary `reward`, not legacy key `3`.
  Score the full Gaussian retry path, including rejected draws. The optional
  physical spatial auxiliary is a heuristic objective, not an exact control
  variate. Reference research configs set its weight to zero; a separate config
  ablates 1.2. Do not call hybrid decoder behaviour unrestricted/pure learned RL.
- `vf-research` / `tools.research` is the headless research entry point, independent
  of Module Lab for synthetic experiments. JSON configs, site/seed separation,
  fixed-budget checkpoint selection, full failure logs, common reference
  evaluator, source archives, RNG checkpoints and resume checks are mandatory.
- Train/test summaries must include FAR failures and separate proxy components.
  A few episodes or an 80-episode single-site curve do not establish convergence.
  Historical results remain intact but use earlier core/loss semantics.
- Use `uv sync --frozen --extra dev` and `uv.lock`. Synthetic CPU CI plus JS
  checks require no sibling renderer/data. Record real hardware for MPS claims.
- Live site rotation uses only the declared training split and switches at
  completed episode boundaries while sharing the policy and optimizer. Treat
  the local top-five archive as outputs for review. Keep held-out evaluation
  evidence separate. Layouts and full episode statistics persist under ignored
  `runs/layouts/`; preview requests do not mutate the run.
- The user's own-code license remains undecided; preserve vendored licenses and
  OSM attribution without inventing a new repository license.

## Scope and design intent
- This is an independent voxel massing project, separate from Module Lab.
- Cached OSM geometry may be read from the sibling Module-Lab checkout using
  `voxelform/context.py`. Do not import its generator/policy code or mutate its
  datasets. Preserve metric coordinates, original triangle meshes and provenance.
- The workspace is 3D only: default orthographic axonometric, optional perspective,
  and six-axis view snapping. Surroundings use flat shading and are opaque by
  default, with an explicit transparency checkbox. Keep display choices separate
  from evaluation. No floor-plan pane is wanted in this project. Solid voxel mass uses flat lit
  shading and receives shadows from the existing shared-engine directional light;
  do not revert it to an unlit uniform tint. Keep evaluation colours separate.
  User mood boards favour pale blue interventions, white context and a soft
  peach plot. Distinguish roofs and wall directions even in complete shadow
  using face-tinted material fill; retain actual shared-engine shadows. Solid
  cubes touch without artificial gaps. Current user preference is surface outlines,
  not silhouettes: merged voxel boundary creases, 3 px at a building reference scale, HSL lightness 50%.
  Preserve the blue hue (~227.90 degrees) and saturation (100%) from #1645ff.
  Solid shading remaps shared-engine illumination/shadows to HSL L=70–95%,
  interpolating in sRGB so hue/saturation remain fixed. No silhouette mask pass.
  Lift display mass AND edges just above the rendered parcel top to keep bottom
  edges out of the plot surface; this is visual only, not occupancy/evaluation.
  The bundled LineMaterial requires resolution updated from renderer.getSize.
  Optional
  camera-facing tree sprites may illustrate fetched green polygons only, never
  invented vegetation on unmapped ground. They are not surveyed trees or part
  of environmental evaluation. The present cached dataset has no green meshes.
- Reuse Module Lab's actual `frontend/public/scripts/urban_viewer.js` graphic
  engine and its dependencies through `/shared-viewer/`. Do not recreate its
  renderer, lighting, camera controller or gizmo. `public/viewer.js` is only a
  VoxelForm overlay/protocol adapter. Do not change the sibling engine for this project.
- Do not import modular dictionaries, module reuse objectives, module equivalence,
  or the 90% full-wall interface requirement into this project.
- Preserve the user's existing Module Lab checkout, live server and saved policies.
- Optimize credible building massing on real plots and surrounding context.
  Visual complexity is a welcome byproduct, never an objective by itself.
- Do not claim a massing proposal establishes detailed rooms, circulation,
  structure, daylight compliance or acoustic compliance.

## Active organic 3D preview (2026-11-18)
- Default demonstration site: `m_bcn_0093` (Barcelona / Eixample), served via
  `main.py` (`unified_server.app`, port 8011; note `uvicorn` runs without
  `--reload`, so restart `main.py` after backend edits).
- Active `UnifiedSession` defaults to `UnifiedLearner(..., organic_3d=True,
  floor_copy=False, net_guided=True, amount_feedback=True)` with `Organic3DNetwork`
  (`coarse_2d1d_v2`, 12 input channels: 10 standard + `long_narrow6` `< 6m` strip
  map + `usable`; strided 2D stem/down -> factored 2D+1D blocks + `1x1`
  coarse `local_skip` -> `3.0 * tanh(coarse / 3.0)` bounded logits; ~20.8× faster
  MPS fwd+bwd than the prior `Conv3d` full-res network).
- **Organic decoder without legacy vertical locks**: In `organic_3d=True`,
  cantilever / vertical-support restrictions (`changed[1:] & ~changed[:-1]`),
  persistent `copy_labels` coupling, `col_mean` / `cummin` column-collapse
  averaging, and hardcoded inference score overrides are removed. Legacy
  `floor_copy=True` remains in `voxelform/unified.py` only for backward
  compatibility with existing unit tests (`tests/test_floor_copy.py`,
  `tests/test_unified_server.py`).
- **Two-tier vectorized anti-sliver rules**:
  1. Hard width guard (`min_room_width = 3` on `plot_area >= 400`): async
     2-pool GPU `clean_floor_tensor` (zero sync stalls) and pure-NumPy
     separable bitwise `clean_floor_numpy` (~0.14 ms) eliminate all 1-voxel
     slits/walls and 2-voxel-wide long strips while permitting arbitrary 3D
     cantilevers, overhangs, and bridges.
  2. `< 6m` long narrow strip defense (`long_narrow_mask(occ, width=6, stub=2)`):
     3-pool vectorized detector flags narrow `< 6m` strips extending > 2 m past
     any 6×6 core (exempting 2 m stubs on stepped facades). Fed as CNN channel
     10, used in GPU carve/grow guards, excluded from `usableGoodArea`, and
     penalized `-0.55 * ln6_frac` in `organic_objective` (`0.65 * usable_good +
     0.55 * sun + 0.45 * diff + 0.10 * roof - 0.20 * deep - 0.55 * ln6_frac -
     0.08 * noise`).
- **Vectorized engine & server speedups**:
  - `gaussian_field` (`voxelform/patch_policy.py`): exact separable 1D (Z) + 2D
    (Y, X) convolutions replacing `F.conv3d` (zero `Conv3d` MPS stall).
  - `diffuse_access` (`voxelform/fields.py`): single preallocated `padded` buffer.
  - `SelfShadowMaps.query` (`voxelform/shadows.py`): batched `(U, V, S)` matmul.
  - `project_organic_3d` (`voxelform/unified.py`): `_remaining_organic_valid`
    avoids duplicate `exterior_air` / `ndi.label` passes.
  - `UnifiedSession` (`unified_server.py`): cached initial-state snapshot reset
    in `next_episode()` (`1.34 ms`), 4-step training burst loop, and cached /
    view-aware `/api/state?view=solid` (`18.1 ms` / `211 KB` uncached, `0.02 ms`
    cached vs `73.2 ms` / `1,186 KB` for `view=all`).
- **Dense Spatial RL Credit Assignment & Live Dashboard**:
  - `UnifiedLearner` (`organic_3d=True`) wraps mask decoding in `no_grad()`,
    combines multi-episode GAE (`rollout_steps=16`, `audit_steps=5`,
    `lr=1.2e-3`, `weight_decay=1e-4`, `mark_done()`) with an experimental physics-shaped
    dense spatial auxiliary objective (not an exact control variate) (`(latent.sigmoid() - mean.detach().sigmoid()) *
    des_c`) using orientation-neutral room-scale physical desirability
    (`compute_spatial_desirability`), weighted by `1.2 / sqrt(N_elig)`.
  - Built-in **Training & Massing Dashboard** (`public/index.html`,
    `public/style.css`, `public/app.js`) shows live KPI cards (`Organic score`,
    `Usable good area`, `Direct sun · Diffuse sky` kept separate, `Geometry
    health`, `RL policy`, `Throughput`), learning curves (`#qualityChart`,
    `#history`), recent episode table, and `Episode-sync 3D during training`.
- **Verified checks (`results/organic-3d-rl-demo-2026-11-18.json`, 2026-11-18)**:
  - Unit tests: **168 Python tests** + **10 JavaScript tests** passed.
  - 35-episode M4 MPS continuous training on `m_bcn_0093` (`53.85 s`, `1.539
    s/ep`, `19` updates, `maxSlimVoxels = 0`): EP 30 reached `obj=0.7041,
    good=3,061 m², sun=47.8%, diff=9.9%`; EP 35 reached `obj=0.8052,
    good=2,893 m², sun=59.9%, diff=13.6%, ln6=5.9%`.
  - 5-ep eval (`trainedEval` vs `untrainedEval` on `m_bcn_0093`):
    `meanOrganicObjective` `0.3260 -> 0.8224`, `meanUsableGoodArea`
    `995.0 -> 2,957.0 m²`, `meanDirectSunProxy` `38.3% -> 61.2%`,
    `meanDiffuseFacadeProxy` `4.0% -> 14.0%`, `meanLongNarrowFraction`
    `13.6% -> 5.5%`, `maxSlimVoxels = 0`, `deepFloorArea = 0.0 m²`; `farSuccessRate`
    was `3/5` (`meanFar = 3.1084`, 2 slight overshoots at `3.14` and `3.16`).
  - Zero-shot held-out eval (before -> after): `m_bcn_0094` `obj 0.1089 -> 0.6307`,
    `usableGoodArea 288.7 -> 1,973.3 m²`, `directSunProxy 25.9% -> 51.0%`,
    `diffuseFacadeProxy 2.4% -> 11.6%`, `ln6 20.6% -> 8.7%`, `slim=0`,
    `farSuccessRate 2/3` (`3.1004` overshoot); `m_nyc_0003` `obj 0.0515 -> 0.4490`,
    `usableGoodArea 130.0 -> 1,944.3 m²`, `directSunProxy 14.6% -> 22.8%`,
    `diffuseFacadeProxy 1.6% -> 6.5%`, `ln6 13.2% -> 4.4%`, `slim=0`,
    `farSuccessRate 3/3`, `0.761 s/ep`. Metrics remain uncalibrated proxies.

## Legacy whole-floor preview (2026-11-14, retained for test compatibility)
- Older `floor_copy=True` behavior (retained for `tests/test_floor_copy.py` and legacy `floorCopy` tests in `tests/test_unified_server.py`) uses vertical support checks, `copy_labels`, and whole-plate extrusion:
- For height increase/decrease and action-space continuation, read
  [HEIGHT_GROWTH_HANDOVER.md](HEIGHT_GROWTH_HANDOVER.md) first. It separates user
  intent from implemented restrictions and records unresolved reversibility limits.
- `main.py` serves `unified_server.app` on port 8011. The `devox` branch was
  integrated, preserving its commit. `voxelform.server` is the historical app.
- Default `UnifiedLearner` in the active session uses net guidance and
  `floor_copy=True`. Whole-floor growth and carving/exchange are available from
  step zero; do not add a carve-first schedule or a floor-perfection threshold.
- Growth copies exposed roof plates or (on plots >= 400 m²) coherent extruded
  tower sub-plates via `copy_plates` and `extract_tower_plate`, filtering out
  thin fins and small upper slivers (`min_plate_area`, `coherent_plate`).
  In an exchange, source cuts also affect the new copy. `env.copy_labels`
  couples copied cells to source cells for partial cuts, while complete
  exposed upper tiers (`unstack_cells`) can unstack reversibly without deleting
  their supporting source floor and release stale vertical links on `commit`.
  Column-consistent removal ranking (`0.85 * col_mean + 0.15 * scores`) with
  outward-to-inward `remove_skin` score ordering, vectorized 2D morphological
  thickness + neighbor-count filtering (`clean_floor_tensor` / `clean_floor_numpy`,
  `min_room_width=3` on `plot_area >= 400`), and upward-propagated slim-fin
  absorption prevent hollow outer walls and slim unusable appendages while
  keeping projection selection-only. Reset copy labels with volume.
- Continuous training in `UnifiedSession.finish_episode` stays running across
  episodes (including `step_budget_exhausted` and rejected-draw episodes where
  `unified_actions > 0`), pausing only when `training=False`, on `pause()`, or
  if `edits == 0 and unified_actions == 0`. During `training=True`, `finish` is
  masked on step 0 (`e.edits == 0`) when other edits are legal.
- Enforce direct vertical support for this decoder. No upper occupied cell may
  sit above an empty cell. This disallows cantilevers deliberately and does not
  certify livability. Legacy `floor_copy=False` preserves roof-patch experiments.
- FAR acceptance enables Finish; exchanges may continue within the band.
  Count every attempt, including rejected draws, against the initial workload
  limit + 8 (minimum 8). Budget exhaustion outside FAR is failure. Terminal
  rollouts bootstrap zero. Net guidance regulates area, not an edit phase.
- Ten input channels expose occupancy, air light, noise, allowed area, height,
  same-floor depth, removal eligibility, copy eligibility, external direct sun
  and diffuse facade access. Direct sun and diffuse access remain separate.
- Starts fresh/paused with six initial storeys and ten-storey ceiling. The user
  explicitly waived preserving previous in-memory weights. Manual policy
  export/import supports whole-floor/net-guided states; do not auto-load an old
  failed roof-patch checkpoint. No active autosave/resume is claimed.
- Devoxelization is display only, off by default, finished inference only.
  Algorithms live in `voxelform/devoxelize.py`; overlay in `public/devox-viewer.js`.
  Keep shared-engine shading and zoom-scaled lines. Cache by occupancy content,
  verify revisions before/after reconstruction, and never run it during training.
  Complex meshes may retain boundary/nonmanifold edges: expose diagnostics and
  never claim arbitrary watertight export. `/devoxelize` is a fixed synthetic lab.
- Updated `v2-tower-growth` real-site controls (`m_nyc_0003`, `m_bcn_0094` ×
  seeds 123/456) reached FAR in 4/4 runs with 0 unsupported upper cells across
  7–9 storeys, raising proxy quality from `0.051–0.096` to `0.166–0.207` and
  `potentialUsableArea` from `36–611 m²` to `1,369–2,257 m²` in 10–12 accepted
  edits; see [results/floor-growth-preview-v2-tower-growth-2026-11-14.json](results/floor-growth-preview-v2-tower-growth-2026-11-14.json).
  This establishes geometry/exposure-guided action-space improvements, not
  converged RL learning or residential compliance.
- Older sections below describe the legacy FAR-only policy and prior research;
  the active contract above takes precedence where they differ.

## Representation and actions
- A voxel is computational resolution, not a construction module.
- Initially model occupied/empty volume; defer corridor and room-use labels.
- Each action is one directly generated multi-voxel mask. Freeze eligibility
  before sampling and commit in bulk. Facade cuts use a frozen 1/2/3-layer lateral band
  (`facade_layers`, default 1). Layers expand horizontally from the current skin
  before sampling; no new eligibility is exposed during commitment. Optional deep eligibility
  is max(facade,clamp((same-floor depth-8)/8,0,1)); it intentionally enables interior
  cuts beyond the shallow band, not repeated facade peeling inside one action.
- The server uses `PatchLearner`: dense CNN fields pool to Gaussian latent means,
  sampled/interpolated/smoothed into a coherent mask. Inference also samples.
  Gaussian-latent A2C log probabilities include only consumed rejection retries.
  Do not use independent-Bernoulli scores for correlated masks or reintroduce a
  candidate-mask bank. Independent and smoothed-Bernoulli modes are controls.
- Depth is exact horizontal centre-to-wall distance on the same floor. Roofs,
  undersides and other floors never shorten it. GPU depth uses native tensor
  planar scans/reductions with independently batched floors; CPU SciPy is the
  reference/CPU option. The 3D edit field is distinct from the 2D depth field.
- Roof-open interior courtyards may later join the side facade. New interior
  cuts must reach exterior air after the complete action. Limit disconnected
  openings per floor, not by a 3D sky-connected component alone. Existing
  openings are preserved, merging is allowed, and inner courtyard walls become
  editable facades. Defaults: at least 4m² per new opening and maximum one per
  floor, adjustable. These are research constraints, not compliance.
- Ordinary retained edit patches have at least four cells and at most four
  connected regions; the adaptive allocator permits smaller patches for the last
  16 cells (maximum four regions and courtyard safeguards still apply). Live edits
  use workload-based amounts with no fixed cell cap; the original 256-cell control
  remains selectable. Sampling may select larger fields, but projection only
  prunes unsafe selected bits; it never synthesizes edits to connect a well.
- Prefer actual GPU tensor execution for sampling, fixed-cap ranking, planar
  depth and physical fields. Cache kernels and keep the common retry path to one
  draw, decoding the remaining seven on-device only after rejection. Native CPU
  connectivity/ground/air/opening checks remain; do not claim full GPU residency.
- `voxelform.patch.v1` saves model/optimizer/config. V1–V5 imports preserve compatible
  weights but reset incompatible sampling semantics/counters. Keep all originals.
  Automatic coupled generation saves and a final atomic manifest preserve weights,
  optimizer, geometry/settings after updates/pauses. Normal startup resumes paused.
- Above the FAR target subtract; below it add; at the quantized target finish.
  Keep this operation rule explicit and separate from the learned spatial mask.
- FAR is gross occupied floor area / source parcel area, not the bounding box
  or potential-usable area. Editable study presets are low 1.5, medium 3.0
  (default), high 4.5, plus custom. They are not zoning or compliance claims.
- Live and headless training accept FAR within ±0.1 of target by default.
  Use integer cell bounds within that range; coarse budgets count work to the
  nearest accepted edge. No centre-chasing penalty inside the band. Explicit
  exact-target experimental mode (`far_tolerance=0`) remains for historical tests. Reject
  unreachable targets. No admissible edit is a failure, not success.
- The adaptive amount allocator freezes an episode budget at its start: let M be
  required voxel change, N the initial lateral-facade capacity (growth eligibility
  for addition), P=ceil(M/(0.5*N)), subject to any explicit per-edit cap. The limit
  is P+2+ceil(0.2*P). Deep permissions do not inflate N. Zero work ends immediately.
  Deadline exhaustion is failure, with continuous remaining-error penalty.
- Adaptive amounts start from remaining work / remaining productive steps and
  compensate for observed applied/generated retention (EMA, initial .65). The
  neural correlated field still directly selects spatial edits. Conditional
  rejected draws may request smaller amounts. The deadline never extends.
- Probability-area, fixed-quota and learned-amount alternatives remain controls.
  The learned variant adds countdown/work/capacity conditioning and a Gaussian
  amount action; include its score for the consumed retry prefix. Short learning
  tests have not established an advantage over feedback scheduling.
- Optimizer rollouts of 32 mask actions are **not episodes**. Bootstrap nonterminal
  rollouts, preserve pause/resume state, and do not reset volume at a rollout
  boundary. Training starts another episode only after completion.
- Track FAR on every edit. Environmental fields are audited every eight masks and
  at completion; expose the audit edit and stale-field status honestly.
- Keep solid connectivity, at least one ground anchor, and exterior-air
  connectivity with whole-mask compiled topology checks. These do not certify
  structural support or residential program.
- Room labels and 1m cells must not be presented as actual livable rooms.

- Keep geometric depth separate from sky access and actual daylight compliance.
  Depth updates every mask; sky/sun audits may lag eight masks. Default reward is
  unchanged; optional depth reward/input remain experiments. The selected depth
  field influences edit eligibility even though the CNN still has five inputs.

## Training research and updated massing objective
- `voxelform/unified.py` / `tools/unified_training.py` implement the unified action protocol, promoted to the active app on Nov 14.
  Recommended experimental variant is `--net-guided`: FAR controls net area,
  while paired removal/roof growth is legal from the first action. Pure net
  growth above the band and pure removal below it are masked; mixed edits remain
  available throughout. It is not a carve-first/grow-later phase schedule.
  Full free-operation pilot failed FAR; retain those failures. Safe removal
  pruning + net guidance reached FAR in 10/10 controls on two plots (3 initial
  seeds and 2 trained checkpoints, not 10 independent sites). This establishes
  feasibility, not a learned daylight/quality advantage. Legacy checkpoint
  flags default false on load to preserve their original decoding behavior.
- Use bounded wall-clock tests on M4; report setup, audit and optimizer time,
  success rates and held-out environmental metrics. Current short pilots do not
  establish convergence or a curriculum advantage. Preserve live preview weights.
- Optional `Environment.precompute_external()` exactly caches immutable neighbour
  receiver visibility once per scene. Account for preparation cost; do not apply
  it blindly to single previews. Self shadows and diffuse fields remain dynamic.
- `light` stores sky diffusion in air, zero in solids. `facade_sky_access` samples
  lateral facade-adjacent air with the measured sky boundary; roofs are excluded.
  Diffuse facade sky access is separate from direct-sun potential and the viewer’s
  artistic shadow. Do not substitute one for the other. The nine-channel research
  actor adds diffuse facade access explicitly; `.environment_patch.v2` is an
  experimental checkpoint format, separate from the live five-channel actor.
- Updated user objective: permit redistributing floor area after FAR is reached,
  potentially into several taller grounded buildings. The active growing preview permits this subject to its whole-floor/support rules. Test paired equal-area neural subtraction/addition masks,
  an explicit height ceiling separate from initial mass, and optional top-storey
  duplication/roof extrusion. Retain facade snapshots and meaningful topology
  guards, but do not preserve the old single-component/stop-at-FAR assumptions
  as universal design requirements when implementing this new action space.

## Future interior and facade evaluation
- If real residential rooms are introduced, retain the user's frontage target:
  at least 2 m of window frontage per 15 m² of room area; more is allowed.
  Do not apply this requirement independently to each numerical voxel.
- Privacy concerns realized window-to-window separation: 25 m and plan-view
  outward sectors of ±30 degrees. Use window midpoints for the fast baseline;
  nearly perpendicular target windows should not be treated as directly facing.
  Document the angular interpretation and test drawn examples before claiming
  conflicts. No 3D cone construction is needed for this plan-view rule.
- Core spaces are exempt from residential window/daylight requirements. Allow
  multiple cores when required, rather than assuming exactly one per building.
- Keep intermediate proxies separate from realized facade/room checks. Do not
  call every occupied cell livable, nor a massing score a compliance certificate.
- For a simple early facade audit, all exterior walls may be treated as windows;
  any later solid-wall/glazing allocation must be explicit. Do not draw decorative
  glass outlines in floor plans.

## Environment and speed
- The grid UI has exactly Auto, World, Principal 1 and Longest edge. Auto is the
  fresh-session default; saved/legacy alignment remains intact on resume. Other
  backend alignments remain readable for compatibility. Auto searches angle and quarter-cell offsets
  to maximize fully contained 1 m squares using the union of ground triangles;
  holes/concavities remain excluded. Manual modes retain centre rasterization.
  This bounded 2D fit optimizes footprint coverage, not light or learned quality.
  Keep original world geometry, north/south and sunlight fixed: simulation meshes
  AND ray directions transform into grid coordinates; the viewer transforms back.
  Changing alignment explicitly resets volume but preserves learned weights,
  source parcel area, current floor count and settings. Save alignment in the
  coupled autosave. Never silently rotate/reset a user's current preview.
- Performance is a primary requirement. Measure setup, per-edit updates,
  receiver queries, learning, rendering and CPU/GPU transfers separately.
- Keep tensor simulation/evaluation/model work on one device where practical.
  CPU scene preparation, control and UI are acceptable; disclose fallbacks.
- Benchmark actual CPU and available GPU execution. Built-with-MPS is not proof
  that GPU execution works. Never report unavailable hardware as measured.
- Test both external-shadow methods: sun-aligned depth maps and cached shadow
  masks on up-sun enclosing-box faces, with multiple sampled sun directions.
- Derive required faces from actual directions; do not assume three or four
  faces cover every season/orientation. Separate external and self-shadowing.
- Cached external results can be reused for old receivers. Self-shadowing can
  change on unchanged receivers after edits; invalidate affected rays or update
  the dynamic field. Do not update only newly exposed receivers blindly.
- Treat highway/rail distance decay as a relative noise prior unless calibrated.
  Distinguish screening/diffraction from distance; sum source intensities rather
  than adding decibels. Do not claim an uncalibrated field is dBA.
- Keep direct sun and diffuse-light/sky-access proxies separate. Fast diffusion
  or decay fields are approximations and need independent checks in cavities.

## Verification and benchmarks
- Reach first for rg/rg --files. Verify this project's Git branch before running
  or editing; use codex/ branches by default. Never alter the Thesis branch.
- Test meaningful geometry/field/action invariants. Small fixtures are for
  correctness, not evidence of throughput or generation quality.
- For scale tests use L/XL-sized envelopes, 4 and 8 floors, 8 and 12 sun samples,
  and approximately 10–20 contextual buildings; retain actual sizes and counts.
  Old 120-module caps are not meaningful for this voxel project.
- Report map accuracy, runtime and memory together, including setup cost.
  Validate cached/approximate results against independent direct calculations.
- Test varied/held-out plot shapes, orientations and contexts. Distinguish
  synthetic method tests from real-site generalization. Compare at matched
  usable-floor-area targets; do not reward deleting most of the building.
- Preserve seeds, settings, revision/source hashes, raw results and failed runs.
  Never fabricate speedups, correlations, convergence or learning improvements.
- Learning must be demonstrably trainable (actor gradients/weight changes), but
  those alone are not evidence that quality improved. Use untrained and simple
  search/greedy controls and report failures. Do not train on v0.9.0 as a teacher.
- Diffusion is a later stage, trained on diverse validated outputs; do not mix
  unfinished intermediate shapes into the accepted-final-shape dataset.
- Pause the active training wall timer when paused. Preserve model state across
  previews; do not restart a live session with unsaved in-memory policy.

## Delivery and governance
- Keep README, instructions, issue/experiment notes and result indexes current.
- Before a commit, inspect the diff, check no unrelated files were changed, and
  run appropriate tests. Keep this an independent Git repository/version.
- Never git push without explicit user consent. Do not publish/deploy or send
  messages to other people without authorization.
- When work takes time, give concise, substantive updates. Be candid about what
  is implemented, untested, incomplete or only hypothesized. Avoid repeated
  permission questions for already authorized/reversible work.
- Respect concurrent user changes. Maintain own clearly named resumption notes;
  do not read another agent's private memories without user instruction.

- Isolated `voxelform.redistribution` tests paired constant-volume removal/roof
  growth, grounded splitting and optional top-storey copying. It is not wired
  into production FAR termination or checkpoint semantics. Preserve this
  distinction; do not claim learned tower layouts from geometric/gradient tests.
