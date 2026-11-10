# VoxelForm RL — Agent Instructions

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
- `voxelform/unified.py` / `tools/unified_training.py` implement the separate
  unified action protocol. Do not overwrite the user's active live policy.
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
  potentially into several taller grounded buildings. Current production rules
  do not yet permit this. Test paired equal-area neural subtraction/addition masks,
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
