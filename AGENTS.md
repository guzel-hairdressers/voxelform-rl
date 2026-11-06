# VoxelForm RL — Agent Instructions

## Scope and design intent
- This is an independent voxel massing project, separate from Module Lab/Thesis.
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
  Keep/undo are edit operations, not building-state categories.
- Use stepwise coherent facade/roof/atrium cuts anchored to exterior space.
  Do not permit arbitrary disconnected interior voxel deletions.
- Allow restoration inside the permitted envelope. State the implemented scope
  precisely (e.g. undo of a previous cut versus unrestricted volume addition).
- Hardcode geometric admissibility; let the policy choose location, shape,
  size, depth, floor range, sequence and stopping. Do not prescribe one layout.
- Room labels and 1m cells must not be presented as actual livable rooms.

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
