> Historical single-voxel stage. The current action space is [direct neural facade masks](../direct_neural_masks_2026-11-07/README.md).

# Single-voxel FAR episodes — 2026-11-07

Repository: `/Users/ruslan_faz/Desktop/Work/Thesis/VoxelForm-RL`.
Branch: `codex/voxel-prototype`. All changes remain local and uncommitted.

The user clarified the intended representation: one exterior voxel per move,
with a sequence of subtractions/additions ending at a configurable FAR condition.
The earlier rectangular cut proposals and freely selected Stop action were the
wrong action space. The old session also conflated 16-step optimizer rollouts
with building episodes. This follow-up replaces those behaviours.

## Resulting behaviour

- **Train ↔ Pause training** is one button. The generation button similarly
  becomes Pause generation while inference runs. Pausing retains the volume,
  committed policy state, pending rollout and active-time clock.
- A move changes one 1 × 1 × 3 m voxel, contributing 1 m² of gross floor area.
  The policy selects an admissible exterior location. The FAR controller permits
  removals above target and additions below it; no macroscopic cut, undo slot or
  early Stop action remains. Additions can fill any admissible adjacent exterior
  air cell within the permitted envelope, not just the last removed cell.
- FAR = occupied gross floor area / source parcel area. The plot bounding box and
  potential-usable area are not substituted for parcel/GFA measurements.
- Residential study presets: low 1.5, medium 3.0 (default), high 4.5, plus custom.
  These are editable research scenarios, not regulatory density limits.
  Applying a target pauses and retargets the existing mass without replacing it.
- Success occurs at the nearest integer occupied-cell count for the target FAR.
  Half-cell ties round upward; tolerance is half a cell divided by parcel area.
  Infeasible targets are rejected. Exhausting admissible edits is recorded as
  failure and pauses, rather than pretending an episode succeeded.
- Training repeats completed building episodes. Inference stops after the goal.
  Optimizer rollouts of 32 edits bootstrap from the nonterminal state and never
  end/reset a building episode. The API reports actual edits and remaining edits.

## Implementation

- `voxelform/actions.py`: incremental CPU exterior/frontier bookkeeping. Explicit
  exterior-air reachability excludes sealed voids. Conservative 3×3×3 local
  6-connectivity checks preserve solid connectivity, a ground anchor and exterior
  air during growth. Deletion incrementally opens newly connected air. This does
  not establish engineering support, legal rooms, daylight or residential compliance.
- `voxelform/environment.py`: exact count/FAR updates per edit, target validation,
  arbitrary adjacent additions, true parcel denominator, terminal condition.
  The old fixed 45% retention penalty was removed because it conflicts with
  deliberately lower density targets.
- `voxelform/policy.py`: single-cell feature gathering instead of dense cut masks;
  actor location selection and FAR-conditioned critic. Per-edit reward is FAR
  progress plus 1 at completion; 0.1 times audited environmental-quality change
  is added at the rollout boundary, with discount 0.99 and value bootstrapping.
- `voxelform/session.py`: complete episodes, partial optimizer rollouts, pause,
  settings, scene replacement and telemetry. `server.py` exposes those operations.
- Environmental sun/sky/usable fields refresh every rollout and at completion;
  FAR/GFA always reflect the current cell state. State includes audit edit and
  current/stale status, and the HUD labels lagging measurements. Rendering can
  show several edits per polling frame; this does not make them one volume action.
- Policy V3 identifies `exterior_single_voxel_far_v1`. Legacy V1/V2 imports retain
  learned encoder/context/local weights, zero obsolete tool-feature columns,
  expand the critic for FAR conditioning, and reset incompatible optimizer moments.
  Original files were saved before migration. A failed checkpoint load cannot
  partially replace the live model. This is migration, not a claim of identical
  learned action semantics or improved design quality.
- The graphic engine remains Module Lab's actual shared `urban_viewer.js`.
  No floor-plan pane or separate camera/renderer was reintroduced.
- `tools/benchmark.py` now identifies and exercises the new action space while
  retaining L/XL cases and audit-per-edit method measurements. No benchmark was
  run for this task; historical rectangular-tool measurements were not overwritten
  or presented as throughput evidence for single-voxel FAR episodes.

## Verification

- **38 Python tests passed**, including 17 new FAR/action/session/migration cases.
  The obsolete undo-slot test was updated to verify one-cell subtraction and
  true one-cell addition back into the permitted envelope.
- **4 browser-adapter tests passed**; JavaScript syntax and `git diff --check` passed.
- Checks cover exterior-only proposals, sealed voids, bridge/ground protection,
  cavity-safe additions, no overshoot/early stop, arbitrary attached growth,
  source-area FAR/rounding, rollout vs episode boundaries, actor gradients,
  pause/resume, infeasible targets, failure provenance and checkpoint migration.
- Browser verified both directions of the primary Train/Pause training toggle,
  medium/high preset application, and complete inference sequences.

Actual cached Barcelona site: `m_bcn_0093`, parcel area **1,450.5 m²**.

| Sequence | Target FAR | Final FAR | Single-voxel edits | Result |
| --- | ---: | ---: | ---: | --- |
| First live training episode, continuing saved mass | 3.0 | 3.0003447087 | 2,881 removals | FAR reached |
| High preset, continuing the paused training mass | 4.5 | 4.4998276456 | 105 additions | FAR reached |
| Return to default medium preset | 3.0 | 3.0003447087 | 2,175 removals | FAR reached |

The two inference sequences left all policy tensors and the update count
(**176**) unchanged. The final state is paused at **4,352 occupied cells/m²**, with
zero edits remaining and a completed environmental audit. Floating-point
comparisons of the half-cell tie use a numerical epsilon; no fractional voxel was
invented. These checks verify mechanics and termination, not convergence or
real-site architectural performance. Raw API evidence and source hashes are in
[verification.json](verification.json).

Screenshots:

- [Training button showing Pause training](pause-training-button.jpg).
- [High-density target reached after additions](high-density-reached.jpg).
- [Final medium-density target reached](workspace.jpg).

## Current runtime and preservation

Preview: **http://127.0.0.1:8011/**, paused, default medium target 3.0.
The original port-8010 prototype process was left untouched.

Ignored checkpoint artifacts under `results/checkpoints/`:

- `pre-far-2026-11-07-*`: initial policy/layout backup.
- `pre-far-deploy-2026-11-07-*`: last paused legacy state and V2 policy before
  replacing the preview process. Its 7,233-cell layout was restored byte-for-byte
  before the new FAR actions ran; its 11 historical updates were retained.
- `far-before-ui-validation-2026-11-07-*`: paused new-action policy/layout before
  changing density presets for the end-to-end addition/removal checks.
- `far-current-2026-11-07-state.json` and `far-current-2026-11-07-policy.pt`:
  final medium-density geometry/settings and V3 policy. State snapshots restore
  geometry/settings, not live autograd traces or a historical episode cursor.

`VOXELFORM_RESUME_STATE` restores an occupied-layout/settings snapshot and
`VOXELFORM_POLICY_PATH` restores the policy. Check listeners and saved state
before restarting a live process. No Git commit or remote push was made.
