# VoxelForm RL

An independent, trainable voxel **building-massing experiment**. Each action
applies a coherent multi-voxel mask using frozen **facade and optional deep-interior
eligibility**, changing all selected cells together. Newly exposed facade cells
become eligible only on the next step. Episodes continue through a sequence of
masks until the configured floor-area-ratio (FAR) condition is met or the
workload-based step budget is exhausted (a failure).

The current workspace uses Module Lab's compact white interface with a single
3D viewport running Module Lab's actual `urban_viewer.js` graphic engine. The
engine, Three.js dependencies, lighting, camera controls and gizmo are served
directly from the sibling checkout without copying or modifying them. VoxelForm
only adapts its voxel mass into an instanced overlay in that engine. It defaults
to an orthographic axonometric view. Switch to Perspective,
click an axis in the orientation gizmo to snap to one of six orthogonal views,
or use **Fit view** / double-click to reframe. Surroundings are flat-shaded and
opaque by default; **Display → Transparent surroundings** makes them translucent.
The solid voxel mass uses flat Lambert shading with the shared engine’s existing
directional light and shadow map, so differently oriented faces and recesses read
distinctly. Evaluation colours use a restrained face-orientation cue.
There is no floor-plan pane.

## Run locally

Keep the Module Lab checkout beside this repository as `../Module-Lab`, with its
cached urban scenes and `frontend/public/scripts/urban_viewer.js` available.
Alternatively set `VOXELFORM_CONTEXT_ROOT` to that checkout. These shared assets
and datasets are read-only runtime dependencies and are not included here.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Open **http://127.0.0.1:8010**. It leaves Module Lab on port 8000 alone.
The currently running development preview is **http://127.0.0.1:8011**; port 8010's
older in-memory session was preserved while verifying this update. To choose a port:

```sh
VOXELFORM_PORT=8011 .venv/bin/python main.py
```

Device selection prefers available CUDA/MPS and otherwise uses CPU. The current preview uses MPS with native planar depth. To explicitly choose a device:

```sh
VOXELFORM_DEVICE=mps .venv/bin/python main.py
# or VOXELFORM_DEVICE=cuda on a supported NVIDIA machine
```

During initial development this project's code can also run with the existing
`/Users/ruslan_faz/Desktop/Work/Thesis/Module-Lab/.venv/bin/python`; the Python backend
does not import Module Lab's generator or policy code. Actual CPU and MPS forward/backward execution were measured on this machine;
CUDA was unavailable.

**Train** changes to **Pause training** while running; the same button pauses
and resumes without resetting the building or learned weights. **Generate one**
runs one inference episode and pauses on FAR success or failure. **One mask** applies one complete
inference facade mask. Training repeats complete episodes, starting each new
one from the full permitted envelope. **Reset volume** explicitly restores that
envelope; it is separate from ordinary voxel editing.

The server now uses **smoothed correlated patches**: one CNN pass supplies dense
logit fields, pooled into a coarse Gaussian latent distribution. The model samples
a coherent field, interpolates/smooths it, applies facade/depth weights, and
thresholds it into one mask. Train and Generate one both sample this distribution.
The A2C score uses the actual Gaussian latent log probability. No candidate patch
library or patch-ranking policy is involved. Independent/smoothed-Bernoulli controls
remain selectable under **Edit patches**.

Depth is horizontal distance to a facade on the **same floor**, computed natively
on the selected GPU. Roofs and other floors do not shorten it. Deep-interior cuts
are optional: facade weights are 1; interior weights ramp from 0 at 8 m to 1 at
16 m. Roof-open courtyards can later expand through the side facade. The preview
limits new openings to one per floor and at least 4 m²; ordinary edit patches
retain at least four cells, with at most four patches. The adaptive edit amount has no fixed voxel cap;
the original 256-cell probability-area control remains available.
The final FAR residual can be smaller. Controls expose opening/edit limits.

**Edit patches → Edit amount & episode length** now includes a workload-based
adaptive allocator. Let M be the required voxel change and N the initial facade
capacity. It freezes P=ceil(M/(0.5×N)) productive steps plus 2+ceil(0.2×P) reserve
steps. An optional explicit voxel cap also limits estimated capacity. Interior
depth eligibility does not inflate N. The model generates the spatial field;
a feedback controller adjusts its selected area for observed pruning losses.
The deadline never moves. The final 16 cells may use smaller edit patches while
retaining topology, opening-area and maximum-region constraints. Unreached FAR
at the deadline is penalized, never reported as success. Fixed-quota and learned
amount variants remain experiments. Set maximum cells per edit to **0** to remove
the fixed cap. Existing saved sessions preserve their original settings.

See [workload-budget experiments](agent_notes/workload_budget_2026-11-09/README.md)
for controls, failed deadline hypotheses, learned-amount tests and CPU/MPS checks.

Save/load uses `voxelform.patch.v1`. Legacy V1–V5 policies retain compatible weights
as initialization and reset incompatible optimizer/sampling counters. Historical
updates are displayed separately. Automatic checkpoints save weights, optimizer,
geometry and settings as a coupled generation and restore the latest pair paused.
They live under ignored `results/checkpoints/autosave/`; original checkpoints are
preserved. No convergence or design-quality improvement is claimed.

See [the coherent-mask implementation and experiments](agent_notes/coherent_masks_2026-11-09/README.md)
for sampler comparisons, exact 2D GPU depth tests, failure cases, CPU/MPS timing,
checkpoint preservation and the verified native-engine preview.

See [the M4 training study](agent_notes/m4_training_2026-11-09/README.md) for a bounded
headless trainer, exact cache preparation, reward/curriculum pilots and a separate
diffuse facade-light input. Reaching FAR still ends the production episode;
[paired redistribution and top-storey growth](agent_notes/m4_training_2026-11-09/REDISTRIBUTION.md)
are the next action-space study, not an implemented live capability.

## FAR targets and episode completion

Residential study presets in **Study settings → Residential density**:

| Preset | Target FAR |
| --- | ---: |
| Low density | 1.5 |
| Medium density (default) | 3.0 |
| High density | 4.5 |
| Custom | Editable positive target within the current envelope |

These are editable research scenarios, not jurisdiction-specific zoning limits.
FAR is **gross occupied floor area / source parcel area**. It is independent of
potential usable area and does not use the parcel bounding box as denominator.
Each floor voxel contributes 1 m². Because cells are discrete, completion uses
`round(target FAR × parcel area)` occupied cells (half-cell ties round upward).
The tolerance is half a cell divided by parcel area. Unreachable targets are
rejected; the UI shows the current envelope's maximum FAR.

Above target, the policy generates removal masks using the selected facade/depth eligibility. Below target, it generates
facade-growth masks attached to existing mass within the permitted envelope. **Apply FAR
target** pauses and retargets the current mass without replacing it. Thus raising
the density of a carved mass produces a series of growth masks, not a whole-volume
undo. At the goal, the episode succeeds. If no admissible move remains, it fails
and pauses; an optimizer update or a single cut cannot count as success.

Facade detection uses vectorized whole-grid neighbour operations. The model's
mask is clipped to the selected eligibility snapshot, including stepped/rotated
facades. Facade-only mode excludes hidden layers; deep mode explicitly permits
occupied cells beyond 8 m, subject to roof/opening safeguards. Facade-edge growth may
extend upward/downward. Roof-only interior cells are not ordinary removal cells.
A final FAR residual may require a one-cell mask; ordinary masks change multiple
cells.

Compiled NumPy/SciPy kernels check six-neighbour solid connectivity, ground
anchors and cavity closure on CPU. Projection only removes unsafe selected bits;
it may reject the generated mask. The selected mask is applied in one bulk tensor
update. The model, correlated sampling, planar depth and environmental tensors use the selected CPU/MPS device.
These checks are massing constraints, not structural certification.

FAR/GFA and geometric depth update after every mask. Sun/sky/usable fields refresh every eight masks
and at completion; the HUD labels their audit step if they lag current geometry.
Training rollouts remain 32 masks and do not reset the episode. The UI separates
mask actions, total changed voxels, and voxels remaining to the FAR target.

See [the direct neural-mask report](agent_notes/direct_neural_masks_2026-11-07/README.md)
for correctness checks, checkpoint migration and measured L/XL CPU/MPS timings.
Earlier [FAR episodes](agent_notes/far_episodes_2026-11-07/README.md) and
[sampled-mask measurements](agent_notes/facade_masks_2026-11-07/README.md) are
historical stages superseded by the direct generator.

## Fetched urban surroundings

By default, the app reads the existing cached OSM scenes in the sibling
`../Module-Lab/frontend/public/` directory (with the legacy `public/` location
as a fallback). It reads cached geometry and browser renderer assets without
changing Module Lab's files or importing its generator/policy code.
Set `VOXELFORM_CONTEXT_ROOT` to use a different Module Lab
checkout. Module Lab does not need to be running. Its browser renderer assets are
served read-only under `/shared-viewer/`; the sibling checkout is therefore a
runtime dependency for both cached scenes and graphics.

The current catalog has 390 cached sites across six cities. City, area tier and
site selectors choose among the saved scenes. The default is `m_nyc_0003`.
The site's actual triangulated parcel is rasterized at 1 m, including its holes;
its original neighbouring building meshes and street meshes retain metric scale,
relative positions and cached heights. Grid bounds currently reject parcels over
128 m per side or 10,000 horizontal cells with an explicit error. Rasterized area
can differ slightly from the source polygon area.

The imported building triangles are used for external direct-sun visibility and
the sky-access boundary, with persistent external-receiver caching. They are not
replaced by bounding boxes. Dynamic self-shadowing retains the voxel method.
The synthetic boundary/depth-map experiments remain available for method tests.
No speedup or mesh-scene throughput claim is made by this integration.

Cached street meshes do not include usable traffic-source data, so noise is
**unmeasured** on fetched sites and its display layer is disabled. The objective
omits the unavailable noise contribution for these sites. Sun directions remain
synthetic, not location/date-specific solar measurements. Heights may include
the original fetcher's estimates; this is reuse of cached OSM data, not a new survey.
Attribution: [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright).

An explicit **Synthetic method fixture** source preserves the original plots,
boxes and line-source noise prior for controlled tests. Switching sites preserves
learned weights. Camera and transparency controls affect the display only.

## What is evaluated

- **Potential usable floor area:** occupied cells within a seven-step horizontal
  path of a facade adjacent to sufficiently sky-accessible air. This is an
  experimental depth proxy, not actual room design or daylight compliance.
- **Direct sunlight:** exposed surface samples against 8 synthetic sun directions,
  separate fixed-neighbour occlusion and changing self-occlusion.
- **Diffuse sky access:** a screened-Laplace decay field in air, with surrounding
  sky openness at the envelope boundary. This scalar approximation can leak
  around corners; it is neither illuminance nor a daylight factor.
- **Relative noise:** finite highway/rail line-source intensity integrals, with
  distance decay. No calibrated dBA, barriers or diffraction are modeled.
- **Sunny roof opportunity:** sun-exposed upward faces, not a demonstration of
  accessible green roofs, soil capacity or planting viability.

The secondary environmental quality is `0.70 usable + 0.18 sun + 0.12 sunny roof
- 0.08 noise`. The old fixed 45% retention penalty was removed so it cannot
conflict with a configured low-density FAR target. The usable fraction is relative to the
**original permitted floor area**, so deleting the building cannot earn a high
usable-area score. Per-mask reward is progress toward the quantized FAR goal, plus 1 at completion.
At each environmental audit, 0.1 times the environmental-quality change is added to the
rollout reward. Nonterminal rollouts bootstrap with discount 0.99.
These weights and approximations are a transparent research baseline, not a
validated design objective. Display all metrics separately when comparing edits.

## Shadow implementations

`voxelform/shadows.py` includes direct ray/rotated-box tests, sun-aligned depth
maps and the proposed enclosing-box surface masks. Face masks store visibility
on each up-sun box face for each sun direction. The union of required faces is
derived, not fixed: the usual unrotated 180-degree test arc needs four; a rotated
grid can require five. Boxes intersecting the envelope require direct fallback.

External receiver visibility is cached by persistent voxel-face IDs. Only newly
exposed external receivers are queried. **Self-shadowing is updated for all
current receivers after every edit**: removing a blocker can illuminate an old
receiver. Cached sun-plane DDA paths gather changing occupancy and reduce depth
on the selected device. Receiver lookup has finite map-resolution error.

Runtime mask/field/network operations use PyTorch on one device. Setup includes
CPU scene creation and DDA path preparation. Python orchestration, action scalar
transfers, dynamic receiver extraction and browser serialization remain; this
is not a claim of a completely GPU-resident simulator.

## Tests and method measurements

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
node --test tests/viewer-data.test.mjs
PYTHONPATH=. .venv/bin/python tools/benchmark.py --device cpu --output results/benchmark-cpu.json
PYTHONPATH=. .venv/bin/python tools/benchmark.py --device mps --output results/benchmark-mps.json
```

The method benchmark uses L (64 × 48 m, 4 floors, 8 sun directions, 10 neighbours)
and XL (96 × 64 m, 8 floors, 12 directions, 20 neighbours). End-to-end edit scenes
currently use 16 neighbours, recorded explicitly. It saves source hashes, raw
timings, memory and shadow disagreement with direct reference calculations.
Small test fixtures verify correctness; they are not throughput evidence.

An [explicit geometric facade-depth field](agent_notes/facade_depth_2026-11-09/README.md)
now updates after every edit and is exposed by the API. It maps depth above 8 m
continuously to badness one at 16 m. Optional model input/reward support is for
experiments; the current policy inputs and default objective are unchanged.

## Architecture hypothesis tests

The [2026-11-08 ablation report](agent_notes/architecture_hypotheses_2026-11-08/README.md)
tests single conditioned output, explicit facade inputs, local 3D features and
multiscale context, individually and in combinations. Twelve configurations,
three seeds, matched action budgets, controls and actual CPU/MPS measurements are
retained. Local 3D is promising on synthetic proxies; no configuration establishes
a general learned-quality advantage. Those experiments used the V5 baseline. The current server retains its floorwise
CNN but uses the correlated latent sampler and safeguards described above.

## Current limits

The cavity probes in `results/field-probes.json` show that the diffusion field
has much stronger attenuation than direct sky visibility. The two are not equal
or calibrated; the 0.15 facade threshold is a provisional heuristic. Treat the
resulting potential usable-area metric as experimental until held-out validation
and calibration establish whether it is useful. Do not interpret it as a count
of habitable rooms or use it to claim successful daylight optimization.

The demo supports cached real plots/context and retains synthetic sun paths.
Fetching new coordinates, astronomical times, facade privacy, actual rooms/corridors/cores,
structural engineering, repair/validation of arbitrary imported disconnected mass,
accurate sky radiance, acoustic
screening and policy-quality/generalization evaluations remain outstanding.
Actions use a frozen current facade/depth eligibility field; their sequence can form
varied surfaces and roof-open courtyards. Additions are adjacent facade growth inside the envelope, subject
to the same FAR and whole-mask topology conditions. Diffusion training is deferred until validated data exists.

Nonzero actor gradients and changed weights establish trainability only.
No convergence, design improvement or real-site generalization is claimed.

See [AGENTS.md](AGENTS.md) for the transferable project instructions and
[the urban workspace report](agent_notes/urban_workspace_2026-11-07/README.md) for
the UI/context changes and verification, and
[results](results/) for raw measurements. The vendored browser renderer is
Three.js. The shared engine uses Module Lab's Three.js distribution; the original
prototype's vendored package and license remain in `public/vendor/`.

See [facade-band and paired-edit experiments](agent_notes/redistribution_2026-11-09/README.md) for configurable 1–3-layer editing and the isolated constant-FAR redistribution prototype. Two layers were the most reliable in this small test; the live action protocol is unchanged.

FAR targets now accept ±0.1 by default in the live session and M4 trainer (3.0 accepts 2.9–3.1). Coarse edit counts and budgets use the nearest accepted boundary; no extra completion reward/penalty depends on reaching the centre. Exact-target mode remains available for controlled historical experiments via `far_tolerance=0`. Experimental exchanges may change total volume if the result stays inside the accepted band.

Fetched plots offer four **Voxel grid alignment** choices: **Auto** (default for
fresh sessions), **World**, **Principal 1**, and **Longest edge**.
Auto searches rotation and sub-cell offset to maximize whole 1 m squares inside
the actual parcel (including concavity/holes); manual modes retain the existing
cell-centre raster. North/south, sun and original context geometry stay fixed.
Applying a new grid starts a fresh volume and retains policy weights and floor
count. The current preview is not switched automatically. See the
[grid and quality study](agent_notes/grid_quality_2026-11-09/README.md).

The solid display follows the supplied mood boards: pale blue mass, lighter roofs,
distinct wall tones and white context, with shared-engine shadows. Camera-facing
tree silhouettes are supported on fetched green-area meshes as illustrative
scenery. The current cache contains no such meshes, so no trees are invented.

Solid surfaces touch without voxel display gaps. The current display uses
**zoom-scaled surface outlines**, with merged boundary creases and no coplanar cube/floor
seams. Hue and saturation stay fixed across outline (HSL L=50%) and fill
(L=70–95% from darkest shadow to brightest light), using the shared renderer's
lighting. See [current display verification](agent_notes/outlines_hsl_2026-11-10/README.md).

The [unified action research path](agent_notes/unified_actions_2026-11-10/README.md)
permits removal and roof growth in the same action before FAR is reached. It is
independent of the current live policy. Run a fresh experiment with
`python -m tools.unified_training --tag net-pilot --arms unified --net-guided --seconds 90`.
`--resume PATH --seeds 123` can continue a compatible experimental checkpoint;
changed decoding options reset Adam state, retaining the network. FAR guides net
area rather than imposing a carve/grow sequence. Initial controls establish FAR
feasibility on two plots, not trained environmental-quality improvement.
