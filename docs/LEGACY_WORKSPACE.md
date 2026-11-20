# Historical workspace documentation

This is the preserved pre-research-tooling README. Statements about active models, floor counts and checkpoints are historical. The current contract is the root README and experiment/model/data cards.

# VoxelForm RL

A trainable voxel building-massing experiment on cached real plots and surrounding
buildings. The **active app is the growing preview**, incorporating the `devox`
branch. It starts with six occupied storeys inside a ten-storey height ceiling.
These are massing cells, not designed rooms.

## Run locally

Keep Module Lab beside this repository as `../Module-Lab`, with its cached urban
scenes and `frontend/public/scripts/urban_viewer.js` available. Alternatively set
`VOXELFORM_CONTEXT_ROOT` to that checkout. Shared assets and datasets are read-only
runtime dependencies and are not included here.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Open **http://127.0.0.1:8011**. `python unified_server.py` runs the same app. Set
`VOXELFORM_PORT` for another port, and `VOXELFORM_DEVICE=mps` / `cpu` / `cuda` to
select hardware explicitly. Auto prefers an available GPU. Module Lab on port
8000 is independent. During development the sibling `.venv/bin/python` can be used.
Shapely 2.1+ is required for constrained cap triangulation.

## Try the growing version

- **Generate one** runs an inference episode and pauses. **Train** changes to
  **Pause training** and repeats complete episodes; pausing keeps current weights.
- **One mask** applies one action. Its adjacent selector defaults to the learned
  operation. **Copy top floor · probe**, **Copy floor + carve**, and **Carve · probe**
  let you inspect geometry directly. Probes bypass FAR direction guidance and
  are inference only; Train / Generate one use the learned operation selector.
- **Reset volume** restores the initial six-storey mass without resetting weights.
  Grid/site changes also reset volume; Auto grid alignment is the default.
- **Policy → Save policy / Load policy** exports/imports the new growing policy.
  Startup begins fresh and paused; no older roof-patch checkpoint is silently
  loaded. Older decoder formats remain available in the historical experiments.

The model selects remove, grow, exchange or finish from one actor pass. Additions
now copy one existing building's **entire highest floor**, preserving its holes.
In an exchange the source floor is carved before its final footprint is copied.
Copied cells remain linked to their source for later cuts: selected cuts must
include the linked vertical stack. Unsafe partial selections are pruned, never
expanded by projection. Every retained upper cell requires an occupied cell
immediately below. This restricts cantilevers deliberately; it does not certify
structure, circulation, minimum room widths or residential livability.

There is **no carve-first phase** and no hardcoded "perfect floor" gate. A learned
correlated field selects cuts and which grounded mass receives a floor copy.
With one building there is only one plate to copy, so that action has no spatial
choice until the geometry offers more masses. No sampled candidate-mask bank is
used. The actor remains a ten-channel floorwise CNN with global operation/value
heads and Gaussian latent A2C training; this change does not introduce 3D feature
exchange inside the CNN. The copy/cohort decoder supplies vertical coherence.

Net FAR guidance permits paired carving/copying above and below the target from
the first action. Pure growth is constrained when it would worsen FAR. The default
acceptance is **3.0 ±0.1**; density presets remain editable. FAR acceptance enables
Finish rather than forcing immediate termination. The frozen deadline is the
initial workload budget plus eight redistribution attempts (minimum eight).
Rejected draws consume attempts and may retry on the next action. At the deadline,
FAR outside the band is failure with a remaining-error penalty. Training audits
environmental fields every eight accepted masks and at completion. A 32-action
optimizer rollout is not an episode.

Current mask settings support correlated/smoothed-correlated fields, optional deep
eligibility, opening limits and a whole-action cell cap (0 means unlimited). A full
copied floor must fit the cap; it is never chopped into roof posts to satisfy it.
The original independent-mask/amount controls remain in the historical trainer.

## Display and devoxelization

The white 3D workspace uses Module Lab's actual graphic engine, cameras, lights and
shadows. Default axonometric, Perspective, orthogonal gizmo snaps and Fit view are
available. Context is flat-shaded and opaque by default; transparency is optional.
Pale blue shading retains HSL lightness 70–95%; rich blue surface outlines use 50%
lightness, 3 px at the reference building scale and become thinner when zoomed out.
There is no floor-plan pane.

**Display → Devoxelize after single generation** is off by default. When enabled it
reconstructs only a completed, paused inference building, never during training:

0. Raw voxels.
1. Correlated support-plane loft.
2. Extruded floor slabs.
3. VSA plane arrangement.

These are display experiments from `devox`; voxel occupancy, FAR, sun/depth inputs
and rewards are unchanged. Complex/topology-changing floors can use the same slab
fallback in several modes. T-junctions are conformed before rendering; returned
mesh diagnostics expose remaining boundary/nonmanifold edges. Arbitrary complex
outputs are **not guaranteed watertight**, and reconstructed volume can differ from
voxel volume. Geometry-content caching and revision checks prevent stale meshes.
The comparison lab at **/devoxelize** shows the other agent's fixed synthetic
benchmark, not the current live building. Its 2D slice panel is a mesh diagnostic.

## Growing-preview checks (2026-11-14)

Four matched-seed, untrained runs on two real sites (New York and Barcelona) all
reached FAR in both arms. Whole-floor/link growth used 10–14 accepted edits versus
14–21 for roof patches, and left zero upper cells without direct support versus
428–480. The final retest took a median 0.80 s on MPS for copies versus
1.25 s for patches, excluding setup (7.3 / 11.4 s), UI and polling. Individual
runs varied (0.57–1.61 s / 1.05–2.63 s); this is not an isolated speed benchmark.
Five-second native MPS probes performed two and four optimizer updates
with nonzero gradients. A separate fixture regression tests the exact source/copy
area cost: the old high-density run stalled at 4.370; the corrected run reaches
the 4.4–4.6 band in two actions. This establishes feasibility/trainability, not convergence,
quality improvement or generalization. Seeds, traces, failures, timing and hashes:
[raw comparison](../results/floor-growth-preview-balanced-2026-11-14.json) and
[implementation/testing notes](../agent_notes/growing_preview_2026-11-14/README.md).

The following sections retain the original FAR-only implementation and research
history. `voxelform.server` / `Session` / `PatchLearner` are the **legacy** app;
`main.py` now uses `unified_server.UnifiedSession` / `UnifiedLearner`.

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

See [the direct neural-mask report](../agent_notes/direct_neural_masks_2026-11-07/README.md)
for correctness checks, checkpoint migration and measured L/XL CPU/MPS timings.
Earlier [FAR episodes](../agent_notes/far_episodes_2026-11-07/README.md) and
[sampled-mask measurements](../agent_notes/facade_masks_2026-11-07/README.md) are
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
site selectors choose among the saved scenes. The default demo is **`m_bcn_0093` — Barcelona (Eixample)**, the L-shaped
plot beside a tall building used to demonstrate removing poorly exposed volume
and rebuilding floor area in taller masses. `VOXELFORM_SITE_ID` can explicitly
override the startup site.
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

An [explicit geometric facade-depth field](../agent_notes/facade_depth_2026-11-09/README.md)
now updates after every edit and is exposed by the API. It maps depth above 8 m
continuously to badness one at 16 m. Optional model input/reward support is for
experiments; the current policy inputs and default objective are unchanged.

## Architecture hypothesis tests

The [2026-11-08 ablation report](../agent_notes/architecture_hypotheses_2026-11-08/README.md)
tests single conditioned output, explicit facade inputs, local 3D features and
multiscale context, individually and in combinations. Twelve configurations,
three seeds, matched action budgets, controls and actual CPU/MPS measurements are
retained. Local 3D is promising on synthetic proxies; no configuration establishes
a general learned-quality advantage. Those experiments used the V5 baseline. The legacy server retains its floorwise
CNN; the active growing decoder is described at the top of this file.

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
varied surfaces and roof-open courtyards. Legacy additions are adjacent facade growth inside the envelope. Active additions
copy whole top floors with the support/link rules described above. Diffusion training is deferred until validated data exists.

Nonzero actor gradients and changed weights establish trainability only.
No convergence, design improvement or real-site generalization is claimed.

See [AGENTS.md](../AGENTS.md) for the transferable project instructions and
[the urban workspace report](../agent_notes/urban_workspace_2026-11-07/README.md) for
the UI/context changes and verification, and
[results](../results/) for raw measurements. The vendored browser renderer is
Three.js. The shared engine uses Module Lab's Three.js distribution; the original
prototype's vendored package and license remain in `public/vendor/`.

See [facade-band and paired-edit experiments](../agent_notes/redistribution_2026-11-09/README.md) for configurable 1–3-layer editing and the isolated constant-FAR redistribution prototype. Two layers were the most reliable in this small test; that report predates promotion of the growing preview.

FAR targets now accept ±0.1 by default in the live session and M4 trainer (3.0 accepts 2.9–3.1). Coarse edit counts and budgets use the nearest accepted boundary; no extra completion reward/penalty depends on reaching the centre. Exact-target mode remains available for controlled historical experiments via `far_tolerance=0`. Experimental exchanges may change total volume if the result stays inside the accepted band.

Fetched plots offer four **Voxel grid alignment** choices: **Auto** (default for
fresh sessions), **World**, **Principal 1**, and **Longest edge**.
Auto searches rotation and sub-cell offset to maximize whole 1 m squares inside
the actual parcel (including concavity/holes); manual modes retain the existing
cell-centre raster. North/south, sun and original context geometry stay fixed.
Applying a new grid starts a fresh volume and retains policy weights and floor
count. The current preview is not switched automatically. See the
[grid and quality study](../agent_notes/grid_quality_2026-11-09/README.md).

The solid display follows the supplied mood boards: pale blue mass, lighter roofs,
distinct wall tones and white context, with shared-engine shadows. Camera-facing
tree silhouettes are supported on fetched green-area meshes as illustrative
scenery. The current cache contains no such meshes, so no trees are invented.

Solid surfaces touch without voxel display gaps. The current display uses
**zoom-scaled surface outlines**, with merged boundary creases and no coplanar cube/floor
seams. Hue and saturation stay fixed across outline (HSL L=50%) and fill
(L=70–95% from darkest shadow to brightest light), using the shared renderer's
lighting. See [current display verification](../agent_notes/outlines_hsl_2026-11-10/README.md).

The [unified action research path](../agent_notes/unified_actions_2026-11-10/README.md)
permits removal and roof growth in the same action before FAR is reached. It is
independent of the current live policy. Run a fresh experiment with
`python -m tools.unified_training --tag net-pilot --arms unified --net-guided --seconds 90`.
`--resume PATH --seeds 123` can continue a compatible experimental checkpoint;
changed decoding options reset Adam state, retaining the network. FAR guides net
area rather than imposing a carve/grow sequence. Initial controls establish FAR
feasibility on two plots, not trained environmental-quality improvement.
