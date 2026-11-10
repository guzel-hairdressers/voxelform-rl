# Urban 3D workspace — 2026-11-07

> Follow-up: [single-voxel FAR episodes](../far_episodes_2026-11-07/README.md) supersedes this report's earlier action/episode behaviour and preview counters. Shared graphic-engine integration remains unchanged.

Repository: VoxelForm-RL. Branch: `codex/voxel-prototype`.

The final implementation uses **Module Lab's actual graphic engine**, served
byte-for-byte from `../Module-Lab/frontend/public/scripts/urban_viewer.js`, with
its original Three.js distribution, styling, lighting, shadows, cameras, gizmo,
axis snapping and projection handling. No sibling files were modified. The
provisional independent camera/renderer was removed at the user's direction.

`public/viewer.html` embeds the engine in an iframe. `public/viewer.js` initializes
it with cached scene geometry and adds one instanced voxel overlay to its exposed
optimizerGroup. It does not instantiate another renderer, camera, light or controls.
The overlay uses the engine's architectural two-tone color convention. Host
projection and Fit view buttons send the exact messages used by Module Lab's
ThreeViewer. Native gizmo clicks and double-click framing remain in the engine.
`public/viewer-data.js` transforms cached-grid coordinates back into the original
site frame, including the renderer's X/up/negative-Y convention.

The surrounding interface follows the compact white Module Lab layout. The final
workspace is 3D only: no plan pane, floor selector or alternate 2D renderer.
Surrounding buildings are opaque by default; the transparency checkbox changes
only their visual material/edges. It does not change simulation or obstruction.

## Context and evaluation

- Read-only catalog of 390 cached OSM scenes across six cities. No fresh fetch was
  needed. Frontend indexes take precedence over legacy public indexes, preserving
  existing custom-site deletions. Public-relative scene paths and cache-busting
  query strings are supported; path escape and remote URLs are rejected.
- Original building and street triangle meshes, cached heights and parcel shape
  are preserved at metric scale. The real parcel is cell-centre rasterized at 1 m
  from its ground triangles, retaining holes/concavity. Grid limit: 128 m per side,
  10,000 horizontal cells. Oversize scenes get an explicit error.
- External sun and sky-boundary queries use imported building triangles with
  persistent receiver caching. Dynamic voxel self-shadowing remains unchanged.
  Box boundary/depth-map methods remain the synthetic method-test paths. This
  task establishes no throughput improvement or benchmark result.
- Street ribbons have no usable traffic-source attributes. Noise is null/unmeasured
  for imported scenes, its layer is disabled and its objective term is omitted.
  Sun directions remain synthetic; source building heights can be estimated.
- Dataset reuse and a trainable policy do not establish architectural compliance,
  generalization or quality improvement. Synthetic fixtures remain an explicit
  optional source. Module Lab generator objectives/policies were not imported.

## Verification

- Python suite: **21 tests passed** (existing geometry/fields/learning plus eight
  imported-context tests). Checks include holes, scale/translation, concave mesh
  shadows, mesh-vs-box reference rays, receiver caching, JSON-only parsing,
  path safety, oversized-scene rejection and missing noise provenance.
- Browser adapter suite: **4 tests passed** with `node --test tests/viewer-data.test.mjs`.
  It checks alignment with original scene coordinates, overlay axis mapping,
  synthetic parcel retention and oriented-box conversion.
- JavaScript syntax checks and `git diff --check` passed.
- HTTP-delivered shared engine compared byte-for-byte with the sibling file;
  SHA-256 recorded in `verification.json`.
- Browser verified New York and Barcelona cached geometry, usable layer, progress
  disclosure, then the final shared engine's perspective toggle, native top-axis
  snap, Fit view and optional transparency. Confirmed no plan canvas remains.
- Final shared-engine layout checked at normal 1710 px and narrow 390 px widths;
  no horizontal page overflow. Temporary viewport override was cleared.

Before replacing the preview process to mount the shared assets, its learned
policy and state were saved under ignored `results/checkpoints/`. The preview
was paused with no volume edits; its ten optimizer updates were restored exactly.
An earlier isolated inference smoke test stopped after one episode with unchanged
weights; that policy chose stop and did not improve or edit the mass. Its API
results are retained in `verification.json`; preview episode counters restarted
when loading the shared engine. The original port-8010 process was left running.

Current preview: **http://127.0.0.1:8011/**, paused, on Barcelona `m_bcn_0093`.
Default future launches still use port 8010. `VOXELFORM_SITE_ID` selects the initial
cached site; `VOXELFORM_POLICY_PATH` restores a saved policy. Check listeners before
starting another process. No commits, pushes or deployment were performed.

## Screenshots

These show actual cached scenes and the shared renderer, not quality benchmarks.

- [Workspace](workspace.jpg): final opaque default and native gizmo.
- [Native top view with transparency](top-transparent.jpg).
- [Narrow workspace](narrow.jpg).
