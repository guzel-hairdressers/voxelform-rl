# Data and simulator card

**Self-contained data:** seeded synthetic rectangle/triangle/courtyard/branch
sites with oriented contextual boxes. These are method fixtures, not sampled
real buildings or population-representative urban data.

**Bundled real contexts** include 27 OpenStreetMap-derived parcels and their
surrounding buildings for the historical and frozen L/XL comparisons.
The checkout uses a sibling Module-Lab cache when available. To select the
bundled files explicitly, set `VOXELFORM_CONTEXT_ROOT="$PWD/data/urban_context"`
from the repository root. VoxelForm extracts embedded JSON from scene HTML.
Run manifests record scene-byte hashes and allowed-grid hashes. Use matching
scene inputs for exact reproduction. Other cached scenes and model checkpoints
are stored outside the repository. [Context bundle](../data/urban_context/public/)

**Units:** occupied voxel = 1 m × 1 m × 3 m, contributing 1 m² gross floor area.
FAR uses original parcel area, not grid bounding area or usable area. Rotating the
grid transforms rays/context consistently and preserves original world north.

**Splits:** use distinct model/evaluation seeds and sites; prefer spatially
separated cities. Different site IDs do not establish independence if their
context overlaps. The research runner records centers/radii and rejects known
cross-split context-circle overlap. The historical benchmark evaluates M-tier
plots. The frozen L/XL split has validation and test plots at all three tiers,
including two Singapore XL test parcels. The cascade uses measured simulation
parcel area. Some legacy site IDs retain an L prefix for parcels over 3,800 m².
Report missing georeferencing when interpreting split independence.
[Split coverage](EXPERIMENT_PROTOCOL.md#600-episode-lxl-follow-up)

**Labels/rewards:** no architectural ground-truth label set or certified daylight
labels exist. Sky diffusion, sampled direct sun, depth, narrow-strip tests and
core proximity are synthetic/approximate proxies. Direct sun and diffuse sky
access remain separate outputs. Solar directions are not an annual ephemeris;
noise is an uncalibrated prior. Fast shadow/diffusion variants trade accuracy
for computation and require independent checks before physical claims.

**Cores:** metric component geometry with outward one-metre reservations and a
greedy grounded placement heuristic. Whole-footprint grid distance is not a
corridor/door/evacuation solution. See [core layout assumptions](CORE_LAYOUTS.md).

**Rights/provenance:** preserve original cached provenance and
[OpenStreetMap attribution/licensing](https://www.openstreetmap.org/copyright).
The MIT License covers the project's original code. Third-party data and
bundled libraries retain their own terms. [Project license](../LICENSE)
