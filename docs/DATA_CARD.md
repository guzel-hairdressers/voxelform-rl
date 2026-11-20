# Data and simulator card

**Self-contained data:** seeded synthetic rectangle/triangle/courtyard/branch
sites with oriented contextual boxes. These are method fixtures, not sampled
real buildings or population-representative urban data.

**Optional real contexts:** cached OSM-derived parcel and surrounding geometry in
the sibling Module-Lab checkout. Set `VOXELFORM_CONTEXT_ROOT` when it is elsewhere.
VoxelForm reads cached scene HTML/JSON and does not execute HTML as Python or
modify Module Lab's dataset. External cache files and checkpoints are excluded
from this repository. Source-byte and allowed-grid hashes are saved per run;
exact reproduction requires obtaining the matching cached inputs. The public
repository alone reproduces synthetic experiments, not all urban experiments.

**Units:** occupied voxel = 1 m × 1 m × 3 m, contributing 1 m² gross floor area.
FAR uses original parcel area, not grid bounding area or usable area. Rotating the
grid transforms rays/context consistently and preserves original world north.

**Splits:** use distinct model/evaluation seeds and sites; prefer spatially
separated cities. Different site IDs do not establish independence if their
context overlaps. The research runner records centers/radii and rejects known
cross-split context-circle overlap. Unknown georeferencing cannot prove absence
of leakage and must be reported.

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
No new license for the repository's own code is inferred from OSM or the vendored
Three.js license. Redistribution of external data requires the applicable terms;
no external data is silently uploaded by the experiment runner.
