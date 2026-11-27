# VoxelForm-RL · Density, daylight, and the urban block

![Paris rollout showing changes to the building and cores, direct-sun exposure, and distance to a circulation core](docs/media/episode-rollout.gif)

*Paris, one edit sequence. Left shows the mass and reserved cores. Center shows sampled direct sun. Right shows walking distance to the closest core.*

## How the form is carved

![Three stages of form-making, from extruding the parcel to carving a court and refining the perimeter](docs/media/form-carving-sequence.gif)

The sequence shows three moments in a massing study. An initial extrusion establishes the envelope, a carve opens a court, and successive edits refine the perimeter towards the target floor area ratio (FAR).

## Paris · A break in the street frontage

![Paris voxel axonometric, aerial Blender maquette, and view from the street](docs/media/blender-maquette-showcase.png)

The Paris study divides the street frontage into two volumes at **FAR 3.09**. The gap makes the two masses readable from the street. The procedural maquette shows how the model’s core reservations and floor plates carry through to the façade.

## Barcelona · Opening the middle of the block

![Barcelona comparison plate with the voxel axonometric at left and the Blender maquette from the same camera at right](docs/media/barcelona-axo-comparison.jpg)

The Eixample study keeps mass along the street edges and opens the middle of the block. Yellow identifies the space reserved for stairs and lifts. In the maquette, the envelope becomes recessed window bays and a ground-floor arcade.

As FAR fell from **5.82 to 2.90**, the share of floor area within **8 m of a façade** rose from **62% to 100%**. The direct-sun and diffuse-sky scores fell over the same edits. The court shortened floor-plate depth, while exposure to light still depended on orientation, enclosure, and neighboring buildings.

## Singapore · Stepped wings on a larger site

![Singapore comparison plate with the voxel axonometric at left and the Blender maquette from the same camera at right](docs/media/singapore-axo-comparison.jpg)

The **3,845 m²** parcel reaches **FAR 2.90** by carving the block into a curved perimeter wing and a companion courtyard volume (**68% direct-sun proxy**, **7 cores**). Recessed curtain-frame bays, solid structural piers, and a ground-floor plinth give the faceted massing a legible façade rhythm.

## Architectural precedents

- **Daylight as Form.** Daniel Koehler’s lab-eds catalogue reads courts, stepped sections, and other light-shaping devices across precedents in the dense city. It provides a typological reference for the Barcelona court and the Singapore terraces. [View the Daylight as Form gallery](https://lab-eds.org/work/daylight-as-form)
- **Maison Partino.** Rasa Navasaityte used reinforcement learning to assemble prefabricated housing components and evaluate the voids between them. It provides a precedent for treating the space created by sequential operations as part of the design objective. [View the Maison Partino gallery](https://lab-eds.org/work/maison-partino)

## From the voxel grid to the architectural model

A floor area target leaves the shape of a building open. VoxelForm-RL uses reinforcement learning to test how courts, stepped wings, and breaks in street frontage emerge from successive edits that account for density, daylight, and circulation.

In the Barcelona and Singapore plates, the voxel diagram and the maquette share the same orthographic projection, crop, and voxel state. The left panel isolates the envelope and cores against simplified context. The Blender exporter adds slabs, parapets, a façade system, and streetscape detail. [Geometry export](tools/export_blender_site.py) · [Comparison plate renderer](tools/render_matched_showcase.py)

Each voxel is **1 × 1 × 3 m** and contributes **1 m²** of gross floor area. FAR divides the total by the original parcel area. The policy proposes where to remove, add, or exchange groups of voxels. A geometric decoder checks those patches, and larger parcels pass through a coarse stage before finer edits. Each committed edit updates the occupancy used in the next evaluation. [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)

![Sequential editing loop and coarse-to-fine network architecture](docs/media/mdp-architecture.png)

The evaluation keeps three design questions distinct. **Plan depth** measures horizontal distance to the nearest exterior wall on the same storey. **Direct sun and diffuse sky access** are separate exposure estimates that account for self-shading and neighbors. **Circulation** is represented by height-graded stair and lift reservations and a same-floor walking-distance check. These are schematic design measures. Detailed daylight, structure, and egress require separate studies. [Core layout library](docs/CORE_LAYOUTS.md)

## What transfers across sites

![24-site multi-tier architectural massing atlas across XS through XL urban parcels](docs/media/atlas-4x6-massing.png)

*Clean 4×6 multi-city evaluation atlas spanning `XS` through `XL` parcels across Barcelona, Hong Kong, New York, Paris, Singapore, and Tokyo, pairing shaded massing and yellow vertical cores with continuous outer silhouettes and hidden-line-removed context.*

![Selected FAR-compliant layouts across five parcel tiers, from small infill to urban blocks](docs/media/multitier-showcase.png)

Selected layouts across five parcel sizes show how the same policy adapts to narrow infill, perimeter wings, and larger blocks with several circulation cores.

![Untrained seed-4211 and 400-episode FAR-gated tier_meso2 rollouts compared on the same New York, Paris, and Singapore parcels](docs/media/heldout-generalization.png)

The upper and lower rows compare untrained and trained rollouts on the same held-out parcels (`l_nyc_0019`, `m_prs_0320`, `l_sgp_0327`) and evaluation seeds. With the unified `2×2×1 → 1×1×1` cascade and FAR-gated editing, five-seed single-stage RL (`0.803`, `100%` FAR) leads the greedy heuristic (`0.734`) across all seven held-out validation and test plots ([L/XL ablation report](results/control-all-tiers-400ep-mps-2026-11-22.md)). Across an untouched **eight-site blind holdout (`XS`–`XL`)**, FAR-gated RL reaches `0.864 ± 0.046` (`99.5%` FAR) stochastically and `0.929 ± 0.027` (`100%` FAR) under deterministic MAP decoding on blind `L/XL` parcels (outscoring `5×` best-of-5 Fair Greedy `0.850` on `4/4` blind `L/XL` parcels), while the matched `400`-episode, five-seed auxiliary ablation confirms `20%`-warmup cosine annealing (`1.2 → 0.0`) holds `0.767` (`99%` FAR) at episode `400` and `0.883` on blind `L/XL` parcels versus `0.420` (`0.501` blind `L/XL`) for pure RL from scratch.

### Research figures

| Blind 8-site holdout & 400-ep aux ablation | Five-seed L/XL ablation (`400` ep) |
| --- | --- |
| [![Blind 8-site zero-shot holdout and matched 400-episode auxiliary ablation](docs/media/blind-8site-and-aux-ablation.png)](docs/media/blind-8site-and-aux-ablation.png) | [![Five-seed L/XL validation curves and held-out ablation](docs/media/l-xl-ablation-400ep.png)](docs/media/l-xl-ablation-400ep.png) |

<details>
<summary>Historical five-seed M-tier baseline chart (<code>600</code> ep, <code>tier_adaptive</code>)</summary>

[![Five-seed M-tier training curves and held-out comparisons](docs/media/training-curves-600ep.png)](docs/media/training-curves-600ep.png)

</details>

Open a figure at full size for metrics and legends. [Technical README](README.md) · [Results index](results/README.md)

## Explore a plot

| 3D workspace | Training diagnostics |
| --- | --- |
| ![Paris study in the interactive voxel workspace](docs/media/voxel-workspace.jpg) | ![Training dashboard showing FAR, exposure, and episode timing](docs/media/training-diagnostics.jpg) |

Run `uv run --frozen python main.py` and open [the local 3D workspace](http://127.0.0.1:8011). Choose a catalog site or open **Context source → Module Lab custom plots** to load a saved parcel from a local Module Lab checkout.

Surrounding building and street data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).

<!-- Barcelona l_bcn_0074 (evaluation seed 9008) and Singapore l_sgp_0379 (evaluation seed 9205)
use the guided L/XL run's local validation-selected episode-450 checkpoint (training seed 4215,
SHA-256 a8d49b006675347b84a3a419a66bcc099aae4ea4eb7c2607756c58806e392423).
The two panels in each comparison use the same occupancy and core masks and the bundled urban contexts. -->
