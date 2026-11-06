# VoxelForm RL — resumption

Independent repository: `/Users/ruslan_faz/Desktop/Work/VoxelForm-RL`.
Branch: `codex/voxel-prototype`. Development staging: `/private/tmp/VoxelForm-RL`.
Do not resume the old Module Lab goal or modify its port 8000 session.

Implemented the first occupied/empty voxel massing demo: exterior coherent
facade/roof cuts, undo, actor–critic, own policy save/load, 3D/floor plan viewer,
static external receiver caching, dynamic sun-plane self-shadow maps, both
external shadow methods, relative line noise and diffusion accessibility fields.
The project-specific AGENTS.md transfers the applicable user requirements.

Tests: 13 correctness tests pass. CPU and actual MPS benchmarks have raw final
results in results/. Initial results remain preserved under results/initial/.
Final edit masks match across devices. MPS did not beat CPU here. Do not claim
policy-quality improvement or exact daylight. Read results/README.md for known
errors, especially the uncalibrated diffusion threshold and map aliasing.

Known work remaining: real site/context import; actual solar times; usable-area
and sky-access calibration against held-out cases; geometric admissibility
(connectivity/support); richer coherent masks and additions beyond undo;
matched-area policy controls/convergence; batched/fused GPU implementation.
No diffusion model, interior labeling or compliance implementation exists yet.

Development Python currently reuses the Thesis venv interpreter; this project
has independent requirements.txt and no source imports from Module Lab.
Run main.py on port 8010. Save policy through its API before any restart.
No remote push is authorized for this independent repository.
