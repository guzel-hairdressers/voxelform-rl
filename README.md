# VoxelForm-RL

**Machine-learning research on structured, constrained 3D actions under a limited
compute budget.** A voxel editing environment supplies dynamic spatial eligibility,
correlated multi-cell actions, geometric constraints and multiple proxy objectives.
Urban massing is the test domain; the research concerns RL representation, credit
assignment, constraint decoding, evaluation and efficiency. The 3D workspace is a
debugging frontend, not the scientific result.

The active actor is a factored 2D+1D network with 14 observation channels, coarse
Gaussian mask latents, a categorical operation head and a scalar critic. Training
uses actor-critic with GAE. An optional physics-shaped auxiliary objective is
explicitly ablated; hardcoded geometry rules are not presented as learned skills.

## Reproducible setup

Python 3.11–3.13; the lockfile is tested with Python 3.12. Linux uses the CPU Torch
index; macOS can use MPS. Install [uv](https://docs.astral.sh/uv/) and run:

```sh
uv sync --frozen --extra dev
uv run --frozen vf-research train --config configs/research-smoke.json --output runs/my-smoke
```

This runs without Module Lab, its renderer, external urban data or a browser.
Choose a new output directory each time; evidence is never silently overwritten.
`pyproject.toml` provides an installable ML package/CLI; `uv.lock` pins its resolved
runtime/development dependencies. `requirements.txt` remains a legacy minimum set,
not the reproducibility lock.

## Train and evaluate

```sh
# Three model seeds, fixed training budget, disjoint cities; requires cached data.
VOXELFORM_CONTEXT_ROOT=/path/to/Module-Lab uv run --frozen vf-research train \
  --config configs/research-real.json --output runs/real-reference

# Matching auxiliary-loss ablation: same budgets, seeds, splits and evaluation.
VOXELFORM_CONTEXT_ROOT=/path/to/Module-Lab uv run --frozen vf-research train \
  --config configs/research-real-aux.json --output runs/real-aux

# Exact episode-boundary resume into a new directory; config must match.
uv run --frozen vf-research train --config configs/research-smoke.json \
  --resume runs/my-smoke/seed-123/latest.pt --output runs/resumed

uv run --frozen vf-research evaluate --checkpoint runs/my-smoke/seed-123/latest.pt \
  --split test --output runs/test-recheck.json
```

Each run saves a validated config, Git/dirty-source hashes, the actual source
archive, dependency/backend versions, scene fingerprints, model/evaluation seeds,
JSONL learning traces, raw held-out episodes and atomic model/optimizer/RNG
checkpoints. Failures remain in summary denominators. A common core-aware referee
makes reward-setting ablations interpretable; uncertainty is computed across model
seeds. Test scenes never select checkpoints. See the
[complete experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

The smoke run verifies the pipeline. It is not evidence of convergence or a
publishable quality gain. Historical 80-episode scores used different core/reward
semantics; they remain historical observations, not validation of the new version.

## Compact core geometry

Core geometry is parameterized and shared by the environment, JSON and dimensioned
SVGs. Plans show two stair flights and two landings, shaft/car separation, lobby
and openings. Height-dependent lift counts are explicit research priors.

| Floors | Stairs | Lifts | Grid reservation |
|---|---:|---:|---|
| 1–5 | 1 | 0 | 3 × 5 m |
| 6–10 | 1 | 1 | 5 × 5 m |
| 11–18 | 1 | 2 | 7 × 5 m |
| 19–30 | 2 interlocking scissor routes | 4 | 7 × 9 m |
| 31–60 | 2 interlocking scissor routes | 4 | 7 × 9 m |

[Metric dimensions, drawings and assumptions](docs/CORE_LAYOUTS.md) ·
[6–10-floor SVG](docs/core-plans/T2.svg). No architectural compliance or lift traffic
capacity is inferred from these templates. They are inspectable task parameters.

## Inspect the live model

From the editable checkout, keep Module Lab beside this repository or set
`VOXELFORM_CONTEXT_ROOT`. Its cached geometry and renderer are read-only dependencies.

```sh
uv run --frozen python main.py
# http://127.0.0.1:8011  — training/debugging dashboard
# http://127.0.0.1:8011/core-plans — canonical metric component plans
```

Default demo: **m_bcn_0093 (Barcelona/Eixample)**, the L-shaped plot beside the tall
neighbour used for the shadow/redistribution example. `VOXELFORM_SITE_ID` overrides
it explicitly. Initial height/headroom can follow contextual heights or be set in
the UI. The active organic decoder permits 3D carving/growth; the old linked-floor
copying remains a compatibility/control implementation.

Direct sunlight and diffuse sky access are separate uncalibrated proxies. Rendered
shadows are not the physical evaluation. Devoxelization is display-only and never
changes occupied floor area or training reward. No main-workspace floor-plan pane
is added; `/core-plans` is a component-geometry diagnostic.

## Checks

```sh
uv run --frozen --extra dev python -m unittest discover -s tests -v
node --test tests/*.test.mjs
uv run --frozen --extra dev ruff check voxelform/core_plans.py voxelform/experiment.py \
  voxelform/objectives.py tools/research.py tools/render_core_plans.py \
  tests/test_core_plans.py tests/test_experiment.py
```

GitHub Actions runs the locked CPU suite, JS geometry checks, static checks and a
self-contained smoke experiment. MPS execution is additionally tested locally when
available; CI does not claim to test unavailable GPU hardware.

## Research documentation

- [Experiment protocol and remaining work](docs/EXPERIMENT_PROTOCOL.md)
- [Data/simulator card](docs/DATA_CARD.md) and [model card](docs/MODEL_CARD.md)
- [Core layouts](docs/CORE_LAYOUTS.md)
- [Latest changes and verification](docs/REVIEW_2026-11-20.md)
- [Historical raw results](results/) and [implementation notes](agent_notes/README.md)
- [Preserved older workspace README](docs/LEGACY_WORKSPACE.md)
- [Contribution rules](CONTRIBUTING.md), [citation metadata](CITATION.cff)

The project's own redistribution license needs an explicit owner decision; no
license is inferred from dependencies. Existing Three.js license/OSM attribution
are retained. Dataset hashes are recorded; matching external urban cache content
must be obtained separately to reproduce real-site runs.
