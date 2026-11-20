# VoxelForm-RL

**Reinforcement learning for constrained 3D editing.** VoxelForm-RL is a PyTorch
research project that uses building massing as a testbed for high-dimensional
spatial actions. A policy starts from a voxelized envelope and learns to carve,
grow, and redistribute volume within a limited number of edits. Each episode
targets a floor-area ratio (FAR) band and evaluates tradeoffs among usable area,
daylight proxies, and access to vertical cores.

The research question is whether a policy can learn coherent, useful 3D edits
under these constraints and transfer what it learns to unseen sites.

![VoxelForm-RL workspace showing a generated voxel mass and its evaluation metrics in Paris](docs/media/voxel-workspace.jpg)

*Live workspace on the Paris demonstration site. The blue mass is the generated
layout. Surrounding geometry is derived from [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).*

## Learning setup

| Component | Implementation |
|---|---|
| **State** | A binary occupancy grid, a permitted envelope, FAR and edit-budget context, and 14 spatial channels. The channels encode occupancy, spatial edit eligibility, environmental signals, and core access. |
| **Policy** | A factored 2D+1D actor–critic samples a categorical operation and coarse Gaussian spatial fields. Correlated masks let one action alter a coherent region across multiple floors. |
| **Transition** | The decoder smooths and ranks the fields, allocates edit volume, and checks constraints before committing the mask. The environment supplies geometry guards and a deterministic core placement heuristic. |
| **Training** | Actor–critic with generalized advantage estimation (GAE). The reference configuration uses the environment reward. A matched ablation adds a heuristic spatial auxiliary signal. |
| **Evaluation** | FAR success and component metrics are reported separately. The evaluator records both successful and failed episodes and uses a shared reference score for settings ablations. |

The environment evaluates direct sunlight with both context and self-shadowing,
diffuse sky as a separate proxy, and same-floor distance to a core. Core voxels
occupy gross floor area and are excluded from usable area and facade daylight
scoring. See the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for the
action representation, objective and evaluation procedure.

## Run a reproducible experiment

Start with the self-contained CPU smoke run. It trains two model seeds on
synthetic geometry and evaluates held-out shapes. The run needs only this
repository, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen --extra dev --python 3.12
uv run --frozen vf-research train \
  --config configs/research-smoke.json --output runs/my-smoke
```

Use a new output directory for each run. The runner saves the resolved config,
source and scene fingerprints, runtime metadata, learning traces, per-episode
evaluation results, and model, optimizer and RNG checkpoints.

```sh
uv run --frozen vf-research evaluate \
  --checkpoint runs/my-smoke/seed-123/latest.pt \
  --split test --output runs/my-test.json
```

See the [resume procedure](docs/EXPERIMENT_PROTOCOL.md#artifacts-and-resume-contract)
for continuing a compatible checkpoint at an episode boundary.

## Evaluate across sites

The [real-site reference config](configs/research-real.json) fixes budgets and
model seeds. It trains on ten plots across Barcelona, Tokyo and Singapore, with
two plots in each size tier (XS, S, M, L and XL). It validates on Hong Kong
and tests on New York and Paris. The
[auxiliary-loss config](configs/research-real-aux.json) shares the split and
budget. Real-site runs require the matching local cache and use Apple MPS by
default. Set `VOXELFORM_CONTEXT_ROOT` to the Module Lab checkout, then run:

```sh
uv run --frozen vf-research train \
  --config configs/research-real.json --output runs/real-reference
```

The protocol evaluates fixed-budget checkpoints against untrained controls
using common evaluation seeds. Summaries report variability across independent
model seeds. The synthetic smoke run verifies the pipeline. Historical pilots
use earlier task definitions and are preserved with their settings in the
[results index](results/README.md). A matched-budget comparison of single-site
and multi-site training on held-out sites remains an open experiment.

## Inspect a live run

The live workspace uses the Module Lab renderer and cached geometry. With Module
Lab beside this checkout, or `VOXELFORM_CONTEXT_ROOT` set, run:

```sh
uv run --frozen python main.py
```

Open [localhost:8011](http://127.0.0.1:8011) to inspect the current massing,
environment fields, and training curves. The default demo is the Barcelona
Eixample site (`m_bcn_0093`). **Rotate training sites** shares one
policy and optimizer across those ten training sites in three-episode blocks
(30 episodes per full rotation). **Saved layouts** previews the five highest-scoring
distinct layouts that reached the target FAR for the site and comparable settings. Episode
statistics and those voxel masks persist locally in `runs/layouts/`.

<details>
<summary>Training diagnostics screenshot</summary>

![Training dashboard with the direct-sun curve highlighted and FAR and timing plotted alongside](docs/media/training-diagnostics.jpg)

The dashboard records exploratory live runs. The headless runner produces the
controlled evaluation artifacts.

</details>

## Documentation and checks

- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) and [model card](docs/MODEL_CARD.md)
- [Data and simulator card](docs/DATA_CARD.md) and [core geometry](docs/CORE_LAYOUTS.md)
- [Results and historical evidence](results/README.md), [contributing](CONTRIBUTING.md), and [citation](CITATION.cff)

```sh
uv run --frozen --extra dev python -m unittest discover -s tests
node --test tests/*.test.mjs
```

[GitHub Actions](.github/workflows/research.yml) runs the locked CPU tests,
viewer geometry tests, static checks, and the synthetic smoke experiment.
The simulator's environmental metrics are approximate research proxies.
