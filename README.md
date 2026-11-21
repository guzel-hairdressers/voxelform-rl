# VoxelForm-RL

[![Research checks](https://github.com/guzel-hairdressers/voxelform-rl/actions/workflows/research.yml/badge.svg)](https://github.com/guzel-hairdressers/voxelform-rl/actions/workflows/research.yml)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-2563eb.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c.svg)
![Protocol](https://img.shields.io/badge/protocol-voxelform.research.v1-0f766e.svg)
![Cores](https://img.shields.io/badge/constraints-compact--core.v5-dc2626.svg)

**Structured deep reinforcement learning for constrained 3D combinatorial optimization.**

VoxelForm-RL is a PyTorch research codebase studying **correlated, high-dimensional 3D spatial actions under hard geometric, topological, and accessibility constraints**. Starting from an initial discrete voxel lattice $`M_0 \in \{0,1\}^{Z \times Y \times X}`$ clipped to a permitted spatial envelope $`A \in \{0,1\}^{Z \times Y \times X}`$, a factored 2D+1D actor–critic sequentially carves, grows, and redistributes volumetric elements over a finite edit horizon while targeting a prescribed volumetric density band (`FAR* ± ε`) and optimizing multiple uncalibrated spatial exposure and accessibility proxies.

Urban building massing provides the concrete constraint structure, proxy physics, and 3D inspection UI; **representation, correlated Gaussian latent action spaces, feasibility projection, spatial credit assignment, and multi-scale/vertical compute scaling** are the primary Machine Learning objects of study.

---

## Visual Overview

### Sequential 3D Editing Trajectory on Held-Out Paris (`1,200`-Episode Policy)

![1,200-episode trained policy zero-shot rollout on held-out Paris m_prs_0315 showing occupied mass with crimson compact-core.v5 shafts, core-exempt direct sun proxy, and geodesic core travel distance](docs/media/episode-rollout.gif)

*Zero-shot rollout of the **1,200-episode multi-site trained checkpoint** (`runs/real-aux-1200ep`, `seed=123`, `compact-core.v5`, target `FAR = 3.00 ± 0.10`, `spatial_aux_weight=1.2`, deployment decoding `temp=2.5, noise=0.08`) on unseen **Paris (`m_prs_0315`, eval seed `9005`)**: reaches `FAR 3.10`, `Obj 1.033`, `67.6%` (`2,917 m²`) usable floor area, `71.3%` `directSunProxy`, `12.4%` `diffuseFacadeProxy`, and `96.9%` core coverage across `2` continuous crimson cores. **Left:** translucent occupied voxels (`1 m × 1 m × 3 m`) with embedded vertical core shafts. **Middle:** exposed-face `directSunProxy` solar visibility (core faces excluded from daylight averaging while casting shadows). **Right:** 8-neighbor geodesic floor-plate travel distance to nearest core without corner cutting. Full clip: [docs/media/episode-rollout.mp4](docs/media/episode-rollout.mp4).*

### Cross-Scale Trained Showcase (`XS` → `XL`) & Zero-Shot Unseen City Transfer

![1,200-episode trained layouts with compact-core.v5 shafts across XS, S, M, L, and XL training sites](docs/media/multitier-showcase.png)

*Best FAR-satisfied evaluation rollouts per training plot-size tier from the **1,200-episode shared policy** (`target_far=3.0 ± 0.10`, 20F ceiling, `compact-core.v5`, eval seeds `9001–9008`, deployment decoding `temp=2.5, noise=0.08`), spanning `xs_sgp_0346` (`0.852` obj) through `xl_tokyo_0185` (`18F`, 7 cores).*

![Paired zero-shot comparison between untrained control and 1,200-episode trained policy across unseen Hong Kong, New York, and Paris plots](docs/media/heldout-generalization.png)

*Zero-shot generalization on unseen held-out cities under **matched evaluation decoding (`temp=1.0, noise=1.0`)**: identical seeds (`9005`, `9004`, `9001`) compare untrained seed-123 initialization (top) against the 1,200-episode trained checkpoint (bottom).*

<details>
<summary>Local Web UI Screenshots (<code>main.py</code>)</summary>

| 3D Volumetric State Inspector (local `main.py`) | Live Training Dashboard (local `main.py`) |
|---|---|
| ![VoxelForm-RL live workspace on the Paris demonstration site](docs/media/voxel-workspace.jpg) | ![Training dashboard plotting direct-sun proxy, FAR convergence, and episode timing](docs/media/training-diagnostics.jpg) |

*Static captures of the optional local interactive workspace started via `uv run --frozen python main.py`. Left: `m_prs_0315` inspection with surrounding context geometry derived from [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Right: live episode monitoring keeping `directSunProxy` and `diffuseFacadeProxy` distinct.*

</details>

---

## Problem Formulation: Constrained 3D Volumetric MDP

We model sequential 3D editing as a finite-horizon Markov Decision Process $`(\mathcal{S}, \mathcal{A}, \mathcal{T}, R, \gamma)`$ over a metric lattice with voxel dimensions `1 m × 1 m × 3 m`:

### 1. State Space & 14-Channel Observation Tensor

At decision step `t`, state $`s_t = (O_t, g_t)`$ consists of a dense voxel observation tensor $`O_t \in \mathbb{R}^{Z \times 14 \times Y \times X}`$ paired with a normalized 5-dimensional goal vector $`g_t \in \mathbb{R}^5`$ (`[far, target_far, far_error, remaining_budget_frac, spent_budget_frac]`):

| Channels | Group | Description (`environment.py`, `unified.py`) |
|---|---|---|
| `0, 3, 4` | Lattice Geometry | Binary occupancy $`M_t`$, permitted 3D envelope $`A`$, normalized floor elevation $`z / Z`$ |
| `6, 7` | Action Eligibility | Boundary exterior-air removal weights (`remove_eligible`), 3D growth mask (`add_eligible`) |
| `1, 2, 8, 9` | Physical Proxies | Screened-Laplace diffuse sky air field, acoustic line prior (synthetic only), cached external solar ray potential, exposed facade diffuse access |
| `5, 10, 11` | Shape Regularity | Planar room-depth excess penalty (`depth_badness`), `< 6 m` narrow-strip detector beyond 2 m stubs (`long_narrow6`), usable interior depth mask |
| `12, 13` | Vertical Cores | Active `compact-core.v5` core mask $`C_t \cap M_t`$, same-floor 8-neighbor geodesic core-distance badness (limit `12 m` default) |

### 2. Hybrid Discrete–Continuous Action Space & Feasibility Projection

Rather than stepping over $`\mathcal{O}(Z \cdot Y \cdot X)`$ per-voxel Bernoulli variables independently, an action $`a_t`$ couples a high-level categorical edit operation with correlated coarse Gaussian spatial fields:

```math
o_t \sim \pi_\theta(o \mid s_t) \in \{\text{remove},\ \text{grow},\ \text{exchange},\ \text{finish}\}, \qquad z_t^{(k)} \sim \mathcal{N}\left(\mu_\theta(s_t),\ \mathrm{diag}(\sigma_\theta^2)\right)
```

**1. Correlated Decoding:** Sampled coarse fields $`z_t^{(k)} \in \mathbb{R}^{|\mathcal{C}_t| \times \lceil Z/2 \rceil \times \lceil Y/4 \rceil \times \lceil X/4 \rceil}`$ are trilinearly upsampled, smoothed via exact separable 1D (`Z`) + 2D (`Y, X`) Gaussian convolutions (avoiding MPS `Conv3d` host stalls), and ranked inside eligible edit frontiers with area budgets scaled to the target FAR band.

**2. Augmented Retry Accounting:** When geometric projection rejects a candidate draw (up to `retries = 8`), `UnifiedLearner` records **all consumed Gaussian draws** (rejected + accepted) on the full coarse lattice without spatial eligibility masking, scoring the unbiased joint action log-probability:

```math
\log \pi_\theta(a_t \mid s_t) = \log \pi_\theta(o_t \mid s_t) + \sum_{k=1}^{K_t} \log \mathcal{N}\left(z_t^{(k)} ; \mu_\theta(s_t), \sigma_\theta^2\right)
```

**3. Deterministic Feasibility Projection** ($`\Pi_{\text{valid}}`$): Committed edits must satisfy the FAR band or move strictly toward it, maintain 6-neighbor solid connectivity and exterior-air cavity validity, enforce `k = 3 m` separable morphological floor width on plots `≥ 400 m²`, and preserve active `compact-core.v5` ground-to-roof shafts. Core placement uses a deterministic height-graded heuristic (`solve_cores`) over compact metric plans ([docs/CORE_LAYOUTS.md](docs/CORE_LAYOUTS.md)). Because $`\Pi_{\text{valid}}`$ and `solve_cores` embed strong domain priors, the policy acts within a **constrained hybrid decoder** rather than an unconstrained end-to-end generator.

### 3. Objective Decomposition & Policy Optimization

The shared live/headless scalar objective (`voxelform/objectives.py`) intentionally keeps directional sunlight (`sun`) and sky diffusion (`diff`) **separate**:

```math
\begin{aligned}
J(M_t) =\;& 0.65\,u_{\text{good}} + 0.55\,\bar{s}_{\text{sun}} + 0.45\,\bar{s}_{\text{diff}} + 0.10\,r_{\text{roof}} + 0.20\,s_{\text{core}} \\
& - 0.20\,\bar{d}_{\text{deep}} - 0.55\,f_{<6\text{m}} - 0.08\,\bar{\eta}_{\text{noise}} - 0.45\,f_{\text{uncov}} - 0.35\,\bar{b}_{\text{core}}
\end{aligned}
```

where $`s_{\text{core}} = \max(0, (1 - f_{\text{uncov}})(1 - 0.5\,\bar{b}_{\text{core}}))`$. Step rewards add $`0.50\,\Delta s_{\text{core}}`$ (plus a small `-0.005` hold penalty when FAR is already satisfied); terminal rewards scale $`5\cdot(J(M_T) - P_{\text{FAR}})`$ where $`P_{\text{FAR}} = 0`$ inside the band and penalizes remaining relative volume error otherwise.

Training runs synchronous actor–critic with GAE (`γ = 0.99`, `λ = 0.95`, rollout horizon `16`), Huber value regression (`0.5×`), operation entropy regularization (`0.005×`), gradient norm clipping (`≤ 2.0`), and Adam (`lr = 1.2e-3`, `wd = 1e-4`). An optional heuristic physics-shaped dense spatial auxiliary objective is isolated via `spatial_aux_weight` (`0.0` in [configs/research-real.json](configs/research-real.json); `1.2` ablation in [configs/research-real-aux.json](configs/research-real-aux.json)).

---

## Neural Architecture (`Organic3DNetwork`, 48,125 Parameters)

![MDP loop and Organic3DNetwork factored 2D+1D architecture with halo-2 sliding vertical window](docs/media/mdp-architecture.png)

Instead of memory-bound full-resolution 3D convolutions across arbitrary building heights $`Z \in [8, 60]`$, `Organic3DNetwork` (`coarse_2d1d_v4`, `voxelform/unified.py`) uses a **factored 2D+1D encoder with an exact halo-2 sliding vertical window**:

- **Strided 2D Spatial Stem:** Compresses each 14-channel slice `(Y, X)` by `4×` (`Conv2d 14→24→32, k=3, s=2, SiLU`).
- **Factored 2D+1D Residual Mixing (`_zmix`):** Applies planar `3×3` convolutions (`ctx_xy1`, `ctx_xy2`) followed by shift-and-add `Z`-neighbor projections (`Z-1, Z, Z+1`) inside overlapping windows (`win=16` floors, `halo=2`), keeping peak activation memory independent of maximum storey count while remaining mathematically identical to global 2-block `Z`-receptive fields.
- **FiLM-Style Additive Goal & Local Skip Conditioning:** Combines exact floor-mean/max global pooling with `goal_encoder(g_t)` and a `4×`-pooled `1×1` `local_skip`, outputting bounded coarse latent means $`\mu_\theta = 3\tanh(\text{head}/3) \in [-3, 3]^{2 \times \lceil Z/2 \rceil \times \lceil Y/4 \rceil \times \lceil X/4 \rceil}`$, categorical operation logits, and scalar critic value $`V_\phi(s_t)`$.
- **Multi-Scale Extension:** `CoarseToFineNetwork` (`voxelform/coarse_to_fine.py`) supports optional coarse-to-fine lattice schedules (`4×4×2 → 2×2×1 → 1×1×1`) for large footprints and tall envelopes.

---

## Reproducible Experiment Protocol & Empirical Benchmarks

The `vf-research` runner (`tools/research.py`, `voxelform/experiment.py`) enforces clean ML evaluation hygiene ([docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)):

- **Strict Split & Leakage Guards:** `configs/research-real.json` trains across 10 balanced XS–XL plots (Barcelona, Tokyo, Singapore), validates on Hong Kong (`m_hk_0199`), and tests on held-out New York and Paris (`m_nyc_0003`, `m_prs_0315`), rejecting overlapping Haversine context circles across splits.
- **Controlled Baselines & Honest Accounting:** Evaluates untrained controls and fixed-budget final checkpoints on shared evaluation seeds (isolating evaluation RNG state), retains failed-FAR episodes in every mean, and computes `meanReferenceQuality` using a standardized `10 m / 12 m` `compact-core.v5` referee so core-on/off and travel-limit ablations remain comparable.
- **Full Provenance & Exact CPU Resume:** Each run archives `config.json`, `manifest.json` (git SHA, source SHA-256s, hardware/packages), `source.tar.gz` (including dirty diffs), `scenes.json`, and atomic optimizer+RNG checkpoints with strict hash verification on `--resume`.

### 1,200-Episode Multi-Tier Training & Held-Out City Generalization

![1,200-episode multi-site learning curve and matched zero-shot held-out city comparison](docs/media/training-curves-1200ep.png)

Trained for **1,200 episodes** (`2,625.6 s`, `690` synchronous GAE updates on Apple M4 MPS, `seed=123`, `spatial_aux_weight=1.2`, [configs/research-real-aux-1200ep.json](configs/research-real-aux-1200ep.json)) rotating uniformly across all 10 balanced XS–XL training sites, and evaluated under the matched research decoding distribution (`temp=1.0, noise=1.0`) against the identical untrained initialization ([results/real-aux-1200ep-mps-2026-11-21.json](results/real-aux-1200ep-mps-2026-11-21.json)):

| Split / Held-Out Site | FAR Success | Objective (`meanQuality` / Ref) | Usable Area (m²) | `directSunProxy` | `diffuseFacadeProxy` |
|---|---:|---:|---:|---:|---:|
| **Validation:** Hong Kong `m_hk_0199` (Untrained → 1,200 ep) | `1.00 → 1.00` | `0.5675 → 0.7729` | `1,051.6 → 3,163.0` | `61.8% → 64.5%` | `6.2% → 12.5%` |
| **Test:** New York `m_nyc_0003` (Untrained → 1,200 ep) | `1.00 → 1.00` | `0.0546 → 0.3406` | `84.4 → 1,016.8` | `14.6% → 26.6%` | `1.8% → 4.0%` |
| **Test:** Paris `m_prs_0315` (Untrained → 1,200 ep) | `1.00 → 1.00` | `0.5042 → 0.8157` | `688.8 → 2,427.2` | `64.1% → 67.7%` | `5.9% → 12.6%` |
| **Test Aggregate** (10 eps, Untrained → 1,200 ep) | `1.00 → 1.00` | `0.2794 → 0.5782` (**+107%**) | `386.6 → 1,722.0` (**4.45×**) | `39.4% → 47.1%` | `3.8% → 8.3%` |

*Scope & Training Dynamics Note:* Because `seed=123` uses unannealed on-policy updates (`lr=1.2e-3`) across heterogeneous XS–XL action scales and `spatial_aux_weight=1.2`, the 30-episode rolling training objective peaks early (`0.617` at ep `101–200`), oscillates mid-run (`0.107` at ep `701–800`), and recovers (`0.422` at ep `1101–1200`). Establishing variance across multiple seeds and isolating `spatial_aux_weight=0.0` at matching budget remain open experiments.

### Infrastructure Verification & Compute Scaling

![Empirical setup cache scaling and historical multi-scale coarse-to-fine Pareto tradeoff on Apple M4 MPS](docs/media/compute-and-ablations.png)

| Measurement Scope | Dataset / Artifact | Key Observed Metrics | Interpretation Boundary |
|---|---|---|---|
| **Pipeline, Determinism & Core Integrals** (`compact-core.v5`, CPU + MPS) | [results/ml-infrastructure-final-check-2026-11-20.json](results/ml-infrastructure-final-check-2026-11-20.json) | 2-seed CPU exact resume (bitwise weight/trace equality); `farSuccessRate = 1.0`; `meanQuality = 0.593 ± 0.045`; `meanDirectSun = 0.293 ± 0.003`; `meanDiffuseSky = 0.177 ± 0.011`; native MPS 3-ep smoke (`mpsFallback=false`) | Software, gradient, and exact-resume pipeline validation on synthetic fixtures; small budgets do not test learned policy quality or convergence |
| **Scene Cache & Vertical Scaling** (`coarse_2d1d_v3`, M4 MPS) | [results/scalable-height-speedup-2026-11-19.json](results/scalable-height-speedup-2026-11-19.json) | Cold→Warm SHA-256 cache setup: `8.16s→37ms` (`10F`), `8.53s→46ms` (`20F`), `4.57s→41ms` (`40F`); warm single-step eval stays `8.54–9.95 ms` across `10F→40F` | Historical pre-`compact-core.v5` compute benchmark; isolates scene cache and GPU/CPU field solver scaling |
| **Multi-Scale Cascade Tradeoff** (M4 MPS, 20–35 ep) | [results/multiscale-ablation-2026-11-19.json](results/multiscale-ablation-2026-11-19.json) | On `xl_sgp_40F` (ep 25), `4×4×2`/`8×8×3` cascades cut wall-clock time vs `1×1×1` (`69.3s` vs `176.6s`) while `c2f_2x2x1` reached `0.707` objective (`40%` FAR success vs `100%` for `8×8×3` at `0.623`) | Historical pre-`compact-core.v5` exploratory ablation on single sites; full multi-seed study under `voxelform.research.v1` remains open |

Complete raw JSON logs and hash indices (including earlier `compact-core.v2` check, single-site pilots, and solver validation probes) are preserved under [results/README.md](results/README.md).

---

## Quickstart & Reproducibility

Requires Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and Node.js 22 (for JS viewer tests). Zero external datasets are needed for the self-contained CPU smoke experiment and unit tests.

### 1. Install & Run Self-Contained CPU Smoke Experiment

```sh
uv sync --frozen --extra dev --python 3.12

# Train two seeds (123, 456) on synthetic geometry with strict CPU determinism
uv run --frozen vf-research train \
  --config configs/research-smoke.json \
  --output runs/smoke-check

# Evaluate fixed-budget checkpoint on held-out synthetic split
uv run --frozen vf-research evaluate \
  --checkpoint runs/smoke-check/seed-123/latest.pt \
  --split test \
  --output runs/smoke-check/eval-test.json
```

### 2. Multi-Tier Urban Context Training (Optional Local Context Cache)

If cached OSM-derived context geometry (`Module-Lab`) is present in a sibling checkout or at `$VOXELFORM_CONTEXT_ROOT`:

```sh
# Reference run (pure GAE return, spatial_aux_weight = 0.0) on Apple MPS
uv run --frozen vf-research train \
  --config configs/research-real.json \
  --output runs/real-reference

# Matched auxiliary-loss ablation (spatial_aux_weight = 1.2)
uv run --frozen vf-research train \
  --config configs/research-real-aux.json \
  --output runs/real-aux-ablation

# Reproduce the 1,200-episode single-seed auxiliary run on Apple MPS
uv run --frozen vf-research train \
  --config configs/research-real-aux-1200ep.json \
  --output runs/real-aux-1200ep
```

### 3. Live 3D Inspector & Test Suite

```sh
# Launch interactive inspector at http://127.0.0.1:8011 (core plans at /core-plans)
uv run --frozen python main.py

# Run complete verification suite (200 Python tests, 11 JS geometry tests, static lint)
uv run --frozen --extra dev python -m unittest discover -s tests
node --check public/*.js
node --test tests/*.test.mjs
uv run --frozen --extra dev ruff check voxelform/core_plans.py voxelform/experiment.py voxelform/objectives.py voxelform/layout_archive.py tools/research.py tools/render_core_plans.py tools/generate_readme_media.py tests/test_core_plans.py tests/test_experiment.py
```

---

## Repository Structure

```text
VoxelForm-RL/
├── voxelform/
│   ├── environment.py       # 3D voxel lattice, FAR band, step dynamics & metrics
│   ├── unified.py           # Organic3DNetwork (48,125 params), GAE learner & projection Pi_valid
│   ├── coarse_to_fine.py    # Multi-scale cascade actor-critic (CoarseToFineNetwork)
│   ├── objectives.py        # Shared live/headless reward & proxy decomposition
│   ├── cores.py             # Geodesic core distance, badness & greedy core solver
│   ├── core_plans.py        # Canonical compact-core.v5 parametric footprints & SVG export
│   ├── fields.py            # Screened-Laplace diffuse sky, facade depth & acoustic priors
│   ├── shadows.py           # FastVoxelShadows (GPU DDA) & MeshShadows (cached raycaster)
│   ├── patch_policy.py      # Separable Gaussian field smoothing & frontier patch allocation
│   ├── experiment.py        # Config schema, atomic RNG checkpoints, source archival & resume
│   └── layout_archive.py    # SQLite episode ledger & top-five distinct layout archive
├── configs/                 # Reproducible experiment specs (smoke, real, real-aux)
├── tools/                   # CLI entrypoints: research.py, render_core_plans.py, media & benchmarks
├── tests/                   # 200 Python unittest specs + 11 Node geometry/syntax tests
├── docs/                    # Protocol, Model Card, Data Card, Core Layouts & media figures
├── results/                 # Indexed raw JSON benchmarks & historical ablation logs
├── public/                  # Three.js inspector UI & standalone /core-plans viewer
├── unified_server.py        # FastAPI inspection server (port 8011)
└── .github/workflows/       # CPU CI: unit tests, JS checks, ruff lint & smoke train
```

---

## Methodological Limitations & Open Research Directions

- **Uncalibrated Physical Proxies:** `directSunProxy`, `diffuseFacadeProxy`, acoustic `noisePrior`, planar depth, and 8-neighbor geodesic core distance are research surrogate signals — not calibrated daylight simulations, acoustic models, structural analyses, or building-code/egress compliance verifications ([docs/DATA_CARD.md](docs/DATA_CARD.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md)).
- **Hand-Designed Structural Priors:** Morphological opening (`k=3`), long-strip filtering (`<6 m`), connectivity/cavity checks, and deterministic greedy core placement impose substantial inductive bias alongside the learned policy.
- **Open Multi-Seed & Multi-Objective Work:** Full-budget multi-seed runs on `configs/research-real.json` vs `configs/research-real-aux.json`, Pareto trade-off learning across conflicting proxy objectives (e.g., goal-conditioned weight preferences, multi-head vector critics, or constrained Lagrangian updates), and batched multi-environment rollout vectorization remain active next experiments rather than settled results.

---

## Documentation & Citation

- [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md) · [Model Card](docs/MODEL_CARD.md) · [Data & Simulator Card](docs/DATA_CARD.md)
- [Compact Core Geometry (`compact-core.v5`)](docs/CORE_LAYOUTS.md) · [2026-11-20 Technical Review](docs/REVIEW_2026-11-20.md)
- [Results Index](results/README.md) · [Contributing Guidelines](CONTRIBUTING.md) · [Citation Metadata](CITATION.cff)

*Repository code licensing remains an explicit owner decision; vendored Three.js and OpenStreetMap context attributions are preserved under their original terms.*

```bibtex
@software{voxelform_rl_2026,
  title   = {VoxelForm-RL: Structured Reinforcement Learning for Constrained 3D Voxel Editing},
  author  = {Ruslan F},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/guzel-hairdressers/voxelform-rl}
}
```
