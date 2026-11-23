# VoxelForm-RL

[![Research checks](https://github.com/guzel-hairdressers/voxelform-rl/actions/workflows/research.yml/badge.svg)](https://github.com/guzel-hairdressers/voxelform-rl/actions/workflows/research.yml)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-2563eb.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c.svg)
![Protocol](https://img.shields.io/badge/protocol-voxelform.research.v2-0f766e.svg)
![Cores](https://img.shields.io/badge/constraints-compact--core.v5-dc2626.svg)

**Structured deep reinforcement learning for constrained 3D combinatorial optimization.**

> **Architectural / Urban Design Overview:** Looking for a non-ML introduction focused on site massing, FAR targets, daylight/sky access, and vertical core layouts? Read [**README_ARCHITECTURE.md**](README_ARCHITECTURE.md).

VoxelForm-RL is a PyTorch research codebase studying **correlated, high-dimensional 3D spatial actions under hard geometric, topological, and accessibility constraints**. Starting from an initial discrete voxel lattice $`M_0 \in \{0,1\}^{Z \times Y \times X}`$ clipped to a permitted spatial envelope $`A \in \{0,1\}^{Z \times Y \times X}`$, a tier-adaptive factored 2D+1D actor–critic sequentially carves, grows, and redistributes volumetric elements over a finite edit horizon while targeting a prescribed volumetric density band (`FAR* ± ε`) and optimizing multiple uncalibrated spatial exposure and accessibility proxies.

Urban building massing provides the concrete constraint structure, proxy physics, and 3D inspection UI; **multi-scale hierarchical representation, correlated Gaussian latent action spaces, feasibility projection, path-normalized credit assignment, and vertical compute scaling** are the primary Machine Learning objects of study.

---

## Visual Overview

### Sequential 3D Editing Trajectory on Held-Out Paris (`400`-Episode FAR-Gated `tier_meso2`)

![400-episode FAR-gated tier_meso2 trained policy zero-shot rollout on held-out Paris m_prs_0315 showing occupied mass with crimson compact-core.v5 shafts, core-exempt direct sun proxy, and geodesic core travel distance](docs/media/episode-rollout.gif)

*Zero-shot rollout of the **400-episode FAR-gated `tier_meso2` policy** (`runs/control_all_tiers`, `seed=4211`, validation-selected `best.pt`, `compact-core.v5`, target `FAR = 3.00 ± 0.10`, `spatial_aux_weight = 1.2`, cosine LR `1.2e-3 → 2.0e-4`) on untouched blind **Paris (`m_prs_0315`, eval seed `9003`, matched decoding `temp=1.0, noise=1.0`)**: over 19 sculpting edits (`20` states) from `FAR 5.70` down to `FAR 2.92`, objective reaches **`Obj 1.195`** with `71.1%` (`2,890 m²`) usable floor area, `79.3%` `directSunProxy`, `17.2%` `diffuseFacadeProxy`, and `99.7%` core coverage across `4` continuous crimson cores. **Left:** translucent occupied voxels (`1 m × 1 m × 3 m`) with embedded vertical core shafts. **Middle:** exposed-face `directSunProxy` solar visibility (core faces excluded from daylight averaging while casting shadows). **Right:** 8-neighbor geodesic floor-plate travel distance to nearest core without corner cutting. Full clip: [docs/media/episode-rollout.mp4](docs/media/episode-rollout.mp4).*

### Procedural Blender Architectural Maquette (`m_prs_0315`, Zero-AI Pipeline)

![Parallel VoxelForm-RL volumetric lattice and procedural Blender architectural maquette with human-level street views on held-out Paris m_prs_0315](docs/media/blender-maquette-showcase.png)

*Procedurally created in Blender from VoxelForm-RL lattice output + OpenStreetMap (`m_prs_0315`, `seed=9003`) — no AI used for modeling or visualization.*

### Cross-Scale Trained Showcase (`XS` → `XL`) & Zero-Shot Unseen City Transfer

![400-episode FAR-gated tier_meso2 trained layouts with compact-core.v5 shafts across XS, S, M, L, and XL parcels](docs/media/multitier-showcase.png)

*Selected FAR-satisfied evaluation rollouts per plot-size tier from the **400-episode FAR-gated `tier_meso2` shared policy** (`runs/control_all_tiers`, `seed=4211`, `best.pt`; `XS/S/M`: `1×1×1`; `L/XL`: `2×2×1 → 1×1×1`; `target_far = 3.0 ± 0.10`, 20F ceiling, `compact-core.v5`, matched decoding `temp=1.0, noise=1.0`), spanning `xs_sgp_0346` (`7F`, `1.054` obj, `1` core, `94%` coverage, seed `9205`), `s_sgp_0345` (`6F`, `1.087` obj, `1,541 m²` usable, `2` cores, `99%` coverage, seed `9204`), `m_tokyo_0135` (`6F`, `1.129` obj, `2,153 m²` usable, `2` cores, `96%` coverage, seed `9204`), `l_bcn_0074` (`6F`, `1.022` obj, `4,675 m²` usable, `6` cores, `99%` coverage, seed `9204`), and `xl_tokyo_0185` (`6F`, `0.910` obj, `4,286 m²` usable, `8` cores, `96%` coverage, seed `9009`).*

![Paired zero-shot comparison between untrained seed-4211 control and 400-episode FAR-gated tier_meso2 trained policy across unseen New York, Paris, and Singapore plots](docs/media/heldout-generalization.png)

*Zero-shot generalization on unseen held-out cities under **matched evaluation decoding (`temp=1.0, noise=1.0`)**: identical seeds (`9201`, `9203`, `9202`) compare untrained `seed-4211` initialization (top) against the 400-episode FAR-gated `tier_meso2` `best.pt` checkpoint (bottom) across **New York (`l_nyc_0019`, Val `2×2×1→1`: `0.280 → 0.994`, `5,963 m²` usable, `100%` coverage)**, **Paris (`m_prs_0320`, Test `1×1×1`: `0.411 → 0.755`, `1,913 m²` usable, `100%` coverage)**, and **Singapore (`l_sgp_0327`, Test `2×2×1→1`: `0.210 → 0.977`, `5,041 m²` usable, `99%` coverage)**.*

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

At decision step `t`, state $`s_t = (O_t, g_t)`$ consists of a dense voxel observation tensor $`O_t \in \mathbb{R}^{Z_c \times 14 \times Y_c \times X_c}`$ (at the active stage resolution) paired with a normalized goal vector $`g_t \in \mathbb{R}^5`$ in `UnifiedLearner` or $`g_t \in \mathbb{R}^6`$ in `CoarseToFineLearner` (`[far, target_far, far_error, remaining_budget_frac, spent_budget_frac, scale_feature]`):

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

**2. Augmented Retry Accounting & Scale-Normalized Gradients:** When geometric projection rejects a candidate draw (up to `retries = 8`), both `UnifiedLearner` and `CoarseToFineLearner` record **all consumed Gaussian draws** (rejected + accepted) on the full coarse lattice without spatial eligibility masking, keeping score-function accounting unbiased over the augmented retry trajectory. Rather than equaling the exact Gaussian density, `CoarseToFineLearner` scales each retry block by $`1 / \sqrt{N_{\text{lat}}^{(t)}}`$ across the stage latent grid size to form the scale-normalized policy-gradient surrogate $`\tilde{\ell}_\theta(a_t \mid s_t)`$ in the A2C/GAE actor objective so `XS` and `XL` updates maintain comparable gradient magnitude:

```math
\tilde{\ell}_\theta(a_t \mid s_t) = \log \pi_\theta(o_t \mid s_t) + \sum_{k=1}^{K_t} \frac{1}{\sqrt{N_{\text{lat}}^{(t)}}} \log \mathcal{N}\left(z_t^{(k)} ; \mu_\theta(s_t), \sigma_\theta^2\right), \qquad \mathcal{L}_{\text{PG}}(\theta) = -\hat{\mathbb{E}}_t\!\left[\tilde{\ell}_\theta(a_t \mid s_t)\, \hat{A}_t\right]
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

Training runs synchronous actor–critic with GAE (`γ = 0.99`, `λ = 0.95`, rollout horizon `16`), Huber value regression (`0.5×`), operation entropy regularization (`0.005×`, raised to `0.02×` under `learned_cap`), gradient norm clipping (`≤ 2.0`), and Adam (`lr = 1.2e-3`, optional cosine annealing to `2.0e-4` via `lr_schedule = "cosine"`, `wd = 1e-4`). An optional heuristic physics-shaped dense spatial auxiliary objective is isolated via `spatial_aux_weight` (`0.0` in [configs/research-pure-rl-all-tiers.json](configs/research-pure-rl-all-tiers.json); `1.2 → 0.0` linear decay when `spatial_aux_schedule = "annealed"` in [configs/research-annealed-all-tiers.json](configs/research-annealed-all-tiers.json); `1.2` constant in [configs/research-control-all-tiers.json](configs/research-control-all-tiers.json)). Optionally, `reward_profile = "dense_potential"` adds potential-difference step shaping (`2.0 * Δ fine_potential` on audited steps, matched in [unified_server.py](unified_server.py)), `decoder_retry_schedule = "learned_cap"` sizes frontier edits by positive spatial scores, gates premature `exchange` until `|FAR - FAR*| <= 0.75`, allows early coarse handoff, and enforces late pure-FAR moves, while `training_objective = "on_policy_distill"` matches frozen tier-expert `teacher_checkpoints` along on-policy student rollouts alongside scaled GAE (`0.25×`).

---

## Neural Architecture (`CoarseToFineNetwork`, 56,321 Parameters)

![MDP loop and CoarseToFineNetwork tier-adaptive factored 2D+1D architecture with halo-2 sliding vertical window](docs/media/mdp-architecture.png)

Instead of memory-bound full-resolution 3D convolutions across arbitrary building heights $`Z \in [8, 60]`$ or flat single-scale decoding on huge blocks, `CoarseToFineNetwork` (`voxelform/coarse_to_fine.py`) extends the `Organic3DNetwork` `48,125`-parameter factored 2D+1D core (`voxelform/unified.py`) with lightweight multi-scale stems and an exact **halo-2 sliding vertical window**:

- **Tier-Adaptive Entry Stems:** Routes parcels by metric area (`tier_cascade_blocks`): under `tier_adaptive`, `XS / S / M` (`< 2,500 m²`) run directly at `1×1×1` via the strided `4×` fine stem; `L` (`2,500–3,799 m²`) runs `2×2×1 → 1×1×1` via `coarse_stem`; and `XL` (`≥ 3,800 m²`) runs `4×4×1 → 2×2×1 → 1×1×1` via `macro_stem → coarse_stem → fine stem`. Alternatively, `tier_meso2` unifies all `L / XL` parcels (`≥ 2,500 m²`) on `2×2×1 → 1×1×1` to preserve `mw = 2` morphological and thin-strip guards at the coarse stage while sharing coarse-to-fine parameters across large plots.
- **Shared Factored 2D+1D Residual Mixing (`_zmix`):** Applies planar `3×3` convolutions (`ctx_xy1`, `ctx_xy2`) followed by shift-and-add `Z`-neighbor projections (`Z-1, Z, Z+1`) inside overlapping windows (`win=16` floors, `halo=2`), keeping peak activation memory independent of maximum storey count while sharing inductive bias across all spatial scales.
- **FiLM-Style Additive Goal & Local Skip Conditioning:** Combines exact floor-mean/max global pooling with `goal_encoder(g_t)` and a pooled `1×1` `local_skip`, outputting bounded coarse latent means $`\mu_\theta = 3\tanh(\text{head}/3)`$, categorical operation logits, and scalar critic value $`V_\phi(s_t)`$.

---

## Reproducible Experiment Protocol & Empirical Benchmarks

The `vf-research` runner (`tools/research.py`, `voxelform/experiment.py`) enforces clean ML evaluation hygiene ([docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)):

- **Strict Split & Leakage Guards:** The canonical L/XL configs train across `10` Barcelona, Tokyo, and Hong Kong plots, validate on `3` New York plots, and test on Paris and Singapore (with `8` additional blind sites having zero Haversine context overlap across splits); historical M-tier splits ([`configs/historical/research-real.json`](configs/historical/research-real.json)) remain archived.
- **Controlled Baselines & Honest Accounting:** Evaluates untrained controls and fixed-budget final checkpoints on shared evaluation seeds (isolating evaluation RNG state), retains failed-FAR episodes in every mean, and computes `meanReferenceQuality` using a standardized `10 m / 12 m` `compact-core.v5` referee so core-on/off and travel-limit ablations remain comparable.
- **Full Provenance & Exact CPU Resume:** Each run archives `config.json`, `manifest.json` (git SHA, source SHA-256s, hardware/packages), `source.tar.gz` (including dirty diffs), `scenes.json`, and atomic optimizer+RNG checkpoints with strict hash verification on `--resume`.

### 400-Episode L/XL Benchmark & Ablation (`tier_meso2` + `dense_potential` + `learned_cap`)

![400-episode 5-seed L/XL validation curves and 7-site held-out comparison across Greedy, Two-Teacher Distillation, Ungated Actor-Critic, and FAR-Gated Actor-Critic](docs/media/l-xl-ablation-400ep.png)

To close the gap on large parcels (`≥ 2,500 m²`), three five-seed `400`-episode arms (`seeds 4211–4215`, `tier_meso2` `2×2×1 → 1×1×1`, `dense_potential`, `learned_cap`) were evaluated on the same train/NYC-val/Paris+Singapore-test split: two-teacher on-policy distillation from frozen `XS_M` + `L_XL` experts ([configs/research-distill-all-tiers.json](configs/research-distill-all-tiers.json), `9.3 min/seed`), and matched single-stage `actor_critic` controls (`spatial_aux_weight = 1.2`, [configs/research-control-all-tiers.json](configs/research-control-all-tiers.json), `7.3–7.4 min/seed`) tested both before and after adding the `|FAR - FAR*| <= 0.75` `exchange` gate and `0.02` op entropy:

| Method (`temp=1.0 / noise=1.0`, `5` seeds unless `0` params) | Val $J$ (`3` NYC) / FAR | Test $J$ (`4` sites) / FAR | Usable Area (Val / Test `m²`) |
|---|---:|---:|---:|
| Guided RL, `best.pt` (`600` ep, `tier_adaptive`) | — | `0.613 ± 0.079` / `93%` | — |
| Greedy physics-heuristic (`tier_adaptive`) | — | `0.681` / `100%` | — |
| Greedy physics-heuristic (`tier_meso2`) | `0.599` / `93.3%` | `0.734` / `100%` | `2,792.6` / `2,786.7` |
| Two-Teacher `on_policy_distill`, `best.pt` (`400` ep) | `0.707 ± 0.022` / **`100%`** | `0.752 ± 0.013` / `99%` | `4,887.0` / `4,479.7` |
| Two-Teacher `on_policy_distill`, ep `400` | `0.691 ± 0.020` / **`100%`** | `0.753 ± 0.013` / `98%` | **`4,994.0`** / **`4,819.2`** |
| Ungated `actor_critic` control, `best.pt` (`400` ep) | **`0.736 ± 0.015`** / **`100%`** | `0.771 ± 0.072` / `97%` | `4,719.0` / `4,182.8` |
| **FAR-Gated `actor_critic`, `best.pt` (`400` ep)** | `0.723 ± 0.025` / **`100%`** | **`0.803 ± 0.018`** / **`100%`** | `4,631.2` / `4,263.8` |
| **FAR-Gated `actor_critic`, ep `400` (`latest.pt`)** | `0.667 ± 0.054` / `97.3%` | `0.762 ± 0.036` / `94%` | `4,785.1` / `4,499.7` |

The strict ablation shows three complementary effects: (1) switching from `tier_adaptive` to `tier_meso2 + dense_potential + learned_cap` supplies the primary improvement over legacy Guided RL (`0.613 → 0.771`, and `0.803 ± 0.016` on four of five ungated seeds); (2) two-teacher `on_policy_distill` prevents catastrophic `exchange`-spam collapse (`±0.013` SD across seeds, stable at `0.753` through ep `400`), though at a conservative quality ceiling; and (3) gating premature `exchange` until `|FAR - FAR*| <= 0.75` (`+ 0.02` op entropy) fixes `seed-4213` directly in single-stage `actor_critic`, lifting validation-selected `best.pt` to **`0.803 ± 0.018`** (`100/100` FAR compliance, 95% CI `[0.789, 0.817]`) and outscoring `tier_meso2` Greedy on **all 7/7 validation and test plots** (`m_prs_0320`: `0.713` vs. `0.693`, `l_sgp_0327`: `0.921` vs. `0.835`, `l_sgp_0367`: `0.741` vs. `0.646`, `l_sgp_0379`: `0.836` vs. `0.762`). Because that `exchange` gate was motivated on these development/test plots, it is evaluated below on an untouched 8-site blind holdout, and constant-aux late drift (`ep 400` at `0.762`) is isolated in the matched 400-episode auxiliary schedule ablation. [Control & ablation report](results/control-all-tiers-400ep-mps-2026-11-22.md) · [Control evidence](results/control-all-tiers-400ep-mps-2026-11-22.json) · [Distillation report](results/distill-all-tiers-400ep-mps-2026-11-22.md) · [Distillation evidence](results/distill-all-tiers-400ep-mps-2026-11-22.json)

#### Blind 8-Site Zero-Shot Evaluation Across `XS`–`XL` (`640` Eval Rollouts, `400`-Ep Models)

![8-site zero-shot blind holdout across XS-XL tiers and matched 400-episode auxiliary-supervision schedule ablation](docs/media/blind-8site-and-aux-ablation.png)

To verify transfer beyond the split used to diagnose `seed-4213`, all archived `400`-ep five-seed arms (`4211–4215`, `best.pt`) and `tier_meso2` Greedy were evaluated zero-shot across **eight untouched catalog parcels** spanning every tier (`xs_sgp_0346`, `s_sgp_0345`, `m_nyc_0003`, `m_prs_0315`, `l_prs_0324`, `l_sgp_0369`, `xl_nyc_0048`, `xl_sgp_0386`; verified zero Haversine context overlap with all `10` train and `3` NYC validation sites; [results/blind-8site-all-tiers-mps-2026-11-23.json](results/blind-8site-all-tiers-mps-2026-11-23.json)):

| Method (`temp=1.0 / noise=1.0`, `5` seeds unless Greedy) | All `8` Blind ($J$ / FAR / Usable m²) | `XS–M` (`4` sites) | `L–XL` (`4` sites) |
|---|---:|---:|---:|
| **`far_gated_actor_critic` (`best.pt`)** | **`0.864 ± 0.046`** / `99.5%` / `3,408.3` | **`0.833 ± 0.084`** / `100%` | **`0.896 ± 0.017`** / `99.0%` |
| `tier_meso2` Greedy (`0` params) | `0.824` / **`100%`** / `2,426.9` | **`0.833`** / `100%` | `0.815` / `100%` |
| `ungated_actor_critic` (`best.pt`) | `0.820 ± 0.101` / `98.0%` / **`3,423.8`** | `0.747 ± 0.206` / `100%` | `0.892 ± 0.014` / `96.0%` |
| `on_policy_distill` (`best.pt`) | `0.659 ± 0.054` / **`100%`** / `3,223.4` | `0.463 ± 0.102` / `100%` | `0.854 ± 0.013` / `100%` |

On the untouched sites, FAR-Gated `actor_critic` cuts `XS_M` inter-seed SD in half (`±0.206 → ±0.084`, lifting `xs_sgp_0346` from `0.522 ± 0.525` to `0.798 ± 0.219`), beats Greedy on **all `4/4` blind `L/XL` parcels** (`l_prs_0324`: `0.885` vs. `0.847`; `l_sgp_0369`: `0.992` vs. `0.931`; `xl_nyc_0048`: `0.902` vs. `0.766`; `xl_sgp_0386`: `0.804` vs. `0.715`) and **`6/8` blind parcels overall** (`s_sgp_0345`: `0.914` vs. `0.721`; `m_nyc_0003`: `0.547` vs. `0.527`), while Greedy stays ahead on tiny `xs_sgp_0346` (`0.990` vs. `0.798`) and marginally on `m_prs_0315` (`1.093` vs. `1.073`, where ungated reaches `1.113`). Two-teacher distillation generalizes well on `L/XL` (`0.854 ± 0.013`) but inherits `XS_M` teacher weakness on `xs_sgp_0346` (`-0.284 ± 0.040`).

#### Matched 400-Episode Auxiliary-Supervision Schedule Ablation (`seeds 4211–4215`)

All three arms share the exact same `400`-episode budget, `5` model seeds (`4211–4215`), cosine LR schedule (`1.2e-3 → 2.0e-4` over `400` episodes), `tier_meso2 + dense_potential + learned_cap` decoder/shaping, and validation-selection / blind-evaluation protocol ([results/aux-ablation-400ep-mps-2026-11-23.json](results/aux-ablation-400ep-mps-2026-11-23.json)):

| Arm (`400` ep, `5` seeds `4211–4215`) | `best.pt` Val ($J$ / FAR) | `best.pt` Test ($J$ / FAR / m²) | `ep 400` Val ($J$ / FAR) | `ep 400` Test ($J$ / FAR / m²) | Blind `8`-Site (`ALL` / `L–XL`) |
|---|---:|---:|---:|---:|---:|
| `pure_rl` (`λ_aux = 0.0`, [config](configs/research-pure-rl-all-tiers.json)) | `0.291 ± 0.046` / `100%` | `0.420 ± 0.063` / `100%` / `1,142.5` | `0.292 ± 0.054` / `94.7%` | `0.418 ± 0.074` / `99.0%` / `1,191.8` | `0.531` / `0.501` |
| **`annealed_aux` (`1.2 → 0.0`, [config](configs/research-annealed-all-tiers.json))** | **`0.727 ± 0.027`** / `100%` | `0.790 ± 0.021` / `99.0%` / `4,280.2` | **`0.688 ± 0.029`** / `96.0%` | **`0.767 ± 0.038`** / **`99.0%`** / `4,558.7` | `0.823` / `0.883` |
| **`constant_aux` (`λ_aux = 1.2`, [config](configs/research-control-all-tiers.json))** | `0.723 ± 0.025` / `100%` | **`0.803 ± 0.018`** / **`100%`** / `4,263.8` | `0.667 ± 0.054` / `97.3%` | `0.762 ± 0.036` / `94.0%` / `4,499.7` | **`0.864`** / **`0.896`** |

With matched horizons and seeds, **pure RL without `des_c` (`λ_aux = 0.0`) stalls through episode `400`** (`0.420` Test; `0.531` blind), confirming the spatial auxiliary objective is essential for `14`-channel spatial credit assignment. **Linearly annealing `λ_aux: 1.2 → 0.0` matches constant `des_c` supervision** (`0.727` vs. `0.723` Val; `0.790` vs. `0.803` Test; beating Greedy on `4/4` blind `L/XL` sites at `0.883 ± 0.024` and `7/8` blind sites overall), while raising fixed-budget episode-`400` FAR compliance from `94.0% → 99.0%` (`0.767` vs. `0.762` Test; `0.688` vs. `0.667` Val) once training finishes under pure RL (`λ_aux = 0.0`), leaving constant `λ_aux = 1.2` slightly stronger only on tiny `xs_sgp_0346` (`0.798` vs. `0.534`).

<details>
<summary>Historical 600-Episode M-Tier Baseline &amp; Component Controls (<code>tier_adaptive</code>, Pre-<code>tier_meso2</code>)</summary>

![600-episode 5-seed tier-adaptive training and 50-episode validation curves plus matched held-out city comparison against scalar-reward RL, flat 1x1x1, and greedy physics-heuristic controls](docs/media/training-curves-600ep.png)

Five earlier baseline runs used **600 episodes** (`443.5–945.3 s` per seed on Apple M4 MPS, `250–368` synchronous GAE updates, cosine LR `1.2e-3 → 2.0e-4`, `checkpoint_interval = 50`, [configs/historical/research-real-aux-tier-600ep.json](configs/historical/research-real-aux-tier-600ep.json)) with model seeds `123, 456, 789, 202, 303` and uniform rotation over 10 XS–XL training sites, validated every 50 episodes on Hong Kong (`m_hk_0199`) and tested zero-shot on `m_nyc_0003` and `m_prs_0315` (`temp=1.0, noise=1.0`, five eval seeds per site, retaining all FAR failures):

1. **Untrained Control (5 seeds):** Untrained initialization baseline ([results/historical/real-aux-tier-600ep-mps-2026-11-21.json](results/historical/real-aux-tier-600ep-mps-2026-11-21.json)).
2. **Flat `1×1×1` (1,200 ep, seed `123`):** Fine-only baseline without coarse-to-fine stages ([results/historical/real-aux-1200ep-mps-2026-11-21.json](results/historical/real-aux-1200ep-mps-2026-11-21.json)).
3. **RL Without Spatial Auxiliary Loss (`spatial_aux_weight = 0.0`, 600 ep, seed `123`):** Cascade trained from scalar reward with the same heuristic-informed decoder ([results/historical/real-tier-600ep-mps-2026-11-21.json](results/historical/real-tier-600ep-mps-2026-11-21.json)).
4. **Greedy Physics-Heuristic (`0` learned parameters):** Ranks frontier edits using `compute_spatial_desirability` and the same projector $\Pi_{\text{valid}}$ ([Greedy results](results/historical/greedy-heuristic-tier-mps-2026-11-21.json)).

| Method (`temp=1.0 / noise=1.0`) | Val `m_hk_0199` ($J$ / FAR / Sun / Sky) | Test `m_nyc_0003` ($J$) | Test `m_prs_0315` ($J$) | **Test Aggregate** ($J$ / FAR / Usable m² / Sun / Sky) |
|---|---:|---:|---:|---:|
| **Untrained Control** (5 seeds) | `0.536 ± 0.056` / `1.00` / `60.8%` / `6.2%` | `0.090 ± 0.045` | `0.612 ± 0.037` | `0.351 ± 0.022` / `1.00` / `507.5` / `38.7%` / `3.9%` |
| **Flat `1×1×1` (`1,200` ep)** (seed `123`) | `0.773` / `1.00` / `64.5%` / `12.5%` | `0.341` | `0.816` | `0.578` / `1.00` / `1,722.0` / `47.1%` / `8.3%` |
| **Tier RL Without Spatial Aux (`λ_aux=0.0`)** (seed `123`) | `0.907` / `1.00` / `66.4%` / `16.9%` | `0.209` | `0.745` | `0.477` / `1.00` / `1,040.4` / `37.5%` / `5.8%` |
| **Tier + Aux, Final Ep 600 (`latest.pt`, 5 seeds)** | `0.979 ± 0.131` / `0.88` / **`82.4%`** / `21.4%` | `0.288 ± 0.477` | `1.004 ± 0.230` | `0.646 ± 0.352` / `1.00` / `1,920.9` / **`52.3%`** / **`11.2%`** |
| **Tier + Aux, 50-Ep Val (`best.pt`, 5 seeds)** | `1.129 ± 0.075` / **`1.00`** / `81.6%` / **`22.2%`** | `0.504 ± 0.133` | `1.047 ± 0.089` | `0.776 ± 0.103` / **`1.00`** / `2,097.3` / `51.3%` / `10.8%` |
| **Greedy Physics-Heuristic** (`0` params) | **`1.131`** / `1.00` / `76.5%` / `18.3%` | `0.527` | `1.093` | **`0.810`** / `1.00` / **`2,190.5`** / `48.4%` / `9.5%` |

Across these five M-tier runs, validation-selected checkpoints lowered inter-seed spread and raised mean test quality vs. episode 600; while `Greedy` held the highest scalar score (`0.810` vs. `0.776`), Guided RL averaged higher direct sun and diffuse sky proxies. Exploratory follow-up studies across L/XL plots (`30`-, `60`-, and `600`-ep pilots under legacy `tier_adaptive`) are archived under [`results/historical/`](results/historical/) and indexed in [results/README.md](results/README.md).

</details>

### Infrastructure Verification & Compute Scaling

| Measurement Scope | Dataset / Artifact | Key Observed Metrics | Interpretation Boundary |
|---|---|---|---|
| **Pipeline, Determinism & Core Integrals** (`compact-core.v5`, CPU + MPS) | [results/historical/ml-infrastructure-final-check-2026-11-20.json](results/historical/ml-infrastructure-final-check-2026-11-20.json) | 2-seed CPU exact resume (bitwise weight/trace equality); `farSuccessRate = 1.0`; `meanQuality = 0.593 ± 0.045`; `meanDirectSun = 0.293 ± 0.003`; `meanDiffuseSky = 0.177 ± 0.011`; native MPS 3-ep smoke (`mpsFallback=false`) | Software, gradient, and exact-resume pipeline validation on synthetic fixtures; small budgets do not test learned policy quality or convergence |
| **Scene Cache & Vertical Scaling** (`coarse_2d1d_v3`, M4 MPS) | [results/historical/scalable-height-speedup-2026-11-19.json](results/historical/scalable-height-speedup-2026-11-19.json) | Cold→Warm SHA-256 cache setup: `8.16s→37ms` (`10F`), `8.53s→46ms` (`20F`), `4.57s→41ms` (`40F`); warm single-step eval stays `8.54–9.95 ms` across `10F→40F` | Historical pre-`compact-core.v5` compute benchmark; isolates scene cache and GPU/CPU field solver scaling |

All retained raw benchmark summaries, pipeline verification checks, and solver probes are indexed in [results/README.md](results/README.md).

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

### 2. Multi-Tier Urban Context Training (Standalone Bundled Dataset)

All 27 bundled XS–XL urban parcels across 6 cities (Barcelona, Hong Kong, New York, Paris, Singapore, Tokyo) and their surrounding building context are stored in [`data/urban_context/public/`](data/urban_context/public/) (overrideable via `$VOXELFORM_CONTEXT_ROOT` or a sibling `Module-Lab` checkout). [See split coverage and the evaluation plan.](docs/EXPERIMENT_PROTOCOL.md#next-experiments)

```sh
# Matched tier_meso2 greedy physics-heuristic baseline
uv run --frozen vf-research heuristic \
  --config configs/research-control-all-tiers.json \
  --output results/greedy-control-all-tiers.json

# FAR-gated single-stage actor-critic (tier_meso2 + dense_potential + learned_cap, seeds 4211-4215)
uv run --frozen vf-research train \
  --config configs/research-control-all-tiers.json \
  --output runs/control_all_tiers

# Two-teacher on-policy distillation across XS-XL tiers (uses bundled data/teachers/*.pt)
uv run --frozen vf-research train \
  --config configs/research-distill-all-tiers.json \
  --output runs/distill_all_tiers

# Matched 400-episode, 5-seed auxiliary supervision ablations (annealed 1.2->0.0 vs. pure-RL lambda_aux=0.0)
uv run --frozen vf-research train \
  --config configs/research-annealed-all-tiers.json \
  --output runs/annealed_all_tiers_400ep
uv run --frozen vf-research train \
  --config configs/research-pure-rl-all-tiers.json \
  --output runs/pure_rl_all_tiers_400ep
# Canonical vs. historical config index: docs/EXPERIMENT_PROTOCOL.md#canonical-vs-historical-configurations
```

### 3. Live 3D Inspector & Test Suite

```sh
# Launch interactive inspector at http://127.0.0.1:8011 (core plans at /core-plans;
# POST /api/policy accepts live checkpoints and voxelform.research.v2 / multiscale weights,
# executing 2x2x1 -> 1x1x1 tier_meso2 cascades on >=2,500 m2 parcels with learned_cap + |FAR - FAR*| <= 0.75)
uv run --frozen python main.py

# Run complete verification suite (219 Python tests, 11 JS geometry tests, ruff + mypy)
uv run --frozen --extra dev python -m unittest discover -s tests
node --check public/*.js
node --test tests/*.test.mjs
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev mypy
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
├── data/
│   ├── urban_context/       # Standalone bundled 27-site XS–XL OSM parcel & context dataset (6 cities)
│   └── teachers/            # Bundled compact weight-only XS_M and L_XL teacher checkpoints
├── configs/                 # Canonical & historical experiment specs (see EXPERIMENT_PROTOCOL.md)
├── tools/                   # CLI entrypoints: research.py, render_core_plans.py, media & benchmarks
├── tests/                   # 219 Python unittest specs + 11 Node geometry/syntax tests
├── docs/                    # Protocol, Model Card, Data Card, Core Layouts & media figures
├── results/                 # Retained raw JSON benchmarks, core checks & solver probes
├── public/                  # Three.js inspector UI & standalone /core-plans viewer
├── unified_server.py        # FastAPI inspection server (port 8011; fine_potential + research import)
├── README_ARCHITECTURE.md   # Architectural & urban massing overview (non-ML reader guide)
└── .github/workflows/       # CPU CI: unit tests, JS checks, ruff lint & smoke train
```

---

## Methodological Limitations & Open Research Directions

- **Uncalibrated Physical Proxies:** `directSunProxy`, `diffuseFacadeProxy`, acoustic `noisePrior`, planar depth, and 8-neighbor geodesic core distance are research surrogate signals — not calibrated daylight simulations, acoustic models, structural analyses, or building-code/egress compliance verifications ([docs/DATA_CARD.md](docs/DATA_CARD.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md)).
- **Hand-Designed Structural & Heuristic Priors:** Morphological opening (`k=3`), long-strip filtering (`<6 m`), connectivity/cavity checks, deterministic greedy core placement, and the `compute_spatial_desirability` auxiliary field (`des_c`) inject strong inductive bias. With `tier_meso2`, `dense_potential`, and `learned_cap`, FAR-gated single-stage `actor_critic` beats `tier_meso2` Greedy on all `7/7` Val/Test sites (`0.803 ± 0.018` vs. `0.734`) and transfers zero-shot across the **8-site blind holdout** (`0.864 ± 0.046` vs. `0.824`, winning `4/4` blind `L/XL` and `6/8` overall). Meanwhile, our matched `400`-ep, `5`-seed ablation confirms pure RL without `des_c` stalls (`0.420 ± 0.063` Test), whereas **annealing `λ_aux: 1.2 → 0.0`** sustains performance even at `λ_aux = 0` (`0.790` `best.pt`, `0.767` / `99.0%` FAR at `ep 400`, `0.883` blind `L/XL`), leaving tiny-parcel stability (`xs_sgp_0346`: `0.798` vs. Greedy `0.990`) and dense M-tier canyons (`m_nyc_0003`: `0.547`) as clear open targets.
- **Open Multi-Objective & Throughput Work:** Pareto trade-off learning across conflicting proxy objectives (e.g., goal-conditioned weight preferences, multi-head vector critics, or Lagrangian FAR constraints) and batched multi-environment rollout vectorization remain active next experiments rather than settled results.

---

## Documentation & Citation

- [Architectural & Urban Design Overview](README_ARCHITECTURE.md)
- [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md) · [Model Card](docs/MODEL_CARD.md) · [Data & Simulator Card](docs/DATA_CARD.md)
- [Compact Core Geometry (`compact-core.v5`)](docs/CORE_LAYOUTS.md) · [Results Index](results/README.md) · [Contributing Guidelines](CONTRIBUTING.md) · [Citation Metadata](CITATION.cff)

Original project code is available under the MIT License. Three.js and OpenStreetMap data retain their respective terms. [Code license](LICENSE) · [Three.js license](public/vendor/THREE-LICENSE.txt) · [Data provenance](docs/DATA_CARD.md)

```bibtex
@software{voxelform_rl_2026,
  title   = {VoxelForm-RL: Structured Reinforcement Learning
             for Constrained 3D Voxel Editing},
  author  = {Fazulzianov, Ruslan},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/guzel-hairdressers/voxelform-rl}
}
```
