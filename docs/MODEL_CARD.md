# Model card — organic voxel-editing actor, 0.2.0 research tooling

**Intended use:** controlled ML experiments on correlated, constrained 3D actions;
resource-limited actor-critic training; visualization and debugging of learned
behaviour. Core plans define environment geometry, not an architectural thesis.

**Model:** 14-channel factored 2D+1D encoder, bounded coarse spatial Gaussian means,
operation categorical head and scalar critic. Training uses GAE; optional physical
auxiliary loss is separately configured. Explicit area allocation, legality masks,
geometry projection and greedy core placement are substantial inductive biases.

**Current evidence** encompasses both the historical `tier_adaptive` baselines and
the updated `tier_meso2 + dense_potential + learned_cap` five-seed suite (`seeds
4211–4215`, Apple M4 MPS, `temp = 1.0 / noise = 1.0`, retaining all FAR
failures):

1. **400-episode matched L/XL split (`3` NYC Val / `4` Paris + Singapore Test):**
   Validation-selected FAR-Gated `actor_critic` (`|FAR - FAR*| <= 0.75` `exchange`
   gate + `0.02` op entropy) achieves **`0.803 ± 0.018`** (**`100%` FAR**, 95% CI
   `[0.789, 0.817]`) on test and `0.723 ± 0.025` (`100%` FAR) on validation,
   sweeping all `7/7` plots over `tier_meso2` Greedy (`0.599` Val / `0.734` Test)
   and surpassing two-teacher `on_policy_distill` (`0.752 ± 0.013`, `99%` FAR)
   and ungated `actor_critic` (`0.771 ± 0.072`, `97%` FAR, pulled down by one
   `exchange`-spam seed). Without validation selection, constant-aux `actor_critic`
   at episode `400` drifts to `0.762 ± 0.036` (`94%` FAR).
   [Control & ablation](../results/control-all-tiers-400ep-mps-2026-11-22.md) ·
   [Distillation](../results/distill-all-tiers-400ep-mps-2026-11-22.md)
2. **8-site zero-shot blind holdout across `XS`–`XL` (`640` episodes, zero train/val overlap):**
   Evaluating the archived `400`-ep policies on eight untouched catalog parcels
   (`xs_sgp_0346`, `s_sgp_0345`, `m_nyc_0003`, `m_prs_0315`, `l_prs_0324`,
   `l_sgp_0369`, `xl_nyc_0048`, `xl_sgp_0386`), FAR-Gated `actor_critic` scores
   **`0.864 ± 0.046`** (`99.5%` FAR; `XS/S/M = 0.833 ± 0.084`, **`L/XL = 0.896 ± 0.017`**)
   vs. `tier_meso2` Greedy **`0.824`** (`100%` FAR; `XS/S/M = 0.833`, `L/XL = 0.815`),
   ungated `actor_critic` `0.820 ± 0.101` (`98.0%` FAR), and two-teacher
   `on_policy_distill` `0.659 ± 0.054` (`100%` FAR; `L/XL = 0.854 ± 0.013`, hurt
   on tiny `xs_sgp_0346 = -0.284`). FAR-Gated `actor_critic` beats Greedy on
   `4/4` blind `L/XL` parcels and `6/8` blind parcels overall, while Greedy stays
   stronger on tiny `xs_sgp_0346` (`0.990` vs. `0.798 ± 0.219`) and marginally on
   `m_prs_0315` (`1.093` vs. `1.073 ± 0.064`).
   [Blind 8-site evidence](../results/blind-8site-all-tiers-mps-2026-11-23.json)
3. **200-episode auxiliary-supervision schedule ablation (`seeds 4211–4213`):**
   Removing `des_c` spatial auxiliary loss (`pure_rl`, `λ_aux = 0.0`) stalls at
   `0.464 ± 0.066` Test (`100%` FAR) within `200` episodes, whereas annealing
   `λ_aux: 1.2 → 0.0` reaches `0.797 ± 0.025` (`best.pt`, `98.3%` FAR) and holds
   **`0.788 ± 0.033`** at episode `200` (when `λ_aux = 0.0`), versus constant
   `λ_aux = 1.2` dropping from `0.801 ± 0.023` (`best.pt`) to `0.746 ± 0.062` by
   episode `200` (`400`-ep cosine LR prefix).
   [Aux ablation evidence](../results/aux-ablation-200ep-mps-2026-11-23.json)
4. **Historical `tier_adaptive` baselines:** `600`-ep M-tier guided RL (`0.776 ± 0.103`
   vs. Greedy `0.810`), `60`-ep L/XL pilot (`0.501` RL / `0.351` Imitation vs. `0.681`
   Greedy), and `600`-ep L/XL follow-up (`0.613 ± 0.079` RL / `0.649 ± 0.021`
   Imitation vs. `0.681` Greedy). [Results index](../results/README.md)

**Released weights & live checkpoint import:** Weight-only frozen tier-expert
teachers (`teacher_xs_m_seed789.pt`, `teacher_l_xl_ep100.pt`) are bundled under
[`data/teachers/`](../data/teachers/) for reproducing `on_policy_distill`. Full
student run directories remain local and gitignored; any saved `voxelform.research.v2` /
`voxelform.multiscale.v1` `14`-channel checkpoint can also be imported into the
live web inspector via `POST /api/policy` (mapping all fine-stage weights identically
by projecting out the `scale=1` zero `stage_feat` column). New-run checkpoints
contain a reproducibility envelope and source archive.

**Known limits:** proxy reward exploitation, imperfect exposure simulation,
partial observability when constraint settings change, site/context distribution
shift, deterministic guard bias, limited seed counts and no long-run convergence
study. Quality means from different FAR acceptances or reward settings require
careful comparison. The common reference evaluator improves consistency but is
itself a hand-designed proxy.

**Excluded uses:** regulatory, structural, accessibility, fire/egress, daylight,
passenger-capacity or occupancy decisions. A stair/lift count based only on height
is a chosen environment prior, not a conclusion about a real building.

**Version boundaries** include the compact core library `compact-core.v5`,
flat sampling semantics `gaussian-retry-path.v1`, tier-adaptive sampling semantics
`gaussian-retry-path-normalized.v1` and research checkpoint format
`voxelform.research.v2`. Exact resume requires the original task, config, scene
and implementation hashes. [Experiment protocol](EXPERIMENT_PROTOCOL.md)
