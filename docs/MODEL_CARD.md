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
   gate + `0.02` op entropy) achieves **`0.803 ± 0.018`** (**`100%` FAR**,
   bootstrap 95% CI `[0.789, 0.817]`, Student-`t4` 95% CI `[0.781, 0.825]`,
   `1.024 s/ep` vs. Legacy Greedy `0.857 s/ep`) on test and `0.723 ± 0.025`
   (`100%` FAR) on validation,
   sweeping all `7/7` plots over `tier_meso2` Greedy (`0.599` Val / `0.734` Test)
   and surpassing two-teacher `on_policy_distill` (`0.752 ± 0.013`, `99%` FAR)
   and ungated `actor_critic` (`0.771 ± 0.072`, `97%` FAR, pulled down by one
   `exchange`-spam seed). Without validation selection, constant-aux `actor_critic`
   at episode `400` drifts to `0.762 ± 0.036` (`94%` FAR).
   [Control & ablation](../results/control-all-tiers-400ep-mps-2026-11-22.md) ·
   [Distillation](../results/distill-all-tiers-400ep-mps-2026-11-22.md)
2. **8-site zero-shot blind holdout across `XS`–`XL` (`640` eval rollouts, `400`-ep models, zero overlap for single-stage RL):**
   Evaluating the archived `400`-ep policies on eight untouched catalog parcels
   (`xs_sgp_0346`, `s_sgp_0345`, `m_nyc_0003`, `m_prs_0315`, `l_prs_0324`,
   `l_sgp_0369`, `xl_nyc_0048`, `xl_sgp_0386`), FAR-Gated `actor_critic` scores
   **`0.864 ± 0.046`** (`99.5%` FAR; `XS/S/M = 0.833 ± 0.084`, **`L/XL = 0.896 ± 0.017`**)
   vs. `tier_meso2` Greedy **`0.824`** (`100%` FAR; `XS/S/M = 0.833`, `L/XL = 0.815`),
   ungated `actor_critic` `0.820 ± 0.101` (`98.0%` FAR), and two-teacher
   `on_policy_distill` `0.659 ± 0.054` (`100%` FAR; `L/XL = 0.854 ± 0.013`, hurt
   on tiny `xs_sgp_0346 = -0.284`; note that only single-stage `actor_critic`
   arms are strictly zero-shot here because bundled `data/teachers/*.pt`
   checkpoints used by `on_policy_distill` had earlier site overlap across
   cities). FAR-Gated `actor_critic` beats Greedy on `4/4` blind `L/XL` parcels
   and `6/8` blind parcels overall, while Greedy stays stronger on tiny
   `xs_sgp_0346` (`0.990` vs. `0.798 ± 0.219`) and marginally on `m_prs_0315`
   (`1.093` vs. `1.073 ± 0.064`). Under matched fairness parity
   ([Fairness & MAP parity](../results/eval-fairness-parity-mps-2026-11-23.json)),
   giving Greedy the same `_learned_quota` and early coarse handoff
   (`greedy_learned_cap=True`) yields `0.730` Test (`0.897 s/ep`) and `0.833`
   Blind (`0.832` `L/XL`, `0.732 s/ep`), with `5×` Best-of-5 Stochastic Fair
   Greedy reaching `0.773` Test / `0.875` Blind / `0.850` Blind `L/XL`
   (`4.36–5.54 s/site`); meanwhile deterministic MAP FAR-Gated RL (`temp=0.0,
   noise=0.0`) reaches `0.815 ± 0.034` Test (`96%` FAR, `p=0.0053` vs. Fair
   Greedy Det), `0.872 ± 0.062` Blind ALL (`98.5%` FAR; overall 8-site `p=0.198–0.525`
   not significant due to `xs_sgp_0346` variance), and **`0.929 ± 0.027`**
   (**`100%` FAR**, `t(4)=8.13, p=0.0012` / `t(3)=6.79, p=0.0065`) on Blind `L/XL`,
   beating `5×` Best-of-5 Fair Greedy on `4/4` blind `L/XL` parcels.
   [Blind 8-site evidence](../results/blind-8site-all-tiers-mps-2026-11-23.json)
3. **Matched 400-episode auxiliary-supervision schedule ablation (`seeds 4211–4215`):**
   With matched `400`-episode cosine LR horizons (`1.2e-3 → 2.0e-4`) and `5` seeds,
   removing `des_c` (`pure_rl`, `λ_aux = 0.0`) stalls at `0.420 ± 0.063` Test
   (`100%` FAR) and `0.531 ± 0.075` blind, whereas holding `λ_aux = 1.2` for a
   `20%` warmup then cosine-decaying `1.2 → 0.0` reaches `0.727 ± 0.027` Val /
   `0.790 ± 0.021` Test (`99.0%` FAR) at `best.pt`,
   holds `0.767 ± 0.038` (`99.0%` FAR) at episode `400`, and scores
   `0.883 ± 0.024` on blind `L/XL` parcels (vs. constant `λ_aux = 1.2` at
   `0.803 ± 0.018` `best.pt`, `0.762 ± 0.036` / `94.0%` FAR at `ep 400`,
   `0.896 ± 0.017` blind `L/XL`).
   [Aux ablation evidence](../results/aux-ablation-400ep-mps-2026-11-23.json)
4. **Historical `tier_adaptive` baselines:** `600`-ep M-tier guided RL (`0.776 ± 0.103`
   vs. Greedy `0.810`), `60`-ep L/XL pilot (`0.501` RL / `0.351` Imitation vs. `0.681`
   Greedy), and `600`-ep L/XL follow-up (`0.613 ± 0.079` RL / `0.649 ± 0.021`
   Imitation vs. `0.681` Greedy). [Results index](../results/README.md)

**Released weights & live checkpoint import:** Weight-only frozen tier-expert
teachers (`teacher_xs_m_seed789.pt`, `teacher_l_xl_ep100.pt`) with embedded
`"provenance"` metadata are bundled under
[`data/teachers/`](../data/teachers/) for reproducing `on_policy_distill`. Full
student run directories remain local and gitignored; any saved `voxelform.research.v2` /
`voxelform.multiscale.v1` `14`-channel checkpoint can also be imported into the
live web inspector via `POST /api/policy`, executing `2×2×1 → 1×1×1`
`tier_meso2` coarse-to-fine evaluation on parcels `>= 2,500 m²` alongside `1×1×1`
fine edits (`UnifiedLearner` fallback weights are also synced; UI inference
defaults to standard `logits / T` temperature `T=0.4` and first-draw noise `0.08`,
or deterministic argmax at `T <= 0.0`). New-run checkpoints
contain a reproducibility envelope and source archive. Opt-in `ExperimentConfig`
controls (`spatial_aux_mode="regression"`, `retry_logp_norm="mean"`,
`potential_shaping_gamma`, `continuous_entropy_coef`) are supported alongside
archived defaults without altering baseline config hashes.

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
