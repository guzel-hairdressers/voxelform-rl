# Results and evidence

Current reproducible protocol: [experiment specification](../docs/EXPERIMENT_PROTOCOL.md).

## Matched Greedy fairness parity, Deterministic MAP evaluation & significance tests — 2026-11-23

Matched Apple M4 MPS evaluation (`eval seeds 9201–9205`, archived `400`-ep `far_gated_actor_critic` `best.pt` seeds `4211–4215`, bundled `VOXELFORM_CONTEXT_ROOT`) comparing Legacy `tier_meso2` Greedy (`greedy_learned_cap=False`) against **Fair Greedy (`greedy_learned_cap=True`)** given identical `_learned_quota` sizing and early coarse handoff, `5×`-compute Best-of-5 Stochastic Fair Greedy, and Deterministic MAP RL (`temp=0.0, noise=0.0`) ([Evidence](eval-fairness-parity-mps-2026-11-23.json)):

| Method | Test $J$ (`4` sites) / FAR / `s/ep` | Blind `ALL` (`8` sites) / FAR / `s/ep` | Blind `XS–M` / FAR | Blind `L–XL` / FAR |
|---|---:|---:|---:|---:|
| Legacy Greedy (`greedy_learned_cap=False`, `noise=0.0`) | `0.734` / `100%` / `0.857` | `0.824` / `100%` / `0.647` | `0.833` / `100%` | `0.815` / `100%` |
| Fair Greedy Det (`greedy_learned_cap=True`, `noise=0.0`) | `0.730` / `100%` / `0.897` | `0.833` / `100%` / `0.732` | `0.833` / `100%` | `0.832` / `100%` |
| Fair Greedy Stoch Mean (`noise=1.0`, `5` seeds) | `0.737` / `95%` / `0.871` | `0.682` / `100%` / `0.641` | `0.568` / `100%` | `0.796` / `100%` |
| Fair Greedy Best-of-5 (`5×` compute, `noise=1.0`) | `0.773` / `100%` / `4.355/site` | **`0.875`** / `100%` / `3.207/site` | **`0.900`** / `100%` | `0.850` / `100%` |
| FAR-Gated `actor_critic` Stoch (`temp=1.0, noise=1.0`) | `0.803 ± 0.018` / **`100%`** / `1.024` | `0.864 ± 0.046` / `99.5%` | `0.833 ± 0.084` / `100%` | `0.896 ± 0.017` / `99.0%` |
| **FAR-Gated `actor_critic` Det MAP (`temp=0.0, noise=0.0`)** | **`0.815 ± 0.034`** / `96%` / `1.262` | `0.872 ± 0.062` / `98.5%` / `0.847` | `0.815 ± 0.135` / `97%` | **`0.929 ± 0.027`** / **`100%`** |

- **Exact Student-`t` 95% CIs & `t`/`p`-tests vs. Fair Greedy Det:**
  - FAR-Gated RL Stoch: Test Student-`t4` CI `[0.781, 0.825]` (`t(4)=9.10, p=0.00081`); Blind `L–XL` (`t(4)=8.53, p=0.00104` seed / `t(3)=3.76, p=0.0330` paired site); Blind `ALL` (`t(4)=1.54, p=0.198` seed / `t(7)=0.81, p=0.444` paired site — not statistically significant overall due to `xs_sgp_0346`).
  - FAR-Gated RL Det MAP: Test Student-`t4` CI `[0.772, 0.858]` (`t(4)=5.50, p=0.0053`); Blind `ALL` Student-`t4` CI `[0.795, 0.949]` (`p=0.229` seed / `p=0.525` paired site); Blind `L–XL` Student-`t4` CI **`[0.896, 0.962]`** (`t(4)=8.13, p=0.0012` seed / `t(3)=6.79, p=0.0065` paired site), outscoring `5×` Best-of-5 Fair Greedy (`0.850`) on **4/4** blind `L/XL` sites (`l_prs_0324: 0.925`, `l_sgp_0369: 1.021`, `xl_nyc_0048: 0.906`, `xl_sgp_0386: 0.863`).

## Blind 8-site zero-shot evaluation across XS–XL tiers — 2026-11-23

Independent zero-shot evaluation (`640` episodes total across eval seeds
`9201–9205`, `temp = 1.0 / noise = 1.0`, `tier_meso2 + dense_potential + learned_cap`)
on eight untouched catalog parcels (`xs_sgp_0346`, `s_sgp_0345`, `m_nyc_0003`,
`m_prs_0315`, `l_prs_0324`, `l_sgp_0369`, `xl_nyc_0048`, `xl_sgp_0386`) with
verified zero Haversine context overlap against all `10` training sites and `3`
NYC validation sites for all single-stage `actor_critic` arms ([Evidence](blind-8site-all-tiers-mps-2026-11-23.json) ·
[Figure](../docs/media/blind-8site-and-aux-ablation.png)). *Note on `on_policy_distill`:* embedded `"provenance"` metadata in bundled [`data/teachers/`](../data/teachers/) shows those historical teachers earlier trained/tested on several holdout sites across cities ([distillation report](distill-all-tiers-400ep-mps-2026-11-22.md)), so only single-stage `actor_critic` is zero-shot here.
Across all `5` archived `400`-ep seeds (`4211–4215`), FAR-Gated `actor_critic`
outscores `tier_meso2` Greedy on **all `4/4` blind `L/XL` parcels** and **`6/8`
blind parcels overall**, cutting `XS_M` cross-seed SD from `±0.206` (ungated)
to `±0.084` (`ALL` `±0.101 → ±0.046`):

| Method (`5` seeds unless Greedy) | All `8` blind ($J$ / FAR / Usable m²) | `XS–M` (`4` sites) | `L–XL` (`4` sites) |
|---|---:|---:|---:|
| **`far_gated_actor_critic` (`best.pt`)** | **`0.864 ± 0.046`** / `99.5%` / `3,408.3` | **`0.833 ± 0.084`** / `100%` | **`0.896 ± 0.017`** / `99.0%` |
| `tier_meso2` Greedy (`0` params) | `0.824` / **`100%`** / `2,426.9` | **`0.833`** / `100%` | `0.815` / `100%` |
| `ungated_actor_critic` (`best.pt`) | `0.820 ± 0.101` / `98.0%` / **`3,423.8`** | `0.747 ± 0.206` / `100%` | `0.892 ± 0.014` / `96.0%` |
| `on_policy_distill` (`best.pt`) | `0.659 ± 0.054` / **`100%`** / `3,223.4` | `0.463 ± 0.102` / `100%` | `0.854 ± 0.013` / `100%` |

Per-site comparison (`Greedy` vs. `FAR-Gated actor_critic`): `xs_sgp_0346`
(`0.990` vs. `0.798 ± 0.219`), `s_sgp_0345` (`0.721` vs. **`0.914 ± 0.089`**),
`m_nyc_0003` (`0.527` vs. **`0.547 ± 0.028`**), `m_prs_0315` (`1.093` vs. `1.073 ± 0.064`,
where ungated reaches `1.113 ± 0.040`), `l_prs_0324` (`0.847` vs. **`0.885 ± 0.017`**),
`l_sgp_0369` (`0.931` vs. **`0.992 ± 0.035`**), `xl_nyc_0048` (`0.766` vs. **`0.902 ± 0.021`**),
`xl_sgp_0386` (`0.715` vs. **`0.804 ± 0.024`**). Two-teacher distillation remains
strong on `L/XL` (`0.854`), but its `XS_M` teacher underperforms on tiny `xs_sgp_0346` (`-0.284 ± 0.040`).

## Matched 400-episode auxiliary-supervision schedule & mode ablation — 2026-11-23

Matched five-seed (`4211–4215`), `400`-episode ablation isolating `spatial_aux_weight`
(`des_c`) and `spatial_aux_mode` under identical `tier_meso2 + dense_potential + learned_cap`
settings and the full `400`-episode cosine LR schedule (`1.2e-3 → 2.0e-4`) across
`pure_rl` ([config](../configs/research-pure-rl-all-tiers.json)),
`regression_aux` ([config](../configs/research-regression-aux-all-tiers.json)),
`annealed_aux` ([config](../configs/research-annealed-all-tiers.json)), and
`constant_aux` ([config](../configs/research-control-all-tiers.json)).
[Ablation evidence](aux-ablation-400ep-mps-2026-11-23.json) ·
[Regression evidence](regression-aux-400ep-mps-2026-11-23.json) ·
[Figure](../docs/media/blind-8site-and-aux-ablation.png)

| Arm (`400` ep, `5` seeds `4211–4215`) | `best.pt` Val ($J$ / FAR) | `best.pt` Test ($J$ / FAR / m²) | `ep 400` Val ($J$ / FAR) | `ep 400` Test ($J$ / FAR / m²) | Blind `8`-Site (`ALL` / `L–XL`) |
|---|---:|---:|---:|---:|---:|
| `pure_rl` (`λ_aux = 0.0`) | `0.291 ± 0.046` / `100%` | `0.420 ± 0.063` / `100%` / `1,142.5` | `0.292 ± 0.054` / `94.7%` | `0.418 ± 0.074` / `99.0%` / `1,191.8` | `0.531` / `0.501` |
| `regression_aux` (`mode="regression"`, `λ_aux = 1.2`) | `0.563 ± 0.027` / **`100%`** | `0.641 ± 0.031` / **`100%`** / `2,341.3` (`0.695` MAP) | `0.550 ± 0.033` / `98.7%` | `0.631 ± 0.032` / **`100%`** / `2,421.6` (`0.680` MAP) | `0.775` / `0.786` (`0.775` / `0.821` MAP) |
| **`annealed_aux` (`"reinforce"`, `1.2 → 0.0`)** | **`0.727 ± 0.027`** / `100%` | `0.790 ± 0.021` / `99.0%` / `4,280.2` | **`0.688 ± 0.029`** / `96.0%` | **`0.767 ± 0.038`** / `99.0%` / `4,558.7` | `0.823` / `0.883` |
| **`constant_aux` (`"reinforce"`, `λ_aux = 1.2`)** | `0.723 ± 0.025` / `100%` | **`0.803 ± 0.018`** / **`100%`** / `4,263.8` | `0.667 ± 0.054` / `97.3%` | `0.762 ± 0.036` / `94.0%` / `4,499.7` | **`0.864`** / **`0.896`** |

With matched `400`-episode cosine LR horizons and `5` seeds, pure RL without `des_c`
(`λ_aux = 0.0`) stalls (`0.420` Test; `0.531` blind). Switching constant `λ_aux = 1.2`
from `"reinforce"` to fixed-point `SmoothL1` `"regression"` confirms the saturation-drift
diagnosis — holding **`100.0%` FAR** and near-zero `best → ep 400` drift (`-0.010 ± 0.016`,
`95%` CI `[-0.029, +0.009]`) at episode `400` (`0.631 ± 0.032` Test; `0.771 ± 0.010` Blind)
and cutting variance on tiny `xs_sgp_0346` (`0.881 ± 0.099`) — **but** over-constrains
the spatial actor to static `2·des_c` (paired `t(4) = -11.77, p = 0.00030` on Test and
`t(4) = -15.39, p = 0.00010` on Blind `L–XL` below `"reinforce"` `constant_aux`; `0.695 ± 0.039`
Test / `0.775 ± 0.058` Blind under Det MAP). Holding `"reinforce"` at `1.2` for a
`20%` warmup then cosine-annealing `1.2 → 0.0` reaches constant-aux peak quality
(`0.727` Val, `0.790` Test, `0.883` blind `L/XL`, winning `7/8` blind sites over Greedy)
while keeping episode-`400` Test at `0.767` with **`99.0%` FAR** (vs. `94.0%` for constant
`"reinforce"`).

## 400-episode matched actor-critic control and exchange-gating ablation — 2026-11-22

Strict 5-seed ablation (`seeds 4211–4215`, `spatial_aux_weight = 1.2`,
`constant` schedule) on the exact same `tier_meso2 + dense_potential + learned_cap`
configuration as two-teacher distillation ([config](../configs/research-control-all-tiers.json)).
Without gating `exchange`, `actor_critic` improves from legacy `0.613` to
`0.771 ± 0.072` (`97%` FAR) on test (`0.803 ± 0.016` across 4/5 seeds; Val
`0.736 ± 0.015` at `100%` FAR), with `seed-4213` collapsing via premature
`exchange` spam (`m_prs_0320 = 0.105`). Adding the `|FAR - FAR*| <= 0.75`
`exchange` gate and `0.02` operation-entropy regularization eliminates that
failure mode and lifts validation-selected `actor_critic` (`best.pt`) to
**`0.803 ± 0.018`** (**`100%` FAR**, bootstrap 95% CI `[0.789, 0.817]`, Student-`t4`
CI `[0.781, 0.825]`, `1.024 s/ep` vs. Legacy Greedy `0.857 s/ep`), sweeping all `7/7`
validation (`0.723 ± 0.025`, `100%` FAR) and test plots over `tier_meso2` Greedy
(`0.599` Val / `0.734` Test) and surpassing two-teacher distillation (`0.752 ± 0.013`).
Without validation selection, episode `400` (`latest.pt`) drifts to `0.762 ± 0.036`
(`94%` FAR).
[Report](control-all-tiers-400ep-mps-2026-11-22.md) ·
[Evidence](control-all-tiers-400ep-mps-2026-11-22.json) ·
[Matched Greedy](greedy-control-all-tiers-mps-2026-11-22.json) ·
[Fairness & MAP Parity](eval-fairness-parity-mps-2026-11-23.json)

## 400-episode two-teacher on-policy distillation — 2026-11-22

Five matched model seeds (`4211–4215`) on the city-separated L/XL split,
combining `tier_meso2` (`2×2×1 → 1×1×1` for `area >= 2,500 m²`), `dense_potential`
step shaping, `learned_cap` edit sizing, and on-policy distillation from frozen
`XS_M` and `L_XL` tier experts ([config](../configs/research-distill-all-tiers.json)).
Upgrading `tier_meso2` alone lifts Greedy on the 4-plot test split from `0.681`
to `0.734`. Two-teacher on-policy distillation surpasses both Greedy baselines
across all five seeds (`0.752 ± 0.013` / `99%` FAR at validation-selected
`best.pt`; `0.753 ± 0.013` / `98%` FAR at episode `400`) and leads Greedy on
all three unseen New York validation plots (`0.707 ± 0.022` at `100%` FAR vs.
`0.599` at `93.3%` FAR), while M-tier held-out Paris (`m_prs_0320`: `0.632` vs.
`0.693`) remains behind Greedy.
[Report](distill-all-tiers-400ep-mps-2026-11-22.md) ·
[Evidence](distill-all-tiers-400ep-mps-2026-11-22.json) ·
[Matched Greedy](greedy-control-all-tiers-mps-2026-11-22.json) ·
[Historical L/XL Greedy](historical/greedy-expert-l-xl-mps-2026-11-22.json)

## Historical results (pre-`tier_meso2` & early checks)

All earlier `tier_adaptive`, `60`-episode L/XL pilot, `600`/`1200`-episode M-tier, and synthetic benchmark JSON/Markdown artifacts are archived under [`results/historical/`](historical/).

### 600-episode L and XL follow-up — 2026-11-22

Five matched model seeds per arm, with checkpoint evaluation every 50 episodes.
Imitation averaged higher test scores than guided RL at both checkpoint choices.
Greedy led the four-plot mean and both XL plots. Guided RL's validation-selected
checkpoints led the Singapore L plot. [Report](historical/l-xl-600ep-mps-2026-11-22.md) ·
[Evidence](historical/l-xl-600ep-mps-2026-11-22.json)

### 60-episode L and XL pilot — 2026-11-22

Greedy led at all four test plots in the 60-episode pilot. Guided
RL outscored imitation at every plot. The study also measured adaptive retries,
tested two alternative rewards during training, and compared policies on a
synthetic two-edit planning case. [Report](historical/confirmatory-l-xl-2026-11-22.md) ·
[Evidence](historical/confirmatory-l-xl-mps-2026-11-22.json)

#### Held-out L and XL comparison — 2026-11-22

On Apple M4 MPS, five archived validation-selected checkpoints and two fresh
30-episode guided RL policies ran on the same three unseen plots as the greedy
heuristic. The new policies trained on the original ten XS–XL plots and visited
each one three times. Validation selected the episode-30 checkpoint for both
pilots. All methods used five evaluation seeds per plot, a 3.0 ± 0.1 FAR band,
the tier-specific cascade and the same feasibility checks. Both pilot seeds
spent about 43 seconds in the training loop, including periodic validation.
Greedy ranked its first proposal deterministically and accepted it on each
decision observed here.

| Plot | Split | Archived RL mean $J$ ± seed SD | 30-episode RL mean $J$ | Greedy $J$ |
|---|---|---:|---:|---:|
| L Hong Kong `l_hk_0218` | Validation | 0.651 ± 0.096 | 0.150 | 0.642 |
| L Paris `l_prs_0324` | Test | 0.872 ± 0.022 | 0.420 | 0.847 |
| XL New York `xl_nyc_0048` | Test | 0.586 ± 0.064 | 0.336 | 0.664 |

All 120 rollouts met the FAR band. Archived RL averaged more usable good area
and diffuse sky access at all three plots. Greedy averaged higher direct sun
access and fewer edits. The 30-episode policies scored below greedy on each
plot. Evaluation covers one XL plot and two L plots.

**Reward sensitivity.** The same terminal layouts were rescored under the
original objective, eight ±25% weight perturbations and a single core coverage
term. Paris favored archived RL in all ten variants and New York favored greedy
in all ten. Hong Kong favored archived RL in eight. The archived logs record
all objective components except depth, which can be recovered from the exact
scalar formula and the stored terms.

**Decoder diagnostics.** The first archived model seed produced 188 draws
across 15 rollouts. The projector rejected 62 draws and retained 94.6% of the
proposed cells on accepted draws. Cells were counted at each stage's grid
resolution. An additional 168 single draws held the initial state fixed at each
site and scale. Within each operation and budget group, all proposal masks and
all accepted masks were distinct. Rejection was common in that probe, including
all first-attempt fine removals on the initial Hong Kong and Paris masses. This
diagnostic covers one checkpoint and the probed states. [Metrics, hashes and per-stage counts](historical/heldout-l-xl-mps-2026-11-22.json)

### Earlier M-tier, pipeline, and prototype results

- **600-episode tier-adaptive comparison, five archived seeds (`compact-core.v5`, M4 MPS, seeds `123, 456, 789, 202, 303`, `checkpoint_interval=50`, `spatial_aux_weight=1.2`, cosine LR `1.2e-3 -> 2.0e-4`):** [600-ep 5-seed tier-adaptive summary](historical/real-aux-tier-600ep-mps-2026-11-21.json) (`configs/historical/research-real-aux-tier-600ep.json`; `XS/S/M` direct `1×1×1`, `L` `2×2×1 → 1×1×1`, `XL` `4×4×1 → 2×2×1 → 1×1×1`, scale-normalized Gaussian retry surrogate; `gitCommit: dccc2d5`, `gitDirty: false`; `best.pt` selected strictly on held-out `m_hk_0199` at `[100, 400, 250, 200, 150]`; Val `1.129 ± 0.075` / `FAR 1.00`, Test `0.776 ± 0.103` / `FAR 1.00` vs episode-600 `latest.pt` Val `0.979 ± 0.131` / `FAR 0.88`, Test `0.646 ± 0.352`).
  The historical runner flushed pending optimizer batches at checkpoint boundaries. Its validation and test plots exercised the M-tier fine pathway.
- **Greedy physics-heuristic non-learned baseline (`compact-core.v5`, M4 MPS, `0` learned params):** [greedy heuristic summary](historical/greedy-heuristic-tier-mps-2026-11-21.json) (`vf-research heuristic --config configs/historical/research-real-aux-tier-600ep.json`; frontier selection via `compute_spatial_desirability` + projector $\Pi_{\text{valid}}$; `gitCommit: da065b4`, `gitDirty: false`; Val `m_hk_0199` `1.1315`, Test `0.8099`, `FAR 1.00`, NYC `0.5273`, Paris `1.0925`).
- **600-episode RL ablation without spatial auxiliary loss (`compact-core.v5`, M4 MPS, `seed=123`, `spatial_aux_weight=0.0`):** [600-ep auxiliary-off summary](historical/real-tier-600ep-mps-2026-11-21.json) (`configs/historical/research-real-tier-600ep.json`; same cascade and decoder, trained from scalar reward; Val `m_hk_0199` `0.9074`, Test `0.4771`, `FAR 1.00`, NYC `0.2090`, Paris `0.7452`).
- **1,200-episode flat `1×1×1` baseline (`compact-core.v5`, M4 MPS, `seed=123`, `spatial_aux_weight=1.2`):** [1,200-ep flat baseline summary](historical/real-aux-1200ep-mps-2026-11-21.json) (`configs/historical/research-real-aux-1200ep.json`; 10 balanced XS–XL sites, zero-shot held-out eval on HK `0.7729` + NYC/Paris `0.5782`, matched decoding `1.0/1.0`, full `1200`-episode traces aggregated in 30-ep bins).
- **Pipeline & compact-core verification:** [ML pipeline](historical/ml-infrastructure-final-check-2026-11-20.json),
[compact cores](historical/core-layout-v2-check-2026-11-20.json). The short checks are not convergence or learned-quality evidence.
- **Historical compute & cascade pilots (`2026-11-19`):** [scene cache & vertical scaling](historical/scalable-height-speedup-2026-11-19.json), [early multi-scale cascade tradeoff](historical/multiscale-ablation-2026-11-19.json) (pre-`compact-core.v5`; their older reward/core semantics must not be conflated with current runs).

### First prototype measurements — 2026-11-06

Raw data: [CPU](historical/benchmark-cpu.json), [MPS](historical/benchmark-mps.json),
[field probes](historical/field-probes.json).
These are synthetic method checks, not a design-quality or convergence study.

The final CPU and MPS runs used identical eight-edit volume sequences and
contexts. Timings synchronize devices; CPU uses one PyTorch thread. The scripts
record hardware/software, source hashes, seeds, all timings and failure lists.
Neither final run recorded a failure. Browser rendering/network time is excluded.

| Scene | CPU median edit evaluation | MPS median edit evaluation |
|---|---:|---:|
| L: 64 × 48 m, 4 floors, 8 suns | 16.5 ms | 55.5 ms |
| XL: 96 × 64 m, 8 floors, 12 suns | 59.8 ms | 61.7 ms |

These are individual edits, **not complete episodes** or throughput estimates
for many parallel environments. Candidate generation, policy decisions, backprop
and setup have separate raw timings. CPU is the initial small-demo default.
MPS works, but this implementation's small tensor operations and host/device
coordination do not establish a GPU speed advantage. Batched environments and
fused field kernels are future hypotheses, not measured improvements.

For 8,192 receiver queries at 1 m map pitch, CPU enclosing-face queries took
4.3/6.4 ms (L/XL), versus 1.3/1.9 ms for sun-aligned depth maps. The face method
had lower disagreement in these scenes: 0.37%/0.32% versus 0.56%/0.44% for depth.
At 0.5 m pitch both methods had smaller errors and larger setup/storage costs.
Both false-clear and false-shadow rates are preserved. This supports neither
method being universally best; scene orientation/resolution matter.

The dynamic self-shadow map disagreed with exact receiver DDA on 1.29% (L) and
0.69% (XL) of sampled front-facing receiver/sun pairs. CPU and MPS produced the
same measured errors. This is map-resolution error, not verified exact sunlight.

The cavity probes confirm wider/shallower wells improve both scalar diffusion
access and direct sampled sky visibility. Their magnitudes differ substantially:
for a 7 m wide well 13.5 m below its opening, the diffusion proxy is 0.00078 while
sampled direct sky visibility is 0.078. **The diffusion field is uncalibrated and
the provisional usable-area threshold is not a livability test.** The two-turn
alcove also exposes nonzero diffusion with zero direct sky visibility. Do not
claim daylight performance based on this field.

The relative line-source noise implementation agreed with independent numerical
quadrature to a maximum absolute error of 3.92e-8 in three probes. This validates
the chosen integral, not real-world acoustics: no calibration, building screening,
ground effects or diffraction are included.

Actor gradients/weight changes and save/load were checked. Policy quality,
matched-area comparisons against untrained/greedy controls, held-out real sites,
multi-seed convergence and actual interior feasibility remain untested.
