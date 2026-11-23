# Results and evidence

Current reproducible protocol: [experiment specification](../docs/EXPERIMENT_PROTOCOL.md).

## Blind 8-site zero-shot evaluation across XS–XL tiers — 2026-11-23

Independent zero-shot evaluation (`640` episodes total across eval seeds
`9201–9205`, `temp = 1.0 / noise = 1.0`, `tier_meso2 + dense_potential + learned_cap`)
on eight untouched catalog parcels (`xs_sgp_0346`, `s_sgp_0345`, `m_nyc_0003`,
`m_prs_0315`, `l_prs_0324`, `l_sgp_0369`, `xl_nyc_0048`, `xl_sgp_0386`) with
verified zero Haversine context overlap against all `10` training sites and `3`
NYC validation sites ([Evidence](blind-8site-all-tiers-mps-2026-11-23.json) ·
[Figure](../docs/media/blind-8site-and-aux-ablation.png)).
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

## 200-episode auxiliary-supervision schedule ablation — 2026-11-23

Three-seed (`4211–4213`) ablation isolating `spatial_aux_weight` (`des_c`) under
FAR-gated `tier_meso2 + dense_potential + learned_cap` (`pure_rl`
[config](../configs/research-pure-rl-all-tiers.json) and `annealed_aux`
[config](../configs/research-annealed-all-tiers.json) trained for `200` episodes;
`constant_aux` uses the `<= 200`-episode `best.pt` checkpoints `150/100/100` plus
evaluated `checkpoint-000200.pt` from `runs/control_all_tiers`, which ran a
`400`-episode cosine LR horizon).
[Ablation evidence](aux-ablation-200ep-mps-2026-11-23.json) ·
[Figure](../docs/media/blind-8site-and-aux-ablation.png)

| Arm (`seeds 4211–4213`) | `best.pt` Val ($J$ / FAR) | `best.pt` Test ($J$ / FAR / m²) | `ep 200` Val ($J$ / FAR) | `ep 200` Test ($J$ / FAR / m²) |
|---|---:|---:|---:|---:|
| `pure_rl` (`λ_aux = 0.0`) | `0.313 ± 0.054` / `100%` | `0.464 ± 0.066` / `100%` / `547.2` | `0.311 ± 0.054` / `100%` | `0.444 ± 0.088` / `100%` / `473.8` |
| **`annealed_aux` (`1.2 → 0.0`)** | **`0.735 ± 0.021`** / `100%` | `0.797 ± 0.025` / `98.3%` / `4,223.7` | **`0.720 ± 0.040`** / `97.8%` | **`0.788 ± 0.033`** / `98.3%` / `4,269.7` |
| `constant_aux` (`λ_aux = 1.2`) | `0.718 ± 0.030` / `100%` | **`0.801 ± 0.023`** / `100%` / `4,274.0` | `0.669 ± 0.018` / `91.1%` | `0.746 ± 0.062` / `98.3%` / `4,287.4` |

Pure RL without `des_c` (`λ_aux = 0.0`) stalls within `200` episodes, whereas
annealing `1.2 → 0.0` achieves constant-aux `best.pt` quality (`0.797` vs. `0.801`)
while eliminating the `-0.055` late drift at `ep 200` (`0.788` vs. `0.746` Test;
`0.720` vs. `0.669` Val) even after `λ_aux` reaches zero.

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
**`0.803 ± 0.018`** (**`100%` FAR**, 95% CI `[0.789, 0.817]`), sweeping all `7/7`
validation (`0.723 ± 0.025`, `100%` FAR) and test plots over `tier_meso2` Greedy
(`0.599` Val / `0.734` Test) and surpassing two-teacher distillation (`0.752 ± 0.013`).
Without validation selection, episode `400` (`latest.pt`) drifts to `0.762 ± 0.036`
(`94%` FAR).
[Report](control-all-tiers-400ep-mps-2026-11-22.md) ·
[Evidence](control-all-tiers-400ep-mps-2026-11-22.json) ·
[Matched Greedy](greedy-control-all-tiers-mps-2026-11-22.json)

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
[L/XL Greedy](greedy-expert-l-xl-mps-2026-11-22.json)

## 600-episode L and XL follow-up — 2026-11-22

Five matched model seeds per arm, with checkpoint evaluation every 50 episodes.
Imitation averaged higher test scores than guided RL at both checkpoint choices.
Greedy led the four-plot mean and both XL plots. Guided RL's validation-selected
checkpoints led the Singapore L plot. [Report](l-xl-600ep-mps-2026-11-22.md) ·
[Evidence](l-xl-600ep-mps-2026-11-22.json)

## 60-episode L and XL pilot — 2026-11-22

Greedy led at all four test plots in the 60-episode pilot. Guided
RL outscored imitation at every plot. The study also measured adaptive retries,
tested two alternative rewards during training, and compared policies on a
synthetic two-edit planning case. [Report](confirmatory-l-xl-2026-11-22.md) ·
[Evidence](confirmatory-l-xl-mps-2026-11-22.json)

## Held-out L and XL comparison — 2026-11-22

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
diagnostic covers one checkpoint and the probed states. [Metrics, hashes and per-stage counts](heldout-l-xl-mps-2026-11-22.json)

## Earlier research results

- **600-episode tier-adaptive comparison, five archived seeds (`compact-core.v5`, M4 MPS, seeds `123, 456, 789, 202, 303`, `checkpoint_interval=50`, `spatial_aux_weight=1.2`, cosine LR `1.2e-3 -> 2.0e-4`):** [600-ep 5-seed tier-adaptive summary](real-aux-tier-600ep-mps-2026-11-21.json) (`configs/research-real-aux-tier-600ep.json`; `XS/S/M` direct `1×1×1`, `L` `2×2×1 → 1×1×1`, `XL` `4×4×1 → 2×2×1 → 1×1×1`, scale-normalized Gaussian retry surrogate; `gitCommit: dccc2d5`, `gitDirty: false`; `best.pt` selected strictly on held-out `m_hk_0199` at `[100, 400, 250, 200, 150]`; Val `1.129 ± 0.075` / `FAR 1.00`, Test `0.776 ± 0.103` / `FAR 1.00` vs episode-600 `latest.pt` Val `0.979 ± 0.131` / `FAR 0.88`, Test `0.646 ± 0.352`).
  The historical runner flushed pending optimizer batches at checkpoint boundaries. Its validation and test plots exercised the M-tier fine pathway.
- **Greedy physics-heuristic non-learned baseline (`compact-core.v5`, M4 MPS, `0` learned params):** [greedy heuristic summary](greedy-heuristic-tier-mps-2026-11-21.json) (`vf-research heuristic --config configs/research-real-aux-tier-600ep.json`; frontier selection via `compute_spatial_desirability` + projector $\Pi_{\text{valid}}$; `gitCommit: da065b4`, `gitDirty: false`; Val `m_hk_0199` `1.1315`, Test `0.8099`, `FAR 1.00`, NYC `0.5273`, Paris `1.0925`).
- **600-episode RL ablation without spatial auxiliary loss (`compact-core.v5`, M4 MPS, `seed=123`, `spatial_aux_weight=0.0`):** [600-ep auxiliary-off summary](real-tier-600ep-mps-2026-11-21.json) (`configs/research-real-tier-600ep.json`; same cascade and decoder, trained from scalar reward; Val `m_hk_0199` `0.9074`, Test `0.4771`, `FAR 1.00`, NYC `0.2090`, Paris `0.7452`).
- **1,200-episode flat `1×1×1` baseline (`compact-core.v5`, M4 MPS, `seed=123`, `spatial_aux_weight=1.2`):** [1,200-ep flat baseline summary](real-aux-1200ep-mps-2026-11-21.json) (`configs/research-real-aux-1200ep.json`; 10 balanced XS–XL sites, zero-shot held-out eval on HK `0.7729` + NYC/Paris `0.5782`, matched decoding `1.0/1.0`, full `1200`-episode traces aggregated in 30-ep bins).
- **Pipeline & compact-core verification:** [ML pipeline](ml-infrastructure-final-check-2026-11-20.json),
[compact cores](core-layout-v2-check-2026-11-20.json). The short checks are not convergence or learned-quality evidence.
- **Historical compute & cascade pilots (`2026-11-19`):** [scene cache & vertical scaling](scalable-height-speedup-2026-11-19.json), [early multi-scale cascade tradeoff](multiscale-ablation-2026-11-19.json) (pre-`compact-core.v5`; their older reward/core semantics must not be conflated with current runs).

# First prototype measurements — 2026-11-06

Raw data: [CPU](benchmark-cpu.json), [MPS](benchmark-mps.json),
[field probes](field-probes.json).
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
