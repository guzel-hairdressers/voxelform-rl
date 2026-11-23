# 400-episode two-teacher on-policy distillation

Five matched model seeds (`4211–4215`) trained from scratch for `400` episodes on the same city-separated splits as the [600-episode L and XL follow-up](historical/l-xl-600ep-mps-2026-11-22.md) ([configs/research-distill-all-tiers.json](../configs/research-distill-all-tiers.json)). The runner validated every `50` episodes on three unseen New York plots and selected checkpoints by FAR success rate first, then mean validation quality, breaking ties toward earlier episodes. Test scores retain every FAR failure. Note that the test plots were used previously in the 60-episode pilot and 600-episode follow-up.

Relative to the 600-episode follow-up, this run combines four changes:

1. **Unified `2×2×1 → 1×1×1` cascade for L and XL (`multiscale_schedule = "tier_meso2"`):** All parcels with `area >= 2,500 m²` start at `2×2×1` instead of introducing a `4×4×1` macro stage at `>= 3,800 m²` (`l_sgp_0367` at `3,938 m²`, `l_sgp_0379` at `3,845 m²`, and `xl_nyc_0062` at `4,687 m²`), keeping the `mw = 2` width and thin-strip guards active and unifying parameter sharing across L and XL sites.
2. **Dense potential step shaping (`reward_profile = "dense_potential"`):** Adds scaled potential-difference step rewards on coarse and audited fine steps.
3. **Budget-aware positive-logit edit sizing (`decoder_retry_schedule = "learned_cap"`):** Caps accepted edits to positive-scoring frontier voxels within budget-dependent floors, allows coarse handoff near the target band (`abs(far - target_far) <= 0.70`), and disables fine `exchange` in the final `<= 4` productive steps when FAR is not yet satisfied.
4. **Two-teacher on-policy distillation (`training_objective = "on_policy_distill"`):** Student rollouts (`temp=1.0`, `noise=1.0`) query frozen tier experts (`XS_M`: `runs/real-aux-tier-600ep-5seed/seed-789/best.pt`; `L_XL`: `runs/expert_l_xl_ep100_backup.pt`, trained on the same training split) at student-visited states, optimizing masked spatial distillation and legal-masked operation KL alongside `0.25×` scaled GAE updates.

Upgrading the cascade alone (`tier_meso2`) raises the non-learned Greedy baseline on the 4-plot test split from `0.681` to `0.734`. Two-teacher on-policy distillation outperforms both Greedy baselines across all five seeds on validation-selected (`0.752 ± 0.013`, `99%` FAR) and episode-400 (`0.753 ± 0.013`, `98%` FAR) checkpoints, and achieves `100%` FAR across all three unseen New York validation sites (`0.707 ± 0.022` vs. Greedy `0.599` at `93.3%` FAR).

## Aggregate comparison

| Method (`temp=1.0 / noise=1.0`) | Cascade | Val $J$ ± SD (`3` NYC sites) | Val FAR | Test $J$ ± SD (`4` sites) | Test FAR |
|---|---|---:|---:|---:|---:|
| Guided RL, validation-selected (`600` ep, 5 seeds) | `tier_adaptive` | — | — | `0.613 ± 0.079` | `93%` |
| Imitation, episode `600` (`tier_adaptive`, 5 seeds) | `tier_adaptive` | — | — | `0.669 ± 0.011` | `100%` |
| Greedy physics-heuristic (`0` params) | `tier_adaptive` | — | — | `0.681` | `100%` |
| Greedy physics-heuristic (`0` params) | `tier_meso2` | `0.599` | `93.3%` | `0.734` | `100%` |
| **On-policy distill, validation-selected (`best.pt`, 5 seeds)** | `tier_meso2` | **`0.707 ± 0.022`** | **`100%`** | **`0.752 ± 0.013`** | **`99%`** |
| **On-policy distill, episode `400` (`latest.pt`, 5 seeds)** | `tier_meso2` | `0.691 ± 0.020` | **`100%`** | **`0.753 ± 0.013`** | `98%` |

- **95% bootstrap CIs across model seeds (`distill_all_tiers`):** Validation `best.pt` `[0.690, 0.724]`; Test `best.pt` `[0.742, 0.762]`; Test episode `400` `[0.745, 0.765]`.
- **Usable floor area:** Validation-selected distilled policies average `4,887.0 m²` usable good area on validation and `4,479.7 m²` on test (`4,994.0 m²` / `4,819.2 m²` at episode `400`), versus `2,792.6 m²` and `2,786.7 m²` for `tier_meso2` Greedy.
- **Validation-selected episodes:** `4211 → 150`, `4212 → 150`, `4213 → 200`, `4214 → 350`, `4215 → 300`. Mean training-loop time was `9.3` minutes (`556.0 s`) per seed on Apple M4 MPS, including periodic validation.

## Per-site breakdown

| Site | Split | Area (`m²`) | Distill selected ($J$ / FAR) | Distill ep `400` ($J$ / FAR) | Greedy `tier_meso2` ($J$ / FAR) | Legacy Greedy `tier_adaptive` ($J$) |
|---|---|---:|---:|---:|---:|---:|
| `m_nyc_0005` | Val | `1,287.1` | **`0.665 ± 0.029`** / `100%` | `0.654 ± 0.032` / `100%` | `0.622` / `100%` | — |
| `l_nyc_0019` | Val | `2,964.8` | **`0.916 ± 0.022`** / `100%` | `0.869 ± 0.031` / `100%` | `0.769` / `80%` | — |
| `xl_nyc_0062` | Val | `4,686.7` | `0.540 ± 0.060` / `100%` | **`0.549 ± 0.033`** / `100%` | `0.407` / `100%` | — |
| `m_prs_0320` | Test | `1,942.6` | `0.632 ± 0.044` / `100%` | `0.597 ± 0.071` / `100%` | **`0.693`** / `100%` | **`0.693`** |
| `l_sgp_0327` | Test | `3,470.9` | `0.944 ± 0.039` / `100%` | **`1.002 ± 0.028`** / `92%` | `0.835` / `100%` | `0.835` |
| `l_sgp_0367` | Test | `3,938.4` | `0.699 ± 0.026` / `100%` | **`0.733 ± 0.037`** / `100%` | `0.646` / `100%` | `0.534` |
| `l_sgp_0379` | Test | `3,844.7` | `0.733 ± 0.055` / `96%` | `0.683 ± 0.025` / `100%` | **`0.762`** / `100%` | `0.663` |

## Per-seed test scores

Every individual seed surpasses both `tier_adaptive` Greedy (`0.681`) and upgraded `tier_meso2` Greedy (`0.734`) at both `best.pt` and episode `400`:

| Seed | Selected episode | Selected quality | Selected FAR | Episode `400` quality | Episode `400` FAR |
|---:|---:|---:|---:|---:|---:|
| `4211` | `150` | `0.757` | `95%` | `0.754` | `100%` |
| `4212` | `150` | `0.767` | `100%` | `0.749` | `100%` |
| `4213` | `200` | `0.741` | `100%` | `0.740` | `90%` |
| `4214` | `350` | `0.761` | `100%` | `0.775` | `100%` |
| `4215` | `300` | `0.736` | `100%` | `0.749` | `100%` |

- [Machine-readable evidence](distill-all-tiers-400ep-mps-2026-11-22.json)
- [Matched `tier_meso2` Greedy baseline](greedy-control-all-tiers-mps-2026-11-22.json)
- [Matched `L/XL` Greedy baseline (`tier_meso2`)](historical/greedy-expert-l-xl-mps-2026-11-22.json)
