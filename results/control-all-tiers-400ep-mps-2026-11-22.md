# 400-episode matched actor-critic control and exchange-gating ablation

Strict ablation testing whether gains over legacy `tier_adaptive` Guided RL (`0.613 ± 0.079`) come from the two neural teachers in `on_policy_distill`, or from the `tier_meso2` cascade, `dense_potential` step shaping, and `learned_cap` decoder changes paired with standard heuristic spatial guidance (`actor_critic`, `spatial_aux_weight = 1.2`, `spatial_aux_schedule = "constant"`, [configs/research-control-all-tiers.json](../configs/research-control-all-tiers.json)).

Both `actor_critic` arms trained five matched seeds (`4211–4215`) from scratch for `400` episodes on the exact same 10-site train / 3-site NYC validation / 4-site Paris+Singapore test split as the [two-teacher distillation benchmark](distill-all-tiers-400ep-mps-2026-11-22.md), selecting `best.pt` every `50` episodes on held-out NYC validation and retaining every FAR failure. Note that the test plots were used previously in the 60-episode pilot, 600-episode follow-up, and distillation run, and that the `|FAR - FAR*| <= 0.75` `exchange` gate was motivated by inspecting `seed-4213`'s failure mode in the ungated control run:

1. **Ungated `actor_critic` (baseline control, `0.005` op entropy, unrestricted `exchange`):** Mean `7.25 min/seed` on Apple M4 MPS (`bestValidationEpisodes = [300, 100, 350, 150, 150]`). Achieves `0.736 ± 0.015` (`100%` FAR) on validation and `0.771 ± 0.072` (`97%` FAR) on test (`0.803 ± 0.016` across four seeds `4211, 4212, 4214, 4215`), while `seed-4213` (`0.645`, `m_prs_0320 = 0.105`) collapsed by spamming `exchange` from `FAR ~ 6.0`.
2. **FAR-Gated `actor_critic` (`|FAR - FAR*| <= 0.75` `exchange` gate, `0.02` op entropy):** Masks `exchange` under `learned_cap` whenever `|far - target_far| > 0.75` (if `remove` or `grow` is legal) and raises operation entropy regularization from `0.005` to `0.02`. Mean `7.42 min/seed` (`bestValidationEpisodes = [150, 100, 100, 400, 100]`). Achieves `0.723 ± 0.025` (`100%` FAR) on validation and **`0.803 ± 0.018` (`100%` FAR, `100/100` rollouts, 95% CI `[0.789, 0.817]`)** on test (`best.pt`), outscoring `tier_meso2` Greedy (`0.734`) on all `7/7` validation and test plots and exceeding `on_policy_distill` (`0.752 ± 0.013`) across all five seeds. Under constant `spatial_aux_weight = 1.2` + `0.02` op entropy, continuing to episode `400` (`latest.pt`) drifts downward (`0.762 ± 0.036` Test, `94%` FAR; `0.667 ± 0.054` Val), whereas `on_policy_distill` stays stable at episode `400` (`0.753 ± 0.013`).

## Aggregate comparison

| Method (`temp=1.0 / noise=1.0`, `5` seeds unless `0` params) | Val $J$ ± SD (`3` NYC) | Val FAR | Test $J$ ± SD (`4` sites) | Test FAR | 95% Test CI |
|---|---:|---:|---:|---:|---:|
| Guided RL, `best.pt` (`600` ep, `tier_adaptive`) | — | — | `0.613 ± 0.079` | `93%` | — |
| Greedy physics-heuristic (`tier_adaptive`, `0` params) | — | — | `0.681` | `100%` | — |
| Greedy physics-heuristic (`tier_meso2`, `0` params) | `0.599` | `93.3%` | `0.734` | `100%` | — |
| Two-teacher `on_policy_distill`, `best.pt` (`400` ep) | `0.707 ± 0.022` | **`100%`** | `0.752 ± 0.013` | `99%` | `[0.742, 0.762]` |
| Two-teacher `on_policy_distill`, ep `400` (`latest.pt`) | `0.691 ± 0.020` | **`100%`** | `0.753 ± 0.013` | `98%` | `[0.745, 0.765]` |
| Ungated `actor_critic` control, `best.pt` (`400` ep) | **`0.736 ± 0.015`** | **`100%`** | `0.771 ± 0.072` | `97%` | `[0.706, 0.812]` |
| Ungated `actor_critic` control, ep `400` (`latest.pt`) | `0.726 ± 0.016` | `97.3%` | `0.766 ± 0.080` | `98%` | `[0.694, 0.805]` |
| **FAR-gated `actor_critic`, `best.pt` (`400` ep)** | `0.723 ± 0.025` | **`100%`** | **`0.803 ± 0.018`** | **`100%`** | **`[0.789, 0.817]`** |
| **FAR-gated `actor_critic`, ep `400` (`latest.pt`)** | `0.667 ± 0.054` | `97.3%` | `0.762 ± 0.036` | `94%` | `[0.736, 0.792]` |

## Per-site breakdown (`best.pt` validation-selected checkpoints)

| Site | Split | FAR-Gated `actor_critic` ($J$ / FAR) | Ungated `actor_critic` ($J$ / FAR) | `on_policy_distill` ($J$ / FAR) | Greedy `tier_meso2` ($J$ / FAR) |
|---|---|---:|---:|---:|---:|
| `m_nyc_0005` | Val | `0.686 ± 0.055` / `100%` | **`0.717 ± 0.067`** / `100%` | `0.665 ± 0.029` / `100%` | `0.622` / `100%` |
| `l_nyc_0019` | Val | `0.935 ± 0.021` / `100%` | **`0.936 ± 0.046`** / `100%` | `0.916 ± 0.022` / `100%` | `0.769` / `80%` |
| `xl_nyc_0062` | Val | `0.548 ± 0.024` / `100%` | **`0.555 ± 0.068`** / `100%` | `0.540 ± 0.060` / `100%` | `0.407` / `100%` |
| `m_prs_0320` | Test | **`0.713 ± 0.019`** / `100%` | `0.597 ± 0.276` / `100%` | `0.632 ± 0.044` / `100%` | `0.693` / `100%` |
| `l_sgp_0327` | Test | `0.921 ± 0.042` / `100%` | `0.928 ± 0.070` / `100%` | **`0.944 ± 0.039`** / `100%` | `0.835` / `100%` |
| `l_sgp_0367` | Test | **`0.741 ± 0.018`** / `100%` | `0.741 ± 0.025` / `92%` | `0.699 ± 0.026` / `100%` | `0.646` / `100%` |
| `l_sgp_0379` | Test | **`0.836 ± 0.020`** / `100%` | `0.819 ± 0.065` / `96%` | `0.733 ± 0.055` / `96%` | `0.762` / `100%` |

## Per-seed test scores

| Seed | FAR-Gated `actor_critic` (selected ep) | FAR-Gated ep `400` | Ungated `actor_critic` (`best.pt`) | `on_policy_distill` (`best.pt`) |
|---:|---:|---:|---:|---:|
| `4211` | **`0.827`** (`ep 150`, `100%`) | `0.773` (`95%`) | `0.810` (`100%`) | `0.757` (`95%`) |
| `4212` | `0.785` (`ep 100`, `100%`) | `0.739` (`90%`) | **`0.820`** (`100%`) | `0.767` (`100%`) |
| `4213` | **`0.791`** (`ep 100`, `100%`) | `0.721` (`90%`) | `0.645` (`90%`) | `0.741` (`100%`) |
| `4214` | **`0.816`** (`ep 400`, `100%`) | **`0.816`** (`100%`) | `0.782` (`100%`) | `0.761` (`100%`) |
| `4215` | `0.796` (`ep 100`, `100%`) | `0.761` (`95%`) | **`0.799`** (`95%`) | `0.736` (`100%`) |

## Zero-retrain diagnostic probe (`seed-4213` ungated `best.pt`)

Evaluating the archived ungated `seed-4213/best.pt` checkpoint in memory while gating `exchange` whenever `|far - target_far| > threshold` (and another legal operation exists):

| Gating threshold (`\|FAR - FAR*\|`) | Test mean $J$ | Test FAR | `m_prs_0320` | `l_sgp_0327` | `l_sgp_0367` | `l_sgp_0379` |
|---|---:|---:|---:|---:|---:|---:|
| `None` (unmodified control) | `0.6445` | `90.0%` | `0.1047` (`100%`) | `0.8069` (`100%`) | `0.7756` (`80%`) | `0.8909` (`80%`) |
| `<= 1.0` | `0.7651` | `100.0%` | `0.6189` (`100%`) | `0.8643` (`100%`) | `0.7488` (`100%`) | `0.8285` (`100%`) |
| `<= 0.75` | `0.7782` | `100.0%` | `0.6833` (`100%`) | `0.8778` (`100%`) | `0.7428` (`100%`) | `0.8089` (`100%`) |

- [Machine-readable evidence](control-all-tiers-400ep-mps-2026-11-22.json)
- [Matched `tier_meso2` Greedy baseline](greedy-control-all-tiers-mps-2026-11-22.json)
