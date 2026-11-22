# Results and evidence

Current reproducible protocol: [experiment specification](../docs/EXPERIMENT_PROTOCOL.md).
- **600-episode 5-seed tier-adaptive run (`compact-core.v5`, M4 MPS, seeds `123, 456, 789, 202, 303`, `checkpoint_interval=50`, `spatial_aux_weight=1.2`, cosine LR `1.2e-3 -> 2.0e-4`):** [600-ep 5-seed tier-adaptive summary](real-aux-tier-600ep-mps-2026-11-21.json) (`configs/research-real-aux-tier-600ep.json`; `XS/S/M` direct `1×1×1`, `L` `2×2×1 → 1×1×1`, `XL` `4×4×1 → 2×2×1 → 1×1×1`, scale-normalized Gaussian retry surrogate; `gitCommit: dccc2d5`, `gitDirty: false`; `best.pt` selected strictly on held-out `m_hk_0199` at `[100, 400, 250, 200, 150]`; Val `1.129 ± 0.075` / `FAR 1.00`, Test `0.776 ± 0.103` / `FAR 1.00` vs episode-600 `latest.pt` Val `0.979 ± 0.131` / `FAR 0.88`, Test `0.646 ± 0.352`; top validation-selected seeds reach `0.863` / `0.840` / `0.811` Test).
- **Greedy physics-heuristic non-learned baseline (`compact-core.v5`, M4 MPS, `0` learned params):** [greedy heuristic summary](greedy-heuristic-tier-mps-2026-11-21.json) (`vf-research heuristic --config configs/research-real-aux-tier-600ep.json`; deterministic frontier selection via `compute_spatial_desirability` + projector $\Pi_{\text{valid}}$; `gitCommit: da065b4`, `gitDirty: false`; Val `m_hk_0199` `1.1315`, Test `0.8099`, `FAR 1.00`, NYC `0.5273`, Paris `1.0925`).
- **600-episode pure-RL ablation (`compact-core.v5`, M4 MPS, `seed=123`, `spatial_aux_weight=0.0`):** [600-ep pure-RL summary](real-tier-600ep-mps-2026-11-21.json) (`configs/research-real-tier-600ep.json`; identical cascade trained solely on GAE return; Val `m_hk_0199` `0.9074`, Test `0.4771`, `FAR 1.00`, NYC `0.2090`, Paris `0.7452`).
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
