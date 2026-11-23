# 60-episode L and XL pilot

Fixed guidance was promoted by the validation screen. The selected actor–critic and imitation control each trained for 60 episodes on five model seeds. The test compared the final checkpoints to greedy under the same decoder and five evaluation seeds per plot. The 600-episode follow-up validates every 50 episodes and reports both validation-selected and final checkpoints. [Follow-up protocol](../../docs/EXPERIMENT_PROTOCOL.md#600-episode-lxl-follow-up)

| Test plot | Promoted RL | Imitation | Greedy |
|---|---:|---:|---:|
| M `m_prs_0320` | 0.475 | 0.351 | 0.693 |
| L `l_sgp_0327` | 0.622 | 0.376 | 0.835 |
| XL `l_sgp_0367` | 0.394 | 0.293 | 0.534 |
| XL `l_sgp_0379` | 0.513 | 0.383 | 0.663 |

Scores use the common reference objective. Failed FAR rollouts stay in the means.

| Method | Mean reference quality | FAR success |
|---|---:|---:|
| Fixed guidance | 0.501 ± 0.075 | 97% |
| Imitation | 0.351 ± 0.009 | 100% |
| Greedy | 0.681 | 100% |

Model seed IDs were `4211, 4212, 4213, 4214, 4215`. Uncertainty in the table is the standard deviation across model seeds. Mean training-loop time per seed was 67 s for fixed guidance and 41 s for imitation. These times exclude scene setup and final evaluation.

## Validation screen

| Method | Common reference quality | FAR success |
|---|---:|---:|
| Fixed guidance | 0.215 | 100% |
| Annealed guidance | 0.211 | 100% |
| Scalar RL | 0.160 | 100% |
| Imitation | 0.180 | 100% |
| Coverage-only reward | 0.186 | 100% |
| Lower daylight reward | 0.185 | 100% |

The adaptive schedule reduced rejections per committed edit by 24% on the three New York validation plots with the archived calibration checkpoint. The mean reference score moved from 0.598 to 0.604, and mean episode time went from 0.757 to 0.616 s. All nine rollouts per schedule hit FAR. The selected schedule reduced the budget immediately after the first rejection.

## Equal-FAR planning probe

The synthetic oracle transfers two roof patches between stacks. FAR stays at 3.0 and quality moves from 1.1363 to 1.1304 to 1.1400. A one-step choice with its best continuation or early stop reaches 1.1372. This is an exploratory geometric witness.

| Free-mask policy | Episodes | Two edits committed | Dip followed by improvement over start at feasible FAR |
|---|---:|---:|---:|
| Fixed guidance | 25 | 11 | 3 |
| Imitation | 25 | 15 | 0 |
| Greedy | 5 | 0 | 0 |

The free-mask comparison gives each policy the same initial state and two decision attempts. It allows early finish. The scripted witness and the policy probe have different action spaces.

All five model seeds, the frozen selection rule, scene hashes, FAR statistics, and paired comparisons are in the evidence file.
[Machine-readable evidence](confirmatory-l-xl-mps-2026-11-22.json)
