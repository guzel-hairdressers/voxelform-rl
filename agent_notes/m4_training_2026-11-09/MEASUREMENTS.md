# M4 training measurements

Apple M4, 16 GB unified memory; Torch MPS with CPU fallback disabled. Warm medians include real actor-critic backward/optimizer updates and episode reset. Five measured episodes per row.

| Device | Case | Original seconds/episode | Prepared seconds/episode | Extra preparation seconds | Quality replay error |
|---|---|---:|---:|---:|---:|
| cpu | L | 0.255 | 0.239 | 0.04 | 0 |
| cpu | XL | 3.553 | 3.945 | 0.26 | 0 |
| cpu | Barcelona | 1.750 | 0.501 | 33.56 | 0 |
| mps | L | 0.680 | 0.519 | 0.44 | 0 |
| mps | XL | 4.132 | 3.029 | 0.54 | 0 |
| mps | Barcelona | 1.417 | 0.757 | 18.20 | 0 |

Preparation caches only immutable external receiver visibility; self-shadow maps, sky diffusion, planar depth and current exposed receivers remain dynamic. Original lazy caching also warms over time, so the short-run speed ratio is not a guaranteed sustained speedup. Barcelona preparation costs ~18 s on MPS and ~34 s on CPU: amortize over repeated episodes, not one-off previews. All 30 before/after training-trial pairs have identical quality and action counts. Profile JSON retains cold/setup costs, memory, checkpoint serialization and timing breakdowns.

## Equal training-time pilot

60 seconds per arm and seed; two seeds; train on a wide plate, irregular site and cached Barcelona. Three held-out scenes per seed: irregular, XL, another Barcelona parcel. Setup and checkpoint evaluation are additional wall time. These are short exploratory runs, not convergence or generalization evidence.

| Arm | Accepted actions/minute | Held-out FAR successes | Matched-success mean quality change |
|---|---:|---:|---:|
| current | 1320 | 6/6 | +0.000990 (5 matched pairs) |
| quality-mixed | 1186 | 6/6 | +0.000238 (5 matched pairs) |
| quality-curriculum | 1112 | 5/6 | +0.000249 (5 matched pairs) |

The differences are too small/inconsistent to choose a quality winner. Curriculum failed one XL test with five voxels still needed. Do not compare quality on incomplete shapes. Independent checkpoint evaluation reproduced all 18 final pilot quality values exactly and separately records usable fraction, direct sun, roof opportunity, noise and diffuse facade access.

The optional eight-channel `quality-informed` follow-up used depth, eligibility and **direct-sun** exposure, not diffuse facade light. It completed 6/6 held-out FAR cases but ran at only about 311 actions/minute. Its preparation was disrupted by concurrent GPU correctness checks, and its timed throughput also needs an isolated attribution study before adoption. It did not establish a quality improvement and is not the selected default. The corrected nine-channel version below has regression/gradient tests but has not undergone a learning comparison.

## Reward experiment

| Component | Current | Quality-led experiment |
|---|---:|---:|
| Normalized FAR progress | 1.0 | 0.1 |
| FAR success bonus | 1.0 | 0.1 |
| Failure | -1 minus remaining fraction | -2 × remaining fraction |
| Step cost / budget | 0.05 | 0.01 |
| Environmental quality delta | 0.1 | 5.0 |
| Same-floor depth-badness improvement | 0 | 1.0 |

This is a coupled reward hypothesis. The workload deadline/controller and geometric rules stay fixed, isolating reward/schedule experiments from a new action space. Mixed difficulty uses 50% easy targets; curriculum changes easy probability from 80% to 20%, retaining all objectives throughout. The final time-truncated rollout bootstraps and is not labelled a terminal failure.

## Known limitations

- All training episodes in this pilot subtract from a full initial mass; it does not establish addition or redistribution performance.
- Current policy still has a small within-floor spatial receptive field; no model-wide quality reasoning was proved.
- Final FAR still ends these episodes, and the production geometry rules still prevent a single mass splitting into multiple buildings. See REDISTRIBUTION.md.
- The viewer’s single display-light shadow is not a map of annual sunlight or diffuse sky access.
- Experimental rewards and extra inputs have not replaced the live preview policy or its weights.
