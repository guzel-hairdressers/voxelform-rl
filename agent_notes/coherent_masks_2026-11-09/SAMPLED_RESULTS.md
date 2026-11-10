| Eligibility | Sampler | FAR completions | Regions/mask | Cells/region | Boundary faces/cell | Synthetic quality | Barcelona quality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| facade | independent | 30/30 | 19.35 | 1.15 | 5.93 | 0.5842 | 0.0958 |
| facade | smooth | 30/30 | 19.31 | 1.15 | 5.93 | 0.5846 | 0.0906 |
| facade | correlated | 30/30 | 6.17 | 4.18 | 5.03 | 0.6069 | 0.1063 |
| facade | combined | 30/30 | 3.26 | 5.53 | 4.94 | 0.6184 | 0.1178 |
| depth | independent | 30/30 | 19.40 | 1.15 | 5.93 | 0.5842 | 0.0923 |
| depth | smooth | 30/30 | 19.19 | 1.15 | 5.93 | 0.5846 | 0.0928 |
| depth | correlated | 30/30 | 6.17 | 4.18 | 5.03 | 0.6069 | 0.1063 |
| depth | combined | 30/30 | 3.26 | 5.53 | 4.94 | 0.6184 | 0.1176 |

These trajectories use min1/unlimited component guards to isolate sampling. Production additionally prunes patches below four cells and limits disconnected patches. Edit sizes vary during episodes; fixed-count probes are a separate comparison. Quality is an uncalibrated proxy, not architectural validation.
