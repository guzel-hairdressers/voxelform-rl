| Device | Grid | Decoder | Warm mask ms | Cold mask ms | Apply + depth ms | Failed draws /20 | Depth error m |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 4×48×64 | sequential | 3.95 | 6.30 | 2.19 | 0 | 0.000000 |
| cpu | 4×48×64 | adaptive 1+7 | 3.50 | 3.79 | 2.08 | 0 | 0.000000 |
| cpu | 8×64×96 | sequential | 14.17 | 25.74 | 5.84 | 0 | 0.000000 |
| cpu | 8×64×96 | adaptive 1+7 | 14.15 | 14.04 | 5.90 | 0 | 0.000000 |
| mps | 4×48×64 | sequential | 7.95 | 64.83 | 10.20 | 0 | 0.000000 |
| mps | 4×48×64 | adaptive 1+7 | 15.32 | 180.24 | 12.84 | 0 | 0.000000 |
| mps | 8×64×96 | sequential | 13.41 | 41.61 | 10.09 | 0 | 0.000000 |
| mps | 8×64×96 | adaptive 1+7 | 10.80 | 34.67 | 9.68 | 0 | 0.000000 |

Fixed full-envelope snapshots; 20 warmed draws. MPS uses native tensor depth, CPU uses SciPy. Geometry safety remains on CPU. This is not browser-render timing; cold setup and backward/update times, memory and raw samples remain in JSON.
