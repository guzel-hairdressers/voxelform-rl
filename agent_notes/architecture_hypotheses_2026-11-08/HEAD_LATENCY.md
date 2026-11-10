| Device | Scale | Encoder | Dual forward ms | Packed single forward ms | Packed single speedup |
| --- | --- | --- | ---: | ---: | ---: |
| cpu | L | 2d | 3.788 | 3.738 | +1.34% |
| cpu | L | 3d | 15.374 | 15.394 | -0.13% |
| cpu | L | multiscale | 20.970 | 20.901 | +0.33% |
| cpu | XL | 2d | 16.931 | 16.630 | +1.78% |
| cpu | XL | 3d | 59.346 | 59.155 | +0.32% |
| cpu | XL | multiscale | 82.356 | 82.343 | +0.02% |
| mps | L | 2d | 2.068 | 2.086 | -0.88% |
| mps | L | 3d | 2.982 | 2.791 | +6.40% |
| mps | L | multiscale | 4.531 | 4.472 | +1.30% |
| mps | XL | 2d | 2.829 | 2.732 | +3.43% |
| mps | XL | 3d | 12.332 | 11.954 | +3.06% |
| mps | XL | multiscale | 14.510 | 14.645 | -0.93% |

30 randomized interleaved forward triples per case; facade inputs enabled. Packing cost is excluded. This is not an end-to-end speedup claim.
