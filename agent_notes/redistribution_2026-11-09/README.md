# Facade bands and paired redistribution — 2026-11-09

Implemented as explicit experimental configuration/API. The live preview and original checkpoints remain unchanged.

## Frozen 1/2/3-layer facade eligibility

`PatchConfig.facade_layers` accepts 1, 2 or 3; default remains 1 for compatibility. The mask starts at the current lateral exterior facade and expands inward using horizontal occupied-cell adjacency for zero/one/two additional rounds. Floors are independent; roofs do not seed the band. Compute once before drawing and commit the generated edit in bulk. It does not repeatedly expose new layers inside an action. Optional deep eligibility remains max(band, planar depth badness). Geometry checks prune/reject enclosed cuts and unsafe solids. Mask generation/ranking remain native Torch; band construction reuses compiled CPU frontier operations.

The same smooth Gaussian latent decoder is retained. Broader permission alone is not an incentive or guarantee to remove whole layers.

## Layer ablation

L=4×48×64, 8 solar samples; XL=8×64×96, 12 samples; 16 synthetic contextual buildings. Three seeds per layer/device, untrained matched spatial models. Same workload controller (.5 initial outermost-skin capacity), exact FAR 3 and audit cadence.

| Device | Size | Layers | Accepted actions | FAR successes |
|---|---|---:|---|---:|
| cpu | L | 1 | 9, 9, 10 | 3/3 |
| cpu | L | 2 | 9, 10, 9 | 3/3 |
| cpu | L | 3 | 9, 9, 9 | 3/3 |
| cpu | XL | 1 | 31, 32, 32 | 1/3 |
| cpu | XL | 2 | 30, 30, 32 | 3/3 |
| cpu | XL | 3 | 32, 32, 32 | 1/3 |
| mps | L | 1 | 10, 10, 9 | 3/3 |
| mps | L | 2 | 9, 9, 9 | 3/3 |
| mps | L | 3 | 9, 10, 10 | 3/3 |
| mps | XL | 1 | 27, 30, 28 | 3/3 |
| mps | XL | 2 | 31, 31, 29 | 3/3 |
| mps | XL | 3 | 32, 32, 32 | 2/3 |

Two layers completed 12/12 cases; one layer 10/12; three layers 9/12. Compare success first: a short failed run is not a speedup. Three layers did not consistently reduce edits. This small synthetic sample does not establish learned architectural quality or real-site generalization.

An aggressive controller follow-up used utilization 1.0 instead of .5, one seed/device/size. Layers 1 and 2 each completed 4/4, layers 3 completed 2/4. Exact action counts and failures are in `results/redistribution-fast-2026-11-09.json`. The 2-layer variant is the more promising option; keep the 3-layer variant experimental. Runs overlapped some checks, so timing rows are not isolated hardware speed comparisons.

## Constant-volume exchange prototype

`voxelform.redistribution.choose` gets both spatial fields from one actor pass, samples a joint Gaussian latent and generates two masks. The masks change the same number of cells. Projection may only discard model-selected bits, then rebalance counts; it cannot create paths or synthesize cells. Invalid paired draws retry at smaller counts. Consumed retry scores include both neural fields. No candidate-mask library or geometric ranker is used.

One combined update permits at most two grounded connected masses by default (configurable), and checks frozen removal eligibility, roof/facade addition eligibility, exterior-air connectivity and selected courtyard/patch limits. Current roofs allow a single new layer within the explicit allowed height envelope. Existing production FAR termination is bypassed only through this isolated exchange API; no automatic refinement phase was added to the UI. The caller controls refinement horizon and bootstrap.

`duplicate_top` supplies an optional addition primitive copying the chosen component’s highest occupied storey into the next permitted layer. In whole-storey mode, insufficient source edits reject the action rather than silently copying a partial floor. Component/macro-action choice is currently supplied by the caller, not yet a learned categorical action head.

## What was actually verified

- A deterministic geometry fixture splits a connected bridge into two grounded masses while adding roof volume and preserving cell count/FAR. One-building validation rejects the same move.
- Top-storey copying moves only one layer and respects the height ceiling.
- Both neural spatial heads get nonzero gradients; actual MPS backward execution passes.
- Eight successive model-generated paired moves ran on CPU and MPS at **exact FAR 3.0**, starting with three occupied floors inside an eight-floor envelope. The final projected-pair experiment uses minimum four cells per patch and maximum four regions, with up to two grounded buildings. Its courtyard-count cap is unlimited; the validator supports stricter opening limits but these must be tested before live adoption.
- Early fixed-count prototypes stalled after only 1–3 exchanges. Adaptive retry counts alone were insufficient. Selected-bit projection plus count rebalancing enabled all eight moves. These failed variants remain archived.

These checks establish representability, balanced edits and trainability, not that an RL policy has learned the user’s two-tower layout or improved design quality. The production Gaussian checkpoint/optimizer semantics describe single-operation FAR edits; do not export paired-trained optimizer state as a production checkpoint. A dedicated paired-policy checkpoint/protocol and an environment with explicit coarse/refinement phases are still required before live integration.

## Raw evidence

`results/redistribution-2026-11-09.json`, `redistribution-fast-2026-11-09.json`, `redistribution-adaptive-pairs-2026-11-09.json` and `redistribution-projected-pairs-2026-11-09.json` retain settings/hashes, failure outcomes, per-action counts and exact FAR. Current code is snapshotted alongside this report. No long convergence study, commit or push was performed.
