# VoxelForm RL project notes

- [ML/core review — 2026-11-20](../docs/REVIEW_2026-11-20.md): compact component plans, corrected reward/retry likelihood, independent core-distance tests, reproducible headless experiments and limitations.

- [Height/growth handover — 2026-11-14](../HEIGHT_GROWTH_HANDOVER.md): user intent, approach history, active floor copying, height-decrease/reversibility limits, evidence and continuation map.

- [Unified removal and roof growth — 2026-11-10](unified_actions_2026-11-10/README.md): failed free-operation pilot, separate feedback/pruning tests, then net-area guidance with growth from step one; 10/10 FAR controls on two sites, no learned-quality claim.
- [3 px outlines and HSL blue — current, 2026-11-10](outlines_hsl_2026-11-10/README.md): surface outlines restored, lower-edge display offset, 50% outline / 70–95% fill lightness at fixed hue/saturation.
- [Thick blue silhouettes — 2026-11-10](silhouette_2026-11-10/README.md): GPU camera-facing edge filtering, 8-pixel saturated blue strokes, orthographic/perspective support and verification.
- [Lighter colours and merged outlines — 2026-11-09](light_contours_2026-11-09/README.md): linear occupancy-based contours, Node timings, four alignment options with Auto default, and explicit training-order limitations.
- [Grid fit and matched quality tests — 2026-11-09](grid_quality_2026-11-09/README.md): full-square 2D fitting, fixed world/sun coordinates, real-site alignment comparisons and bounded training with post-FAR redistribution.
- [Facade bands and paired redistribution — 2026-11-09](redistribution_2026-11-09/README.md): 1/2/3-layer tests, model-generated balanced roof growth/removal, storey copying and grounded splitting; isolated prototype.
- [M4 training speed and facade light — 2026-11-09](m4_training_2026-11-09/README.md): timed actor–critic training, exact external cache preparation, reward/curriculum pilots, separate diffuse facade input, and the proposed redistribution action.
- [Massing shading — 2026-11-09](shading_2026-11-09/README.md): flat lit solid view, self-shadow reception and before/after browser verification.
- [Workload-based episode budgets — current, 2026-11-09](workload_budget_2026-11-09/README.md): capacity-based deadlines, adaptive edit amounts, learned-amount ablations and exact-FAR checks.
- [Coherent masks and GPU planar depth — 2026-11-09](coherent_masks_2026-11-09/README.md): correlated direct masks, per-floor GPU depth, courtyard merging, safety limits, autosave and measured CPU/MPS execution.
- [Explicit facade-depth field — 2026-11-09](facade_depth_2026-11-09/README.md): exact geometric field, optional input/reward, current-vs-proposed eligibility and visibility tests.
- [Architecture hypothesis tests — 2026-11-08](architecture_hypotheses_2026-11-08/README.md): twelve configurations, three seeds, matched training budgets, controls and CPU/MPS measurements; production unchanged.
- [Direct neural facade masks — prior baseline](direct_neural_masks_2026-11-07/README.md): dense model output, constraints, migration, live verification and CPU/MPS timings.
- [Sampled facade masks — superseded](facade_masks_2026-11-07/README.md): historical proposal-ranking stage.
- [Single-voxel FAR episodes — superseded](far_episodes_2026-11-07/README.md): original episode and density implementation.
- [Urban 3D workspace](urban_workspace_2026-11-07/README.md): fetched context and actual shared Module Lab graphic engine.
- [Raw measurements](../results/).

- 2026-11-20 scissor follow-up: active T4 now uses compact-core.v3, two
  interlocking routes with separate +1.5 m landings, three lifts; metric
  6.55×8.38 m / grid 7×9 m. See docs/CORE_LAYOUTS.md for the area tradeoff.
