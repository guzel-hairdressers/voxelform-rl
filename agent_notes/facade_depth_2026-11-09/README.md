> This initial CPU-reference implementation is superseded by [coherent masks and native GPU planar depth](../coherent_masks_2026-11-09/README.md). Production now uses depth-weighted eligibility when enabled. Earlier MPS timings in the initial field report measured CPU geometry plus transfer, not native GPU distance computation.

# Explicit facade depth — 2026-11-09

A geometric diagnostic field is now recomputed after every bulk action, before the
next policy decision, independently of the eight-action environmental audit.
Distance is horizontal Euclidean distance in metres from voxel centre to the
nearest actual lateral exterior-wall segment on the same floor. Open courtyards
count; sealed voids and roof/underside exposure do not. CPU SciPy EDT on a half-cell
lattice is exact for the axis-aligned voxel faces and centre queries. Empty cells
return zero. This is a geometric depth proxy, not measured daylight penetration.

`depthBadness = clip((depthMetres - 8) / 8, 0, 1)`. The adjacent facade cell has
centre depth 0.5 m. Zero through 8 m, 0.5 at 12 m, one from 16 m onward. An open
courtyard fixture reduces an unchanged cell's depth from 11.5 m to 3.5 m; the same
sealed void does not reduce its depth.

The API exposes `facadeDepth`, `depthBadness`, deep floor area and update timing.
`Environment.observation(include_depth=True)` optionally appends badness, and
`Network(input_channels=6)` supports research use. Production remains five-channel;
no policy/checkpoint or default reward was silently changed. An optional nonnegative
`depth_weight` subtracts original-capacity-normalized badness from quality; default
zero preserves the previous objective. No learned benefit is claimed.

Multiplication of facade eligibility by badness gives zero on every facade;
multiplication by inverted badness gives the original facade mask. The user's
later proposal is instead an additive/max union: facade=1 plus the deep-interior
ramp. That deliberately enables interior subtraction, so it is a distinct action
hypothesis, not yet installed in the production facade-only guard. Interior-void
connectivity and access from exterior/roof need to be tested before promotion.

The current local actor has a 7×7 receptive field. Appending only >8m badness cannot
affect its facade outputs directly: that nonzero field lies beyond its receptive
field. A regression test establishes this exactly. Wider context or an explicit
projection of interior pressure to the facade is needed; merely adding a channel
is insufficient. Global critic pooling can still observe the field.

Tests cover independent continuous wall-segment distance references, metric spacing,
roof exclusion, sealed/open voids, dynamic updates, exact remapping, optional reward,
naive gates, and local actor visibility. Prior architecture benchmarks predate
this new per-action geometric diagnostic; their timing/source hashes remain
historical evidence rather than current end-to-end timings.

All 56 Python tests pass. Isolated L/XL field timings (including transfer/remapping) and zero-error analytical rectangle checks are in [raw measurements](../../results/facade-depth-2026-11-09.json).
