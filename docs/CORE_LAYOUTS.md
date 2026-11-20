# Compact core geometry, version `compact-core.v2`

This is a parameterized service-space prior for an ML environment. It replaces
large opaque rectangles with inspectable metric component plans. The experiment
still reserves the conservative enclosing rectangle on its one-metre grid.

## Plans

| Height band | Stairs | Lifts | Metric enclosure | Metric area | Grid reservation | Previous grid area |
|---|---:|---:|---|---:|---|---:|
| 1–5 | 1 | 0 | 2.80 × 4.94 m | 13.83 m² | 3 × 5 = 15 m² | 30 m² |
| 6–10 | 1 | 1 | 4.90 × 4.94 m | 24.21 m² | 5 × 5 = 25 m² | 48 m² |
| 11–18 | 1 | 2 | 7.00 × 4.94 m | 34.58 m² | 7 × 5 = 35 m² | 72 m² |
| 19–30 | 2 | 3 | 11.90 × 4.94 m | 58.79 m² | 12 × 5 = 60 m² | 108 m² |
| 31–60 | 2 | 4 | 14.00 × 4.94 m | 69.16 m² | 14 × 5 = 70 m² | 160 m² |

[1–5 floor plan](core-plans/T1.svg) · [6–10](core-plans/T2.svg) ·
[11–18](core-plans/T3.svg) · [19–30](core-plans/T4.svg) · [31–60](core-plans/T5.svg).
The same drawings are available in the live app at `/core-plans`.

Each stair shows **two separate flights and two landings**. At 3 m storey height,
each flight has nine risers and eight 0.28 m treads: 18 × 166.67 mm total rise,
2.24 m run per flight, 1.20 m clear width and 1.20 m landing depth. Two 0.15 m
walls plus two flights and a 0.10 m gap give 2.80 m enclosure width. The two
landings, run and end walls give 4.94 m length. Thus a six-metre stair length
was not an unavoidable consequence of the selected geometry.

Each lift reservation is 2.10 × 2.20 m outside the walls; clear shaft 1.80 ×
1.90 m, drawn car 1.10 × 1.40 m and 0.80 m door. A shared lobby connects the
stair floor landing to the lift doors. Green marks show openings; the pale outer
band shows outward grid rounding. Plans distinguish shaft from car, flight from
landing, and metric enclosure from voxel reservation.

These dimensions are selected **research assumptions**. For scale comparison,
[KONE's published planning table](https://www.kone.se/Images/planning-guide-kone-elevator-monospace-700_tcm27-17969.pdf)
includes a 630 kg configuration with a 1.10 × 1.40 m car and 1.75 × 1.81 m shaft.
That manufacturer example does not establish universal shaft dimensions or a
height-to-lift-count rule. Stair dimensions can be compared with the parameter
ranges in [Approved Document K](https://www.gov.uk/government/publications/protection-from-falling-collision-and-impact-approved-document-k);
this is a dimensional reference, not a regulatory assessment of these plans.

The 1–5-storey stairs-only rule follows the user's requested experimental band.
Higher lift counts are explicit configurable priors (`HeightBand`), not a lift
traffic model. Number of stairs, fire separation, accessibility, evacuation,
occupancy and jurisdiction-specific requirements are not inferred from height.
Heights beyond 60 storeys are outside this library; no last-tier lift count is
silently extrapolated. No compliance, minimum architectural footprint or actual passenger capacity is
claimed. Within a band, reserved volume grows linearly with floors; area jumps
at the configured band boundaries. The earlier “strictly super-linear” claim was
not mathematically justified.

## Source of truth and limits

`voxelform/core_plans.py` defines `CorePlanParameters`, `HeightBand`, `make_plan`
and `render_plan_svg`. `voxelform/cores.py` derives its placement templates from
those plans. The API and exported SVGs use the same geometry; there is no
independent hardcoded fallback drawing library. Regenerate exports with:

```sh
uv run --frozen python -m tools.render_core_plans
```

Height schedules can be changed through `HeightBand` definitions and dimensions
through `CorePlanParameters`. A changed library must receive a new version and
new experiments; historical checkpoints/results retain their earlier semantics.

The core placer is a **deterministic greedy heuristic**, not an RL-learned core
layout planner. It reserves complete grid footprints through grounded columns.
Distances are per-floor eight-neighbour shortest paths through occupied cells,
with blocked-corner diagonals forbidden. They are measured to the reserved core
footprint, not to the drawn door or through a modeled corridor network. The
independent Dijkstra tests validate that grid graph, not actual circulation or
evacuation. Bounded screening passes can leave distant cells unresolved; the
final default pass runs to convergence. No universal optimal placement is claimed.

A core-on/off ablation changes the task reward. New research evaluation therefore
also reports a common **core-enabled, spacing 10 m, distance 12 m referee** after
the acting policy has finished. It does not expose additional core information
to a cores-disabled actor during the episode.
