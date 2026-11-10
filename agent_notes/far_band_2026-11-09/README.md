# FAR acceptance band — 2026-11-09

Live and M4 headless sessions now default to absolute FAR tolerance ±0.1:
target 3 accepts 2.9–3.1. Integer cell bounds are rounded inward, clipped to the
permitted capacity. Explicit zero-tolerance mode preserves historical exact-cell
experiments. Coarse policy counts, rewards and workload budgets use distance to
the nearest acceptable boundary, so there is no centre-chasing penalty inside
the band. Prototype redistribution accepts unequal additions/removals if their
result stays in the band; balanced exchanges remain available.

Tolerance persists in coupled session snapshots and is exposed in API settings
and the FAR display. FAR-only target changes preserve geometry. The saved live
mass and policy were preserved, and the preview restarted paused. The existing
mass at FAR 3.0003447087 is accepted immediately.

The original 118-test suite passed; six dedicated band tests then passed,
including boundaries, one-cell completion at the upper edge, unchanged centre
inside the band, unequal exchanges, invalid tolerances and policy changes while
already successful. JavaScript syntax and whitespace checks passed.
