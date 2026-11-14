# Active growing preview — 14 November 2026

User requested promotion of the other agent's growing preview, integration of
`origin/devox` (`ec0c38f`), and growth that avoids scattered roof cells. Preserved
that commit via fast-forward before local changes. Current policy starts fresh
as requested; no session-weight migration or automatic checkpoint resume.

## Hypotheses and changes

1. Copying complete top plates, rather than arbitrary roof bits, should prevent
   fragmentary growth. Copies preserve courtyards and share source cuts on the
   copy action. Persistent per-cell source/copy labels require later cuts to
   select the linked stack. Direct column support is an explicit restriction.
2. Coherent top-down ranking and selection-only safety pruning should let the
   neural field still carve a complete mask without manufacturing a repair cut.
3. These changes together should retain FAR feasibility and fast MPS training.
   They are tested as one coupled decoder; individual causal contributions have
   not been isolated. No perfect-floor heuristic or carve-before-growth gate.

`tools/check_floor_growth.py` compares roof-patch net-guided controls with the
linked-floor decoder at matched seeds (123/456) on m_nyc_0003/m_bcn_0094. Initial
six floors, ceiling ten, tolerance ±0.1, two facade layers and same-floor deep
eligibility; both untrained. Four successes out of four in each arm. New decoder
10–14 accepted edits / 11–16 attempts, old 14–21 / 15–21. Upper cells without
support: new zero, old 428–480. New cached-scene generation 0.50–0.68 s versus
0.80–1.04 s; setup 5.53/9.38 s, 19/20 context buildings. Five-second MPS training
probes yielded 8/6 updates, nonzero gradients. This small test establishes
feasibility, not convergence or residential/lighting quality.

Raw results: `results/floor-growth-preview-2026-11-14.json`, including generated
geometry summaries, full operation traces and source hashes at measurement time.
Subsequent fixes to attempt bookkeeping/UI/devoxelization do not invalidate these
geometry tests; hashes retain the exact measured state rather than being rewritten.

## Preview integration

`main.py` now runs unified_server on 8011, same graphic engine and requested blue
palette/outlines. Rejected draws consume the attempt clock rather than immediately
ending an episode; deadline in-band succeeds, outside-band fails with remaining
error penalty. Terminal A2C bootstrap is zero. FAR retarget resets the clock.
Manual operation probes bypass FAR guidance only for the One mask button; all
training/generation uses normal legal-operation guidance. Validate invalid probes
before dispatch so they cannot kill the worker. Unsupported old mask settings are
removed/rejected instead of silently ignored. New policy export/import works with
whole-floor/net-guided semantics. Normal startup is fresh and paused.

The imported devoxelization algorithms were extracted into voxelform/devoxelize.py,
with content-keyed cache, method validation, T-junction conformance and boundary/
nonmanifold diagnostics. The original static synthetic lab and its builder remain.
Live conversion runs only after finished inference; before/after revision checks
reject stale work. Viewer avoids repeat geometry rebuilds, reuses the actual ink
shader, and uses zoom-scaled LineMaterial for reconstruction creases. Evaluation
layers restore raw voxels with outlines hidden. No sibling renderer changes.

A complex live NYC sample originally had 622 boundary/nonmanifold edge incidences;
conformance reduced that sample to 8 boundary edges and 1 nonmanifold edge. This
is still an experimental surface, not a guaranteed watertight construction/export
mesh. All three algorithms can fall back to identical slabs on complex floors.
Reconstruction volume is displayed separately and never used for FAR/reward.

## Verification

- Complete/source-linked growth, courtyard preservation, first-action exchange,
  frozen eligibility, direct support, selection-only pruning, decoder persistence.
- Actual MPS forward/backward through copied-cell score reductions.
- Box/courtyard/empty mesh fixtures, valid indices, content cache isolation,
  finished-generation/revision gate and unknown-mode errors.
- Live server attempts, terminal bootstrap, FAR retarget, settings and probe safety.
- Full Python and existing JavaScript suites; browser generation, growth probe,
  training/pause, 0–3 display and static lab checked before delivery.

Important remaining limits: a copied poor footprint remains poor; no room widths,
cores or circulation exist. One component offers only one plate choice to its add
field. Column support and persistent links reduce architectural freedom. Learned
quality and longer-run stability still need user feedback and further tests.

## Source/copy area regression and final retest

An additional fixture exposed stalling below FAR 4.5: copying a plate and cutting
its source counts the source cut twice (existing cell and withheld new cell).
The initial decoder used 13 attempts and stopped at FAR 4.36979. The corrected
projection prunes the ONE neural cut ranking using exact area costs and linked
stack support. It never trims a copied plate into roof posts or invents cut bits.
The fixed seed now reaches FAR 4.40104 in 2 attempts. Targets 1.5 and 7.5 were also
checked. Full before/after traces, including failures, are preserved in
`results/floor-copy-far-gap-regression-2026-11-14.json`.

Final real-site retest: `results/floor-growth-preview-balanced-2026-11-14.json`.
Both arms still reached FAR in 4/4 runs, with the same supported-cell/edit-count
results as above. Timings varied with background UI/system activity: copy median
0.80 s (range 0.57–1.61), patches median 1.25 s (1.05–2.63), setup 7.33/11.35 s.
Five-second training probes made 2/4 updates, versus 8/6 before balancing. Do not
claim a universal speedup; exact balancing adds safety work and the machine was
not isolated. Median generation is well below the previous minutes-per-plot
concern. The first measurements remain intact rather than being overwritten.
