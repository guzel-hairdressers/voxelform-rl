# Direct neural facade masks — 2026-11-07

Current implementation on `codex/voxel-prototype`. Supersedes the sampled mask
ranker and single-voxel action prototypes. Preview: http://127.0.0.1:8011.

## Action and learning

The network emits two dense `(z,y,x)` logit fields from its convolutional mask
head, one for subtraction and one for addition. The permitted operation follows
the FAR residual. Training draws independent Bernoulli bits from the corresponding
neural field; deterministic inference uses its expected active area to set the
number of pixels, then selects neural confidence maxima. There is no geometric
patch template, candidate list, candidate scoring or search over candidate masks.
A rejected mask can be resampled from the same neural distribution (eight attempts
maximum); failure is reported honestly rather than replaced with a fallback patch.

The facade is frozen before each action. Eligibility removes interior/sealed-air
bits; the FAR budget and native topology projection may only prune neural-selected
bits. Whole-mask connectivity, ground anchoring and air-cavity checks run through
compiled NumPy/SciPy CPU kernels. Bulk application changes every retained voxel
together, followed by the next facade refresh. Multiple separated mask regions
are permitted. Ordinary actions change at least two cells; the final FAR residual
may be one. Projection is conservative and may reject an output; it is not a
structural engineering certificate.

A2C trains the actual dense head with policy log probabilities, critic loss and
entropy. Nonterminal 32-mask rollouts bootstrap without ending/resetting the
building episode. Rewards retain the existing FAR and environmental audit logic.
Gradient/parameter checks establish trainability, not useful learned shape quality.

V5 checkpoints store this architecture. Earlier checkpoint imports retain the
encoder/critic but initialize a new mask head, goal conditioning and optimizer.
The preview retains 176 historical updates and starts with zero direct-head
updates, explicitly separated in the UI. Original state/weights were saved under
`results/checkpoints/pre-direct-neural-2026-11-07-*`; saved occupancy was compared
exactly after restart. Module Lab and the older port-8010 process were untouched.

## Verification

- 42 Python tests pass, including exact irregular/disjoint masks supplied by fixed
  model fields, separate addition output, interior exclusion, next-layer deferral,
  stale-mask rejection, topology pruning, FAR clipping, real parcel area,
  multi-action episodes, actor gradients/weight changes and transactional migration.
- Four browser geometry-adapter tests pass. JavaScript syntax check passes.
- Live cached Barcelona site `m_bcn_0093`, six floors, grid `(6,79,51)`, source
  parcel 1,450.5 m², 29 fetched neighbouring buildings. One neural mask generated
  141 eligible bits and applied 140 after topology pruning. Independent occupancy
  XOR confirmed every changed cell belonged to the pre-action facade.
- Live inference reached medium FAR with 30 subtraction masks / 2,175 voxels, then
  high FAR with 47 addition masks / 2,175 voxels. Model parameters were unchanged
  across these inference sequences. Full states/policy backups are ignored under
  `results/checkpoints/`; compact evidence is beside this report.
- The original shared Module Lab renderer, opaque surroundings option and camera
  controls remain the graphic engine used in the browser.

## Measured method latency

Median milliseconds, 15 mask actions per case, one CPU thread, seed 123. L is
`(4,48,64)` with eight sun samples; XL is `(8,64,96)` with twelve. Each synthetic
fixture has 16 contextual buildings. These are method timings, not generalization
or quality evidence. Actual CPU and MPS forward/backward ran; CUDA was unavailable.
MPS fallback was disabled. Geometry remains on CPU, including for MPS cases.

| Device | Scale | Facade | Neural mask incl. projection | Projection subset | Apply/refresh | Environmental audit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu | L | 0.206 | 2.794 | 0.152 | 0.372 | 9.352 |
| cpu | XL | 0.639 | 11.154 | 0.312 | 0.961 | 31.437 |
| mps | L | 0.237 | 4.452 | 0.114 | 0.780 | 49.377 |
| mps | XL | 0.679 | 3.874 | 0.310 | 1.391 | 36.109 |

Environmental audits normally run every eight masks and on completion; the
benchmark samples every five. Model forward, setup, p95/raw samples, memory,
source hashes and four-mask optimizer-update timings are retained in
[`results/direct-neural-masks-2026-11-07.json`](../../results/direct-neural-masks-2026-11-07.json).
No rendering is included in timed calls. CPU remains the preview default because
these timings do not establish an end-to-end MPS advantage.

## Resume

Current server uses the sibling environment:

```sh
VOXELFORM_RESUME_STATE=results/checkpoints/direct-current-state.json \
VOXELFORM_POLICY_PATH=results/checkpoints/direct-current-policy.pt \
../Module-Lab/.venv/bin/python -m uvicorn voxelform.server:app \
  --host 127.0.0.1 --port 8011 --no-access-log
```

Do not restart a running session without pausing and saving its current state and
policy first. The direct head has not undergone a quality/convergence study.
