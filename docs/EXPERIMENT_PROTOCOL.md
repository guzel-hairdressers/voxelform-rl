# ML experiment protocol

## Research scope

VoxelForm-RL studies **structured reinforcement learning in a constrained 3D
binary editing environment**. Domain geometry and exposure proxies provide a
controlled test bed. The contributions to test are representation, correlated
latent actions, constraint decoding, credit assignment and compute efficiency.
The 3D UI is an inspection/debugging tool. Architectural presentation or visual
complexity is not the research objective.

This infrastructure makes experiments inspectable and repeatable. It does not
itself establish a thesis contribution, convergence, physical validity or a
state-of-the-art result.

## Task and algorithm

State includes occupied/allowed grids, geometric/physical proxy fields and FAR/
budget context. The active factored 2D+1D encoder consumes 14 channels: occupancy,
air diffuse field, noise, allowed space, elevation, planar depth badness, removal
eligibility, addition eligibility, immutable external direct-sun potential,
diffuse facade access, narrow-strip map, usable proxy, core mask and core-distance
badness. Three-dimensional context is mixed through factored layers and global
pooling; this is different from both the earlier floor-independent CNN and a
full-resolution Conv3d actor.

An action is an operation (remove/grow/exchange/finish) plus a coarse Gaussian
latent field for each consumed spatial head. Interpolation, smoothing, ranking,
area allocation, geometry guards and greedy core placement are explicit parts
of the environment/decoder. They include hand-designed inductive biases; the
method should not be called unrestricted or “pure” learned design. The active
organic decoder permits genuine 3D additions and cantilevers. Legacy whole-floor
copying remains only as a compatibility/control path.

Training is synchronous actor-critic with GAE (`gamma=.99`, `lambda=.95`), Huber
value loss (coefficient .5), operation entropy regularization (.005), Adam
(`lr=.0012`, `weight_decay=.0001`) and gradient clipping (2). This is not PPO.
Terminal masks prevent return propagation across episode boundaries; checkpoint
flushes happen only at completed episodes. The terminal/environment reward is
shared with the live app through `voxelform/objectives.py`.

The augmented action includes **all Gaussian retries consumed by decoding**,
including rejected draws. The corrected actor scores their full joint log
probability. Eligibility masking of spatial log probabilities was removed:
smoothing may transmit information across eligibility boundaries. Only the
operation/heads actually drawn are scored. The historical implementation saved
only the successful spatial draw and lost rejected attempts, so those old runs
are not an unbiased validation of this corrected estimator.

An optional physics-shaped spatial auxiliary objective is weighted by
`spatial_aux_weight`. The research reference config uses **0**. The separate
`research-real-aux.json` uses **1.2** for an ablation. This term is a heuristic
auxiliary learning signal, **not an exact control variate for the original return**.
The live UI retains 1.2 by default so its hybrid behaviour is explicit. Another
corrected bug wrote step reward into dictionary key `3` while the trainer read
`reward`; the new implementation stores and tests the actual step reward.

GAE follows [Schulman et al.](https://arxiv.org/abs/1506.02438). Evaluation practices
are motivated by [Henderson et al.](https://arxiv.org/abs/1709.06560) and
[Agarwal et al.](https://arxiv.org/abs/2108.13264); using a few seeds still leaves
substantial uncertainty.

## Configuration and separation

- JSON configurations are validated, reject unknown fields, invalid ranges,
  duplicate/overlapping seeds and overlapping train/validation/test site IDs.
- `research-smoke.json` is a self-contained CPU pipeline check with two model
  seeds and synthetic held-out shapes. It is not a quality benchmark.
- `research-real.json` trains on ten Barcelona, Tokyo and Singapore plots (two
  each for XS, S, M, L and XL), validates on Hong Kong, and tests on New York
  and Paris. The runner rejects overlapping context circles across splits
  when coordinates and radii are available. The headless runner visits each
  training site once per ten episodes. The interactive rotation
  spends three consecutive episodes on each site with the same policy and
  optimizer. Historical results may use the original two-site pilot pool.
- Test scenes are not used for training or checkpoint selection. Final fixed-
  budget checkpoints are evaluated; no best-test checkpoint is selected.
- Untrained policies and trained policies use common evaluation seeds. The
  untrained test baseline is evaluated only after training has finished.
- Evaluation stores/restores Python/NumPy/Torch/device RNG state, so evaluation
  does not perturb the training RNG stream. Environment scene seeds are distinct
  stable hashes of the scene IDs; model seeds are not reused as scene seeds.
- Research evaluation defaults to the training distribution: operation multiplier
  1 and latent noise scale 1. The UI's default multiplier 2.5 / first-draw noise
  .08 is a different deployment decoding policy. Both values are recorded and
  saved; comparisons must specify which is used.

FAR success and quality are separate. All episodes, including failed FAR, are
retained in raw logs and summary denominators. Report quality alongside FAR
success; an unacceptable FAR result is not an improvement solely because it
removed more volume. Raw reward scores from different core distances/objective
weights are not directly comparable. The fixed post-episode referee is reported
as `meanReferenceQuality` to support such ablations; it is still an uncalibrated
research proxy. The referee uses its own reserved core mask to exclude core
faces from the direct sun and diffuse sky averages. See the
[core access and daylight objective](CORE_LAYOUTS.md#core-access-and-daylight-in-the-objective).

Uncertainty is aggregated over independent **model seeds**, not by pretending
many stochastic episodes of one trained model are independent training runs.
Summaries include standard deviation across model seeds; a fixed-seed bootstrap
interval is supplied only with at least three model seeds. Such small-sample
intervals are descriptive and do not establish significance/generalization.

## Artifacts and resume contract

Every new output directory contains:

- `config.json`: resolved configuration and its fingerprint in the manifest.
- `manifest.json`: UTC time, Python/dependency versions, hardware/backend flags,
  Git revision/dirty status, source hashes and core-layout version.
- `source.tar.gz`: the actual dirty or clean Python/config source used, not just
  a Git SHA that could omit uncommitted changes. No credentials, external scene
  cache, browser history or saved user policies are included.
- `scenes.json` / `setup.json`: source/allowed-mask hashes, scene seed, shape,
  parcel area, context count, alignment, coordinates/radius and preparation time.
- Per-model-seed `train.jsonl`, raw validation/test episodes, untrained controls,
  checkpoint and summary; top-level cross-model-seed aggregation.

Checkpoints include model, optimizer, counters, next episode index, complete RNG
states, config hash, source hashes, scene fingerprints and library version.
Pending rollouts must be flushed before saving. Changing checkpoint interval
therefore changes the update schedule and is part of the configuration.

Resume writes into a **new directory**, retains parent-checkpoint provenance and
continues one model seed. Exact resume rejects changed source/config/scene/core
semantics. Interrupted partial episodes are not silently relabeled successes;
resume uses the last complete atomic checkpoint. Historical checkpoints without
this envelope are transfer/inference artifacts, not exact-resume artifacts.

CPU strict determinism is tested. Seeding MPS/CUDA does not guarantee bitwise
identity across devices, library versions or hardware. `strict_determinism=true`
requests deterministic operations and raises if unsupported; MPS real runs default
false and record that fact. Wall-clock timing itself is not deterministic.

The editable checkout supports the UI. Installed ML packages/entry points support
headless experiments; the renderer and external urban cache are optional data/UI
dependencies, not prerequisites for the synthetic smoke run.

## Research that remains to be done

No full new multi-seed real-site study has been claimed from the smoke check.
Longer budgets, training curves, sensitivity to reward weights/area budgets,
independent exposure calibration, cross-city confidence, and paired comparisons
against reproduced geometric/random and architecture baselines remain research
work. Existing older ablation JSON is historical evidence, not automatically a
reproduction under the corrected loss/core library.

Preference conditioning, vector critics, PCGrad, Lagrange multipliers and parallel
environment throughput were proposals in the other agent's response. They were
not implemented or validated by that response; “<80 lines”, universal Pareto
learning and 3–5× throughput are not established results. Add them only through
separate matched-budget, versioned experiments.
