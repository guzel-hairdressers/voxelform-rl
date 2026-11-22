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
Terminal masks prevent return propagation across episodes. The runner updates
when a completed episode takes the rollout to `rollout_steps`, and uses any
remaining transitions at the end of the training budget. The terminal and
environment rewards are shared with the live app through `voxelform/objectives.py`.

The augmented action includes **all Gaussian retries consumed by decoding**,
including rejected draws. The flat learner (`gaussian-retry-path.v1`) scores
their joint log probability. The tier-adaptive learner
(`gaussian-retry-path-normalized.v1`) divides each spatial draw's log probability
by the square root of the number of latent elements. This scaled score is a
surrogate. Smoothing can transmit information across eligibility boundaries, so
each consumed head is scored over its full latent grid. Only the drawn operation
and spatial heads contribute to the score.

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
- The runner evaluates on validation at `checkpoint_interval` and at the end of
  training. `best.pt` maximizes validation FAR success first and mean quality
  second, with the earliest checkpoint winning ties. Test evaluation runs after
  training for both `best.pt` and the fixed-budget `latest.pt`.
- A development config can set `test_sites` to `[]` to skip final test evaluation.
  Alternative reward profiles select checkpoints using common reference quality
  after FAR success, so the selector uses the same objective across rewards.
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

Research v2 checkpoints store the model, optimizer, counters, next episode
index, pending rollout tensors, RNG states, config and source hashes, scene
fingerprints and core-layout version. Checkpointing and validation leave the
pending batch intact. A clean interruption saves at the next completed episode
and preserves the usual validation cadence. The historical v1 runner flushed
pending batches before saving, so its published update counts include that effect.

Resume writes into a **new directory**, retains parent-checkpoint provenance and
continues one model seed. Exact resume rejects changed source/config/scene/core
semantics. Interrupted partial episodes are not silently relabeled successes;
resume uses the last complete checkpoint. Use the recorded source revision for
exact resume of a v1 run. The first launch fixes the config and model-seed roster.
`--only-seed` accepts seeds on that roster with matching config and source hashes.
`--finalize-only` requires completed summaries for the entire roster.

CPU strict determinism is tested. Seeding MPS/CUDA does not guarantee bitwise
identity across devices, library versions or hardware. `strict_determinism=true`
requests deterministic operations and raises if unsupported; MPS real runs default
false and record that fact. Wall-clock timing itself is not deterministic.

The editable checkout supports the UI. Installed ML packages/entry points support
headless experiments; the renderer and external urban cache are optional data/UI
dependencies, not prerequisites for the synthetic smoke run.

## Next experiments

The published five-seed comparison tests M-tier plots. The new bundled
[held-out config](../configs/research-real-aux-tier-heldout.json) retains the
ten training plots and expands evaluation to these sites.

| Split | M | L | XL |
|---|---|---|---|
| Validation | Hong Kong `m_hk_0199` | Hong Kong `l_hk_0218` | — |
| Test | New York `m_nyc_0003`, Paris `m_prs_0315` | Paris `l_prs_0324` | New York `xl_nyc_0048` |

Added plots were selected by size, city separation and grid fit. The validation
city has no XL parcel in the current cache. An XL result would measure transfer
to one site without validation on the macro pathway. Aggregate results should
include per-site and per-tier results so failures on a larger tier are visible.
[L/XL comparison and diagnostic results](../results/README.md#held-out-l-and-xl-comparison--2026-11-22)

1. **Measure what RL adds.** Compare greedy, imitation-only, scalar-reward RL,
   fixed auxiliary loss and an auxiliary weight that anneals to zero. Use the
   same training sites, decoder, model and episode budget for learned arms. Log
   environment steps, optimizer updates and wall time on MPS. Start with short
   validation-only screens and freeze the arms, weights, seeds and budgets
   before a held-out comparison. On a paired scene, test whether a policy
   accepts a temporary quality loss that leads to a higher final score at
   comparable FAR and edit count.
2. **Check reward sensitivity.** Rescore the same terminal layouts under a
   predeclared range of weights first. Compare FAR feasibility, usable area,
   direct sun, diffuse sky, depth and core coverage individually. Include a
   variant with one core-access term to measure the effect of double-counting
   access penalties. Retrain the variants whose rankings materially change and
   report tradeoffs among feasible layouts.
3. **Measure decoder effects.** Log retries, rejection rate, distinct decoded
   masks, the fraction of selected voxels changed by projection, and per-stage
   time. Use those measurements to decide whether changing the sampler or
   constraint projection merits a matched-budget experiment.

## L/XL development screen and pilot

[The confirmation config](../configs/research-confirmatory-l-xl.json) separates
training in Barcelona, Tokyo, and Hong Kong, validation in New York, and test
in Paris and Singapore. The training set has ten plots covering all five size
tiers, including one XL plot. The New York XL validation and two Singapore XL
test plots first appear in this protocol. The Singapore site IDs retain their
catalog prefixes. The simulator uses measured areas of 3,938.4 and 3,844.7 m²
to route both through the XL cascade.

The [research suite](../tools/research_suite.py) records its plan before evaluating
the decoder. It adopts faster budget reduction after a rejected mask only if
rejections per accepted edit drop by at least 10%, FAR success holds, and mean
reference quality falls by at most 0.02. It then screens six arms for 30 episodes
per seed using two development model seeds. The screen sets `test_sites` to an
empty list. Every learned arm uses the selected decoder and the same train sites.

| Arm | Training signal |
|---|---|
| Scalar RL | Actor–critic with no spatial auxiliary loss |
| Fixed guidance | Actor–critic with auxiliary weight 1.2 |
| Annealed guidance | Weight 1.2 in the first fifth, then cosine decay to zero |
| Imitation | Greedy teacher trajectories, operation cross-entropy, and masked score-field regression |
| Coverage-only reward | Fixed guidance with one terminal core term, `0.20 − 0.65 × uncovered` |
| Lower daylight reward | Fixed guidance with terminal sun and sky weights reduced by 25% |

The imitation actor generates masks from its learned scores at evaluation.
The reward arms keep the same step shaping and use the common reference for
checkpoint selection and comparisons. The pilot uses the highest-ranked
RL arm from validation and the imitation control. Both train from scratch for
60 episodes with five new seeds, and both keep the final checkpoint. Greedy
uses the same split, edit constraints, and evaluation seeds. The suite records
every seed and includes failed FAR episodes in reported means.

The [equal-FAR planning probe](../tools/lookahead_probe.py) compares one-step
and two-step oracles in an exploratory synthetic scene at a fixed two-edit
horizon. It can also evaluate the trained actor and heuristic from that start
using their own generated masks and a common two-decision cap. The staged
oracles use explicit action templates and exact volume conservation. The policy
probe uses the production FAR band and allows finishing early.
[Pilot results](../results/confirmatory-l-xl-2026-11-22.md)

## 600-episode L/XL follow-up

Fixed-guidance RL and imitation each train from scratch for 600 episodes on the
same five model seeds and ten training sites as the pilot. The full cosine
learning-rate schedule runs from the first episode. Both arms use the adaptive
decoder and a validation checkpoint every 50 episodes. Selection ranks
checkpoints by validation FAR success, then common quality, and favors the
earliest checkpoint in a tie. The comparison reports validation-selected and
episode-600 results on the same four test plots as greedy, with every FAR
failure included. The test sites were previously used in the 60-episode pilot.
[Guided RL config](../configs/research-l-xl-600ep-guided.json) ·
[Imitation config](../configs/research-l-xl-600ep-imitation.json) ·
[Result](../results/l-xl-600ep-mps-2026-11-22.md)
