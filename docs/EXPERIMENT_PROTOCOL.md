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
value loss (coefficient `.5`), operation entropy regularization (`.005` baseline,
raised to `.02` under `decoder_retry_schedule="learned_cap"`), Adam (`lr=.0012`,
optional cosine decay via `lr_schedule="cosine"`, `weight_decay=.0001`), and gradient
clipping (`2.0`). This is not PPO. When `multiscale_schedule="tier_meso2"` is
selected, all parcels with `area >= 2,500 m²` (`L` and `XL`) share the `2×2×1 → 1×1×1`
cascade. Under `decoder_retry_schedule="learned_cap"`, accepted voxel budgets scale
with positive frontier logits, early coarse handoff is enabled near target FAR,
and `exchange` is gated off outside `|FAR - FAR*| <= 0.75` (with late fine `exchange`
also suppressed in the final `<= 4` productive steps if FAR is not yet met).
`training_objective="on_policy_distill"` adds masked spatial MSE and legal-masked
operation KL toward frozen tier experts along student trajectories alongside `0.25×`
GAE updates. `reward_profile="dense_potential"` adds scaled coarse/fine potential
differences (`1.5 * delta coarse_potential` at coarse stages and
`2.0 * delta fine_potential` across `audit_steps = 5` shadow/diffuse refreshes,
optionally discounted via `potential_shaping_gamma` for exact `γΦ(s') - Φ(s)`
shaping). Both terminal/step
objectives and audited `fine_potential` shaping match the live app via
`voxelform/objectives.py` (`POST /api/policy` imports `voxelform.research.v2` /
`voxelform.multiscale.v1` `14`-channel checkpoints into `UnifiedLearner` and
`CoarseToFineLearner`, running `2×2×1 → 1×1×1` `tier_meso2` coarse-to-fine
execution on `L/XL` parcels with `area >= 2,500 m²` as well as fine `1×1×1`
refinement). Terminal masks prevent return
propagation across episodes. The runner updates when a completed episode takes
the rollout to `rollout_steps`, and uses any remaining transitions at the end
of the training budget.

The augmented action includes **all Gaussian retries consumed by decoding**,
including rejected draws. The flat learner (`gaussian-retry-path.v1`) scores
their joint log probability. The tier-adaptive learner
(`gaussian-retry-path-normalized.v1`) divides each spatial draw's log probability
by the square root of the number of latent elements. This scaled score is a
surrogate. Smoothing can transmit information across eligibility boundaries, so
each consumed head is scored over its full latent grid. Only the drawn operation
and spatial heads contribute to the score.

An optional physics-shaped spatial auxiliary objective is weighted by
`spatial_aux_weight`. The canonical pure-RL ablation uses **0** (`research-pure-rl-all-tiers.json`),
`research-control-all-tiers.json` uses **1.2** constant, and `research-annealed-all-tiers.json`
holds **1.2** for the first `20%` warmup then cosine-decays to `0.0` (`spatial_aux_schedule="annealed"`). This term is a heuristic
auxiliary learning signal, **not an exact control variate for the original return**;
in constant `"reinforce"` mode (`local_adv` RMS-normalized per step), it has no
finite equilibrium before `±3 tanh` saturation. Opt-in `ExperimentConfig` controls
(`spatial_aux_mode="regression"`, `retry_logp_norm="mean"`, `potential_shaping_gamma`,
and `continuous_entropy_coef`) are available without changing default config hashes
(`to_dict()` omits them when equal to defaults).
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
- Legacy `configs/historical/research-real.json` trains on ten Barcelona, Tokyo and Singapore plots (two
  each for XS, S, M, L and XL), validates on Hong Kong, and tests on New York
  and Paris, while the canonical L/XL configs train across Barcelona, Tokyo, and Hong Kong,
  validate on New York, and test on Paris and Singapore. The runner rejects overlapping context circles across splits
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
- Research evaluation defaults to the training distribution (`logits / T`):
  operation temperature `T = 1.0` and latent noise scale `1.0`, while
  deterministic MAP evaluation uses `T = 0.0` (exact legal argmax) and noise
  `0.0`. The UI's default temperature `0.4` (`1 / 2.5`) / first-draw noise
  `0.08` is a different deployment decoding policy. Both values are recorded and
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
Summaries include standard deviation across model seeds; fixed-seed bootstrap
(`bootstrap95AcrossModelSeeds`) and exact small-sample Student-`t`
(`tCi95AcrossModelSeeds`) 95% intervals are supplied with at least three model
seeds. Such small-sample intervals are descriptive and should not be treated as
definitive proof of generalization on their own.

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

The published five-seed comparison tests M-tier plots. The historical
[held-out config](../configs/historical/research-real-aux-tier-heldout.json) retains the
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

[The confirmation config](../configs/historical/research-confirmatory-l-xl.json) separates
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
[Pilot results](../results/historical/confirmatory-l-xl-2026-11-22.md)

## 600-episode L/XL follow-up

Fixed-guidance RL and imitation each train from scratch for 600 episodes on the
same five model seeds and ten training sites as the pilot. The full cosine
learning-rate schedule runs from the first episode. Both arms use the adaptive
decoder and a validation checkpoint every 50 episodes. Selection ranks
checkpoints by validation FAR success, then common quality, and favors the
earliest checkpoint in a tie. The comparison reports validation-selected and
episode-600 results on the same four test plots as greedy, with every FAR
failure included. The test sites were previously used in the 60-episode pilot.
[Guided RL config](../configs/historical/research-l-xl-600ep-guided.json) ·
[Imitation config](../configs/historical/research-l-xl-600ep-imitation.json) ·
[Result](../results/historical/l-xl-600ep-mps-2026-11-22.md)

## 400-episode `tier_meso2` suite, blind 8-site holdout, and auxiliary ablation

To address L/XL performance without macro `4×4×1` loss of morphological width guards, the current suite combines `multiscale_schedule="tier_meso2"`, `reward_profile="dense_potential"`, and `decoder_retry_schedule="learned_cap"` across five seeds (`4211–4215`, `400` episodes, cosine LR `1.2e-3 → 2.0e-4`) on the city-separated Barcelona/Tokyo/Hong Kong train, New York validation (`3` plots), and Paris+Singapore test (`4` plots) split:

1. **Two-teacher on-policy distillation** ([config](../configs/research-distill-all-tiers.json), [report](../results/distill-all-tiers-400ep-mps-2026-11-22.md)): queries bundled frozen tier experts with embedded `"provenance"` metadata under [`data/teachers/`](../data/teachers/) (`teacher_xs_m_seed789.pt`, `teacher_l_xl_ep100.pt`) at student-visited states, achieving `0.752 ± 0.013` (`99%` FAR) test (`0.753 ± 0.013` at episode `400`) vs. `tier_meso2` Greedy `0.734`. Because those historical teachers had earlier exposure to several holdout parcels across cities, `on_policy_distill` is not zero-shot on the 8-site blind holdout (unlike single-stage `actor_critic`).
2. **Matched `actor_critic` control & `exchange` gating** ([config](../configs/research-control-all-tiers.json), [report](../results/control-all-tiers-400ep-mps-2026-11-22.md)): before gating `exchange`, `actor_critic` reached `0.771 ± 0.072` (`97%` FAR; `seed-4213` suffered premature `exchange`-spam collapse on `m_prs_0320`). Adding the `|FAR - FAR*| <= 0.75` `exchange` gate and `0.02` op entropy eliminates that failure mode and lifts validation-selected `best.pt` to **`0.803 ± 0.018`** (`100%` FAR, bootstrap 95% CI `[0.789, 0.817]`, Student-`t4` CI `[0.781, 0.825]`, `1.024 s/ep` vs. Legacy Greedy `0.857 s/ep`, leading Greedy on all `7/7` Val/Test sites), though fixed-budget episode `400` drifts to `0.762 ± 0.036` (`94%` FAR).
3. **Blind 8-site zero-shot holdout** ([evidence](../results/blind-8site-all-tiers-mps-2026-11-23.json), `640` eval rollouts across `400`-ep models): evaluating all archived `400`-ep seeds (`4211–4215`) on eight completely untouched catalog parcels (`xs_sgp_0346`, `s_sgp_0345`, `m_nyc_0003`, `m_prs_0315`, `l_prs_0324`, `l_sgp_0369`, `xl_nyc_0048`, `xl_sgp_0386`; strictly zero Haversine overlap with train or validation for single-stage `actor_critic`), FAR-Gated `actor_critic` achieves **`0.864 ± 0.046`** (`99.5%` FAR; `XS/S/M = 0.833 ± 0.084`, `L/XL = 0.896 ± 0.017`) vs. `tier_meso2` Greedy `0.824` (`100%` FAR; `XS/S/M = 0.833`, `L/XL = 0.815`), ungated `actor_critic` `0.820 ± 0.101`, and distillation `0.659 ± 0.054` (`L/XL = 0.854`, `xs_sgp_0346 = -0.284`). FAR-Gated `actor_critic` outscores Greedy on `4/4` blind `L/XL` parcels and `6/8` parcels overall, while Greedy retains an edge on tiny `xs_sgp_0346` (`0.990` vs. `0.798`).
4. **Matched 400-episode auxiliary-supervision ablation** ([pure-RL config](../configs/research-pure-rl-all-tiers.json), [regression config](../configs/research-regression-aux-all-tiers.json), [annealed config](../configs/research-annealed-all-tiers.json), [aux evidence](../results/aux-ablation-400ep-mps-2026-11-23.json), [regression evidence](../results/regression-aux-400ep-mps-2026-11-23.json), seeds `4211–4215`): with identical `400`-episode cosine LR horizons and `5` seeds, removing `des_c` (`pure_rl`, `λ_aux = 0.0`) stalls at `0.420 ± 0.063` Test (`0.418 ± 0.074` at `ep 400`; `0.531 ± 0.075` blind). Switching constant `λ_aux = 1.2` from `"reinforce"` to fixed-point `SmoothL1` `"regression"` (`regression_aux`) eliminates late-episode FAR collapse (`0.641 ± 0.031` `best.pt` vs. `0.631 ± 0.032` / `100%` FAR at `ep 400`, drift `-0.010 ± 0.016`; `0.695` MAP Test; `0.775` Blind / `0.786` Blind `L/XL`, `0.821` MAP) **but** over-constraints the spatial actor to static `2·des_c` (paired `p = 0.00030` Test below `"reinforce"`). Holding `"reinforce"` `λ_aux = 1.2` for a `20%` warmup then cosine-decaying `1.2 → 0.0` reaches `0.727 ± 0.027` Val / `0.790 ± 0.021` Test (`99.0%` FAR) at `best.pt`, holds `0.767 ± 0.038` (`99.0%` FAR) at episode `400`, and transfers to `0.823 ± 0.087` overall (`0.883 ± 0.024` `L/XL`) on the 8-site blind holdout, versus constant `λ_aux = 1.2` (`0.803 ± 0.018` `best.pt`, `0.762 ± 0.036` / `94.0%` FAR at `ep 400`, `0.864 ± 0.046` blind).
5. **Matched Greedy fairness parity & deterministic MAP evaluation** ([evidence](../results/eval-fairness-parity-mps-2026-11-23.json)): giving Greedy both `_learned_quota` and early coarse handoff (`greedy_learned_cap=True`) yields `0.730` Test (`0.897 s/ep`), `0.833` Blind ALL (`0.732 s/ep`), and `0.832` Blind `L/XL` (`1.259 s/ep`), while `5×` Best-of-5 Stochastic Fair Greedy reaches `0.773` Test (`4.355 s/site`), `0.875` Blind ALL (`3.207 s/site`), and `0.850` Blind `L/XL` (`5.538 s/site`). Under deterministic MAP decoding (`temp=0.0, noise=0.0`), FAR-Gated `actor_critic` achieves `0.815 ± 0.034` (`96%` FAR, `1.262 s/ep`, `t(4)=5.50, p=0.0053`) on Test, `0.872 ± 0.062` (`98.5%` FAR, `0.847 s/ep`) on Blind ALL, and **`0.929 ± 0.027`** (**`100%` FAR**, Student-`t4` CI `[0.896, 0.962]`, `t(4)=8.13, p=0.0012` / `t(3)=6.79, p=0.0065` vs. Fair Greedy Det) on Blind `L/XL`, outscoring even `5×` Best-of-5 Fair Greedy on all `4/4` blind `L/XL` parcels. Overall 8-site blind gains (`p=0.198–0.525`) are not statistically significant across all 8 parcels due to tiny-parcel `xs_sgp_0346` variance.

## Canonical vs. historical configurations

| Config (`configs/`) | Status | Role |
|---|---|---|
| [`research-control-all-tiers.json`](../configs/research-control-all-tiers.json) | **Canonical** | 5-seed `400`-ep FAR-gated `actor_critic` (`tier_meso2 + dense_potential + learned_cap`, `λ_aux = 1.2` constant) |
| [`research-annealed-all-tiers.json`](../configs/research-annealed-all-tiers.json) | **Canonical** | 5-seed `400`-ep `λ_aux: 1.2 → 0.0` annealed ablation (`tier_meso2 + dense_potential + learned_cap`) |
| [`research-regression-aux-all-tiers.json`](../configs/research-regression-aux-all-tiers.json) | **Canonical** | 5-seed `400`-ep single-variable `spatial_aux_mode = "regression"` control (`λ_aux = 1.2` constant) |
| [`research-pure-rl-all-tiers.json`](../configs/research-pure-rl-all-tiers.json) | **Canonical** | 5-seed `400`-ep scalar/potential pure-RL control (`λ_aux = 0.0`) |
| [`research-distill-all-tiers.json`](../configs/research-distill-all-tiers.json) | **Canonical** | 5-seed `400`-ep two-teacher `on_policy_distill` using bundled `data/teachers/*.pt` |
| [`research-smoke.json`](../configs/research-smoke.json) | **Canonical (CI)** | Fast deterministic CPU synthetic smoke check |
| [`research-real-aux-tier-600ep.json`](../configs/historical/research-real-aux-tier-600ep.json), [`research-real-tier-600ep.json`](../configs/historical/research-real-tier-600ep.json), [`research-real-aux-1200ep.json`](../configs/historical/research-real-aux-1200ep.json) | Historical (M-tier, under `configs/historical/`) | Pre-`tier_meso2` `600`/`1200`-ep M-tier baseline & flat/aux controls |
| [`research-real.json`](../configs/historical/research-real.json), [`research-real-aux.json`](../configs/historical/research-real-aux.json), [`research-real-aux-tier-heldout.json`](../configs/historical/research-real-aux-tier-heldout.json) | Historical (under `configs/historical/`) | Early M-tier single-stage and initial held-out transfer specs |
| [`research-expert-l-xl.json`](../configs/historical/research-expert-l-xl.json), [`research-confirmatory-l-xl.json`](../configs/historical/research-confirmatory-l-xl.json), [`research-l-xl-600ep-guided.json`](../configs/historical/research-l-xl-600ep-guided.json), [`research-l-xl-600ep-imitation.json`](../configs/historical/research-l-xl-600ep-imitation.json) | Historical (L/XL pilots, under `configs/historical/`) | Pre-`tier_meso2` `tier_adaptive` L/XL expert, `60`-ep pilot, and `600`-ep follow-up |
