# Model card — organic voxel-editing actor, 0.2.0 research tooling

**Intended use:** controlled ML experiments on correlated, constrained 3D actions;
resource-limited actor-critic training; visualization and debugging of learned
behaviour. Core plans define environment geometry, not an architectural thesis.

**Model:** 14-channel factored 2D+1D encoder, bounded coarse spatial Gaussian means,
operation categorical head and scalar critic. Training uses GAE; optional physical
auxiliary loss is separately configured. Explicit area allocation, legality masks,
geometry projection and greedy core placement are substantial inductive biases.

**Current evidence** includes the five-seed 600-episode tier-adaptive comparison
and deterministic CPU pipeline checks. Mean test quality was `0.776 ± 0.103`
for the validation-selected guided policy and `0.810` for the
greedy baseline. The guided policy had higher mean direct sun and diffuse sky
proxies. The comparison covers two M-tier test parcels. The earlier Eixample
pilot used different cores, walking distance, reward and retry semantics.
[Results index](../results/README.md)

The 60-episode pilot found mean common reference quality of
`0.501 ± 0.075` for guided RL and `0.351 ± 0.009` for imitation across five
model seeds. Greedy scored `0.681`. The test included two XL plots and retained
all FAR failures. [L/XL pilot results](../results/confirmatory-l-xl-2026-11-22.md)

The 600-episode L/XL follow-up checked validation every 50 episodes. Validation-selected
checkpoints averaged `0.613 ± 0.079` for guided RL and `0.649 ± 0.021` for
imitation on test, with greedy at `0.681`. The episode-600 means were `0.616`
for guided RL and `0.669` for imitation. [600-episode results](../results/l-xl-600ep-mps-2026-11-22.md)

**No released validated checkpoint:** generated/saved user checkpoints remain
local. Some historical JSON references private scratch paths that are not present
in the repository. Do not describe those paths as downloadable/reproducible model
releases. New-run checkpoints contain a reproducibility envelope and source archive.

**Known limits:** proxy reward exploitation, imperfect exposure simulation,
partial observability when constraint settings change, site/context distribution
shift, deterministic guard bias, limited seed counts and no long-run convergence
study. Quality means from different FAR acceptances or reward settings require
careful comparison. The common reference evaluator improves consistency but is
itself a hand-designed proxy.

**Excluded uses:** regulatory, structural, accessibility, fire/egress, daylight,
passenger-capacity or occupancy decisions. A stair/lift count based only on height
is a chosen environment prior, not a conclusion about a real building.

**Version boundaries** include the compact core library `compact-core.v5`,
flat sampling semantics `gaussian-retry-path.v1`, tier-adaptive sampling semantics
`gaussian-retry-path-normalized.v1` and research checkpoint format
`voxelform.research.v2`. Exact resume requires the original task, config, scene
and implementation hashes. [Experiment protocol](EXPERIMENT_PROTOCOL.md)
