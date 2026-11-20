# Model card — organic voxel-editing actor, 0.2.0 research tooling

**Intended use:** controlled ML experiments on correlated, constrained 3D actions;
resource-limited actor-critic training; visualization and debugging of learned
behaviour. Core plans define environment geometry, not an architectural thesis.

**Model:** 14-channel factored 2D+1D encoder, bounded coarse spatial Gaussian means,
operation categorical head and scalar critic. Training uses GAE; optional physical
auxiliary loss is separately configured. Explicit area allocation, legality masks,
geometry projection and greedy core placement are substantial inductive biases.

**Current evidence:** historical pilot JSON and new deterministic CPU pipeline /
geometry / gradient / checkpoint tests. The other agent's 80-episode Eixample
A/B results used older larger core templates, different walking-distance semantics
and the earlier reward/retry implementation. Their increasing score is not proof
of convergence, nor a result for the new compact-core model. Raw historical files
are retained and indexed with hashes rather than overwritten.

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

**Version boundaries:** compact core library `compact-core.v4`; corrected Gaussian
retry path `gaussian-retry-path.v1`; experiment wrapper `voxelform.research.v1`.
Network checkpoint compatibility does not imply task/reward compatibility. See
[experiment protocol](EXPERIMENT_PROTOCOL.md) for exact resume requirements.
