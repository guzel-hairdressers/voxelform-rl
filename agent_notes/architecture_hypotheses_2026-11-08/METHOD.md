# Method and interpretation

The production architecture is unchanged. `tools/architecture_hypotheses.py`
contains research-only variants which reuse the production direct mask decoder,
facade projection, A2C loss/Adam settings and environmental evaluation. The exact
original forward pass is checked against the experiment's baseline with zero
numerical tolerance.

## Factors

1. Dual outputs versus one operation-conditioned output. This is intentionally a
   coupled design change: adding the explicit operation signal and sharing the
   final head. Quality differences cannot distinguish the benefit of conditioning
   from the benefit/cost of sharing that head. Frozen-input timing tests measure
   the compute cost directly.
2. Five original spatial channels versus seven: append the current subtraction
   and addition eligibility grids. Extra input weights start at zero so baseline
   and facade versions begin with identical predictions; gradients can train them.
   The same hard facade constraints remain in every configuration.
3. Original floorwise 2D convolutions versus local 3D convolutions. The latter
   inflate compatible 2D kernels across depth, normalized by kernel depth.
4. Local 3D versus multiscale 3D: two horizontal average-pooling stages with 3D
   convolution, trilinear upsampling and additive skip fusion. Vertical resolution
   is preserved; there are only four/eight floors. This expands context, but is
   not guaranteed to cover every large footprint.

These form 12 configurations. Width is held at 16/24 features. The original
baseline has 11,267 parameters, local 3D 29,987 and multiscale 61,139 before facade
or output changes. Architectural effects are therefore not capacity-matched.
Mechanistic gradients establish which information paths exist; they do not prove
that trained models use those paths productively.

## Learning and evaluation

Three paired seeds, 512 attempted training actions per model, exactly 64 in each
of four shapes × two operations. Plot dimensions are `(4,12,16)`, four sun samples,
16 synthetic contextual boxes, scene seeds 100–103. Starting states are the full
envelope for subtraction and two occupied bottom floors for addition. Target FAR
is 3.0 for both. This is a short learning probe, not a throughput fixture.

Held-out synthetic plots are `(4,16,20)`, seeds 9100–9103, and the cached Barcelona
site `m_bcn_0093` is `(6,79,51)`, source area 1450.5 m², 29 neighbour meshes. Each is
evaluated for addition and subtraction before and after training. There are ten
evaluations per model/stage, 720 model evaluations plus 60 heuristic evaluations after control replication.
The deterministic decoder retains production stochastic retries on rejected masks.
Every episode is capped at 192 successful mask actions. Failed/capped episodes
remain failures; FAR equality is required for success.

Uniform-probability and simple sky-access greedy fields are additional controls.
No candidate-mask library is introduced by either. Quality is the existing
uncalibrated environmental proxy, not human architectural quality or compliance.
All evaluations use the same exact target cell count for a given scene. Sun/sky
fields are refreshed every eight masks and on completion.

Compare completion first. Quality deltas use only common successful cases, with
pair counts recorded. Aggregate separately for synthetic scenes and the single
real site. Seeds are the replication unit; scene/operation pairs are not treated
as independent statistical replicates. No significance or generalization claim.
Trained-minus-initial deltas separate learning from attractive random initial
architectures. Equal action budgets do not imply equal wall time or optimizer
update counts; both are retained.

## Runtime

Actual CPU and MPS, no MPS fallback, one CPU thread, synchronized calls. L uses
`(4,48,64)`, eight suns; XL `(8,64,96)`, twelve suns; both have 16 context boxes.
Ten end-to-end mask iterations per variant/device/size retain inputs, forward,
projection and application separately; backward is also executed. Setup and raw
samples are retained. RSS is cumulative process high-water, not an isolated peak
for each architecture. MPS allocation includes the resident environment. Geometry
runs on CPU for every device. These tests do not measure browser rendering.

Supplementary head timing uses 30 randomized interleaved forward pairs on the
same frozen observation, after warmup, excluding input preparation and topology.
This avoids attributing different constraint work to the extra output channel.
Small differences remain hardware/timing noise; inspect raw samples.
