# Coherent mask experiment plan

Four samplers: original independent Bernoulli, Gaussian-smoothed means with
independent Bernoulli, coarse Gaussian latent interpolation, and that latent plus
Gaussian smoothing. Common floor CNN and value critic; coarse latent means pool
model outputs. No candidate masks/templates. For correlated samplers, score-function
A2C uses the Gaussian latent draw's actual log probability; thresholding/quantile
selection, eligibility weighting and safety projection are deterministic functions
of that latent and state, not independent Bernoulli decisions.

Two eligibility settings tested independently: current lateral facade, versus
max(facade,clamp((horizontal depth-8)/8,0,1)) for occupied subtraction cells. Addition
remains adjacent growth in both. Frozen current fields; bulk commit; whole-mass
connectivity/grounding and air reachability. Interior cuts must connect to outside
air after the complete action, e.g. an opening through the roof. Sealed pockets are
pruned/rejected; no unselected voxels are added to repair them.

Primary structural hypotheses: correlated sampling reduces component count and
increases largest-component share; extra Gaussian smoothing may improve boundary
regularity. Smoothing probabilities with independent draws may not help. Measure
both raw generated and post-projection masks at matched edit budget in fixed-field
probes, plus actual neural FAR episodes. Do not equate reduced fragmentation with
better architectural/environmental quality.

Three seeds. Balanced short training, 256 attempted actions per model (32 for each
of four plots × two operations). Initial and trained held-out synthetic/real-site
FAR episodes, failure/cap retention, paired quality at matched FAR and explicit
fragmentation measures. Compare strict facade first, then deep union. Record
actual updates, failed attempts, correction fraction, actor gradients and weights.

CPU and actual available MPS L/XL timings, setup, sampling, projection, application,
learning and memory separate. CPU topology is not GPU-resident. Preserve all raw
results/hashes. Optional minimum patch size/component limit is tested separately
from sampling; never enforce it only on the comparator. Final FAR residuals may be
smaller than the ordinary patch minimum. Do not overwrite historical checkpoints.

## Recorded follow-ups

Adaptive 3D pooling was unavailable on actual MPS; replace it with native strided
average pooling on both devices and retain the interrupted pilot file. No silent
CPU fallback. The primary run uses 24 models (four samplers × two eligibility
settings × three seeds), 256 matched actions each.

Small learning plots have no >8m interior; Barcelona's maximum is ~9.19m with only
42 deep cells. Add a held-out 4x40x48 wide plate (max19.5m) for a genuine deep-cut
probe. No wide-scene training/generalization claim. Preserve results separately.

Primary rollout evaluations inherited greedy-first inference; despite fixed-count
random-field coherence gains, mean-field trajectory masks remained fragmented.
Retain that negative diagnostic and separately evaluate the already trained
checkpoints using actual sampled inference for all four modes, with paired seeds.
The product preview must sample correlated latents as training does. Test min4/max4
component guards separately and do not attribute their guarantees to smoothing.
