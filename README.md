# VoxelForm RL

An independent, trainable voxel **building-massing experiment**. This is not
Module Lab and has no construction-module dictionary. The first prototype edits
an occupied envelope using coherent facade cuts, roof/atrium cuts, stop and exact
undo of the latest cut. It does not remove independent random interior cells.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Open **http://127.0.0.1:8010**. It leaves Module Lab on port 8000 alone.
CPU is the initial low-latency default. To explicitly test another device:

```sh
VOXELFORM_DEVICE=mps .venv/bin/python main.py
# or VOXELFORM_DEVICE=cuda on a supported NVIDIA machine
```

During initial development this project's code can also run with the existing
`/Users/ruslan_faz/Desktop/Work/Thesis/.venv/bin/python`; there are no imports of
that project's source. PyTorch reports MPS unavailable in the restricted agent
sandbox on this machine, but actual MPS tensor execution works outside it.

**Train** updates an actor–critic after each rollout of up to 16 edits. Shared
floor CNN embeddings are aggregated across the building. The network selects
among fresh continuous-parameter rectangular cut proposals; it does not yet
directly generate arbitrary free-form masks. **Inference** uses the saved
network without updating weights. **One edit** previews an inference step.
Action scores also see CNN features pooled over the cells each cut would edit;
they do not rely only on global building averages and cut coordinates.
Pause stops active time. Save/load downloads or restores this project's own
policy checkpoint. Changing the plot preserves the policy and clears the rollout.

## What is evaluated

- **Potential usable floor area:** occupied cells within a seven-step horizontal
  path of a facade adjacent to sufficiently sky-accessible air. This is an
  experimental depth proxy, not actual room design or daylight compliance.
- **Direct sunlight:** exposed surface samples against 8 synthetic sun directions,
  separate fixed-neighbour occlusion and changing self-occlusion.
- **Diffuse sky access:** a screened-Laplace decay field in air, with surrounding
  sky openness at the envelope boundary. This scalar approximation can leak
  around corners; it is neither illuminance nor a daylight factor.
- **Relative noise:** finite highway/rail line-source intensity integrals, with
  distance decay. No calibrated dBA, barriers or diffraction are modeled.
- **Sunny roof opportunity:** sun-exposed upward faces, not a demonstration of
  accessible green roofs, soil capacity or planting viability.

The provisional quality is `0.70 usable + 0.18 sun + 0.12 sunny roof - 0.08 noise
- 2 max(0, 0.45 - retained area)`. The usable fraction is relative to the
**original permitted floor area**, so deleting the building cannot earn a high
usable-area score. Rewards are changes in quality minus 0.002 per non-stop edit.
These weights and approximations are a transparent research baseline, not a
validated design objective. Display all metrics separately when comparing edits.

## Shadow implementations

`voxelform/shadows.py` includes direct ray/rotated-box tests, sun-aligned depth
maps and the proposed enclosing-box surface masks. Face masks store visibility
on each up-sun box face for each sun direction. The union of required faces is
derived, not fixed: the usual unrotated 180-degree test arc needs four; a rotated
grid can require five. Boxes intersecting the envelope require direct fallback.

External receiver visibility is cached by persistent voxel-face IDs. Only newly
exposed external receivers are queried. **Self-shadowing is updated for all
current receivers after every edit**: removing a blocker can illuminate an old
receiver. Cached sun-plane DDA paths gather changing occupancy and reduce depth
on the selected device. Receiver lookup has finite map-resolution error.

Runtime mask/field/network operations use PyTorch on one device. Setup includes
CPU scene creation and DDA path preparation. Python orchestration, action scalar
transfers, dynamic receiver extraction and browser serialization remain; this
is not a claim of a completely GPU-resident simulator.

## Tests and method measurements

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=. .venv/bin/python tools/benchmark.py --device cpu --output results/benchmark-cpu.json
PYTHONPATH=. .venv/bin/python tools/benchmark.py --device mps --output results/benchmark-mps.json
```

The method benchmark uses L (64 × 48 m, 4 floors, 8 sun directions, 10 neighbours)
and XL (96 × 64 m, 8 floors, 12 directions, 20 neighbours). End-to-end edit scenes
currently use 16 neighbours, recorded explicitly. It saves source hashes, raw
timings, memory and shadow disagreement with direct reference calculations.
Small test fixtures verify correctness; they are not throughput evidence.

## Current limits

The cavity probes in `results/field-probes.json` show that the diffusion field
has much stronger attenuation than direct sky visibility. The two are not equal
or calibrated; the 0.15 facade threshold is a provisional heuristic. Treat the
resulting potential usable-area metric as experimental until held-out validation
and calibration establish whether it is useful. Do not interpret it as a count
of habitable rooms or use it to claim successful daylight optimization.

The demo uses synthetic plots/context and synthetic sun paths. Real plot/context
loading, astronomical times, facade privacy, actual rooms/corridors/cores,
structural support, disconnected-mass rejection, accurate sky radiance, acoustic
screening and policy-quality/generalization evaluations remain outstanding.
The action profile is rectangular at this stage; it does not prescribe a
rectangular final mass. Restoring volume currently means undo, not arbitrary
additive sculpting. Diffusion training is deferred until validated data exists.

Nonzero actor gradients and changed weights establish trainability only.
No convergence, design improvement or real-site generalization is claimed.

See [AGENTS.md](AGENTS.md) for the transferable project instructions and
[results](results/) for raw measurements. The vendored browser renderer is
Three.js; its license is included in `public/vendor/THREE-LICENSE.txt`.
