# Measurements

Accepted masks, not voxel changes or rejected sampling retries. Success means the exact quantized FAR target. Mean steps below includes only successful episodes; compare success counts first.

| Variant | Initial success | Initial mean steps | Short-trained success | Trained mean steps |
|---|---:|---:|---:|---:|
| baseline | 35/36 | 99.7 | 35/36 | 89.3 |
| uncapped | 35/36 | 94.5 | 35/36 | 84.1 |
| quota | 14/36 | 7.1 | 16/36 | 7.9 |
| learned | 13/36 | 8.3 | 17/36 | 9.9 |
| Adaptive feedback, CPU | 36/36 | 17.1 | Not trained | — |
| Adaptive feedback, MPS | 35/36 | 16.8 | Not trained | — |
| Adaptive feedback, fourth seed CPU | 12/12 | 17.2 | Not trained | — |

Feedback scheduling uses the original random-initialized spatial model; the live preview separately preserves the saved spatial weights. These two observations distinguish scheduling gains from learning.

## Initial CPU comparisons by scene

| Scene | Operation | Original steps (successes/3) | Adaptive steps (successes/3) | Mean matched-FAR proxy-quality difference |
|---|---|---:|---:|---:|
| S | subtract | 26–29 (3/3) | 3–4 (3/3) | +0.0189 |
| irregular | subtract | 25–26 (3/3) | 4–4 (3/3) | +0.0221 |
| wide | subtract | 40–46 (3/3) | 7–9 (3/3) | +0.0450 |
| L | subtract | 39–41 (3/3) | 9–10 (3/3) | +0.0501 |
| XL | subtract | 280–302 (3/3) | 30–30 (3/3) | +0.0165 |
| Barcelona | subtract | 94–98 (3/3) | 12–13 (3/3) | +0.0302 |
| S | add | 60–65 (3/3) | 10–11 (3/3) | -0.0023 |
| irregular | add | 65–67 (3/3) | 9–10 (3/3) | +0.0008 |
| wide | add | 147–159 (3/3) | 25–27 (3/3) | +0.0053 |
| L | add | 177–186 (3/3) | 30–31 (3/3) | +0.0073 |
| XL | add | 128–130 (3/3) | 42–42 (3/3) | +0.0047 |
| Barcelona | add | 75–76 (2/3) | 19–20 (3/3) | -0.0066 |

Proxy quality is uncalibrated. Differences are descriptive and use only pairs that both reached FAR. They do not establish architectural quality, daylight compliance or generalization.

## Final adaptive failures

- MPS: seed 456, L add, mask_generation_failed after 30 steps with 248/3072 voxels remaining. Budget 36.

## Training budgets

| Variant | Seed | Actual actions | Optimizer updates |
|---|---:|---:|---:|
| baseline | 123 | 791 | 54 |
| uncapped | 123 | 791 | 54 |
| quota | 123 | 768 | 170 |
| learned | 123 | 770 | 158 |
| baseline | 456 | 776 | 70 |
| uncapped | 456 | 776 | 70 |
| quota | 456 | 769 | 170 |
| learned | 456 | 772 | 154 |
| baseline | 789 | 779 | 63 |
| uncapped | 789 | 779 | 63 |
| quota | 789 | 769 | 171 |
| learned | 789 | 770 | 154 |

Training stops at an episode boundary after at least 768 accepted actions (actual counts above). The short runs use small synthetic shapes and FAR progress rewards; the live session also retains its environmental-audit reward. This is a trainability/short-run comparison, not a convergence study.

## Run provenance

The main run was resumed after preserving every completed row and checkpoint, to reuse immutable scene/shadow setup. Its JSON records both source hash sets and the starting row count. Model/action semantics in the four main configurations are unchanged by the feedback-only follow-ups. Runtime measurements include concurrent test activity and are not an isolated hardware speed benchmark.
