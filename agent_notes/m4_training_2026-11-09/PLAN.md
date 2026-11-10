# M4 training plan — 2026-11-09

User priority: wall-clock training speed on Apple M4, while keeping environmental
quality meaningful. Avoid long FAR-only pretraining or an expensive new RL stack.

1. Profile actual training, including reset, neural generation, projection,
   environment/depth updates, environmental audits and backward/optimizer steps.
   Compare CPU and MPS, L/XL and cached real Barcelona, five warm episodes each.
2. Test exact precomputation of immutable external receiver visibility, paying
   setup once per plot. Preserve dynamic self-shadowing and every-eight-action
   audits. Measure setup and amortization, not only warm throughput.
3. Controlled pilot on MPS: current reward with mixed difficulty; quality-led
   reward with mixed difficulty; quality-led reward with a light curriculum.
   All retain identical geometry safeguards, FAR target, workload controller,
   network architecture, optimizer and audit resolution. 60 training seconds per
   arm, two seeds, training on wide/irregular/real scenes and evaluating on three
   held-out scenes. Includes a hard XL case. Rotate arm order by seed.
4. Current reward vs quality-led is a coupled reward hypothesis: less FAR
   progress/success pressure, continuous failure error without the -1 cliff,
   stronger environmental deltas and a cheap exact planar depth delta. Curriculum
   changes easy-problem probability from .8 to .2; mixed stays at .5, so both have
   the same expected average mix. No objective is switched off by the curriculum.
5. Count partial last rollouts as time truncations, bootstrap them and never
   label them terminal failures. Log actual wall time, actions, updates, every
   component/final metric, success and failure separately. Full-FAR test cases
   are used for quality comparisons; exclude incomplete shapes from quality
   means and report their failure rates.

A one-minute pilot cannot establish eventual design quality or convergence.
Preserve raw results/checkpoints. The live preview weights and training reward
stay unchanged until there is evidence to adopt an experimental reward.
