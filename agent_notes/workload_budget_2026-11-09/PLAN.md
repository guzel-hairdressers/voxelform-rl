# Workload-based episode budgets — 2026-11-09

User request: implement a feasible capacity/workload budget, test alternatives,
preserve directly generated coherent masks, and reduce accepted edit count. No
fixed 15-step limit; larger/less accessible workloads may need more actions.

## Hypotheses and controls

1. Remove the 256-cell cap without changing the probability-area decoder.
2. Set edit count from remaining work / remaining productive steps. Freeze an
   episode deadline from initial capacity; reserve steps for projection losses.
3. Learn a multiplicative edit amount around that quota, with countdown,
   remaining work, current capacity and operation as neural conditioning.
4. Follow up if necessary with feedback from applied/generated counts, retaining
   the neural spatial mask. This is scheduling, not a bank of geometric masks.

Compare initial and short-trained policies on six scenes and both operations,
three seeds. Preserve exact failures; report final quality only at matched FAR.
Train small shapes; larger/wider/real-site scenes are evaluation cases. These
are method tests, not evidence of architectural quality or generalization.

## Pilot amendments

- Eligibility-count deadline was too optimistic: deep permissions are not all
  simultaneously usable as coherent roof-open cuts. Use initial lateral facade
  count for subtraction; initial facade-growth count for addition. Deep cuts
  remain permitted, but do not shorten the initial deadline estimate.
- Baseline/uncapped decoder stayed slow (Barcelona 95 actions in seed 123).
- Pure quota lost substantial selected volume to existing topology/opening/patch
  pruning. Retry fresh fields at smaller requested sizes after rejection.
- A separate feedback follow-up scales requests by an EMA of applied/generated
  retention, initially .65. Episode limit stays fixed; it is not extended.
- Strict target completion often failed on tiny tails. Test two versus four
  reserved actions and a small final-adjustment phase. Only the adaptive variant
  relaxes minimum edit-patch size near completion. Maximum patch count, frozen
  eligibility, topology, courtyard minimum area and exact FAR target remain.

## Evidence requirements

- Initial policies distinguish decoder changes from learning.
- Learned amount must get nonzero gradients; consumed retries must include its
  Gaussian sampling score. Check optimizer and fixed-budget checkpoint recovery.
- Execute actual MPS sampling, same-floor depth and backward without CPU fallback.
- Preserve live weights/geometry before changes; do not overwrite original files.
- Archive raw JSON, source hashes and relevant source versions, including failed
  pilot variants. Do not equate near-target timeouts with success.
