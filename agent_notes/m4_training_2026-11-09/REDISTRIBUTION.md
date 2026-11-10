# New requirement: redistribute floor area after reaching FAR

User's screenshot shows a long low mass beside a tall contextual building; the
user proposes removing poorly exposed volume and adding it to taller, potentially
separate towers. This supersedes the assumption that reaching FAR alone should
end all design optimization. It is a requirement for the next action-space study,
not a claim that the current model can learn the proposed form.

## Current blockers confirmed in code

- `Environment.apply` chooses subtraction/addition solely from FAR error and
  ends immediately at the quantized target. It forbids overshoot or editing at
  the target. Starting near FAR does not solve this restriction.
- `ExteriorFacades._unsafe` disallows increasing the number of components. A
  single mass therefore cannot become two separate grounded buildings.
- Real scenes have a fixed six-floor permitted envelope, filled at reset. Growth
  cannot move above that ceiling. Current addition eligibility only includes
  lateral neighbours and upper/lower facade edges, not all exposed roof cells.

## Recommended hypothesis

Retain the fast coarse FAR-matching controller as phase one. Then permit a small
budget of **paired volume exchanges**: model-generated subtraction and addition
fields select the same number of cells, applied together as one action. FAR stays
exact without a large FAR reward or failure cliff in this phase. Existing two
spatial output heads can supply both fields in one network pass. Each frozen
eligibility field is computed from the current state; no hidden layer peeling or
geometric candidate-mask library. Audit the final combined geometry once.

The user also proposes top-storey duplication as a fast growth action. Test it
as an optional parameterized roof extrusion in the same exchange mechanism, not
a required irreversible footprint-first stage. It may duplicate a selected
building’s entire top storey or extend a learned roof patch. Growth changes both
exposure and self-shading, so continue allowing footprint corrections afterward.
A complete storey may add hundreds of cells in one action while advancing only
one exposed roof layer. Higher ceilings must be explicit. Do not infer that all
newly added tower floors are useful without evaluating diffuse sky access,
direct sun and planar depth at their actual heights.

Use a separate height ceiling from initial occupied height. Permit upward growth
from exposed roofs, one newly exposed layer per action. Permit a bounded number
of grounded buildings; reject floating pieces. Evaluate inter-building shading,
diffuse facade-sky access, planar depth and floor area separately. Do not assert
that two towers are always better or infer annual performance from a screenshot's
single display-light shadow. The shared viewer's artistic light is not the same
thing as the environmental evaluation's sampled solar directions.

Compare equal wall time and exact final FAR: carve-only baseline versus paired
refinement, with fixed contexts and several initial heights. Report time, accepted
actions, failure rate and all environmental metrics. This next experiment is not
implemented in the production action space in this work session. Preserve old
checkpoints and label new action/checkpoint semantics explicitly.
