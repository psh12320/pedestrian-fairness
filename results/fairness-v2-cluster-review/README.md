# Completed fairness-v2 cluster evidence

This is a compact export of the original cluster study, not a new experiment.
Thirty models completed 512 training episodes each, followed by 420 evaluation
groups (21 scenarios × 20 seeds), containing 13,860 controller runs.

- [Original report](archived_REPORT.md): paired vehicle differences and service violations.
- [Frozen protocol](archived_protocol.json): scenarios, seeds, budgets, source/runtime identities.
- [Primary inference](archived_primary_inference.json): the original joint inference.
- [Recovered absolute summaries](../service-priority-cluster/summary.json): includes
  the original study's all-scenario absolute outcomes, recovered during E05 review.
- [Historical interpretation](../project-review/DATA.md) and
  [current experiment catalogue](../../docs/experiments/README.md).

None of the six learned configurations establishes the original joint target.
Vehicle-only policies' low completed-trip delay accompanies serious pedestrian
under-service; missing crossing entries are not evidence of zero waiting.

**Artifact limitation:** the original status collector capped each file at 2 MiB.
The downloaded `runs/cluster-final-review/comparisons.json` is truncated and
invalid JSON. It is deliberately not exported as usable results. Full raw
outputs and models require the original cluster study; see the
[artifact guide](../../docs/experiments/ARTIFACTS.md). These compact records cannot
independently replay every run or recover every missing comparison interval.
