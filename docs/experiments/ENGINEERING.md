# Engineering probes and run-folder history

The [full local inventory](../../results/experiment-index/local_artifacts.json)
records every top-level directory under `runs/`, including retained diagnostics,
partial attempts and presentation artifacts. These folders are intentionally
ignored by Git. The inventory is a dated record, not a claim that each directory
contains a complete scientific experiment. The main catalogue indexes the
research evidence separately.

| Local name or family | Purpose / interpretation | Public explanation |
|---|---|---|
| `fixed-seed38`, `actuated-seed38`, `3j-fixed-seed38`, `turn-*`, `pressure-smoke` | Baseline, turning movement and safety smoke checks. | [Milestone and setup](../../README.md) |
| `ped-entry-pilot`, `speedprobe*`, `dense-probe*` | Entrance detection, simulator speed and observation/TraCI probes. Not a learned-policy performance comparison. | [Measurement audit](../measurement_safety_audit.md) |
| `pilot-*` | Original learner, matched-critic, parallel execution, prepared inputs, report generation, cardinality/version and short-spacing checks. Small engineering trials, not the later repeated-seed study. | [Legacy study](../../results/final-study/REPORT.md) |
| `burst-*`, `models-safety-one-episode`, `training-safety-one-episode` | Burst loading, safety failure diagnosis and corrected-phase preflight checks. | [Safety revision and excluded attempts](../safety_revision_v3.md) |
| `overload-probe-*`, `short-spacing-*`, `jam-diagnostic*` | Calibrating synthetic stress and examining physical jams. These informed the later fixed protocol; they are not held-out tests. | [Assumptions](../fairness_assumptions.md), [legacy audits](../../results/reproducibility/) |
| `evaluation*`, `training*` | Original pipeline smoke, interrupted safety versions and final linear study. Version and seed matter; do not combine folders indiscriminately. | [E01/E02](README.md) |
| `nus-derived-probe*`, `singapore-*` | Import, protected-crossing feasibility and simplified geometry transfer. | [Singapore limitations](../singapore.md) |
| `fairness-v2-*`, `cluster-final-review`, `cluster-logs` | Neural tracker/load checks, local preflights, partial progress, downloaded final metadata and scheduler diagnostics. | [Verification record](../fairness_validation.md), [E04 archive caveat](../../results/fairness-v2-cluster-review/README.md) |
| `service-priority-*`, `service-policy-audit-*` | Priority intervention, smoke and sampled/greedy replay checks. | [E05/E06](README.md) |
| `critic-fit-*`, `critic-horizon-*` | Local fixed-behavior critic diagnostics; downloaded cluster protocols are exported separately. | [Critic checks](../critic_checks.md) |
| `ppo-repair-*`, `guided-exploration-*` | Packaging, training/evaluation and exact-resume smoke checks for later cluster studies. | [E09/E10](README.md) |
| `local-timing-*` | Conventional-controller admission-duration checks and completed sensitivity study. | [E11](README.md) |
| `deep-audit*`, `audit-repair-*` | Implementation audit, 56-run smoke and completed factorial trial; I/O recovery is retained. | [Deep audit](../deep_audit_2026_11_03.md), [E12](README.md) |
| `pedestrian-finetune*` | Teacher, learner and scheduling preflights; partial local duplicate stopped after cluster transfer. | [E13](README.md), [stopped status](../../results/experiment-index/partial_local_finetune_status.json) |
| `guided-failure-replay` | Eight preserved congested replays, not retraining. | [E14](README.md) |
| `demand-service-preflight*`, `demand-service-test` | Connection/serialization checks, verified preflight, then the complete 720-run frozen evaluation. | [E15](README.md) |
| `friend-marl-tsc-review` | Selected external files used for an independent comparison audit, not our training outputs. | [E16](README.md) |
| `gui`, `final-visuals`, `report-*` | Visualization and report-rendering checks; not new outcome evidence. | [Historical presentation guide](../../deliverables/presentation/PRESENTER_GUIDE.md) |

Some old scratch folders do not have a standalone reviewed report. They are
listed rather than promoted into scientific findings. A file count, a model file
or a successful exit code does not establish matched demand, adequate sample
sizes or completion. Cite the specific reviewed study and its protocol in the
report; use these engineering artifacts only for the checks they actually support.
