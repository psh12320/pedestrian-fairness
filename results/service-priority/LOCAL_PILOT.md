# Service priority development results

Date: 2 November 2026. Branch: fairness-service-priority.

These are exploratory validation runs, not a successful final proposal result.
No old held-out seeds were reused to tune the new controller.

## Full length conventional controller check

Two matched demand seeds, 47001 and 47002, medium vehicle and pedestrian demand;
original 2400-second horizon, 300-second warm-up, 2100-second measurement cutoff.
All four controllers use unchanged physical phases, minimum green and clearance.

| Controller | Vehicle time-loss difference in seconds seed 47001 / 47002 | Crossings with worse p95 | Largest p95 increase in seconds |
| --- | --- | --- | --- |
| Existing pressure | -10.55 / -10.60 | 9 / 11 | 16.6 / 19.0 |
| Immediate priority pressure | -0.44 / -1.07 | 4 / 6 | 3.2 / 4.4 |
| Age triggered pressure | -4.15 / -1.07 | 5 / 6 | 4.2 / 5.2 |

Differences are relative to matched pedestrian-actuated control; negative vehicle
values favour the candidate. Immediate priority pressure matched the baseline's
vehicle completed/unfinished/uninserted counts in both runs. It matched person
completion counts, had no unresolved crossing requests after follow-up, and had
zero unknown/lost crossing events. Its strict service score was 1 in both runs:
crossing-specific pending requests at the measurement cutoff account for the
failure even though the network-wide pending-count differences were -1 and +1.
This is why net totals alone cannot establish every-crossing service fairness.

None passed the full every-crossing criterion. Two development seeds do not
establish confidence intervals or superiority. The evidence motivates learning
vehicle allocations under explicit pedestrian priority; it does not establish
that the proposed learned controller will improve on actuated control.

## Software and learning smoke

- Ten unit tests passed (four service-priority and six existing analysis tests).
- Four learned arms completed: plain IPPO/MAPPO and priority IPPO/MAPPO, one
  initialisation and two training episodes each; eight updates total.
- Thirty-two evaluation records across four scenarios and eight controller arms.
- No unknown/lost pedestrian events in these evaluation records; the environment
  completed without its conflict/occupancy/collision assertions firing.
- Service masks never add an action to the physical safety mask. The executed
  action and stored PPO sampling mask agree; forced service is critic-only.
- Completed-run resume left all eight latest/best checkpoint hashes unchanged.
- The diagnostic ZIP collector was exercised on one existing engineering group
  containing nine controller records. Structured JSON is stored whole.

Smoke runs are 180 seconds with only two learning episodes. Their sparse crossing
samples and immature policies are not evidence of algorithm performance. They
exist to verify execution before a controlled training pilot.

Local detailed outputs: runs/service-priority-medium-pilot and
runs/service-priority-verified. The first restricted execution failed because
Windows denied the local TraCI socket; the subsequent authorised execution
completed. Local torch emitted a NumPy compatibility warning; the executed paths
used list/tensor conversions and updates completed. Cluster pilot reuses the
existing environment pinned to torch 2.3 and NumPy 1.26.4.

## Next experiment

Run the isolated 32-episode, three-replica controlled pilot described in
docs/service_priority_followup.md. Compare protected and plain PPO under matched
budgets and seeds, with the protected conventional controller as an essential
baseline. The cluster script first exports the original study's decomposed
service failures. Reserve seeds 58001-58020 for a later frozen final evaluation;
do not use them for method selection.
