# NUS-derived corridor transfer check

**Uncalibrated synthetic-demand simulation.** The three approximate candidate-site centroids and spacings came from a local OSM audit. The connecting roads, approaches, crossings, routes, pedestrian arrivals and phase plans are project assumptions. These results do not estimate an improvement to deployed Singapore signals.

The frozen transfer protocol used 10 matched demand seed(s), 5 independent training replicas per PPO configuration, and 330 SUMO runs. Each controller received the same demand within each seed and used the shared legal phase and pedestrian-clearance machine.

| Controller | Vehicle time loss (s) | Paired difference vs actuated [95% CI] (s) | Completed | Pending requests | Worst crossing p95 difference [95% CI] (s) | Descriptive crossing target |
|---|---:|---:|---:|---:|---:|---|
| fixed | 42.7 | 6.3 [4.0, 8.4] | 192.0 | 0.0 | 42.0 [32.0, 66.0] | no / inconclusive |
| actuated | 36.5 | reference | 192.0 | 0.0 | reference | reference |
| pressure | 29.4 | -7.1 [-10.2, -4.1] | 192.0 | 0.0 | 16.0 [n/a, n/a] | no / inconclusive |
| ippo-w0 | 36.0 | -0.4 [-2.5, 1.7] | 192.0 | 0.0 | 11.6 [n/a, n/a] | no / inconclusive |
| ippo-w10 | 36.0 | -0.5 [-2.3, 1.5] | 192.0 | 0.0 | 5.0 [4.0, 41.0] | no / inconclusive |
| ippo-w3 | 35.5 | -1.0 [-2.9, 1.0] | 192.0 | 0.0 | 4.2 [3.0, 40.4] | no / inconclusive |
| mappo-w0 | 37.2 | 0.7 [-1.2, 2.8] | 192.0 | 0.0 | 16.4 [n/a, n/a] | no / inconclusive |
| mappo-w10 | 36.0 | -0.5 [-2.3, 1.5] | 192.0 | 0.0 | 5.0 [4.0, 40.6] | no / inconclusive |
| mappo-w3 | 35.5 | -0.9 [-2.9, 1.1] | 192.0 | 0.0 | 4.4 [3.0, 40.0] | no / inconclusive |

Each crossing's entry denominator, mean wait, p95 wait, pending count and paired p95 interval is in `crossing_summary.csv` and `crossing_comparison.csv`. The complete per-run results are in `runs.csv`; raw crossing events are compressed in `crossing_events.csv.gz`. A p95 based on fewer than 100 pooled entries per training replica is labelled inadequate for a strong every-crossing claim.

## Limits

- Three approximate candidate-site centroids and spacings are map-derived; site identities, geometry and crossing locations are not field-verified.
- Connecting links, side arms, sidewalks, protected crossings and all demand are synthetic project assumptions; no Singapore turning or pedestrian counts or deployed signal timing plans were used.
- The raw OSM import had no signal-controlled pedestrian crossings, so the protected derived network is not a reconstructed field junction.
- A sparse crossing p95 or incomplete bootstrap makes the every-crossing criterion inconclusive even if its point estimate appears favorable.

This transfer check is deliberately separate from the locked 13-scenario synthetic study. The earlier seed-61 Singapore pilot and its v2 policies were exploratory and trained under the superseded clearance rule; neither is mixed into this table.


## Final console-warning audit

The retained console logs contain **2 pedestrian-jam warning lines**. This is a warning-line count, not a count of unique people or failed runs. SUMO's striping jam recovery permits movement ignoring obstacles, and does not register pedestrian collisions while jammed. Passing the signal-state checks therefore does not establish collision-free trajectories. Crossing wait ends at entry and excludes delays during traversal. See the [official jamming explanation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming) and [compact log audit](../reproducibility/final_log_audit.json). A read-only replay of one short-spacing case reproduced every saved outcome measure and kept vehicle links red while the person occupied the crossing; its [trace and audit](../reproducibility/pedestrian_jam_audit.json) provide a diagnostic, not a validation of physical safety.
