# Measurement and signal-safety audit

This is a development audit of the constructed SUMO networks, not a certification for a real junction. It describes the entrance-event measurement rule used by the current study. Older results in `results/REPORT.md` used an earlier stopped-approach rule and are not directly comparable.

## Crossing event definition

At each one-second TraCI observation, the tracker reads a person's current road edge, next edge, position, and SUMO waiting time. A crossing request begins at the first observation within 1.2 m of either endpoint of the next crossing lane. If the person is already stopped on the immediate approach farther away, the start is the beginning of that first stopped one-second interval. Crossing service begins at the first observation on the crossing edge. The difference is the recorded wait. The `arrival_source` field records which rule triggered; `entry_without_approach` is flagged and assigned zero observed wait. A reached crossing without an entry at the run horizon is retained as unserved, including its elapsed wait. SUMO waiting time is only used to detect an approach queue; it is not the reported complete crossing wait.

The unit trace in `tests/test_milestone.py` checks a moving person reaching the entrance at 3 s and entering at 9 s (6 s wait), a person whose first stopped interval starts at 5 s and remains unserved at 12 s (7 s elapsed), and a person first observed on the crossing (zero imputed wait). A live single-junction test separately compares tracker timestamps with TraCI positions and crossing-edge entries. Because positions are sampled every second, either boundary can fall between observations; individual waits have roughly one-step timing resolution. The 1.2 m threshold is operational for this network, where the lateral sidewalk/crossing offset leaves the stop line about a metre from the crossing lane endpoint. Recheck it before using another imported network.

In an **earlier exploratory** three-junction, medium-demand actuated run (seed 6001, SUMO 1.27.1, before the three-empty-step guard), 160 crossing entries were observed across 12 crossings. Arrival sources were 147 entrance, 3 queued approach, and 10 direct-entry imputations; no request remained unserved. The smallest per-crossing entry count was 10 in that run. Direct entries are 6.25% of entries; their zero assignment may lower a reported mean or tail and must remain visible beside final results. This pilot seed is excluded from the locked test seeds 9001–9010; its wait values are not combined with the revised protocol.

## Direct check of sampling error

The repeatable `python -m src.traffic_control.measurement_resolution` audit runs
one fine-step SUMO trajectory, then observes it with both a 0.1-second tracker
and a one-second tracker. It uses development seed 6001, 24 pedestrians, light
vehicle demand, the common fixed-controller phase replay, and a 180-second
horizon. This isolates observation sampling on the same movements. It does not
measure the effect of changing SUMO's simulation step on traffic dynamics, or
validate the 1.2 m entrance boundary against physical ground truth.

The two trackers found **32 requests: 30 entered and 2 pending**. Across matched
entries, the mean absolute wait difference was **0.49 s**, the mean signed
difference (coarse minus fine) was **+0.33 s**, and the largest absolute
difference was **1.30 s**. The queued-approach rule backdates the first stopped
interval, so the wait difference can exceed one second. One coarse observation
missed an approach and imputed zero wait; the fine tracker had no such case.

| Person / crossing | Case | Fine arrival / entry | Fine wait | One-second wait |
| --- | --- | --- | ---: | ---: |
| `p002 / :C0_c1` | Immediate entry | 147.1 / 148.0 s | 0.9 s | 0.0 s, direct-entry imputation |
| `p001 / :C0_c1` | Queued wait | 110.4 / 143.1 s | 32.7 s | 33.0 s |
| `p000 / :C0_c0` | Pending at horizon | 156.0 / no entry | 24.0 s elapsed at 180 s | Same pending request |

The fine trace for `p002` places the person 1.200011 m from the entrance at
147.0 s, just outside the threshold, and 1.098154 m away at 147.1 s. The coarse
tracker next observes that person on the crossing at 148 s. For `p001`, the
coarse timestamps are 111 and 144 s, so its reported wait is 33 s. These cases
can be checked in [the hand trace](../results/measurement-resolution/hand_trace.csv),
[event comparison](../results/measurement-resolution/event_comparison.csv), and
[audit summary](../results/measurement-resolution/summary.json). The repeated
audit reproduced its demand and all measured values. This small fixture is an
observed error check, not a universal 1.30-second bound.

## Shared phase machine

Every controller chooses only whether to request an earlier change after a 12 s vehicle minimum. The same phase machine applies the green, 3 s yellow, 2 s all-red, 12 s pedestrian green, and pedestrian all-red clearance. The constructed network's crossing length is 6.4 m; at the design minimum walking speed 0.8 m/s plus a 3 s buffer, calculated clearance is 11 s. Clearance extends if any person remains on a crossing and requires three consecutive empty crossing observations before vehicle green. Each state is checked against the generated SUMO junction `foes` conflict matrix, including the compatible near-side turns in stress scenarios. The live loop also checks that observed signal states equal the commanded states and that no person occupies a crossing during a vehicle green. An attempted 80-person burst failed this assertion in all ten inspected seeds; its groups are excluded. The revised 20-person burst passed a 330-run development safety pilot. See [the safety revision](safety_revision_v3.md).

The tests verify minimum green, yellow and all-red durations, pedestrian requests reaching a pedestrian phase, clearance extension while a crossing remains occupied, the three-step empty guard, turning-link foe checks, and deterministic matched demand. All 43 tests passed on SUMO 1.27.1 on 30 September 2026, including the later reporting and transfer checks. The protected all-walk design is conservative for the synthetic corridors. It does not establish that the raw NUS-area OSM import has legal phases: that import currently has no signal-controlled crossing edges and is excluded from controller comparisons.

## Vehicle service and congestion checks

Scheduled IDs from the demand generator are reconciled with TraCI departure and arrival IDs: completed + unfinished + uninserted = scheduled. Completed-trip time loss comes from SUMO TripInfo; the other two categories remain in every result. In an exploratory fixed-time overload run (seed 81), 860 vehicles were scheduled, 715 completed, 94 departed but remained unfinished, and 51 were never inserted. The 16.9% service shortfall shows that the constructed overload can expose misleading completed-trip averages; the locked seeds must establish whether this persists. An exploratory 65 m short-spacing run (seed 6002) left 35 of 624 vehicles unfinished or uninserted. Its internal connector was above a simple 80% space-occupancy proxy for 261 of 600 seconds while an upstream edge had a halted vehicle. This is a queue-coupling diagnostic, not proof of the cause of each stop.

Run the audit tests with `python -m unittest discover -s tests -v` after putting the portable SUMO `tools` directory on `PYTHONPATH`. Inspect `crossing_events.json`, `crossings.csv`, and `phase_history.csv` when reviewing final results. Consult the official [SUMO pedestrian simulation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html), [TraCI crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html), and [TripInfo output](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html) documentation for the underlying simulation fields.
