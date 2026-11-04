# Replayed congested failures

Verified replays: **8/8**. This snapshot includes completed runs only.

| Controller | Demand seed | Longest continuous clearance | Longest stationary crossing occupancy | Unresolved measured requests |
|---|---:|---|---|---:|
| mappo-age-r2 | 890004 | C2: 986.0 s | p000235: 991.6 s | 76 |
| mappo-guided-r1 | 890004 | C1: 1662.0 s | p000221: 1293.8 s | 105 |
| mappo-guided-r2 | 890005 | C0: 2080.0 s | p000076: 2084.2 s | 130 |
| actuated | 890004 | C0: 11.0 s | p000237: 0.6 s | 0 |
| actuated | 890005 | C0: 11.0 s | p000424: 0.4 s | 0 |
| imitation-r2 | 890004 | C0: 11.0 s | p000530: 0.4 s | 0 |
| imitation-r1 | 890004 | C0: 11.0 s | p000530: 0.4 s | 0 |
| imitation-r2 | 890005 | C0: 11.0 s | p000424: 0.4 s | 0 |

Demand (normalised line endings), network, vehicle/person/crossing outcomes and discrete diagnostics reproduce the archived study. Floating diagnostic reductions may differ by CPU roundoff; every observed difference is retained in summary.json. Traffic metrics have no roundoff exemption.

The initial verified case, guided replica 2 / seed 890005, has pedestrian p000076 stopped on :C0_c2 from 315.8 until 2400 seconds. C0 remains continuously in pedestrian clearance from 320 to 2400 seconds. A stationary northbound vehicle v000252 occupies internal edge :C0_5 in the recorded prolonged-stall snapshots. This establishes occupied-crossing lockout of the actor; the coincident vehicle is a candidate obstruction. It does not by itself establish why that vehicle stopped or isolate the causal effect of removing it.

This person entered the blocked crossing at 315.6 seconds after a measured wait of only 1.4 seconds, but never exited it within the run. Arrival-to-entry wait and safe completion of crossing are distinct outcomes. Keep entered-but-not-exited visits and full person-trip completion alongside queued requests; a short recorded entry wait does not establish successful pedestrian service.

The safety machine correctly refuses to release conflicting vehicles while the crossing remains occupied. Once clearance is active, the actor has no legal choice. Earlier policy choices may contribute to the state that produces the obstruction; this replay does not absolve the policy or prove that the simulator alone is responsible.

Historical settings are preserved for exact reproduction, including collision settings and disabling obstacle-ignoring pedestrian jam recovery. No detected collision in this replay is not a comprehensive safety certificate. Do not remove travellers, shorten clearance, enable unsafe movements, or replace the archived score to manufacture an improvement.

The remaining matched replays determine whether the same demand jams under actuated and imitation-only control. These cases were selected because they failed, so they cannot estimate the frequency of rare failures. Separate any later all-controller geometry/simulator sensitivity study from this replay.

Rebuild this snapshot with `python scripts/summarize_guided_failures.py`. Raw positions, event timestamps, demand and phase histories are retained under `runs/guided-failure-replay/`. See [the staged research protocol](../../docs/failure_explanation_protocol.md).
