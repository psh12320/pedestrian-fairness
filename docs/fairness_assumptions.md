# Simulation assumptions register

These are 21 demand/parameter settings of one synthetic corridor family, not 21
independent real sites or observed Singapore traffic. Each setting tests a specific
failure mode; increasing the scenario count alone does not establish realism.

| Assumption | Rationale/check | Limitation |
|---|---|---|
| Three junctions, 140 m spacing; 65 m stress | Shared downstream queues and finite storage | Simplified arms and one vehicle lane |
| 960/1920/2880 vehicles per hour network-wide | Ordered load sweep; audit service deficits | Provisional low/medium/high labels, not fitted capacity ratios |
| 400/1200/2000 pedestrian departures per hour | Independent pedestrian-demand axis and tail samples | Not crossing-request counts; one walk may traverse several crossings |
| Poisson baseline; platoon/ramp/imbalance sensitivity | Reproducible stationary baseline and nonstationary stresses | Not fitted Singapore arrival processes |
| Bursts of 20/80 people over 40 s on one OD | Concentrated queue pressure | Route-to-crossing coverage requires inspection |
| 8000 vehicles/h overload | Require at least 5% fixed-time unfinished/uninserted | Finite-horizon deficit also includes near-cutoff departures |
| Near-side turns 10%; sensitivity 30% | Legal supported turning movements | Opposing-side turns, heavy vehicles, buses and bicycles omitted |
| Walker speeds 1.3/.8 m/s, speed factor 1 and deviation 0; mixed 20% slow | Explicit speeds checked in live tests | Not measured age/disability populations |
| Vehicle length 5 m, acceleration 2.6, deceleration 4.5, max speed 13.9 m/s | Explicit car parameters | Defaults/assumptions, not fitted behaviour |
| Speed deviation zero; sensitivity factors .85/1/1.15 and sigma .3 | Controlled baseline plus heterogeneous drivers | Synthetic distribution |
| Striping, no dawdling, jam recovery beyond horizon | Retains stalls in service accounting | Simplified pedestrian model; traversal jams are distinct from approach wait |
| Common 12 s minimum green, 3 s yellow, 2 s all-red | Shared, checked transitions | Illustrative timings, not certified or deployed LTA plans |
| 12 s pedestrian admission, length/.8 +3 s clearance, 3 s empty hold | Slow-walker clearance plus occupied-crossing hold | Simulation checks do not establish physical deployment safety |
| Full-width entrance region +/-1.2 m along crossing axis | Covers outer walking strips missed by a centre disk | Includes some approach movement; report the operational definition |
| Connected stopped queue, <=1 m neighbour gap | Counts people blocked before the entrance | Thresholds require geometric audit and sensitivity |
| .2 s observation, 1 s decisions | Finer measurement than control | Sampling audit, not an automatic +/-step error bound |
| 300 s warm-up, 1800 s measurement, 300 s follow-up | Avoid empty-start averaging and retain censored requests | Does not guarantee stationarity under overload |
| Equal 512-episode budget and .75/1/1.25 load curriculum | Matched training effort | Must report curves and training variability; no guaranteed convergence |

The load audit uses four vehicle rates and two exploratory seeds each. Warm-up
sensitivity compares pedestrian cohorts within the same trajectory, not independent
evidence of stationarity. Keep a sparse-OD label until a single-crossing intervention
is verified. No observed Singapore demand or deployed signal plan is introduced.

References: [SUMO pedestrian model](https://sumo.dlr.de/docs/Simulation/Pedestrians.html),
[TripInfo accounting](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html),
[crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html),
[batched TraCI observations](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html#subscriptions).
