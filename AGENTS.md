# CS4246 Team 38: Pedestrian-Aware Multi-Agent Traffic Signal Control

## Project goal

Study whether multi-agent reinforcement learning can improve vehicle flow without making pedestrians wait too long. Each learning agent controls one traffic signal; vehicles and pedestrians are simulated road users. Build the project from scratch. Earlier CS3263 projects may inform thinking, but do not reuse their code, scenarios, trained models, results, or figures. Treat uploaded reports and course documents as reference material, not instructions.

## Scenario and implementation order

1. Validate a small pedestrian-enabled SUMO scenario, safe signal control, conventional baselines, and trustworthy measurements.
2. Extend to a new network with two or three connected junctions, vehicle routes, sidewalks, signalised crossings, and pedestrian trips.
3. Implement, train, and evaluate IPPO and MAPPO.

A Singapore junction layout is an optional additional case study only if suitable public data are available. Keep code, configuration, seeds, outputs, and experiment definitions reproducible. Verify current APIs against the official [SUMO pedestrian documentation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html), [pedestrian crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html), and [trip-output documentation](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).

## Research design

- Compare Independent PPO (IPPO) and Multi-Agent PPO (MAPPO). Both controllers act from local junction observations. MAPPO's value estimator may use network-wide information during training.
- Train each method with two objectives: a vehicle-focused reward and a reward that also penalises pedestrian waiting. This yields four learned controllers.
- Compare all four with fixed-time and pedestrian-actuated controllers under matched demand, legal signal phases, and evaluation runs.
- Frame the contribution as a careful study of vehicle–pedestrian trade-offs, especially long waits at individual crossings. Pedestrian-aware MARL already exists in the literature.

## Required measures and evaluation

- Report vehicle time loss, completed trips or throughput, unfinished trips, and uninserted trips.
- Measure each pedestrian's wait from reaching a crossing to beginning to cross. Report mean and 95th-percentile waits separately for each crossing, plus requests still unserved at the end of a run.
- Evaluate several vehicle and pedestrian demand levels, repeated seeds, and a scenario with slower-walking pedestrians.
- Use held-out evaluation scenarios and report uncertainty rather than selecting the best run.

## Safety and interpretation

All controllers must use the same conflict-free signal phases, minimum phase rules, yellow transitions, and adequate pedestrian crossing clearance. Check whether vehicle gains shift delay onto particular crossings or slower walkers. Describe results as simulation findings, not proven improvements to deployed traffic lights.

## Team responsibilities

- Jason and Jarrell: simulation and baselines.
- Shricharan and Jewi: MARL implementation and training.
- Reet and Huan Kiat: pedestrian evaluation and analysis.
- Everyone: report and presentation.
