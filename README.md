# CS4246 Team 38 — pedestrian-aware multi-agent traffic signals

This project is an original synthetic SUMO study of vehicle–pedestrian signal-control trade-offs. It starts with one validated junction and extends to three connected junctions. Each junction has sidewalks, four protected crossings, and a controller. The six compared controllers are fixed-time, pedestrian-actuated, Independent PPO (IPPO) with vehicle or pedestrian-aware reward, and Multi-Agent PPO (MAPPO) with the same two rewards. Earlier CS3263 code, scenarios, models, results and figures were not used.

The optional Singapore case study is **not included**: no suitable public junction geometry and demand data have been established for a defensible comparison. The results are from a synthetic corridor and are not evidence of improvement to deployed lights.

## Space and dependencies

The source and saved models are small; generated SUMO episodes go under Git-ignored `runs/`. A complete portable SUMO 1.27.1 binary and Python-tools subset has been extracted into Git-ignored `.local/` for this workspace (about 163 MB). Its download archive was deleted. There are no large ML-framework dependencies: PPO uses the Python standard library. `runs/` can be deleted after preserving `results/` if disk space is needed.

For a fresh checkout, use Python 3.10+ and install [SUMO](https://sumo.dlr.de/docs/Downloads.php) with `sumo` and `netconvert` on `PATH`. Install the small TraCI package with `python -m pip install -r requirements.txt`, or point `PYTHONPATH` at a SUMO installation's `tools` folder. The present MSYS Python installation uses the latter route:

```powershell
$env:PYTHONPATH = (Resolve-Path .local\sumo-1.27.1\tools).Path
```

The code also finds the local portable `sumo.exe` and `netconvert.exe` when they are present. SUMO 1.27.1 was used for the saved runs. No administrative installation is required.

## Watch the simulation

With the `PYTHONPATH` setting above, open two SUMO GUI windows showing the same medium-demand, held-out seed:

```powershell
python -m src.traffic_control.viewer --compare
```

The left window replays fixed-time control (before RL); the right replays the trained pedestrian-aware IPPO controller. The viewer converts each controller's recorded evaluation phase decisions into a SUMO signal program; it does not retrain an agent. Both use the same network, vehicles, pedestrian trips, and seed, at about five simulated seconds per real second. Zoom into a junction to see pedestrians waiting and crossing. The windows stay open when the 600-second simulation finishes. To watch one controller, run `python -m src.traffic_control.viewer --controller fixed` or substitute `ippo-pedestrian`, `mappo-vehicle`, or another controller. `--delay-ms 500` slows playback, `--seed 3002` picks another held-out seed, and `--slow-pedestrians` shows the slower-walker scenario. If raw runs have been removed, the viewer regenerates the selected headless episode first. GUI outputs go under Git-ignored `runs/gui/`.

## Reproduce

Run these commands from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m src.traffic_control.cli --controller fixed --seed 38 --junctions 1
python -m src.traffic_control.cli --controller actuated --seed 38 --junctions 3
python -m src.traffic_control.train
python -m src.traffic_control.evaluate
python -m src.traffic_control.report
```

Use `--slow-pedestrians` with the baseline CLI for a slower-walker run. `--output` selects a different run directory. Training seeds, hyperparameters and counts are in `config/training.json`; the held-out grid and five evaluation seeds are in `config/evaluation.json`. Training creates four small JSON models under `models/`. Evaluation creates raw runs under `runs/evaluation/`; report generation copies aggregate CSV/JSON to `results/` and creates `REPORT.md`, `PRESENTATION.md`, and an original SVG trade-off figure there. An eight-slide presentation is saved as `results/team38-presentation.pptx`.

## Network, signals and safety

The first scenario is specified in `scenario/single.*.xml`. `src/traffic_control/connected.py` writes a new three-junction corridor using SUMO plain XML. `netconvert` builds the network. Vehicles travel along the corridor or through local north–south arms; pedestrians have explicit trips. Demand is generated in departure order from a Python seed, while SUMO receives the same seed. Demand SHA-256 values are checked across controllers in every paired evaluation.

All six controllers use `signals.py`. East–west, north–south and pedestrian movements have separate green groups. Vehicle green lasts at least 12 s; changes pass through 3 s yellow and 2 s all-red. Pedestrian green lasts 12 s, then an all-red clearance is computed using the longest crossing length, 0.8 m/s minimum walking speed and a 3 s buffer. Vehicle green is postponed further if a person remains on a crossing. SUMO's applied signal state is checked each step for unexpected changes and conflicting greens. These are conservative experimental timing rules, not certified jurisdiction-specific plans. Only straight vehicle movements are included.

The actuated baseline can shorten a vehicle green after the minimum when a pedestrian request is waiting. The PPO actors choose whether to end the current vehicle phase when eligible; the state machine still enforces the minimum and a 30 s maximum vehicle green, the phase order and all pedestrian clearance. An outstanding pedestrian request is served on the next pedestrian phase.

## Learning design

Each actor sees only its local east–west and north–south queues, pending crossing requests, current phase and elapsed phase time. IPPO has one local value estimator per junction and optimises local reward. MAPPO has local actors with a single value estimator that sees all three local observations during training and optimises their summed reward. Both use clipped PPO updates and generalised advantage estimation. Vehicle-focused reward penalises local vehicle queues; pedestrian-aware reward also penalises the number of waiting crossing requests. The policy and value functions are **linear** to keep storage and dependencies small. This limits what the learned controllers can represent; the report treats performance as an empirical result of this implementation and training budget, not a general verdict on IPPO or MAPPO.

## Measures and analysis

`CrossingTracker` records the first stopped crossing-approach step when `getNextEdge()` identifies a crossing, then the step when `getRoadID()` becomes that crossing. The difference is the observed crossing wait. Immediate entry is zero wait. SUMO's person `waitingTime` is used only to detect the approach event, never as the final crossing-wait metric. A one-second sampling resolution and possible approach congestion are limitations. Requests not entered by the horizon stay in the event file and are counted separately. A hand-checkable unit trace verifies waits of 0 and 5 s, mean 2.5 s, p95 5 s, and one unserved request.

For vehicles, TraCI departure/arrival IDs reconcile scheduled, completed, unfinished and uninserted trips. SUMO `tripinfo.xml` gives completed-trip `timeLoss`; `--tripinfo-output.write-unfinished true` also preserves on-road trip records at the horizon. Completed-trip mean time loss must always be read with the service counts. The report includes per-crossing mean and p95 wait, unserved requests, a slower-walker check, and bootstrap intervals over held-out seeds. Raw and aggregated data are in `runs/evaluation/` and `results/` respectively.

## Source map

- `src/traffic_control/scenario.py`, `connected.py`: plain-XML network and deterministic demand.
- `signals.py`: shared signal groups, minimum timing, yellow and clearance.
- `metrics.py`: crossing events and vehicle trip accounting.
- `cli.py`: one- or three-junction SUMO episode and local observation interface.
- `ppo.py`, `train.py`: four learned controllers and their training runs.
- `evaluate.py`, `report.py`: held-out study, uncertainty, crossing results, report and figure.
- `viewer.py`: matched fixed-time and learned-controller SUMO GUI runs.
- `tests/`: hand-checkable measurements, safety, determinism, and live SUMO smoke tests.

## Official SUMO references

- [Pedestrian infrastructure and routing](https://sumo.dlr.de/docs/Simulation/Pedestrians.html)
- [TraCI pedestrian-crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html)
- [TripInfo and unfinished-trip output](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html)
- [Plain-XML sidewalks and crossings](https://sumo.dlr.de/docs/Networks/PlainXML.html)
