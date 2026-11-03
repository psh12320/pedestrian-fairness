# CS4246 Team 38: pedestrian-aware traffic signal control

**Local correction experiment:** [audit-repair trial](docs/audit_trial.md) compares
legacy, controller-memory, separate-clipping and combined settings for IPPO/MAPPO.
Its 56-run engineering smoke passed. The full 1,396-simulation CPU study was
launched on 3 November; use `python scripts/collect_audit_trial_status.py` for
current evidence. Launching the experiment is not evidence of a performance gain.

**3 November implementation audit:** [findings, verified repairs and remaining experiments](docs/deep_audit_2026_11_03.md).
Three independent audits identified missing controller memory, a junction-collision
check gap, platform-dependent demand bytes and missing insertion-delay reporting.
Bounded fixes are verified; no new RL performance gain is claimed. Existing
cluster archives/results are preserved. The newer neural `fairness` pipeline uses
a different tracker/protocol from the original study described further below;
do not combine their results or measurement definitions.

**Latest development study:** [guided exploration](docs/guided_exploration.md)
tests crossing-age rewards, actuated-guided MAPPO and training-only curiosity
against matched controls, including imitation without RL. The
[completed PPO-repair review](results/ppo-repair-cluster/REVIEW.md) found no joint
vehicle/pedestrian win. The new study is a separate experiment, not a claimed
solution. See the [v2 execution guide](docs/fairness_v2.md) for the earlier neural
IPPO/MAPPO implementation, 21 demand settings, constraints and cluster workflow.

A separate [local timing study](docs/local_timing_study.md) tests 12-, 18- and
24-second pedestrian admission periods under matched conventional controllers.
It is a timing sensitivity experiment; any gains are not attributable to RL.

This project asks: **When, if ever, does multi-agent PPO save vehicle time without worsening long waits at any crossing, compared with a pedestrian-responsive controller?** It is an original SUMO simulation, built from scratch, with a single-junction safety and measurement milestone and a three-junction research corridor. Each learning actor controls one signal using local observations. Both PPO methods use the same per-junction reward; IPPO has local value estimators, while MAPPO's value estimators can see the whole corridor during training. Vehicles and pedestrians are simulated road users. Earlier CS3263 code, scenarios, models, results, and figures were not reused.

The primary study uses **synthetic geometry and demand**. A separate [NUS Kent Ridge–NUH pilot](docs/singapore.md) includes a small OpenStreetMap extract, clipped public LTA signal-hardware points, and a safe *simplified* left-hand corridor through approximate candidate-site centroids. Its connectors are straight segments and its side arms are synthetic. It has **no observed turning or pedestrian counts and no actual LTA timing plans**. Neither case establishes an improvement to deployed traffic lights.

## Current evidence and status

- The single-junction and connected-junction SUMO smoke tests exercise moving cars and walkers, protected crossing service, conflict checks, phase timing, crossing events, and repeatable seeded demand. The hand-checkable event trace includes immediate entry, a queue, and a request left waiting at the horizon.
- Fixed-time, pedestrian-actuated, and a **pressure-inspired** queue-responsive baseline use the same phase machine as learned policies. The pressure controller is an adaptation for this project, not an exact implementation of a published max-pressure algorithm.
- The older [report](results/REPORT.md) and [crossing table](results/crossing_summary.csv) are **exploratory**. They used five evaluation seeds, one trained model per method, and an earlier stopped-approach crossing-wait definition. Their numbers must not be combined with, or presented as a confirmation of, the new protocol.
- The [completed study](results/final-study/REPORT.md) contains **30 independent replicas trained for 150 episodes each and 4,290 locked-test runs** across 13 scenarios and ten demand seeds. The frozen validation rule selected the common 150-episode budget. At primary medium demand, IPPO weight 3 averaged **33.3 s** completed-vehicle time loss versus **34.7 s** for actuated control; the paired difference was **−1.4 s [95% CI −4.2, +1.5]**. Four crossings had higher p95 waits. **None of the six learned configurations met the every-crossing criterion.** The pressure-inspired baseline had the lowest mean vehicle loss (29.6 s) but also failed the crossing target.
- Every primary crossing had at least 100 entries per training replica and baseline. Fixed-time left **17.1%** of overload vehicles unfinished or uninserted on average; every test seed exceeded 5% (range 15.3–18.4%). The [output audit](results/reproducibility/final_output_audit.json) checks all 130 matched-demand groups, vehicle conservation, all crossing pending counts and frozen source hashes. Re-run it with `python scripts/verify_study_outputs.py`.
- The first test attempt exposed a pedestrian-burst safety failure; its 7001–7010 checkpoints are excluded. A partial 8001–8010 run was also stopped when a training/evaluation phase-rule mismatch was found. All models were retrained before testing the untouched 9001–9010 seeds; see the [safety revision](docs/safety_revision_v3.md). Neither interrupted attempt is included in the final performance tables.
- The final short-spacing test logged **28 pedestrian-jam warnings**. A [recorded replay](results/reproducibility/pedestrian_jam_audit.json) exactly reproduced its saved results and held vehicle links red while the person occupied the crossing. SUMO's jam recovery ignores obstacles, so these stress runs do not establish collision safety. Pre-entry waiting time also excludes jams during traversal; see the [official explanation](https://sumo.dlr.de/docs/Simulation/Pedestrians.html#jamming).

## Completed deliverables

- [Ten-page report](output/pdf/cs4246-team38-final-report.pdf), with all twelve crossings, confidence intervals, training curves, stress results and the separate geometry test.
- [Thirteen-slide presentation](deliverables/presentation/output/team38-research-talk-v2f-final.pptx) and [presenter guide](deliverables/presentation/PRESENTER_GUIDE.md).
- [Full main-study analysis and tables](results/final-study/REPORT.md).
- [Separate Singapore-derived study](results/singapore-transfer/REPORT.md): 330 matched runs; uncalibrated geometry sensitivity, with synthetic demand. No learned setting met the crossing target there either.
- [Labelled moving comparison](deliverables/presentation/assets/final-protocol/sumo-fixed-vs-ippo-pedestrian-seed-6001.mp4): 20 distinct frames per controller, matched seed 6001, 10-second playback. Each frame is an original SUMO rendering from the same deterministic replay.
- [One-click SUMO comparison](Watch-Comparison.cmd): fixed-time and selected IPPO weight-3 replica on identical exploratory demand.

The 43-test suite passed in the recorded environment (`runs/resume-tests.log`). The final saved-output audit reconciles all 4,620 evaluation runs. The report and all 13 slides have been rendered and visually inspected. The individual contributions remain explicitly provisional; replace them with actual work and confirm the contradictory course deadline before submission. LLM assistance is disclosed in both documents.

## Setup within a 2 GB storage budget

The original implementation's setup is below. The new v2 neural implementation
also requires PyTorch; its [execution guide](docs/fairness_execution.md) uses the
existing local CPU installation or an isolated, pinned environment on the cluster.

Use Python 3.10+ and SUMO with `sumo`, `sumo-gui`, and `netconvert`. This workspace has a Git-ignored portable SUMO 1.27.1 installation under `.local/sumo-1.27.1` (about 163 MB); a fresh checkout can instead use an installed SUMO. The PPO implementation uses the Python standard library. The only Python requirement is TraCI:

```powershell
python -m pip install -r requirements.txt
```

For the portable installation, run this in each new PowerShell session from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .local\sumo-1.27.1\tools).Path
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
```

The bundled portable archive omits SUMO's optional data directory. Pointing `SUMO_HOME` at it makes the OSM import smoke fail during XML validation; leaving it unset produces a harmless validation warning and passes the tests. A complete system SUMO installation may use its own correctly configured `SUMO_HOME` instead.

The code locates the portable executables when present. Generated episodes and GUI files go under Git-ignored `runs/`; training and evaluation use temporary episode directories and retain compact logs, group results, and models. Keep `models/v3/`, the active `runs/training-v3/` and `runs/evaluation-v4/`, and `results/final-study/` until the study is complete. Preserve `models/v2/` and its older runs only as a clearly separated safety audit. The 518 MB LTA count archive is **not** part of the workspace; the Singapore intake can inspect a locally supplied ZIP and export only small selected data on a machine with enough space.

The tested simulation environment is recorded in
[`simulation_environment.json`](results/reproducibility/simulation_environment.json):
Python 3.12.7, SUMO 1.27.1, Windows 11, and four concurrent simulation workers.
[`artifact_environment.json`](results/reproducibility/artifact_environment.json)
records the separate document-rendering runtime and exact library versions.
The development runtime comparison reproduced both checked demand files,
crossing events and metrics byte for byte; its scope is documented in
[`runtime_comparison.json`](results/reproducibility/runtime_comparison.json).

## Reproduce the baseline and measurement milestone

Run from the repository root, after making TraCI available:

```powershell
python -m unittest discover -s tests -v
python -m src.traffic_control.cli --controller fixed --seed 38 --junctions 1
python -m src.traffic_control.cli --controller actuated --seed 38 --junctions 3
python -m src.traffic_control.cli --controller pressure --seed 38 --junctions 3
python -m src.traffic_control.measurement_resolution
```

Each CLI run saves `metrics.json`, `crossing_events.json`, its deterministic demand file, and SUMO trip output under `runs/`. Use `--output runs/my-run` to choose a directory and `--slow-pedestrians` for the slow-walker check. The tests cross-check tracker timestamps against a live TraCI position trace and inspect the SUMO conflict matrix; if SUMO is absent, integration tests are skipped and the output says so.

The resolution audit observes the same 0.1-second SUMO trajectory at both 0.1-second and one-second intervals. Its hand trace covers immediate entry, a queue and an unserved request. In the recorded development run, 30 entered events had a mean absolute wait difference of 0.49 s and a maximum of 1.30 s; both trackers retained two pending requests. These fixture results are not a universal error bound. The final report requires this audit's compact outputs in `results/measurement-resolution/`.

## Full research protocol

The primary comparison has three conventional controllers and IPPO/MAPPO trained with pedestrian reward weights **0, 3, and 10**. Weights 0 and 3 reproduce the vehicle-focused and pedestrian-aware comparison; weight 10 probes the trade-off. Five independent initialisation/training seeds are specified for each PPO method and weight, with 150 episodes per replica and checkpoint selection on separate validation seeds. The frozen [validation extension rule](docs/validation_extension_rule.md) decides whether **every** configuration must instead receive 300 episodes. Do not select checkpoints using final-test seeds.

The locked evaluation uses ten new demand seeds, **9001–9010**. Earlier seeds 3001–3005, 5001, 6001–6002, 7001–7010, and 8001–8010 were inspected during development or interrupted safety audits and are excluded from the final test. It crosses low/medium/high vehicle and pedestrian loads, then adds slow walkers, a 20-person concentrated pedestrian burst, overload, and short junction spacing with turning traffic. Runs with the same scenario and seed share a demand SHA-256 across controllers. The primary result is the medium-vehicle, medium-pedestrian case. Use these commands for the full protocol with at most four independent SUMO workers:

```powershell
python -m src.traffic_control.train_parallel --episodes 150 --output runs/training-v3 --model-dir models/v3 --workers 4
python -m src.traffic_control.validation_decision
python -m src.traffic_control.evaluate_parallel --workers 4
```

For a single resumable experiment command, run `python scripts/finish_study.py`.
It completes training, applies the validation-only budget rule, evaluates the locked
seeds and runs the separate Singapore transfer test. It no longer generates reports
or audits report artifacts. Stage logs are under `runs/final-pipeline/`.
Use `--wait-for-training` only when initial training is already running; run one
copy at a time. The script checks saved training-source hashes before resuming.

Report figures use Matplotlib and export both SVG and PNG. The local installation
keeps its plotting packages in `.local/report-python/` (about 126 MB) and records
the matching Python executable in `.local/report-python-runtime.txt`. A fresh
checkout can instead install `requirements-report.txt` into its normal Python
environment. Simulation and PPO training still need only TraCI and the standard
library.

Run final evaluation only after inspecting the new `runs/training-v3/extension_decision.json`. If `extend_all_to_300` is true, train all 30 models afresh with `python -m src.traffic_control.train_parallel --episodes 300 --output runs/training-v4 --model-dir models/v4 --workers 4`, then evaluate with `python -m src.traffic_control.evaluate_parallel --model-dir models/v4 --output runs/evaluation-v5 --workers 4`. Keep the 150-episode artifacts for the learning-curve comparison; do not mix model budgets in one final table.

For a machine with limited memory or CPU, reduce `--workers` to 1 or 2. The sequential alternatives are `python -m src.traffic_control.train` and `python -m src.traffic_control.evaluate`; do not run them simultaneously with the parallel commands against the same output folders. Completed replica and scenario-seed checkpoints can be reused after an interruption.

Training writes replica models and `training_manifest.json` to `models/v3/`, with episode and validation logs in `runs/training-v3/`. Evaluation writes paired run, crossing, event, action, and phase tables plus bootstrap summaries in `runs/evaluation-v4/`; completed scenario/seed groups can be reused on restart only while model, protocol, simulation-source, and milestone-configuration hashes match. It prepares each scenario/seed's network and routes once, then reuses those read-only inputs across the matched controllers. The full protocol is computationally substantial; keep a copy of the configuration and manifest used for any reported result.

The experimental report and PDF generators have been removed. Prepare the report from the saved evaluation outputs; existing historical artifacts remain available.

For a **separate pilot** that cannot be mistaken for the full study:

```powershell
python -m src.traffic_control.train --smoke
python -m src.traffic_control.evaluate --smoke
```

The pilot uses one training seed, one episode, one weight, one scenario, and one evaluation seed. Its outputs remain under `runs/` and must be labelled incomplete.

## Watch the cars and walkers

On this Windows computer, double-click **`Watch-Comparison.cmd`**. It makes the
portable SUMO tools available and opens the fixed-time and pedestrian-aware IPPO
windows with identical demand. It automatically uses the validation-selected
current model directory. The first launch prepares the source runs; subsequent
launches reuse them while their model and safety hashes match. You can zoom into
a crossing or use the SUMO play/pause controls. The quantitative result remains
the repeated evaluation below, not this single illustrative replay.

The viewer shows a fixed-time run before RL next to the selected weight-3 replica-0 policy with **the same medium-demand scenario and exploratory seed**:

```powershell
python -m src.traffic_control.viewer --compare --model-dir models/v3 --seed 6001 --pedestrian-count 120
```

If the frozen validation rule selects 300 episodes, use `--model-dir models/v4` after all replicas have been retrained. Seed 6001 is excluded from the locked test and the GUI is illustrative, even with final-budget models. `--model-dir models/v2` and `--model-dir models` select older **exploratory** policies from superseded safety protocols; their replays remain useful for a historical demo but are not final evidence. `--rl-controller mappo-pedestrian` chooses the other pedestrian-aware method. The viewer replays each recorded legal phase history in SUMO GUI; it does not train a policy or replace numerical evaluation. Use `--controller actuated`, `--controller pressure`, or `--controller ippo-pedestrian` to open one window; `--delay-ms 500` slows playback. Zoom into a junction to see the pedestrians. GUI files are under `runs/gui/`.

For the Singapore-area map audit and the separate safe, NUS-derived simplified corridor:

```powershell
python -m src.traffic_control.singapore --smoke
python -m src.traffic_control.singapore --derived-baselines
```

The raw OpenStreetMap import has crossing edges but **none under signal control**, so it is unsuitable for the common protected-crossing controller comparison. The derived corridor connects **approximate candidate-site centroids** whose straight-line gaps are 161 m and 196 m; candidate identities, connectivity and physical road distances need manual verification. Its straight connectors, side arms, lanes, sidewalk widths, crossings, phases, and all arrivals are project assumptions. See [Singapore pilot](docs/singapore.md) for exact provenance, GUI commands, audit results, and the data needed for a calibrated follow-up.

After the validation-selected training budget is complete, rerun the same simplified corridor as a **single-seed candidate-centroid geometry and left-hand-drive sensitivity pilot** with the final-budget replica-zero IPPO and MAPPO policies alongside the baselines. For the 150-episode selection:

```powershell
python -m src.traffic_control.singapore --derived-baselines --model-dir models/v3 --output runs/singapore-transfer-final-150
```

If validation selects 300 episodes, use `python -m src.traffic_control.singapore --derived-baselines --model-dir models/v4 --output runs/singapore-transfer-final-300` instead. The previous `models/v2` pilot is preserved only as an exploratory interface check; those policies were trained under superseded safety rules even when replayed through the revised runtime. The final-budget rerun still uses synthetic demand, one seed, approximate candidate sites, and assumed phases; it is **not** an observed-data Singapore evaluation or a calibrated transfer result. Its own `derived-baselines.json` carries the model paths and hashes.

For a **repeated** test on that same simplified layout, the separate transfer protocol uses ten new seeds (9101–9110), all 30 final-budget learned replicas, the three conventional controllers, paired uncertainty, and an entry count at every crossing. Run it **after the main locked evaluation is complete** and the validation-selected model budget is frozen. For the 150-episode selection:

```powershell
python -m src.traffic_control.singapore_transfer --model-dir models/v3 --output results/singapore-transfer
```

If validation selects 300 episodes, use `python -m src.traffic_control.singapore_transfer --model-dir models/v4 --output results/singapore-transfer` instead. The output includes `REPORT.md`, per-run and per-crossing CSVs, and compressed event records. Keep it separate from the main locked evaluation and the old v2 pilot. Repetition makes uncertainty visible but **does not calibrate** turning counts, pedestrian arrivals, actual site connectivity, crossing locations, or deployed signal plans. Sparse crossing tails remain inconclusive. See [Singapore pilot](docs/singapore.md) for the audit and specific limitations.

For presentation stills from the revised policy, run `python -m src.traffic_control.capture_gui --controllers fixed ippo-pedestrian --model-dir models/v3 --seed 6001 --second 147 --boundary 240 70 320 150 --output deliverables/presentation/assets/final-protocol` (use `models/v4` after a 300-episode extension). The final visual comparison uses weight 3, replica 0, selected before inspecting test outcomes. The checked [presentation preview](deliverables/presentation/output/team38-research-talk-v2k-preflight-preview.pptx) contains older v2 **illustrative** images and a final-results placeholder. The deck builder writes `deliverables/presentation/output/team38-research-talk-v2f-final.pptx` once the complete final study passes its data gate.

To record a 10-second visual comparison, run `python -m src.traffic_control.capture_clip --controllers fixed ippo-pedestrian --model-dir models/v3 --seed 6001 --start-second 138 --frames 20 --fps 2 --isolated-frames --output deliverables/presentation/assets/final-protocol` (use `models/v4` after extension). The clip shows 20 simulation seconds at twice real-time speed; pedestrian icons are enlarged three times for visibility without changing the simulation. This requires `ffmpeg` on `PATH` and an interactive desktop: Windows sandbox sessions can produce blank GUI captures. The older clips outside `assets/final-protocol/` belong to superseded v2 models and are exploratory only.

Once the study and PDF finish, `python scripts/finish_visuals.py --node <node-executable>` creates both current-budget stills, both clips, a side-by-side MP4 (fixed time left; IPPO with pedestrian weight 3 right), and the final presentation. It needs the bundled presentation runtime; see [the presenter guide](deliverables/presentation/PRESENTER_GUIDE.md) for individual commands. Use `--wait-for-study` only when `scripts/finish_study.py` is already running. Render the resulting deck with `node scripts/render_presentation.mjs` and inspect every slide before submission.

## Signal rules and measurements

The common controller cycles through east–west, north–south, and pedestrian groups. Vehicle green lasts at least 12 s. Changes use 3 s yellow and 2 s all-red; pedestrian green lasts 12 s, followed by an all-red clearance calculated from crossing length, a design speed of 0.8 m/s, and a 3 s buffer. Clearance extends while a person remains on a crossing and requires three consecutive empty crossing observations before vehicle green. The network's SUMO `foes` matrix is checked against every phase; observed states are checked during simulation. These are experimental safety constraints, **not** certified Singapore timing plans. The default corridor uses straight vehicle trips; turn movements are added in the stress cases only where the constructed phases allow them.

The crossing tracker records each person's first observed arrival within **1.2 m of the crossing-lane endpoint**, or the beginning of the first stopped interval on the immediate approach when a queue forms farther back. Entry is the first step on the crossing edge. An entry with no observed approach is tagged and assigned zero observed wait. Each event includes its source and timestamps; a reached crossing that has not been entered by the horizon remains unserved. TraCI's built-in person `waitingTime` is only an approach-queue trigger, **not** the reported complete crossing wait. One-second steps impose about one-step timestamp resolution. Report the mean, nearest-rank p95, entry count, and unserved count **separately for all 12 crossings**; sparse p95 estimates remain inconclusive.

For vehicles, scheduled trips are reconciled with departures and arrivals. Completed-trip mean and total `timeLoss` come from SUMO TripInfo; report them beside scheduled, completed, unfinished, and uninserted counts. A faster mean among only completed vehicles is not a gain if service deteriorates. The strict primary test compares every crossing's pooled p95 with pedestrian-actuated control, plus vehicle loss and service counts. It reports the largest crossing-level difference and paired uncertainty over demand seeds and independent training runs. A descriptive pass requires lower vehicle loss, no crossing with a higher p95 point estimate, and no deterioration in service; a stronger claim additionally needs supporting paired intervals and adequate crossing entries.

## Files and research notes

| Path | Purpose |
| --- | --- |
| `scenario/single.*.xml`, `src/traffic_control/connected.py`, `scenario.py` | New single and connected networks; seeded vehicle/pedestrian demand |
| `src/traffic_control/signals.py`, `cli.py` | Shared phase rules, conflict checks, three baselines, simulation interface |
| `src/traffic_control/metrics.py` | Crossing events and vehicle service accounting |
| `src/traffic_control/ppo.py`, `train.py`, `train_parallel.py`, `evaluate.py`, `evaluate_parallel.py`, `report.py` | Learning, held-out evaluation, uncertainty, tables and figures |
| `src/traffic_control/viewer.py` | Matched-demand graphical phase replay |
| `src/traffic_control/capture_gui.py`, `deliverables/presentation/` | Original matched-seed GUI stills and presentation scaffold |
| `src/traffic_control/singapore.py`, `singapore_transfer.py`, `config/singapore*.json`, `scenario/singapore/` | Singapore map/data audit, simplified pilot and repeated uncalibrated transfer test |
| `docs/literature.md`, `docs/measurement_safety_audit.md`, `docs/course_report.md` | Prior-work positioning, audited measurement and safety assumptions, and the course report draft |
| `tests/` | Hand-checkable measurement, deterministic demand, phase safety, and SUMO smoke tests |

Relevant official SUMO documentation: [pedestrians](https://sumo.dlr.de/docs/Simulation/Pedestrians.html), [TraCI pedestrian-crossing tutorial](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html), [TripInfo and unfinished trips](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html), and [OpenStreetMap import](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html).
