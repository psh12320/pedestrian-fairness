# Cluster data-volume diagnostic review

Both saved-policy experiments completed. All 12 method/critic-initialisation gates pass at 512 updates: the 32-trajectory arm improves on four trajectories and beats both constant predictors. This is a positive value-prediction finding; no traffic actor was updated.

The archive contains 12 fit-job files covering 72 fresh critic fits, 216 evaluation rows and 2,160 assessments. ZIP integrity, exact member coverage, manifest/source/protocol/actor hashes, seed splits, identical initial predictions, scalar metric identities, sample counts, summaries, individual cases, gates and paired intervals passed checks. The protocol calls for 80 short SUMO trajectories. Raw trajectories and actor weights are absent: this verifies scalar results, not a replay of those simulations.

## Matched 4 versus 32 trajectories at 512 updates

| Frozen behavior | Critic inputs | Check MSE reduction | Exploratory 95% interval for reduction | Initialisation gates | Individual cases improved | Cases below own constant |
|---|---|---:|---|---:|---:|---:|
| plain-mappo-r0 | Local | 41.6% | [28.3%, 49.2%] | 3/3 | 60/72 | 69/72 |
| plain-mappo-r0 | Network-wide | 55.9% | [44.4%, 62.6%] | 3/3 | 69/72 | 71/72 |
| priority-mappo-r0 | Local | 28.0% | [16.9%, 37.1%] | 3/3 | 60/72 | 69/72 |
| priority-mappo-r0 | Network-wide | 37.8% | [24.6%, 46.1%] | 3/3 | 64/72 | 66/72 |

Each set of 72 individual cases is three junctions × eight check demand seeds × three critic initialisations. They share data and are correlated. The intervals resample whole check seeds and critic initialisations, conditional on the two fixed training datasets; they exclude training-dataset sampling variation and are exploratory. Error reductions are squared prediction errors in reward units, not percentages of vehicle or pedestrian waiting time.

## All junctions, including remaining failures

| Behavior | Inputs | Junction | MSE32 / MSE4 | MSE32 / own constant |
|---|---|---|---:|---:|
| plain-mappo-r0 | ippo | C0 | 0.634 | 0.647 |
| plain-mappo-r0 | ippo | C1 | 0.492 | 0.581 |
| plain-mappo-r0 | ippo | C2 | 0.687 | 0.625 |
| plain-mappo-r0 | mappo | C0 | 0.527 | 0.654 |
| plain-mappo-r0 | mappo | C1 | 0.354 | 0.569 |
| plain-mappo-r0 | mappo | C2 | 0.499 | 0.692 |
| priority-mappo-r0 | ippo | C0 | 0.695 | 0.452 |
| priority-mappo-r0 | ippo | C1 | 0.631 | 0.609 |
| priority-mappo-r0 | ippo | C2 | 0.889 | 0.498 |
| priority-mappo-r0 | mappo | C0 | 0.487 | 0.512 |
| priority-mappo-r0 | mappo | C1 | 0.665 | 0.659 |
| priority-mappo-r0 | mappo | C2 | 0.676 | 0.532 |

All 12 behavior/input/junction averages improve and beat their own constant reference. Some individual demand/initialisation cases still worsen or fail to beat the constant; retain all cases in summary.json. These are junction-level prediction checks, not the proposal's 12 crossing-level waiting-time checks.

## Train versus check errors

| Behavior | Inputs | Trajectories | Updates | Train RMSE | Check RMSE | Check bias | Check MSE / own constant |
|---|---|---:|---:|---:|---:|---:|---:|
| plain-mappo-r0 | ippo | 4 | 128 | 34.06 | 49.76 | 10.56 | 1.033 |
| plain-mappo-r0 | ippo | 4 | 512 | 25.01 | 46.97 | 10.38 | 0.921 |
| plain-mappo-r0 | ippo | 32 | 128 | 42.94 | 42.12 | -0.40 | 0.846 |
| plain-mappo-r0 | ippo | 32 | 512 | 33.87 | 35.89 | -0.22 | 0.614 |
| plain-mappo-r0 | mappo | 4 | 128 | 26.15 | 54.88 | 15.67 | 1.257 |
| plain-mappo-r0 | mappo | 4 | 512 | 9.58 | 54.76 | 18.08 | 1.252 |
| plain-mappo-r0 | mappo | 32 | 128 | 41.09 | 43.13 | 1.26 | 0.887 |
| plain-mappo-r0 | mappo | 32 | 512 | 25.68 | 36.38 | 0.87 | 0.631 |
| priority-mappo-r0 | ippo | 4 | 128 | 36.51 | 53.82 | 12.31 | 0.874 |
| priority-mappo-r0 | ippo | 4 | 512 | 21.50 | 46.81 | 10.37 | 0.661 |
| priority-mappo-r0 | ippo | 32 | 128 | 43.37 | 47.87 | 4.47 | 0.770 |
| priority-mappo-r0 | ippo | 32 | 512 | 34.32 | 39.72 | 3.22 | 0.530 |
| priority-mappo-r0 | mappo | 4 | 128 | 28.97 | 54.81 | 13.48 | 0.906 |
| priority-mappo-r0 | mappo | 4 | 512 | 9.98 | 52.52 | 16.16 | 0.832 |
| priority-mappo-r0 | mappo | 32 | 128 | 41.07 | 47.47 | 4.98 | 0.757 |
| priority-mappo-r0 | mappo | 32 | 512 | 26.71 | 41.41 | 3.61 | 0.576 |

The larger dataset reduces the train/check gap and prediction bias, especially for the network-wide critic. Network-wide inputs still do not outperform local inputs on average: with 32 trajectories, check RMSE is 36.38 versus 35.89 under plain behavior and 41.41 versus 39.72 under priority behavior. These are critic architectures fitted on the same saved MAPPO actor data, not an IPPO-versus-MAPPO traffic ranking.

## What the experiment supports

Within this diagnostic, four fixed trajectories gave a poor basis for generalisation. More independent simulation runs, at the same optimiser update count, improved prediction on eight new demand seeds. At plain C1, the training target mean shifts from -96.15 with four runs to -123.11 with 32; the new check mean is -126.87. This is consistent with better training coverage. It does not prove a unique cause of the original PPO failures.

Both arms already use normalised outputs and remaining episode time, and share a transform fitted only to the four-run subset. The larger arm processes eight times as many states per update. Thus this is not an equal-compute or equal-sample-exposure experiment. No new claim separating sample diversity from exposure is supported. The original PPO study generated fresh episodes over training; it did not train on just four trajectories.

The new check seeds differ from the previous horizon diagnostic. Do not compare its numbers directly with these to infer a treatment effect; use the matched four-run control in this archive. In particular the four-run priority/network critic already beats its constant on this new check set. The matched improvement, rather than a before/after pass label, is the evidence.

## Next bounded experiment

Move from offline critic diagnosis to an on-policy pilot. Compare the existing PPO formulation against verified value-output normalisation plus critic remaining-time inputs, then separately test multiple independent rollout seeds collected under the same frozen actor before each update. Preserve local actor observations, categorical sampling at evaluation, rewards, common phase safety, and the original controller checkpoints. Use equal total SUMO transitions and report both actor-update counts and compute; fitting 32 offline trajectories is not a reason to prescribe a 32-episode PPO batch.

Use repeated training initialisations and separate development demand seeds. Keep ordinary and service-priority logic separate so any effect is attributable. Compare against pedestrian-actuated and uniform-choice controls. The decision metrics remain vehicle time loss, completed/unfinished/uninserted vehicles, per-crossing mean/p95 and unresolved requests. Report crossings helped out of 12 as a secondary measure; the original every-crossing criterion stays unchanged.

Do not claim a solution or launch a full study solely because this diagnostic passed. First demonstrate an actual vehicle/pedestrian improvement in the bounded pilot; then freeze the selected configuration and run held-out tests. All previous negative outcomes remain part of the study.
