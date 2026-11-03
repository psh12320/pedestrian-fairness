# Cluster critic-check review

All four diagnostic reports and 24 fit-job files are present, covering 144 fitted critics. ZIP integrity, exact member coverage, source/protocol hashes, seed splits, metric identities and all reported averages/gates passed checks. The protocol describes 16 short simulations. Raw trajectory tensors and saved actor weights are absent, so this review verifies the supplied statistics and their aggregation rather than recomputing predictions from raw samples.

## Decision

Remaining-time information helps on both learned behavior policies. The local-input critics pass the predeclared average gate in all three initialisations for each policy. The network-wide critics improve relative to their no-clock controls but fail the constant-predictor comparison in every initialisation. Therefore neither complete horizon diagnostic passes its all-methods gate, and neither scaling-only diagnostic passes. The earlier local success on actuated-generated trajectories did not fully carry over.

## Matched remaining-time comparison at 512 updates

Both arms already use normalised value outputs. Error is MSE divided by the error of predicting the training-set mean on the same check data. Below 1 is better than that constant reference; these are reward-prediction errors, not traffic delays.

| Frozen behavior | Critic inputs | No clock | Clock | Error reduction vs no clock | Initialisation gates | Individual cases beating constant |
|---|---|---:|---:|---:|---:|---:|
| plain-mappo-r0 | Local | 1.196 | 0.833 | 30.3% | 3/3 | 15/18 |
| plain-mappo-r0 | Network-wide | 1.503 | 1.123 | 25.3% | 0/3 | 6/18 |
| priority-mappo-r0 | Local | 1.373 | 0.833 | 39.3% | 3/3 | 16/18 |
| priority-mappo-r0 | Network-wide | 1.733 | 1.236 | 28.7% | 0/3 | 8/18 |

The 18 cases per row are three critic initialisations x three junctions x two check demand seeds. They share trajectories and are not 18 independent statistical trials. These critics were fitted afresh; both behavior actors are MAPPO replica 0. The table is not evidence that trained IPPO traffic control beats MAPPO.

## Scaling-only experiment at 512 updates

| Behavior | Inputs | Raw-output error ratio | Normalised-output error ratio |
|---|---|---:|---:|
| plain-mappo-r0 | ippo | 5.450 | 1.443 |
| plain-mappo-r0 | mappo | 5.495 | 1.434 |
| priority-mappo-r0 | ippo | 4.577 | 1.042 |
| priority-mappo-r0 | mappo | 4.612 | 1.475 |

Normalisation removes much of the raw-output bias, but it is not sufficient to learn a reliable state-dependent value function on these four training trajectories. The scaling and horizon experiments use different check seed pairs (49201/49202 and 49401/49402); do not combine their numbers into a single raw-versus-clock effect estimate.

## Statistics hidden by the averages

| Behavior | Inputs | Junction | Clock error ratio | Training target mean | Check target mean | Prediction bias |
|---|---|---|---:|---:|---:|---:|
| plain-mappo-r0 | ippo | C0 | 0.809 | -115.80 | -124.31 | 9.79 |
| plain-mappo-r0 | ippo | C1 | 1.066 | -96.15 | -140.29 | 44.11 |
| plain-mappo-r0 | ippo | C2 | 0.625 | -116.41 | -125.59 | 11.12 |
| plain-mappo-r0 | mappo | C0 | 1.033 | -115.80 | -124.31 | 18.34 |
| plain-mappo-r0 | mappo | C1 | 1.455 | -96.15 | -140.29 | 62.13 |
| plain-mappo-r0 | mappo | C2 | 0.881 | -116.41 | -125.59 | 15.00 |
| priority-mappo-r0 | ippo | C0 | 0.720 | -110.95 | -122.07 | 4.66 |
| priority-mappo-r0 | ippo | C1 | 0.769 | -102.16 | -148.95 | 37.83 |
| priority-mappo-r0 | ippo | C2 | 1.011 | -130.49 | -149.78 | 29.17 |
| priority-mappo-r0 | mappo | C0 | 1.808 | -110.95 | -122.07 | 18.06 |
| priority-mappo-r0 | mappo | C1 | 0.737 | -102.16 | -148.95 | 39.50 |
| priority-mappo-r0 | mappo | C2 | 1.164 | -130.49 | -149.78 | 42.31 |

Local critics still have failures: C1 under plain behavior has a pooled error ratio of 1.066; C2 under priority behavior is 1.011. For network-wide critics, plain C1 is 1.455 and priority C0 is 1.808. All individual failures are retained in summary.json.

The network-wide clock critic fits its training data closely (ratios about 0.050 and 0.049) while its check ratios are 1.123 and 1.236. This is a strong training/generalisation gap. More updates on the same four trajectories do not resolve it. The check trajectories also have different return distributions: at C1 the mean target changes from -96.15 to -140.29 for plain behavior, and -102.16 to -148.95 for priority behavior. Larger input/capacity, limited independent trajectories, stochastic outcomes and distribution differences are plausible contributors; these experiments do not identify one unique cause.

## What follows

The scale and remaining-time mechanisms justify a controlled repair, but the combined implementation is not yet validated for both critic types. Do not launch another unchanged 512-episode traffic study or claim the traffic target has been met.

The next narrow prediction test should address generalisation: increase independent training trajectories while keeping the architecture, target definition and update budget matched, and evaluate on a newly reserved development check set. A simple constant or regularised predictor should remain a comparator. If simplifying the central critic is tested, separate that from the data-volume change. Improving the number of nearly adjacent states is not equivalent to adding independent traffic seeds.

For an eventual on-policy PPO pilot, separate raw/no-clock, normalised/no-clock and normalised/clock variants under equal actor settings, rewards, demand and safety rules. These offline fits use only four trajectories, whereas original PPO generated new traffic episodes throughout training; do not infer that the original 512-episode study trained on only four trajectories. Improved offline values do not guarantee an actor improvement. All original vehicle/pedestrian results remain unchanged.

Keep the deadline in view: preserve these results as diagnosis and avoid open-ended method searching. A final paper should distinguish demonstrated evaluation-mode effects, value-prediction findings, and actual traffic-control outcomes. No final evaluation seeds have been consumed by these checks.
