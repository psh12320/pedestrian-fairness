# Guided exploration development study

Engineering smoke: **False**. 12 trained models; 3 imitation-only actors. Three junctions, 12 crossings; identical physical phases and clearance.

Control = previous repaired PPO. Age = additional bounded quadratic oldest-wait cost at each crossing. Guided = age plus actuated actor pretraining and decaying online teacher cross entropy. RND = age plus local training-only novelty. These are independent adaptations, not reproductions of the cited papers.

Imitation-only outcomes and validation fidelity are included even if poor. No replica is discarded and fidelity does not choose checkpoints. Guidance adds demonstration and actor fitting work; RND adds predictor updates. PPO simulation steps, 4 PPO passes per episode and final episode are matched.

All intervals are exploratory paired bootstrap intervals across whole training replicas and demand seeds. One action stream per demand. Crossing waits are pooled across seeds within a replica before averaging replica p95. Sparse estimates (<100 entries) are inconclusive. A descriptive pass is not a statistical guarantee or a real-world benefit.

## vmedium-pmedium

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 32.76 | 921.50 | 38.17 | 0.33 | 12.67 | 0.00 |
| actuated | 50.55 | 919.33 | 40.33 | 0.33 | 3.33 | 0.00 |
| pressure | 39.67 | 923.83 | 35.83 | 0.33 | 3.83 | 0.00 |
| uniform | 49.16 | 921.00 | 38.50 | 0.50 | 12.00 | 0.00 |
| mappo-control | 40.61 | 924.50 | 35.17 | 0.33 | 6.11 | 0.00 |
| mappo-age | 43.45 | 926.56 | 33.11 | 0.33 | 5.39 | 0.00 |
| mappo-guided | 49.57 | 920.17 | 39.50 | 0.33 | 3.44 | 0.00 |
| mappo-rnd | 43.30 | 925.11 | 34.56 | 0.33 | 6.11 | 0.00 |
| imitation | 51.22 | 921.28 | 38.39 | 0.33 | 3.83 | 0.00 |

| Model vs actuated | Vehicle delta s | 95% exploratory interval | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---:|---|---:|---:|---:|---:|---|
| mappo-control | -9.95 | [-11.849548657802654, -8.522804605793166] | 0/12 | 17.87 | 18/18 | 258 | False; sparse=False |
| mappo-age | -7.10 | [-8.68591841532265, -5.335572723912123] | 0/12 | 15.93 | 16/18 | 259 | False; sparse=False |
| mappo-guided | -0.99 | [-2.1873131187947465, 0.29030483079761166] | 3/12 | 1.53 | 13/18 | 260 | False; sparse=False |
| mappo-rnd | -7.25 | [-9.169707177439376, -5.703478196257092] | 0/12 | 14.40 | 18/18 | 258 | False; sparse=False |
| imitation | 0.66 | [-0.5701526571835243, 1.7651962974229063] | 7/12 | 1.07 | 13/18 | 259 | False; sparse=False |

Full crossing means, p95s, counts, service accounting and component ablations are in summary.json. Per-run waits, speed groups, phases and actor diagnostics remain in evaluation/*.json.

## pedestrian-burst

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 32.76 | 921.50 | 38.17 | 0.33 | 12.67 | 0.00 |
| actuated | 50.55 | 919.33 | 40.33 | 0.33 | 3.33 | 0.00 |
| pressure | 39.47 | 925.33 | 34.33 | 0.33 | 4.83 | 0.00 |
| uniform | 49.16 | 921.00 | 38.50 | 0.50 | 12.00 | 0.00 |
| mappo-control | 40.89 | 924.22 | 35.44 | 0.33 | 6.33 | 0.00 |
| mappo-age | 43.55 | 924.67 | 35.00 | 0.33 | 5.83 | 0.00 |
| mappo-guided | 49.47 | 920.50 | 39.17 | 0.33 | 3.50 | 0.00 |
| mappo-rnd | 43.08 | 924.94 | 34.72 | 0.33 | 5.67 | 0.00 |
| imitation | 51.20 | 921.33 | 38.33 | 0.33 | 3.83 | 0.00 |

| Model vs actuated | Vehicle delta s | 95% exploratory interval | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---:|---|---:|---:|---:|---:|---|
| mappo-control | -9.67 | [-11.646672733432883, -8.095521774486942] | 0/12 | 17.13 | 17/18 | 258 | False; sparse=False |
| mappo-age | -7.01 | [-8.68591841532265, -5.163101936855101] | 0/12 | 16.00 | 17/18 | 259 | False; sparse=False |
| mappo-guided | -1.08 | [-2.211954393113622, 0.08129043002080473] | 4/12 | 1.40 | 12/18 | 260 | False; sparse=False |
| mappo-rnd | -7.47 | [-9.594180583521794, -5.840098997502169] | 0/12 | 14.53 | 18/18 | 256 | False; sparse=False |
| imitation | 0.65 | [-0.6201537528808316, 1.7651962974229063] | 7/12 | 0.60 | 13/18 | 259 | False; sparse=False |

Full crossing means, p95s, counts, service accounting and component ablations are in summary.json. Per-run waits, speed groups, phases and actor diagnostics remain in evaluation/*.json.

## short-spacing

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 38.27 | 2277.83 | 82.67 | 31.00 | 13.17 | 0.00 |
| actuated | 43.13 | 780.67 | 72.83 | 1538.00 | 1.50 | 0.00 |
| pressure | 47.38 | 1671.00 | 76.33 | 644.17 | 32.83 | 21.67 |
| uniform | 91.38 | 1487.17 | 103.50 | 800.83 | 13.00 | 0.00 |
| mappo-control | 63.99 | 1701.44 | 92.22 | 597.83 | 6.17 | 0.00 |
| mappo-age | 61.35 | 1528.33 | 90.22 | 772.94 | 10.33 | 4.22 |
| mappo-guided | 47.11 | 805.39 | 65.61 | 1520.50 | 17.72 | 13.06 |
| mappo-rnd | 64.34 | 1558.33 | 89.22 | 743.94 | 6.33 | 0.00 |
| imitation | 47.86 | 852.17 | 67.72 | 1471.61 | 4.72 | 0.00 |

| Model vs actuated | Vehicle delta s | 95% exploratory interval | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---:|---|---:|---:|---:|---:|---|
| mappo-control | 20.86 | [7.5641562124540815, 34.15008188929625] | 0/12 | 29.53 | 18/18 | 295 | False; sparse=False |
| mappo-age | 18.22 | [7.098983873102181, 28.9773628984897] | 0/12 | 138.47 | 18/18 | 268 | False; sparse=False |
| mappo-guided | 3.98 | [-4.618079377822929, 11.474976004012001] | 5/12 | 1.20 | 18/18 | 245 | False; sparse=False |
| mappo-rnd | 21.21 | [10.084014734451303, 32.85868234239661] | 0/12 | 19.87 | 18/18 | 295 | False; sparse=False |
| imitation | 4.74 | [-2.6251458772844334, 15.512513937759405] | 7/12 | 0.87 | 17/18 | 294 | False; sparse=False |

Full crossing means, p95s, counts, service accounting and component ablations are in summary.json. Per-run waits, speed groups, phases and actor diagnostics remain in evaluation/*.json.

## slow-walkers

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 32.77 | 921.50 | 38.17 | 0.33 | 20.50 | 0.00 |
| actuated | 51.57 | 920.33 | 39.33 | 0.33 | 6.50 | 0.00 |
| pressure | 44.60 | 922.50 | 37.17 | 0.33 | 6.00 | 0.00 |
| uniform | 49.16 | 921.00 | 38.50 | 0.50 | 14.67 | 0.00 |
| mappo-control | 42.77 | 924.17 | 35.50 | 0.33 | 8.78 | 0.00 |
| mappo-age | 44.72 | 921.89 | 37.78 | 0.33 | 7.33 | 0.00 |
| mappo-guided | 52.06 | 917.78 | 41.89 | 0.33 | 7.89 | 0.00 |
| mappo-rnd | 45.56 | 922.83 | 36.83 | 0.33 | 8.17 | 0.00 |
| imitation | 51.46 | 917.89 | 41.78 | 0.33 | 6.50 | 0.00 |

| Model vs actuated | Vehicle delta s | 95% exploratory interval | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---:|---|---:|---:|---:|---:|---|
| mappo-control | -8.80 | [-10.716063991227408, -6.43101465125273] | 0/12 | 21.53 | 18/18 | 257 | False; sparse=False |
| mappo-age | -6.85 | [-8.787065666852753, -4.783381904537175] | 0/12 | 17.07 | 16/18 | 254 | False; sparse=False |
| mappo-guided | 0.49 | [-0.6702547924035956, 1.6939141144693721] | 7/12 | 0.40 | 18/18 | 257 | False; sparse=False |
| mappo-rnd | -6.01 | [-8.08596316279764, -3.811566946983865] | 0/12 | 17.47 | 16/18 | 257 | False; sparse=False |
| imitation | -0.11 | [-1.1292708962072497, 0.9285382084065428] | 8/12 | 0.60 | 16/18 | 255 | False; sparse=False |

Full crossing means, p95s, counts, service accounting and component ablations are in summary.json. Per-run waits, speed groups, phases and actor diagnostics remain in evaluation/*.json.

## Limits

Synthetic demand, exclusive pedestrian phase, four development settings, three initialisations. Short spacing also increases demand. No final test seeds used. Traffic outcomes, not reward or teacher accuracy, determine success. QMIX/COMA have not been tested by this study.
