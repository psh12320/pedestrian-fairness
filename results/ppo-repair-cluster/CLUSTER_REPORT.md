# On-policy PPO repair pilot

Engineering smoke only: **False**. Models completed: 18. Final episode checkpoints; no best-run selection. All training demand hashes match across methods/variants within each replica. Evaluation uses categorical sampling, three junctions, 12 crossings and unchanged safety.

A = original PPO; B = normalised critic with remaining time; C = B with four independent rollouts per update. All use pedestrian weight 3. This follow-up supplements the original four-controller reward comparison.

Every confidence interval below is exploratory: resample whole training replicas and demand seeds together, preserving paired references. A single action-sampling stream is used per demand; action randomness is not separately replicated. Crossing p95 is pooled across demand seeds within each replica, then averaged across replicas. These are not averages of individual run p95s. Sparse crossing estimates (<100 entries in any replica or baseline) are inconclusive. No confirmatory test or final seeds are used.

## vmedium-pmedium

| Controller | Vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved after followup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.29 | 957.33 | 924.33 | 32.83 | 0.17 | 11.00 | 0.00 |
| actuated | 50.78 | 957.33 | 919.50 | 37.67 | 0.17 | 3.33 | 0.00 |
| pressure | 40.16 | 957.33 | 928.67 | 28.50 | 0.17 | 4.83 | 0.00 |
| uniform | 49.33 | 957.33 | 918.50 | 37.67 | 1.17 | 9.00 | 0.00 |
| ippo-original | 54.20 | 957.33 | 919.61 | 37.56 | 0.17 | 5.72 | 0.00 |
| ippo-normalized-clock | 52.19 | 957.33 | 920.44 | 36.56 | 0.33 | 4.39 | 0.00 |
| ippo-normalized-clock-batch4 | 54.29 | 957.33 | 919.61 | 36.89 | 0.83 | 8.44 | 0.00 |
| mappo-original | 54.24 | 957.33 | 921.78 | 35.33 | 0.22 | 5.00 | 0.00 |
| mappo-normalized-clock | 55.14 | 957.33 | 916.33 | 40.56 | 0.44 | 4.89 | 0.00 |
| mappo-normalized-clock-batch4 | 53.58 | 957.33 | 919.50 | 37.50 | 0.33 | 9.33 | 0.00 |

| Learned controller vs actuated | Vehicle delta s (95% interval) | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---|
| ippo-original | 3.42 ([-0.87, 8.81]) | 0/12 | 26.33 | 17/18 | 274 | False  |
| ippo-normalized-clock | 1.41 ([-1.40, 4.62]) | 0/12 | 26.20 | 15/18 | 274 | False  |
| ippo-normalized-clock-batch4 | 3.51 ([-1.22, 9.16]) | 0/12 | 41.13 | 18/18 | 274 | False  |
| mappo-original | 3.46 ([-1.53, 8.66]) | 0/12 | 27.07 | 17/18 | 272 | False  |
| mappo-normalized-clock | 4.36 ([0.91, 8.21]) | 0/12 | 28.33 | 17/18 | 273 | False  |
| mappo-normalized-clock-batch4 | 2.80 ([-1.23, 8.25]) | 0/12 | 39.20 | 17/18 | 273 | False  |

| Paired training change | Vehicle delta s | Worst crossing p95 delta s | Crossings no worse |
|---|---:|---:|---:|
| ippo:normalized-clock-vs-original | -2.01 | 7.07 | 7/12 |
| ippo:normalized-clock-batch4-vs-normalized-clock | 2.10 | 21.40 | 0/12 |
| mappo:normalized-clock-vs-original | 0.90 | 5.47 | 6/12 |
| mappo:normalized-clock-batch4-vs-normalized-clock | -1.56 | 23.93 | 0/12 |

Full crossing means, p95s, sample counts, pending requests, paired intervals and comparisons against fixed/pressure/uniform controls are in summary.json. Per-run waits, slower-walker groups and diagnostics remain in evaluation JSON.

## pedestrian-burst

| Controller | Vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved after followup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.29 | 957.33 | 924.33 | 32.83 | 0.17 | 11.00 | 0.00 |
| actuated | 50.75 | 957.33 | 919.50 | 37.67 | 0.17 | 3.33 | 0.00 |
| pressure | 40.27 | 957.33 | 929.67 | 27.50 | 0.17 | 4.50 | 0.00 |
| uniform | 49.33 | 957.33 | 918.50 | 37.67 | 1.17 | 9.00 | 0.00 |
| ippo-original | 54.50 | 957.33 | 919.61 | 37.56 | 0.17 | 5.89 | 0.00 |
| ippo-normalized-clock | 52.25 | 957.33 | 920.28 | 36.67 | 0.39 | 4.50 | 0.00 |
| ippo-normalized-clock-batch4 | 54.31 | 957.33 | 920.00 | 36.50 | 0.83 | 8.28 | 0.00 |
| mappo-original | 54.38 | 957.33 | 921.22 | 35.89 | 0.22 | 4.33 | 0.00 |
| mappo-normalized-clock | 55.09 | 957.33 | 916.61 | 40.28 | 0.44 | 4.89 | 0.00 |
| mappo-normalized-clock-batch4 | 53.47 | 957.33 | 919.22 | 37.78 | 0.33 | 9.33 | 0.00 |

| Learned controller vs actuated | Vehicle delta s (95% interval) | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---|
| ippo-original | 3.75 ([-0.55, 9.08]) | 0/12 | 26.33 | 16/18 | 274 | False  |
| ippo-normalized-clock | 1.49 ([-1.49, 4.84]) | 0/12 | 26.13 | 15/18 | 272 | False  |
| ippo-normalized-clock-batch4 | 3.55 ([-1.14, 9.18]) | 0/12 | 41.13 | 18/18 | 274 | False  |
| mappo-original | 3.63 ([-1.60, 9.06]) | 0/12 | 27.27 | 16/18 | 272 | False  |
| mappo-normalized-clock | 4.33 ([0.55, 8.19]) | 0/12 | 24.00 | 17/18 | 273 | False  |
| mappo-normalized-clock-batch4 | 2.72 ([-1.29, 8.17]) | 0/12 | 39.67 | 17/18 | 273 | False  |

| Paired training change | Vehicle delta s | Worst crossing p95 delta s | Crossings no worse |
|---|---:|---:|---:|
| ippo:normalized-clock-vs-original | -2.25 | 5.20 | 8/12 |
| ippo:normalized-clock-batch4-vs-normalized-clock | 2.06 | 20.47 | 0/12 |
| mappo:normalized-clock-vs-original | 0.71 | 6.33 | 6/12 |
| mappo:normalized-clock-batch4-vs-normalized-clock | -1.62 | 23.33 | 0/12 |

Full crossing means, p95s, sample counts, pending requests, paired intervals and comparisons against fixed/pressure/uniform controls are in summary.json. Per-run waits, slower-walker groups and diagnostics remain in evaluation JSON.

## short-spacing

| Controller | Vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved after followup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 39.28 | 2431.17 | 2316.67 | 86.67 | 27.83 | 14.83 | 0.00 |
| actuated | 46.86 | 2431.17 | 788.17 | 74.50 | 1568.50 | 9.83 | 0.00 |
| pressure | 53.05 | 2431.17 | 1934.00 | 85.17 | 412.00 | 9.00 | 0.00 |
| uniform | 90.46 | 2431.17 | 1484.33 | 105.67 | 841.17 | 12.17 | 0.00 |
| ippo-original | 84.19 | 2431.17 | 1448.22 | 102.56 | 880.39 | 8.72 | 0.00 |
| ippo-normalized-clock | 76.56 | 2431.17 | 1473.94 | 96.78 | 860.44 | 7.61 | 0.00 |
| ippo-normalized-clock-batch4 | 93.15 | 2431.17 | 1438.33 | 103.44 | 889.39 | 11.33 | 0.00 |
| mappo-original | 82.91 | 2431.17 | 1460.61 | 102.94 | 867.61 | 8.44 | 0.00 |
| mappo-normalized-clock | 84.80 | 2431.17 | 1468.78 | 103.72 | 858.67 | 8.78 | 0.00 |
| mappo-normalized-clock-batch4 | 92.87 | 2431.17 | 1447.83 | 103.61 | 879.72 | 12.11 | 0.00 |

| Learned controller vs actuated | Vehicle delta s (95% interval) | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---|
| ippo-original | 37.33 ([26.95, 48.39]) | 2/12 | 34.53 | 18/18 | 252 | False  |
| ippo-normalized-clock | 29.70 ([18.74, 42.82]) | 2/12 | 31.60 | 18/18 | 252 | False  |
| ippo-normalized-clock-batch4 | 46.29 ([35.33, 57.25]) | 2/12 | 47.13 | 18/18 | 252 | False  |
| mappo-original | 36.05 ([26.04, 46.69]) | 2/12 | 30.73 | 18/18 | 252 | False  |
| mappo-normalized-clock | 37.94 ([26.60, 50.32]) | 2/12 | 33.53 | 18/18 | 252 | False  |
| mappo-normalized-clock-batch4 | 46.02 ([35.36, 56.21]) | 2/12 | 45.87 | 18/18 | 252 | False  |

| Paired training change | Vehicle delta s | Worst crossing p95 delta s | Crossings no worse |
|---|---:|---:|---:|
| ippo:normalized-clock-vs-original | -7.63 | 5.93 | 8/12 |
| ippo:normalized-clock-batch4-vs-normalized-clock | 16.59 | 25.87 | 1/12 |
| mappo:normalized-clock-vs-original | 1.89 | 7.53 | 7/12 |
| mappo:normalized-clock-batch4-vs-normalized-clock | 8.07 | 21.33 | 0/12 |

Full crossing means, p95s, sample counts, pending requests, paired intervals and comparisons against fixed/pressure/uniform controls are in summary.json. Per-run waits, slower-walker groups and diagnostics remain in evaluation JSON.

## slow-walkers

| Controller | Vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved after followup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.29 | 957.33 | 924.33 | 32.83 | 0.17 | 21.00 | 0.00 |
| actuated | 52.73 | 957.33 | 919.17 | 38.00 | 0.17 | 6.67 | 0.00 |
| pressure | 44.95 | 957.33 | 926.00 | 31.17 | 0.17 | 6.67 | 0.00 |
| uniform | 49.33 | 957.33 | 918.50 | 37.67 | 1.17 | 15.67 | 0.00 |
| ippo-original | 54.84 | 957.33 | 918.11 | 38.72 | 0.50 | 10.39 | 0.00 |
| ippo-normalized-clock | 52.23 | 957.33 | 919.67 | 37.39 | 0.28 | 8.17 | 0.00 |
| ippo-normalized-clock-batch4 | 55.36 | 957.33 | 919.28 | 37.78 | 0.28 | 13.50 | 0.00 |
| mappo-original | 54.64 | 957.33 | 919.06 | 38.00 | 0.28 | 10.11 | 0.00 |
| mappo-normalized-clock | 56.42 | 957.33 | 917.44 | 39.28 | 0.61 | 9.17 | 0.00 |
| mappo-normalized-clock-batch4 | 54.17 | 957.33 | 919.50 | 37.56 | 0.28 | 13.72 | 0.00 |

| Learned controller vs actuated | Vehicle delta s (95% interval) | Crossings no worse | Worst p95 delta s | Service failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---|
| ippo-original | 2.12 ([-1.28, 6.01]) | 0/12 | 36.53 | 16/18 | 276 | False  |
| ippo-normalized-clock | -0.50 ([-2.79, 2.38]) | 0/12 | 34.73 | 17/18 | 277 | False  |
| ippo-normalized-clock-batch4 | 2.64 ([-1.16, 7.75]) | 0/12 | 54.40 | 18/18 | 276 | False  |
| mappo-original | 1.92 ([-1.94, 5.04]) | 0/12 | 36.07 | 17/18 | 276 | False  |
| mappo-normalized-clock | 3.69 ([0.55, 6.78]) | 0/12 | 36.33 | 15/18 | 278 | False  |
| mappo-normalized-clock-batch4 | 1.44 ([-1.79, 6.41]) | 0/12 | 53.80 | 18/18 | 276 | False  |

| Paired training change | Vehicle delta s | Worst crossing p95 delta s | Crossings no worse |
|---|---:|---:|---:|
| ippo:normalized-clock-vs-original | -2.62 | -0.40 | 12/12 |
| ippo:normalized-clock-batch4-vs-normalized-clock | 3.14 | 29.80 | 0/12 |
| mappo:normalized-clock-vs-original | 1.77 | 5.20 | 8/12 |
| mappo:normalized-clock-batch4-vs-normalized-clock | -2.25 | 30.40 | 0/12 |

Full crossing means, p95s, sample counts, pending requests, paired intervals and comparisons against fixed/pressure/uniform controls are in summary.json. Per-run waits, slower-walker groups and diagnostics remain in evaluation JSON.

## Limits and decision

These synthetic settings are hypotheses about traffic demand, not calibrated Singapore traffic. The original objective remains lower vehicle loss with no worse p95 at every crossing and no service deterioration. Better critic fits alone do not meet that objective. Descriptive passage is not statistical confirmation. Baseline service at a finite cutoff is strict and must be interpreted alongside followup counts.

A and B perform one update batch per episode; C performs one per four episodes. At 64 episodes, A/B make 64 batches and C 16; each uses four full-batch PPO passes. Equal simulated transitions do not imply equal update counts or equal policy trajectories. B combines output normalisation and clock inputs; it does not separate their individual causal contributions. C varies batch composition and policy-update frequency together.

Inspect repeated-seed traffic outcomes and action/critic diagnostics before extending training. Preserve negative results, and freeze any subsequent larger evaluation before using untouched test seeds.
