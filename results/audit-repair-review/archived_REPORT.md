# Local SUMO audit-repair ablation

Engineering smoke only: **False**. Completed models: **16**, each trained for **64 episodes**. Every final checkpoint and prescribed evaluation is present. No best-run selection. All paired demand and network hashes match.

Legacy = previous normalized-clock PPO; memory = known controller state added; separate = actor/critic gradients clipped separately; memory-separate = both changes. Pedestrian reward weight remains 3. All controllers share the corrected collision checks, phase safety and measurements. This isolates learning changes; the earlier study and original four-controller proposal comparison remain separate.

Uncertainty is exploratory: 2 training replicas and 4 demand seeds. Resample whole paired replicas and demand seeds. Crossing p95 pools events across seeds within each replica, then averages replica tails. Sparse estimates (<100 entries for any crossing/replica/reference) are inconclusive. Baseline intervals resample demand only. Vehicle delay entries are means of run means, not traffic-volume-weighted means. Action randomness has one controlled stream per demand.

## vmedium-pmedium

| Controller | Time loss s | Insertion delay s | Combined s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.37 | 0.32 | 33.69 | 963.50 | 928.25 | 35.00 | 0.25 | 13.25 | 0.00 |
| actuated | 51.42 | 0.32 | 51.74 | 963.50 | 923.50 | 39.75 | 0.25 | 2.00 | 0.00 |
| pressure | 39.91 | 0.33 | 40.24 | 963.50 | 928.75 | 34.50 | 0.25 | 3.25 | 0.00 |
| uniform | 56.05 | 0.56 | 56.61 | 963.50 | 925.25 | 38.00 | 0.25 | 11.25 | 0.00 |
| ippo-legacy | 71.80 | 0.46 | 72.26 | 963.50 | 909.25 | 53.38 | 0.88 | 4.75 | 0.00 |
| ippo-memory | 71.31 | 0.59 | 71.90 | 963.50 | 911.50 | 51.62 | 0.38 | 4.12 | 0.00 |
| ippo-separate | 75.34 | 0.71 | 76.05 | 963.50 | 916.50 | 46.75 | 0.25 | 3.25 | 0.00 |
| ippo-memory-separate | 70.08 | 0.65 | 70.73 | 963.50 | 910.62 | 51.50 | 1.38 | 4.00 | 0.00 |
| mappo-legacy | 62.91 | 0.42 | 63.33 | 963.50 | 917.38 | 45.88 | 0.25 | 7.00 | 0.00 |
| mappo-memory | 59.70 | 0.44 | 60.14 | 963.50 | 919.00 | 44.25 | 0.25 | 3.88 | 0.00 |
| mappo-separate | 67.29 | 1.00 | 68.29 | 963.50 | 914.12 | 49.12 | 0.25 | 3.38 | 0.00 |
| mappo-memory-separate | 65.05 | 0.56 | 65.61 | 963.50 | 915.38 | 47.88 | 0.25 | 5.25 | 0.00 |

Whole-horizon service below includes **all scheduled departures, including warmup**. It is a different cohort and time window from the primary table above.

| Controller | Scheduled | Completed at end | Unfinished at end | Uninserted at end |
|---|---:|---:|---:|---:|
| fixed | 1110.00 | 1110.00 | 0.00 | 0.00 |
| actuated | 1110.00 | 1110.00 | 0.00 | 0.00 |
| pressure | 1110.00 | 1110.00 | 0.00 | 0.00 |
| uniform | 1110.00 | 1110.00 | 0.00 | 0.00 |
| ippo-legacy | 1110.00 | 1109.38 | 0.62 | 0.00 |
| ippo-memory | 1110.00 | 1107.75 | 2.25 | 0.00 |
| ippo-separate | 1110.00 | 1108.25 | 1.75 | 0.00 |
| ippo-memory-separate | 1110.00 | 1109.12 | 0.88 | 0.00 |
| mappo-legacy | 1110.00 | 1110.00 | 0.00 | 0.00 |
| mappo-memory | 1110.00 | 1109.25 | 0.75 | 0.00 |
| mappo-separate | 1110.00 | 1109.88 | 0.12 | 0.00 |
| mappo-memory-separate | 1110.00 | 1107.12 | 2.88 | 0.00 |

| Learned vs actuated | Vehicle delta s (95% exploratory interval) | Worst crossing p95 delta s | Crossings no worse | Strict service failures | Cutoff-only failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---:|---|
| ippo-legacy | 20.37 [14.99, 28.61] | 14.30 | 0/12 | 8/8 | 1 | 171 | False  |
| ippo-memory | 19.89 [10.60, 28.67] | 14.50 | 0/12 | 8/8 | 1 | 171 | False  |
| ippo-separate | 23.91 [15.27, 35.31] | 22.10 | 0/12 | 7/8 | 1 | 171 | False  |
| ippo-memory-separate | 18.65 [11.79, 28.02] | 18.80 | 0/12 | 7/8 | 1 | 171 | False  |
| mappo-legacy | 11.48 [2.11, 22.67] | 30.80 | 0/12 | 8/8 | 3 | 171 | False  |
| mappo-memory | 8.28 [4.48, 13.78] | 24.10 | 0/12 | 8/8 | 1 | 171 | False  |
| mappo-separate | 15.86 [3.90, 34.15] | 19.80 | 0/12 | 8/8 | 1 | 171 | False  |
| mappo-memory-separate | 13.62 [11.14, 17.52] | 26.60 | 0/12 | 8/8 | 2 | 171 | False  |

| Paired correction | Vehicle delta s (95% exploratory interval) | Worst crossing delta s (95% exploratory interval) | Crossings no worse | Strict service failures |
|---|---|---|---:|---:|
| ippo:memory-vs-legacy | -0.49 [-13.28, 10.31] | 5.90 [5.30, 17.20] | 6/12 | 7/8 |
| ippo:separate-vs-legacy | 3.54 [-8.42, 18.49] | 14.60 [8.80, 28.20] | 4/12 | 5/8 |
| ippo:memory-separate-vs-memory | -1.24 [-10.38, 7.99] | 11.70 [6.90, 26.60] | 2/12 | 6/8 |
| ippo:memory-separate-vs-separate | -5.26 [-10.97, -0.81] | 10.50 [6.80, 15.60] | 3/12 | 5/8 |
| ippo:memory-separate-vs-legacy | -1.72 [-11.86, 11.21] | 10.60 [9.50, 24.60] | 1/12 | 6/8 |
| mappo:memory-vs-legacy | -3.21 [-17.73, 6.97] | 13.10 [3.40, 34.40] | 8/12 | 8/8 |
| mappo:separate-vs-legacy | 4.38 [-17.15, 27.52] | 4.20 [0.00, 17.00] | 10/12 | 7/8 |
| mappo:memory-separate-vs-memory | 5.35 [-2.44, 13.04] | 3.70 [2.80, 19.40] | 7/12 | 7/8 |
| mappo:memory-separate-vs-separate | -2.24 [-22.99, 12.47] | 6.80 [3.20, 15.80] | 5/12 | 8/8 |
| mappo:memory-separate-vs-legacy | 2.14 [-6.56, 9.94] | 1.20 [0.90, 17.00] | 10/12 | 8/8 |

A negative paired vehicle difference means the correction reduced completed-trip time loss relative to its named control. That alone does not establish the proposal target. Service-failure categories overlap; cutoff-only counts identify timing-sensitive failures without dropping the original criterion. Full crossing means/p95s, replica entry counts, slower-walker groups, service components and insertion-delay intervals are in summary.json; raw per-run waits remain in evaluation JSON.

## pedestrian-burst

| Controller | Time loss s | Insertion delay s | Combined s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.37 | 0.32 | 33.69 | 963.50 | 928.25 | 35.00 | 0.25 | 13.25 | 0.00 |
| actuated | 51.42 | 0.32 | 51.74 | 963.50 | 923.50 | 39.75 | 0.25 | 2.00 | 0.00 |
| pressure | 40.80 | 0.32 | 41.12 | 963.50 | 932.25 | 31.00 | 0.25 | 2.75 | 0.00 |
| uniform | 56.05 | 0.56 | 56.61 | 963.50 | 925.25 | 38.00 | 0.25 | 11.25 | 0.00 |
| ippo-legacy | 71.63 | 0.46 | 72.09 | 963.50 | 912.25 | 50.38 | 0.88 | 4.75 | 0.00 |
| ippo-memory | 71.34 | 0.59 | 71.93 | 963.50 | 910.75 | 52.38 | 0.38 | 3.75 | 0.00 |
| ippo-separate | 75.34 | 0.71 | 76.05 | 963.50 | 916.50 | 46.75 | 0.25 | 3.25 | 0.00 |
| ippo-memory-separate | 70.08 | 0.65 | 70.73 | 963.50 | 910.62 | 51.50 | 1.38 | 4.00 | 0.00 |
| mappo-legacy | 63.68 | 0.46 | 64.14 | 963.50 | 918.38 | 44.88 | 0.25 | 7.00 | 0.00 |
| mappo-memory | 59.70 | 0.44 | 60.14 | 963.50 | 919.00 | 44.25 | 0.25 | 3.88 | 0.00 |
| mappo-separate | 67.71 | 1.17 | 68.88 | 963.50 | 912.62 | 50.62 | 0.25 | 3.75 | 0.00 |
| mappo-memory-separate | 65.05 | 0.56 | 65.61 | 963.50 | 915.38 | 47.88 | 0.25 | 5.25 | 0.00 |

Whole-horizon service below includes **all scheduled departures, including warmup**. It is a different cohort and time window from the primary table above.

| Controller | Scheduled | Completed at end | Unfinished at end | Uninserted at end |
|---|---:|---:|---:|---:|
| fixed | 1110.00 | 1110.00 | 0.00 | 0.00 |
| actuated | 1110.00 | 1110.00 | 0.00 | 0.00 |
| pressure | 1110.00 | 1110.00 | 0.00 | 0.00 |
| uniform | 1110.00 | 1110.00 | 0.00 | 0.00 |
| ippo-legacy | 1110.00 | 1109.38 | 0.62 | 0.00 |
| ippo-memory | 1110.00 | 1107.75 | 2.25 | 0.00 |
| ippo-separate | 1110.00 | 1108.25 | 1.75 | 0.00 |
| ippo-memory-separate | 1110.00 | 1109.12 | 0.88 | 0.00 |
| mappo-legacy | 1110.00 | 1110.00 | 0.00 | 0.00 |
| mappo-memory | 1110.00 | 1109.25 | 0.75 | 0.00 |
| mappo-separate | 1110.00 | 1109.88 | 0.12 | 0.00 |
| mappo-memory-separate | 1110.00 | 1107.12 | 2.88 | 0.00 |

| Learned vs actuated | Vehicle delta s (95% exploratory interval) | Worst crossing p95 delta s | Crossings no worse | Strict service failures | Cutoff-only failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---:|---|
| ippo-legacy | 20.21 [14.99, 28.13] | 14.30 | 0/12 | 8/8 | 2 | 185 | False  |
| ippo-memory | 19.92 [10.66, 28.67] | 14.50 | 0/12 | 7/8 | 0 | 185 | False  |
| ippo-separate | 23.91 [15.27, 35.31] | 22.10 | 0/12 | 7/8 | 1 | 185 | False  |
| ippo-memory-separate | 18.65 [11.79, 28.02] | 18.80 | 0/12 | 7/8 | 1 | 185 | False  |
| mappo-legacy | 12.25 [1.80, 22.67] | 30.80 | 0/12 | 8/8 | 3 | 181 | False  |
| mappo-memory | 8.28 [4.48, 13.78] | 24.10 | 0/12 | 8/8 | 1 | 184 | False  |
| mappo-separate | 16.29 [3.90, 35.84] | 19.80 | 0/12 | 8/8 | 1 | 185 | False  |
| mappo-memory-separate | 13.62 [11.14, 17.52] | 26.60 | 0/12 | 8/8 | 2 | 185 | False  |

| Paired correction | Vehicle delta s (95% exploratory interval) | Worst crossing delta s (95% exploratory interval) | Crossings no worse | Strict service failures |
|---|---|---|---:|---:|
| ippo:memory-vs-legacy | -0.29 [-12.97, 10.31] | 5.90 [5.30, 17.20] | 5/12 | 7/8 |
| ippo:separate-vs-legacy | 3.71 [-7.75, 18.49] | 14.50 [8.80, 27.40] | 4/12 | 5/8 |
| ippo:memory-separate-vs-memory | -1.26 [-10.38, 7.99] | 12.30 [7.50, 29.60] | 2/12 | 6/8 |
| ippo:memory-separate-vs-separate | -5.26 [-10.97, -0.81] | 11.50 [7.40, 25.00] | 3/12 | 5/8 |
| ippo:memory-separate-vs-legacy | -1.55 [-11.18, 11.21] | 10.60 [9.30, 24.60] | 1/12 | 6/8 |
| mappo:memory-vs-legacy | -3.98 [-17.73, 4.87] | 13.50 [3.40, 34.40] | 8/12 | 8/8 |
| mappo:separate-vs-legacy | 4.03 [-17.15, 25.99] | 4.40 [0.30, 17.00] | 9/12 | 7/8 |
| mappo:memory-separate-vs-memory | 5.35 [-2.44, 13.04] | 3.70 [2.80, 19.40] | 7/12 | 7/8 |
| mappo:memory-separate-vs-separate | -2.66 [-24.68, 12.47] | 6.80 [3.20, 20.00] | 5/12 | 8/8 |
| mappo:memory-separate-vs-legacy | 1.37 [-6.93, 10.07] | 1.20 [0.90, 16.90] | 10/12 | 8/8 |

A negative paired vehicle difference means the correction reduced completed-trip time loss relative to its named control. That alone does not establish the proposal target. Service-failure categories overlap; cutoff-only counts identify timing-sensitive failures without dropping the original criterion. Full crossing means/p95s, replica entry counts, slower-walker groups, service components and insertion-delay intervals are in summary.json; raw per-run waits remain in evaluation JSON.

## short-spacing

| Controller | Time loss s | Insertion delay s | Combined s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 37.70 | 9.74 | 47.44 | 2353.50 | 2251.00 | 86.50 | 16.00 | 14.75 | 0.00 |
| actuated | 46.64 | 21.44 | 68.07 | 2353.50 | 709.50 | 64.50 | 1579.50 | 5.25 | 0.00 |
| pressure | 53.38 | 78.27 | 131.65 | 2353.50 | 1903.50 | 86.25 | 363.75 | 9.00 | 0.00 |
| uniform | 88.42 | 242.93 | 331.34 | 2353.50 | 1474.00 | 103.50 | 776.00 | 15.25 | 0.00 |
| ippo-legacy | 84.55 | 166.77 | 251.32 | 2353.50 | 1265.88 | 92.00 | 995.62 | 5.75 | 0.00 |
| ippo-memory | 78.14 | 135.71 | 213.85 | 2353.50 | 1339.88 | 87.62 | 926.00 | 3.88 | 0.00 |
| ippo-separate | 72.65 | 131.86 | 204.50 | 2353.50 | 1304.50 | 88.12 | 960.88 | 5.38 | 0.00 |
| ippo-memory-separate | 76.26 | 141.39 | 217.65 | 2353.50 | 1313.38 | 96.62 | 943.50 | 6.00 | 0.00 |
| mappo-legacy | 76.45 | 146.06 | 222.50 | 2353.50 | 1361.00 | 98.25 | 894.25 | 7.12 | 0.00 |
| mappo-memory | 77.57 | 141.04 | 218.61 | 2353.50 | 1429.38 | 98.62 | 825.50 | 6.88 | 0.00 |
| mappo-separate | 85.27 | 179.83 | 265.10 | 2353.50 | 1371.25 | 101.50 | 880.75 | 7.38 | 0.00 |
| mappo-memory-separate | 85.28 | 178.66 | 263.94 | 2353.50 | 1353.25 | 102.00 | 898.25 | 8.38 | 0.00 |

Whole-horizon service below includes **all scheduled departures, including warmup**. It is a different cohort and time window from the primary table above.

| Controller | Scheduled | Completed at end | Unfinished at end | Uninserted at end |
|---|---:|---:|---:|---:|
| fixed | 2743.50 | 2743.50 | 0.00 | 0.00 |
| actuated | 2743.50 | 1265.25 | 77.50 | 1400.75 |
| pressure | 2743.50 | 2499.25 | 23.25 | 221.00 |
| uniform | 2743.50 | 2121.50 | 81.50 | 540.50 |
| ippo-legacy | 2743.50 | 1847.38 | 72.38 | 823.75 |
| ippo-memory | 2743.50 | 1902.88 | 62.50 | 778.12 |
| ippo-separate | 2743.50 | 1875.62 | 66.00 | 801.88 |
| ippo-memory-separate | 2743.50 | 1883.62 | 64.88 | 795.00 |
| mappo-legacy | 2743.50 | 1934.25 | 68.50 | 740.75 |
| mappo-memory | 2743.50 | 2018.62 | 63.00 | 661.88 |
| mappo-separate | 2743.50 | 1972.00 | 69.50 | 702.00 |
| mappo-memory-separate | 2743.50 | 1967.75 | 73.75 | 702.00 |

| Learned vs actuated | Vehicle delta s (95% exploratory interval) | Worst crossing p95 delta s | Crossings no worse | Strict service failures | Cutoff-only failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---:|---|
| ippo-legacy | 37.91 [25.65, 50.44] | 18.80 | 1/12 | 8/8 | 1 | 176 | False  |
| ippo-memory | 31.50 [18.27, 44.34] | 21.10 | 1/12 | 8/8 | 1 | 176 | False  |
| ippo-separate | 26.01 [14.25, 38.49] | 30.20 | 1/12 | 8/8 | 2 | 176 | False  |
| ippo-memory-separate | 29.62 [16.74, 42.96] | 27.20 | 1/12 | 8/8 | 0 | 175 | False  |
| mappo-legacy | 29.81 [18.27, 41.31] | 39.80 | 0/12 | 8/8 | 0 | 175 | False  |
| mappo-memory | 30.93 [19.26, 42.68] | 31.20 | 0/12 | 8/8 | 0 | 176 | False  |
| mappo-separate | 38.63 [28.17, 51.08] | 31.70 | 0/12 | 8/8 | 0 | 176 | False  |
| mappo-memory-separate | 38.64 [25.69, 51.93] | 34.80 | 1/12 | 8/8 | 0 | 176 | False  |

| Paired correction | Vehicle delta s (95% exploratory interval) | Worst crossing delta s (95% exploratory interval) | Crossings no worse | Strict service failures |
|---|---|---|---:|---:|
| ippo:memory-vs-legacy | -6.41 [-12.15, 0.73] | 8.00 [6.20, 25.80] | 6/12 | 5/8 |
| ippo:separate-vs-legacy | -11.90 [-19.22, -6.52] | 14.80 [13.60, 35.60] | 3/12 | 8/8 |
| ippo:memory-separate-vs-memory | -1.88 [-5.79, 2.72] | 10.70 [9.70, 23.00] | 3/12 | 7/8 |
| ippo:memory-separate-vs-separate | 3.61 [0.35, 7.85] | 5.10 [3.80, 15.40] | 7/12 | 7/8 |
| ippo:memory-separate-vs-legacy | -8.29 [-16.00, -1.41] | 13.30 [10.50, 32.80] | 3/12 | 6/8 |
| mappo:memory-vs-legacy | 1.12 [-5.67, 11.76] | 8.00 [3.30, 24.60] | 6/12 | 8/8 |
| mappo:separate-vs-legacy | 8.83 [3.29, 13.74] | 4.60 [2.70, 30.60] | 9/12 | 8/8 |
| mappo:memory-separate-vs-memory | 7.71 [4.07, 10.22] | 8.60 [3.80, 29.60] | 7/12 | 8/8 |
| mappo:memory-separate-vs-separate | 0.01 [-7.75, 10.92] | 9.50 [7.60, 24.00] | 7/12 | 8/8 |
| mappo:memory-separate-vs-legacy | 8.83 [0.99, 20.56] | 5.30 [1.90, 17.40] | 11/12 | 8/8 |

A negative paired vehicle difference means the correction reduced completed-trip time loss relative to its named control. That alone does not establish the proposal target. Service-failure categories overlap; cutoff-only counts identify timing-sensitive failures without dropping the original criterion. Full crossing means/p95s, replica entry counts, slower-walker groups, service components and insertion-delay intervals are in summary.json; raw per-run waits remain in evaluation JSON.

## slow-walkers

| Controller | Time loss s | Insertion delay s | Combined s | Scheduled | Completed | Unfinished | Uninserted | Pending at cutoff | Unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.45 | 0.32 | 33.77 | 963.50 | 928.25 | 35.00 | 0.25 | 21.00 | 0.00 |
| actuated | 52.19 | 0.32 | 52.52 | 963.50 | 925.50 | 37.75 | 0.25 | 6.50 | 0.00 |
| pressure | 44.05 | 0.32 | 44.37 | 963.50 | 930.75 | 32.50 | 0.25 | 7.00 | 0.00 |
| uniform | 56.05 | 0.56 | 56.61 | 963.50 | 925.25 | 38.00 | 0.25 | 19.00 | 0.00 |
| ippo-legacy | 72.88 | 0.56 | 73.45 | 963.50 | 912.38 | 50.88 | 0.25 | 7.25 | 0.00 |
| ippo-memory | 73.14 | 0.64 | 73.78 | 963.50 | 912.12 | 50.62 | 0.75 | 8.00 | 0.00 |
| ippo-separate | 74.88 | 1.13 | 76.01 | 963.50 | 913.62 | 49.25 | 0.62 | 7.25 | 0.00 |
| ippo-memory-separate | 69.07 | 0.74 | 69.82 | 963.50 | 920.12 | 43.12 | 0.25 | 7.50 | 0.00 |
| mappo-legacy | 67.17 | 0.90 | 68.07 | 963.50 | 921.62 | 41.50 | 0.38 | 6.62 | 0.00 |
| mappo-memory | 61.14 | 0.41 | 61.55 | 963.50 | 921.88 | 41.38 | 0.25 | 9.00 | 0.00 |
| mappo-separate | 67.01 | 0.83 | 67.85 | 963.50 | 913.38 | 49.75 | 0.38 | 7.62 | 0.00 |
| mappo-memory-separate | 67.39 | 0.48 | 67.86 | 963.50 | 918.38 | 44.88 | 0.25 | 6.38 | 0.00 |

Whole-horizon service below includes **all scheduled departures, including warmup**. It is a different cohort and time window from the primary table above.

| Controller | Scheduled | Completed at end | Unfinished at end | Uninserted at end |
|---|---:|---:|---:|---:|
| fixed | 1110.00 | 1110.00 | 0.00 | 0.00 |
| actuated | 1110.00 | 1110.00 | 0.00 | 0.00 |
| pressure | 1110.00 | 1110.00 | 0.00 | 0.00 |
| uniform | 1110.00 | 1110.00 | 0.00 | 0.00 |
| ippo-legacy | 1110.00 | 1109.88 | 0.12 | 0.00 |
| ippo-memory | 1110.00 | 1108.12 | 1.88 | 0.00 |
| ippo-separate | 1110.00 | 1109.12 | 0.88 | 0.00 |
| ippo-memory-separate | 1110.00 | 1110.00 | 0.00 | 0.00 |
| mappo-legacy | 1110.00 | 1109.12 | 0.88 | 0.00 |
| mappo-memory | 1110.00 | 1109.62 | 0.38 | 0.00 |
| mappo-separate | 1110.00 | 1109.25 | 0.75 | 0.00 |
| mappo-memory-separate | 1110.00 | 1107.88 | 2.12 | 0.00 |

| Learned vs actuated | Vehicle delta s (95% exploratory interval) | Worst crossing p95 delta s | Crossings no worse | Strict service failures | Cutoff-only failures | Min entries | Descriptive pass |
|---|---|---:|---:|---:|---:|---:|---|
| ippo-legacy | 20.69 [13.99, 27.19] | 24.60 | 0/12 | 8/8 | 2 | 180 | False  |
| ippo-memory | 20.95 [10.46, 31.93] | 23.40 | 0/12 | 8/8 | 1 | 180 | False  |
| ippo-separate | 22.69 [11.99, 35.19] | 33.70 | 0/12 | 8/8 | 1 | 180 | False  |
| ippo-memory-separate | 16.88 [7.20, 29.01] | 26.00 | 0/12 | 8/8 | 2 | 180 | False  |
| mappo-legacy | 14.98 [3.50, 29.90] | 34.70 | 0/12 | 7/8 | 2 | 180 | False  |
| mappo-memory | 8.95 [2.45, 13.59] | 33.30 | 0/12 | 8/8 | 2 | 180 | False  |
| mappo-separate | 14.82 [1.91, 28.49] | 33.60 | 0/12 | 8/8 | 2 | 180 | False  |
| mappo-memory-separate | 15.19 [8.55, 22.10] | 33.60 | 0/12 | 7/8 | 2 | 180 | False  |

| Paired correction | Vehicle delta s (95% exploratory interval) | Worst crossing delta s (95% exploratory interval) | Crossings no worse | Strict service failures |
|---|---|---|---:|---:|
| ippo:memory-vs-legacy | 0.26 [-8.21, 11.72] | 8.20 [6.30, 35.20] | 5/12 | 8/8 |
| ippo:separate-vs-legacy | 2.00 [-6.79, 10.12] | 13.90 [11.00, 35.00] | 3/12 | 7/8 |
| ippo:memory-separate-vs-memory | -4.07 [-19.53, 9.38] | 8.40 [7.60, 18.80] | 2/12 | 8/8 |
| ippo:memory-separate-vs-separate | -5.81 [-16.63, 0.63] | 7.50 [2.40, 16.80] | 7/12 | 6/8 |
| ippo:memory-separate-vs-legacy | -3.81 [-11.94, 9.04] | 7.40 [7.40, 25.80] | 3/12 | 8/8 |
| mappo:memory-vs-legacy | -6.03 [-23.08, 6.89] | 6.30 [3.00, 18.00] | 7/12 | 8/8 |
| mappo:separate-vs-legacy | -0.16 [-24.44, 22.21] | 3.80 [1.50, 33.00] | 7/12 | 8/8 |
| mappo:memory-separate-vs-memory | 6.24 [0.80, 13.06] | 3.00 [2.30, 21.60] | 9/12 | 8/8 |
| mappo:memory-separate-vs-separate | 0.37 [-13.34, 14.80] | 15.60 [3.20, 21.90] | 8/12 | 7/8 |
| mappo:memory-separate-vs-legacy | 0.22 [-11.81, 12.83] | 6.60 [-0.10, 17.00] | 11/12 | 7/8 |

A negative paired vehicle difference means the correction reduced completed-trip time loss relative to its named control. That alone does not establish the proposal target. Service-failure categories overlap; cutoff-only counts identify timing-sensitive failures without dropping the original criterion. Full crossing means/p95s, replica entry counts, slower-walker groups, service components and insertion-delay intervals are in summary.json; raw per-run waits remain in evaluation JSON.

## Interpretation limits

No result here proves deployed benefit, safety outside the checked model, or guaranteed PPO superiority. The joint criterion remains lower vehicle time loss, no higher p95 at any of 12 crossings, and no strict service deterioration. Sparse samples and the small number of training replicas limit inference; even a descriptive pass is not a confirmatory finding. The unchanged 600-second training versus 2400-second evaluation mismatch, local rewards, limited corridor-through traffic, and unobserved external backlog remain separate hypotheses. These ablations do not claim to fix them.

Inspect effective free decisions, entropy, KL and gradients in learning.json and summary.json. Both improvements and failures must be retained. Reserved final test seeds remain unused.
