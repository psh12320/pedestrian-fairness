# Local pedestrian admission timing study

Engineering smoke: **False**. Completed: 144 simulations. Single CPU worker; no neural training.

12, 18 and 24 seconds are admission periods. Separate protected clearance is unchanged. Every controller receives the same timing at each setting. Effects against each controller at 12 seconds are timing changes, not RL gains. Controllers are also compared against actuated at the same duration. No comparison with a differently timed baseline may be used as the original algorithm claim.

Metrics include scheduled/completed/unfinished/uninserted vehicles, whole person trips and crossing-cohort requests. P95 pools crossing events across matched demand seeds. Intervals resample whole paired seeds; four seeds provide limited uncertainty estimates. Sparse crossing samples (<100) are inconclusive. No final test seeds are used.

## vmedium-pmedium

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Worst p95 change vs itself at 12s | Crossings no worse | Service failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-12s | 33.77 | 928.25 | 36.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| actuated-12s | 54.14 | 922.25 | 42.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| pressure-12s | 40.87 | 930.25 | 34.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| fixed-18s | 37.20 | 946.50 | 18.00 | 0.00 | 2.00 | 8/12 | 4/4 |
| actuated-18s | 69.48 | 920.00 | 44.50 | 0.00 | 1.00 | 10/12 | 3/4 |
| pressure-18s | 47.23 | 927.00 | 37.50 | 0.00 | 10.80 | 4/12 | 4/4 |
| fixed-24s | 41.63 | 921.00 | 43.50 | 0.00 | 1.60 | 8/12 | 4/4 |
| actuated-24s | 94.28 | 907.25 | 57.25 | 0.00 | 1.80 | 11/12 | 4/4 |
| pressure-24s | 51.15 | 931.25 | 33.25 | 0.00 | 13.20 | 2/12 | 4/4 |

All crossing means/p95/counts, service outcomes, same-duration actuated comparisons and exploratory intervals are retained in summary.json. Raw event timestamps and phase histories are in compressed per-run records.

## pedestrian-burst

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Worst p95 change vs itself at 12s | Crossings no worse | Service failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-12s | 33.77 | 928.25 | 36.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| actuated-12s | 53.13 | 924.25 | 40.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| pressure-12s | 42.31 | 929.50 | 35.00 | 0.00 | 0.00 | 12/12 | 0/4 |
| fixed-18s | 37.20 | 946.50 | 18.00 | 0.00 | 1.80 | 8/12 | 4/4 |
| actuated-18s | 70.40 | 923.00 | 41.50 | 0.00 | 2.60 | 9/12 | 4/4 |
| pressure-18s | 47.75 | 925.25 | 39.25 | 0.00 | 7.40 | 3/12 | 4/4 |
| fixed-24s | 41.63 | 921.00 | 43.50 | 0.00 | 1.60 | 8/12 | 4/4 |
| actuated-24s | 94.28 | 907.25 | 57.25 | 0.00 | 2.00 | 8/12 | 4/4 |
| pressure-24s | 52.53 | 927.75 | 36.75 | 0.00 | 14.20 | 0/12 | 4/4 |

All crossing means/p95/counts, service outcomes, same-duration actuated comparisons and exploratory intervals are retained in summary.json. Raw event timestamps and phase histories are in compressed per-run records.

## slow-walkers

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Worst p95 change vs itself at 12s | Crossings no worse | Service failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-12s | 33.77 | 928.25 | 36.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| actuated-12s | 54.82 | 926.25 | 38.25 | 0.00 | 0.00 | 12/12 | 0/4 |
| pressure-12s | 46.80 | 930.75 | 33.75 | 0.00 | 0.00 | 12/12 | 0/4 |
| fixed-18s | 37.20 | 946.50 | 18.00 | 0.00 | 3.00 | 7/12 | 4/4 |
| actuated-18s | 68.07 | 915.00 | 49.50 | 0.00 | 0.20 | 11/12 | 4/4 |
| pressure-18s | 49.95 | 921.25 | 43.25 | 0.00 | 10.80 | 7/12 | 4/4 |
| fixed-24s | 41.63 | 921.00 | 43.50 | 0.00 | 0.60 | 11/12 | 4/4 |
| actuated-24s | 96.43 | 902.25 | 62.25 | 0.00 | 0.20 | 11/12 | 4/4 |
| pressure-24s | 55.15 | 924.25 | 40.25 | 0.00 | 9.00 | 4/12 | 3/4 |

All crossing means/p95/counts, service outcomes, same-duration actuated comparisons and exploratory intervals are retained in summary.json. Raw event timestamps and phase histories are in compressed per-run records.

## short-spacing

| Controller | Vehicle loss s | Completed | Unfinished | Uninserted | Worst p95 change vs itself at 12s | Crossings no worse | Service failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-12s | 38.80 | 2282.75 | 81.75 | 33.75 | 0.00 | 12/12 | 0/4 |
| actuated-12s | 25.79 | 664.25 | 59.75 | 1674.25 | 0.00 | 12/12 | 0/4 |
| pressure-12s | 52.81 | 1840.00 | 82.25 | 476.00 | 0.00 | 12/12 | 0/4 |
| fixed-18s | 45.38 | 2261.00 | 75.75 | 61.50 | 5.00 | 8/12 | 4/4 |
| actuated-18s | 37.64 | 614.25 | 60.00 | 1724.00 | 0.60 | 10/12 | 4/4 |
| pressure-18s | 61.02 | 1751.00 | 86.75 | 560.50 | 4.20 | 3/12 | 4/4 |
| fixed-24s | 53.36 | 2148.25 | 100.00 | 150.00 | 3.60 | 6/12 | 4/4 |
| actuated-24s | 65.40 | 578.50 | 68.25 | 1751.50 | 1.40 | 9/12 | 4/4 |
| pressure-24s | 63.22 | 1656.75 | 94.50 | 647.00 | 14.20 | 7/12 | 4/4 |

All crossing means/p95/counts, service outcomes, same-duration actuated comparisons and exploratory intervals are retained in summary.json. Raw event timestamps and phase histories are in compressed per-run records.

Synthetic geometry/demand; four development scenarios. Short spacing also increases vehicle demand. This study does not evaluate the Gini-based learning method or demonstrate deployed benefits.
