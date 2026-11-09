# Demand and early pedestrian service: frozen-model evaluation

**Complete development experiment: 720 evaluations, no retraining.**

Three saved MAPPO replicas per learned arm. Early service uses the continued-policy weights and requests PED whenever a local request is pending and PED is legal. Vehicle decisions and empty pedestrian requests otherwise follow the saved policy. It is a hybrid controller, not a newly trained algorithm.

All controllers share the same geometry and physical safety rules. Demand and networks are hash-matched within each setting and seed. Final reserved test seeds are untouched. Earlier unfavorable studies remain unchanged.

Vehicle time loss is conditional on completion. Vehicle cohort: scheduled departures in [300,2100) seconds, status at 2100. Pedestrians reaching crossings in that window are followed to 2400. Whole-run service is also retained separately in summary.json. Crossing p95 pools seeds within each replica, then averages replica tails. Intervals resample paired whole seeds and replicas; they are exploratory, not a simultaneous noninferiority claim. Fewer than 100 entries at any pooled replica/reference crossing is flagged as sparse. Only 1e-9 second arithmetic ties are ignored.

## vlow-plow

| Controller | Vehicle loss s | Change vs actuated s | Paired 95% interval | Crossings no worse /12 | Worst crossing excess s | Sparse? |
|---|---:|---:|---|---:|---:|---|
| fixed | 29.35 | -6.25 | [-7.21, -5.30] | 0 | 46.20 | False |
| actuated | 35.60 | 0.00 | [0.00, 0.00] | 12 | 0.00 | False |
| pressure | 29.54 | -6.06 | [-7.26, -4.85] | 0 | 13.40 | False |
| mappo-original | 36.32 | 0.73 | [-1.78, 3.09] | 0 | 22.07 | False |
| mappo-continue | 33.83 | -1.77 | [-3.97, 0.21] | 0 | 27.53 | False |
| mappo-tail | 36.65 | 1.05 | [-1.16, 3.04] | 0 | 16.60 | False |
| mappo-guided | 41.07 | 5.48 | [2.38, 7.88] | 0 | 8.13 | False |
| mappo-early | 41.47 | 5.87 | [4.04, 7.37] | 5 | 3.53 | False |

| Controller | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup | Insertion delay s | Strict service failures / pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed | 482.60 | 464.90 | 17.70 | 0.00 | 6.60 | 0.00 | 0.20 | 10/10 |
| actuated | 482.60 | 465.60 | 17.00 | 0.00 | 2.00 | 0.00 | 0.20 | 0/10 |
| pressure | 482.60 | 468.00 | 14.60 | 0.00 | 3.00 | 0.00 | 0.20 | 7/10 |
| mappo-original | 482.60 | 468.37 | 14.23 | 0.00 | 3.43 | 0.00 | 0.20 | 23/30 |
| mappo-continue | 482.60 | 468.77 | 13.83 | 0.00 | 3.13 | 0.00 | 0.20 | 24/30 |
| mappo-tail | 482.60 | 467.07 | 15.53 | 0.00 | 2.63 | 0.00 | 0.20 | 27/30 |
| mappo-guided | 482.60 | 467.17 | 15.43 | 0.00 | 1.83 | 0.00 | 0.20 | 17/30 |
| mappo-early | 482.60 | 465.90 | 16.70 | 0.00 | 2.43 | 0.00 | 0.20 | 24/30 |

| Crossing | Actuated p95 | Original | Continue | Tail | Guided | Early service | Minimum entries across all replicas/controllers |
|---|---:|---:|---:|---:|---:|---:|---:|
| :C0_c0 | 28.20 | 44.80 | 44.53 | 39.20 | 36.33 | 28.00 | 156 |
| :C0_c1 | 27.00 | 45.00 | 43.67 | 38.73 | 34.20 | 27.67 | 304 |
| :C0_c2 | 26.20 | 32.00 | 39.07 | 41.40 | 28.53 | 25.67 | 165 |
| :C0_c3 | 26.60 | 44.27 | 44.27 | 37.07 | 32.80 | 26.87 | 181 |
| :C1_c0 | 25.60 | 31.40 | 39.60 | 35.93 | 29.20 | 27.13 | 151 |
| :C1_c1 | 27.00 | 41.93 | 41.60 | 35.87 | 31.80 | 27.13 | 334 |
| :C1_c2 | 27.40 | 34.07 | 37.73 | 31.47 | 28.27 | 26.67 | 174 |
| :C1_c3 | 26.60 | 35.73 | 40.73 | 36.27 | 29.07 | 28.00 | 303 |
| :C2_c0 | 23.80 | 42.13 | 47.00 | 37.60 | 28.73 | 27.33 | 151 |
| :C2_c1 | 26.80 | 47.07 | 54.33 | 43.40 | 34.40 | 27.33 | 187 |
| :C2_c2 | 27.20 | 49.27 | 47.33 | 41.33 | 34.80 | 26.93 | 155 |
| :C2_c3 | 27.20 | 44.27 | 43.47 | 41.13 | 32.13 | 26.93 | 314 |

Early-service minus unchanged continued MAPPO: {"crossing_p95_delta_seconds": {":C0_c0": -16.53333333333336, ":C0_c1": -16.0, ":C0_c2": -13.3999999999999, ":C0_c3": -17.39999999999988, ":C1_c0": -12.46666666666662, ":C1_c1": -14.46666666666666, ":C1_c2": -11.066666666666606, ":C1_c3": -12.733333333333311, ":C2_c0": -19.666666666666742, ":C2_c1": -27.000000000000057, ":C2_c2": -20.400000000000016, ":C2_c3": -16.533333333333417}, "crossings_no_worse": 12, "insertion_delay_delta_seconds": 0.0018695057534629151, "missing_crossings": [], "time_loss_plus_insertion_delay_delta_seconds": 7.641154167046842, "vehicle_delta_seconds": 7.639284661293378, "worst_crossing_delta_seconds": -11.066666666666606}

The complete summary includes crossing means, per-replica tails, pointwise intervals, count differences, whole-run vehicle/person service, phase occupancy, decision audits, and the strict service-gate components.

## vlow-pmedium

| Controller | Vehicle loss s | Change vs actuated s | Paired 95% interval | Crossings no worse /12 | Worst crossing excess s | Sparse? |
|---|---:|---:|---|---:|---:|---|
| fixed | 29.35 | -15.69 | [-16.98, -14.54] | 0 | 42.40 | False |
| actuated | 45.04 | 0.00 | [0.00, 0.00] | 12 | 0.00 | False |
| pressure | 41.22 | -3.82 | [-5.04, -2.30] | 3 | 1.00 | False |
| mappo-original | 40.06 | -4.98 | [-6.43, -3.22] | 0 | 14.80 | False |
| mappo-continue | 38.74 | -6.29 | [-7.89, -4.63] | 0 | 13.33 | False |
| mappo-tail | 41.93 | -3.11 | [-4.50, -1.74] | 0 | 9.60 | False |
| mappo-guided | 45.47 | 0.43 | [-1.07, 1.82] | 0 | 4.07 | False |
| mappo-early | 48.24 | 3.20 | [1.82, 4.57] | 8 | 0.73 | False |

| Controller | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup | Insertion delay s | Strict service failures / pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed | 482.60 | 464.90 | 17.70 | 0.00 | 13.70 | 0.00 | 0.20 | 10/10 |
| actuated | 482.60 | 467.00 | 15.60 | 0.00 | 4.10 | 0.00 | 0.20 | 0/10 |
| pressure | 482.60 | 467.80 | 14.80 | 0.00 | 3.90 | 0.00 | 0.20 | 8/10 |
| mappo-original | 482.60 | 466.17 | 16.43 | 0.00 | 6.07 | 0.00 | 0.20 | 28/30 |
| mappo-continue | 482.60 | 466.03 | 16.57 | 0.00 | 5.40 | 0.00 | 0.20 | 27/30 |
| mappo-tail | 482.60 | 466.17 | 16.43 | 0.00 | 4.87 | 0.00 | 0.20 | 29/30 |
| mappo-guided | 482.60 | 464.73 | 17.87 | 0.00 | 4.83 | 0.00 | 0.20 | 29/30 |
| mappo-early | 482.60 | 464.17 | 18.43 | 0.00 | 4.73 | 0.00 | 0.20 | 29/30 |

| Crossing | Actuated p95 | Original | Continue | Tail | Guided | Early service | Minimum entries across all replicas/controllers |
|---|---:|---:|---:|---:|---:|---:|---:|
| :C0_c0 | 27.40 | 41.73 | 40.73 | 37.00 | 30.47 | 27.73 | 469 |
| :C0_c1 | 28.00 | 42.20 | 40.20 | 34.87 | 29.53 | 28.27 | 976 |
| :C0_c2 | 27.60 | 38.93 | 35.67 | 32.60 | 28.20 | 27.73 | 527 |
| :C0_c3 | 27.60 | 42.40 | 38.60 | 34.67 | 30.60 | 27.40 | 507 |
| :C1_c0 | 27.60 | 28.33 | 30.80 | 28.13 | 28.00 | 27.33 | 469 |
| :C1_c1 | 28.00 | 35.00 | 36.80 | 30.40 | 28.40 | 27.60 | 1005 |
| :C1_c2 | 27.00 | 30.93 | 28.60 | 28.80 | 28.47 | 27.73 | 528 |
| :C1_c3 | 28.20 | 32.67 | 33.93 | 30.33 | 28.53 | 27.93 | 935 |
| :C2_c0 | 27.80 | 33.33 | 34.60 | 28.53 | 28.00 | 27.20 | 535 |
| :C2_c1 | 27.60 | 42.07 | 40.40 | 37.00 | 31.67 | 27.53 | 536 |
| :C2_c2 | 27.60 | 40.67 | 39.80 | 34.73 | 31.60 | 27.33 | 516 |
| :C2_c3 | 28.00 | 41.07 | 39.20 | 35.00 | 29.20 | 27.87 | 1052 |

Early-service minus unchanged continued MAPPO: {"crossing_p95_delta_seconds": {":C0_c0": -12.999999999999963, ":C0_c1": -11.933333333333318, ":C0_c2": -7.933333333333356, ":C0_c3": -11.199999999999969, ":C1_c0": -3.466666666666621, ":C1_c1": -9.199999999999932, ":C1_c2": -0.8666666666666932, ":C1_c3": -5.999999999999943, ":C2_c0": -7.4000000000000155, ":C2_c1": -12.86666666666675, ":C2_c2": -12.46666666666664, ":C2_c3": -11.333333333333371}, "crossings_no_worse": 12, "insertion_delay_delta_seconds": 0.002100782921427313, "missing_crossings": [], "time_loss_plus_insertion_delay_delta_seconds": 9.496116812672406, "vehicle_delta_seconds": 9.494016029750979, "worst_crossing_delta_seconds": -0.8666666666666932}

The complete summary includes crossing means, per-replica tails, pointwise intervals, count differences, whole-run vehicle/person service, phase occupancy, decision audits, and the strict service-gate components.

## vmedium-plow

| Controller | Vehicle loss s | Change vs actuated s | Paired 95% interval | Crossings no worse /12 | Worst crossing excess s | Sparse? |
|---|---:|---:|---|---:|---:|---|
| fixed | 33.09 | -7.11 | [-8.57, -5.74] | 0 | 43.80 | False |
| actuated | 40.20 | 0.00 | [0.00, 0.00] | 12 | 0.00 | False |
| pressure | 30.36 | -9.84 | [-11.01, -8.49] | 0 | 18.00 | False |
| mappo-original | 36.89 | -3.31 | [-6.30, -0.01] | 0 | 28.60 | False |
| mappo-continue | 34.06 | -6.14 | [-8.75, -2.91] | 0 | 27.53 | False |
| mappo-tail | 37.15 | -3.05 | [-5.07, -1.07] | 0 | 19.60 | False |
| mappo-guided | 41.06 | 0.86 | [-2.01, 3.36] | 0 | 15.33 | False |
| mappo-early | 45.02 | 4.82 | [3.59, 6.44] | 4 | 1.67 | False |

| Controller | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup | Insertion delay s | Strict service failures / pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed | 953.70 | 917.40 | 36.10 | 0.20 | 4.40 | 0.00 | 0.32 | 10/10 |
| actuated | 953.70 | 918.70 | 34.80 | 0.20 | 1.10 | 0.00 | 0.32 | 0/10 |
| pressure | 953.70 | 925.30 | 28.20 | 0.20 | 3.30 | 0.00 | 0.32 | 9/10 |
| mappo-original | 953.70 | 921.53 | 31.97 | 0.20 | 1.73 | 0.00 | 0.32 | 23/30 |
| mappo-continue | 953.70 | 923.30 | 30.20 | 0.20 | 2.60 | 0.00 | 0.32 | 23/30 |
| mappo-tail | 953.70 | 922.20 | 31.30 | 0.20 | 1.60 | 0.00 | 0.32 | 21/30 |
| mappo-guided | 953.70 | 919.00 | 34.50 | 0.20 | 2.17 | 0.00 | 0.33 | 24/30 |
| mappo-early | 953.70 | 918.80 | 34.70 | 0.20 | 1.43 | 0.00 | 0.32 | 20/30 |

| Crossing | Actuated p95 | Original | Continue | Tail | Guided | Early service | Minimum entries across all replicas/controllers |
|---|---:|---:|---:|---:|---:|---:|---:|
| :C0_c0 | 27.00 | 49.40 | 49.27 | 43.67 | 41.73 | 26.60 | 173 |
| :C0_c1 | 27.20 | 47.53 | 47.07 | 43.60 | 39.07 | 27.07 | 327 |
| :C0_c2 | 25.60 | 43.27 | 42.87 | 40.87 | 40.93 | 27.27 | 162 |
| :C0_c3 | 26.40 | 47.80 | 47.33 | 42.53 | 38.33 | 26.67 | 166 |
| :C1_c0 | 27.00 | 34.53 | 37.27 | 36.00 | 32.40 | 27.40 | 173 |
| :C1_c1 | 26.60 | 43.07 | 44.27 | 41.60 | 33.80 | 27.27 | 355 |
| :C1_c2 | 26.60 | 36.33 | 40.93 | 38.33 | 28.40 | 27.60 | 185 |
| :C1_c3 | 27.00 | 42.20 | 44.13 | 40.40 | 34.53 | 27.53 | 318 |
| :C2_c0 | 25.60 | 44.07 | 44.40 | 42.00 | 31.13 | 27.20 | 178 |
| :C2_c1 | 27.40 | 56.00 | 54.93 | 47.00 | 37.27 | 27.27 | 146 |
| :C2_c2 | 27.80 | 49.87 | 54.27 | 43.67 | 42.00 | 27.73 | 177 |
| :C2_c3 | 27.20 | 48.27 | 46.33 | 42.47 | 39.27 | 27.60 | 352 |

Early-service minus unchanged continued MAPPO: {"crossing_p95_delta_seconds": {":C0_c0": -22.66666666666661, ":C0_c1": -20.000000000000227, ":C0_c2": -15.599999999999929, ":C0_c3": -20.666666666666476, ":C1_c0": -9.866666666666712, ":C1_c1": -17.0, ":C1_c2": -13.333333333333409, ":C1_c3": -16.599999999999834, ":C2_c0": -17.19999999999997, ":C2_c1": -27.666666666666668, ":C2_c2": -26.533333333333303, ":C2_c3": -18.73333333333335}, "crossings_no_worse": 12, "insertion_delay_delta_seconds": -0.0017762931409045927, "missing_crossings": [], "time_loss_plus_insertion_delay_delta_seconds": 10.951234053096865, "vehicle_delta_seconds": 10.953010346237768, "worst_crossing_delta_seconds": -9.866666666666712}

The complete summary includes crossing means, per-replica tails, pointwise intervals, count differences, whole-run vehicle/person service, phase occupancy, decision audits, and the strict service-gate components.

## vmedium-pmedium

| Controller | Vehicle loss s | Change vs actuated s | Paired 95% interval | Crossings no worse /12 | Worst crossing excess s | Sparse? |
|---|---:|---:|---|---:|---:|---|
| fixed | 33.09 | -18.08 | [-18.94, -17.24] | 0 | 42.60 | False |
| actuated | 51.17 | 0.00 | [0.00, 0.00] | 12 | 0.00 | False |
| pressure | 40.43 | -10.74 | [-11.96, -9.56] | 2 | 8.80 | False |
| mappo-original | 40.85 | -10.32 | [-12.03, -8.32] | 0 | 16.33 | False |
| mappo-continue | 38.43 | -12.74 | [-14.38, -11.02] | 0 | 17.67 | False |
| mappo-tail | 41.94 | -9.23 | [-10.10, -8.37] | 0 | 14.07 | False |
| mappo-guided | 46.91 | -4.26 | [-5.26, -3.09] | 0 | 11.27 | False |
| mappo-early | 54.97 | 3.80 | [1.98, 5.68] | 6 | 0.87 | False |

| Controller | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup | Insertion delay s | Strict service failures / pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed | 953.70 | 917.40 | 36.10 | 0.20 | 14.90 | 0.00 | 0.32 | 10/10 |
| actuated | 953.70 | 914.10 | 39.40 | 0.20 | 4.40 | 0.00 | 0.32 | 0/10 |
| pressure | 953.70 | 920.70 | 32.80 | 0.20 | 6.70 | 0.00 | 0.32 | 10/10 |
| mappo-original | 953.70 | 919.40 | 34.10 | 0.20 | 6.13 | 0.00 | 0.32 | 25/30 |
| mappo-continue | 953.70 | 921.27 | 32.20 | 0.23 | 7.10 | 0.00 | 0.32 | 24/30 |
| mappo-tail | 953.70 | 920.10 | 33.40 | 0.20 | 6.17 | 0.00 | 0.34 | 27/30 |
| mappo-guided | 953.70 | 919.27 | 34.23 | 0.20 | 5.27 | 0.00 | 0.32 | 21/30 |
| mappo-early | 953.70 | 912.60 | 40.83 | 0.27 | 4.87 | 0.00 | 0.36 | 26/30 |

| Crossing | Actuated p95 | Original | Continue | Tail | Guided | Early service | Minimum entries across all replicas/controllers |
|---|---:|---:|---:|---:|---:|---:|---:|
| :C0_c0 | 27.40 | 43.73 | 42.07 | 39.27 | 33.87 | 27.60 | 506 |
| :C0_c1 | 27.80 | 43.07 | 42.40 | 40.93 | 36.73 | 28.20 | 938 |
| :C0_c2 | 27.80 | 41.53 | 35.73 | 32.47 | 28.47 | 27.73 | 506 |
| :C0_c3 | 27.60 | 42.47 | 42.33 | 39.93 | 35.80 | 27.53 | 471 |
| :C1_c0 | 27.60 | 32.73 | 29.20 | 31.40 | 27.93 | 27.60 | 510 |
| :C1_c1 | 28.00 | 39.07 | 41.13 | 38.20 | 29.67 | 27.67 | 963 |
| :C1_c2 | 27.40 | 33.93 | 36.80 | 32.93 | 28.20 | 28.00 | 476 |
| :C1_c3 | 27.80 | 37.93 | 40.27 | 38.87 | 29.67 | 27.93 | 983 |
| :C2_c0 | 27.60 | 37.80 | 41.27 | 34.93 | 29.00 | 27.00 | 493 |
| :C2_c1 | 26.40 | 42.07 | 44.07 | 40.47 | 37.67 | 27.27 | 476 |
| :C2_c2 | 27.60 | 41.87 | 43.33 | 39.40 | 33.60 | 27.07 | 535 |
| :C2_c3 | 27.80 | 42.27 | 43.00 | 40.47 | 33.00 | 27.93 | 976 |

Early-service minus unchanged continued MAPPO: {"crossing_p95_delta_seconds": {":C0_c0": -14.466666666666677, ":C0_c1": -14.199999999999969, ":C0_c2": -8.000000000000114, ":C0_c3": -14.800000000000031, ":C1_c0": -1.6000000000000039, ":C1_c1": -13.46666666666662, ":C1_c2": -8.800000000000201, ":C1_c3": -12.333333333333561, ":C2_c0": -14.266666666666765, ":C2_c1": -16.799999999999994, ":C2_c2": -16.266666666666577, ":C2_c3": -15.066666666666643}, "crossings_no_worse": 12, "insertion_delay_delta_seconds": 0.043061961248991594, "missing_crossings": [], "time_loss_plus_insertion_delay_delta_seconds": 16.580717749583194, "vehicle_delta_seconds": 16.537655788334202, "worst_crossing_delta_seconds": -1.6000000000000039}

The complete summary includes crossing means, per-replica tails, pointwise intervals, count differences, whole-run vehicle/person service, phase occupancy, decision audits, and the strict service-gate components.

## Limitations

- Synthetic network-wide Poisson arrival rates, not calibrated Singapore counts
- Demand matches across controllers within a cell; changing vehicle rate changes shared-RNG pedestrian draws across cells
- Three trained replicas; new development analysis, not a confirmatory test
- Early service is a hybrid rule plus MAPPO, not a newly trained MARL algorithm or a p95 guarantee
- Normal walking speed only in this factorial; existing slow-walker results remain part of the project
- 600-second empty-reset training versus 2400-second evaluation remains unchanged

No sparse or missing crossing is counted as zero wait. A partial crossing count or favorable setting does not replace the original every-crossing, no-service-deterioration target. An unchanged model at low demand tests transfer; it is not evidence about a model retrained for low demand.

Artifact verification: 720 complete records and both compressed event/decision hashes per record. 449874 recorded entries checked against the signal active during movement.
