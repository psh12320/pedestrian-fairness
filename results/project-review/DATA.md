# Project data review - 2 November 2026

This index reads saved results. It does not rerun experiments or combine different study versions.

## What is available

| Study | Setting | Available evidence |
|---|---|---|
| Single-junction milestone | One signal, four crossings | Local raw network, demand, events and metrics |
| Initial exploratory study | Three signals, ten demand/speed scenarios | Complete local summaries; one trained model per learned configuration |
| Earlier repeated-training study | Three signals, 13 scenarios; linear hold/change policies; five training seeds, ten evaluation seeds | Complete local absolute results and event tables |
| Fairness v2 cluster study | Three signals, 21 scenarios; neural phase-selection policies; five training seeds, twenty evaluation seeds | Learned-controller vehicle differences for all scenarios and primary crossing differences. Complete held-out baseline/raw evaluation records are still on the cluster |
| NUS-derived transfer | Three synthetic junctions positioned using approximate map-derived centroids; synthetic demand | Complete local earlier-controller transfer results; not a calibrated Singapore network |
| Service-priority follow-up | Same physical v2 environment; short learning smoke and two full-length conventional development seeds | Engineering checks and pilot results only |

The latest cluster run has 21 x 20 = 420 evaluation groups. Each group contains three conventional controllers and six learned configurations x five replicas = 33 instances, or 13,860 episodes. The four original learned configurations remain the core comparison; the two constrained configurations are extensions.

The downloaded comparisons.json is a truncated tail, not valid complete JSON. Do not reconstruct missing absolute results from it. The full REPORT.md, protocol.json and primary_inference.json are readable. No latest held-out absolute baseline values are invented here.

## Junction audit

Saved corridor network: signals C0, C1, C2; 12 crossing edges. The cluster protocol independently records actor observations for C0, C1 and C2. Four crossings per junction is not four junctions.

The five environment/demand/signal/configuration source hashes checked here exactly match the cluster protocol. This verifies which layout and rules the pipeline specified; it is not a fresh end-to-end audit of every cluster trajectory.

## Latest study: primary crossing result

A crossing is descriptively no worse when its pooled p95 point difference is <= 0. This is a secondary count, not proof of statistical non-inferiority. Missing/insufficient events must remain unevaluable, never successful zero waits. The original all-crossings target and service counts remain primary.

| Configuration | Vehicle difference vs actuated, s | Crossings no worse / 12 | Largest p95 difference, s | Minimum reported pooled crossing entries | Service/measurement gate failed runs / 100 |
|---|---:|---:|---:|---:|---:|
| ippo-constrained | +1.950 | 4/12 | 1.16 | 969 | 49 |
| ippo-pedestrian | +1.524 | 3/12 | 1.20 | 968 | 47 |
| ippo-vehicle | -36.761 | unevaluable from available summary | unavailable | 0 | 100 |
| mappo-constrained | +1.643 | 3/12 | 1.16 | 968 | 42 |
| mappo-pedestrian | +1.344 | 3/12 | 1.16 | 969 | 50 |
| mappo-vehicle | -36.498 | unevaluable from available summary | unavailable | 0 | 100 |

Service/measurement flags combine several checks. They are not collision counts, and their summed score is not a count of unique unserved people.

### Every measured primary crossing

| Crossing | IPPO pedestrian p95 difference, s | MAPPO pedestrian p95 difference, s |
|---|---:|---:|
| :C0_c0 | +0.16 | +0.24 |
| :C0_c1 | +1.20 | +1.16 |
| :C0_c2 | -0.68 | -0.76 |
| :C0_c3 | +0.20 | +0.24 |
| :C1_c0 | -0.60 | -0.64 |
| :C1_c1 | +0.72 | +0.72 |
| :C1_c2 | +0.40 | +0.48 |
| :C1_c3 | +0.48 | +0.52 |
| :C2_c0 | -0.60 | -0.60 |
| :C2_c1 | +0.32 | +0.20 |
| :C2_c2 | +1.00 | +1.00 |
| :C2_c3 | +0.16 | +0.08 |

Positive means a longer wait than actuated. Differences are point estimates, not confidence intervals. Primary crossing data: [runs/cluster-final-review/primary_inference.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/cluster-final-review/primary_inference.json)

## Latest study: all 21 scenarios

Rates below are synthetic expected arrivals per hour across the network, not per junction or lane. The default demand interval is 0-2100 s; run end is 2400 s. Crossing-arrival measurement cohort: 300-2100 s, with follow-up to 2400 s. Baseline vehicle/pedestrian labels are scenario assumptions, not field calibration or guaranteed saturation levels.

| Scenario | Vehicles/hour | People/hour | Additional change |
|---|---:|---:|---|
| vlow-plow | 960 | 400 | none |
| vlow-pmedium | 960 | 1200 | none |
| vlow-phigh | 960 | 2000 | none |
| vmedium-plow | 1920 | 400 | none |
| vmedium-pmedium | 1920 | 1200 | none |
| vmedium-phigh | 1920 | 2000 | none |
| vhigh-plow | 2880 | 400 | none |
| vhigh-pmedium | 2880 | 1200 | none |
| vhigh-phigh | 2880 | 2000 | none |
| slow-walkers | 1920 | 1200 | {"slow_fraction": 1.0} |
| mixed-walkers | 1920 | 1200 | {"slow_fraction": 0.2} |
| pedestrian-burst | 1920 | 1200 | {"burst_count": 20} |
| extreme-burst | 1920 | 1200 | {"burst_count": 80} |
| vehicle-overload | 8000 | 1200 | none |
| short-spacing | 4800 | 1200 | {"junction_spacing_m": 65} |
| directional-imbalance | 1920 | 1200 | {"directional_ratio": 3.0} |
| sparse-pedestrian-route | 1920 | 1200 | {"sparse_route_factor": 0.1} |
| vehicle-platoons | 1920 | 1200 | {"arrival_profile": "platoons"} |
| demand-ramp | 1920 | 1200 | {"arrival_profile": "ramp"} |
| higher-turning-share | 1920 | 1200 | {"turn_fraction": 0.3} |
| heterogeneous-drivers | 1920 | 1200 | {"heterogeneous_drivers": true} |

### Vehicle time-loss differences vs actuated (seconds)

| Scenario | IPPO vehicle | IPPO pedestrian | MAPPO vehicle | MAPPO pedestrian |
|---|---:|---:|---:|---:|
| vlow-plow | -20.522 | +10.960 | -20.146 | +11.238 |
| vlow-pmedium | -28.950 | +2.988 | -28.553 | +2.916 |
| vlow-phigh | -30.044 | +1.739 | -29.420 | +1.575 |
| vmedium-plow | -24.749 | +12.378 | -24.404 | +12.721 |
| vmedium-pmedium | -36.761 | +1.524 | -36.498 | +1.344 |
| vmedium-phigh | -38.044 | +0.246 | -37.529 | +0.084 |
| vhigh-plow | -37.238 | +36.652 | -36.932 | +34.254 |
| vhigh-pmedium | -82.012 | -3.765 | -81.717 | -4.912 |
| vhigh-phigh | -87.449 | -5.847 | -86.977 | -7.919 |
| slow-walkers | -37.766 | +0.805 | -37.405 | +0.756 |
| mixed-walkers | -37.512 | +0.711 | -37.231 | +0.748 |
| pedestrian-burst | -36.747 | +1.548 | -36.452 | +1.372 |
| extreme-burst | -36.732 | +1.559 | -36.416 | +1.384 |
| vehicle-overload | -86.777 | -16.946 | -86.978 | -27.177 |
| short-spacing | -13.125 | +9.772 | -12.712 | +9.529 |
| directional-imbalance | -90.665 | -7.141 | -90.108 | -4.398 |
| sparse-pedestrian-route | -36.595 | +1.517 | -36.067 | +1.388 |
| vehicle-platoons | -37.039 | +2.039 | -36.958 | +1.742 |
| demand-ramp | -47.589 | +1.433 | -47.341 | +1.679 |
| higher-turning-share | -36.141 | +1.622 | -35.775 | +1.786 |
| heterogeneous-drivers | -40.335 | +1.093 | -39.820 | +1.191 |

Read the differences with the service failures below. A negative completed-trip mean does not establish success when traffic remains unserved.

### Paired runs failing the service/measurement gate (out of 100)

| Scenario | IPPO vehicle | IPPO pedestrian | MAPPO vehicle | MAPPO pedestrian |
|---|---:|---:|---:|---:|
| vlow-plow | 100 | 57 | 100 | 66 |
| vlow-pmedium | 100 | 32 | 100 | 31 |
| vlow-phigh | 100 | 64 | 100 | 55 |
| vmedium-plow | 100 | 62 | 100 | 67 |
| vmedium-pmedium | 100 | 47 | 100 | 50 |
| vmedium-phigh | 100 | 38 | 100 | 39 |
| vhigh-plow | 100 | 92 | 100 | 94 |
| vhigh-pmedium | 100 | 89 | 100 | 88 |
| vhigh-phigh | 100 | 85 | 100 | 80 |
| slow-walkers | 100 | 27 | 100 | 37 |
| mixed-walkers | 100 | 31 | 100 | 37 |
| pedestrian-burst | 100 | 47 | 100 | 50 |
| extreme-burst | 100 | 47 | 100 | 49 |
| vehicle-overload | 100 | 92 | 100 | 90 |
| short-spacing | 100 | 69 | 100 | 67 |
| directional-imbalance | 100 | 98 | 100 | 98 |
| sparse-pedestrian-route | 100 | 33 | 100 | 42 |
| vehicle-platoons | 100 | 51 | 100 | 54 |
| demand-ramp | 100 | 52 | 100 | 52 |
| higher-turning-share | 100 | 37 | 100 | 45 |
| heterogeneous-drivers | 100 | 53 | 100 | 49 |

Full six-configuration comparison: [runs/cluster-final-review/REPORT.md](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/cluster-final-review/REPORT.md)

## Earlier repeated-training study: absolute results

Historical results, NOT the latest cluster study. Five independent training replicas, ten demand seeds, 150 training episodes; linear hold/change controllers. Pedestrian weights 0, 3 and 10. Means and existing 95% bootstrap intervals are reproduced from the original summary; no new uncertainty calculation is performed here. Counts are mean counts per run.

Source: [results/final-study/summary.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/final-study/summary.json)

### pedestrian-burst

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 35.13 [33.91, 36.34] | 25.91 [23.80, 28.11] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| actuated | 34.09 [32.85, 35.46] | 17.39 [16.94, 17.85] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| pressure | 29.48 [28.35, 30.50] | 17.00 [16.33, 17.80] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 33.26 [31.82, 34.85] | 19.11 [17.78, 20.70] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 34.08 [31.93, 36.43] | 16.89 [16.01, 17.77] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 33.10 [30.72, 35.72] | 17.05 [16.16, 17.76] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 33.66 [31.57, 35.78] | 20.76 [18.27, 23.78] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 33.90 [31.58, 36.21] | 17.15 [16.16, 18.30] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 33.69 [31.63, 35.97] | 16.94 [16.07, 17.88] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |

### short-spacing

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 59.47 [57.88, 61.64] | 25.72 [24.72, 26.68] | 624.00 | 582.50 | 23.30 | 18.20 | 0.00 |
| actuated | 81.76 [77.84, 86.06] | 16.04 [15.30, 16.70] | 624.00 | 527.20 | 47.30 | 49.50 | 0.00 |
| pressure | 80.69 [77.80, 83.71] | 17.47 [16.00, 18.95] | 624.00 | 532.20 | 47.70 | 44.10 | 0.00 |
| ippo-w0 | 78.91 [76.76, 81.87] | 17.64 [15.86, 19.75] | 624.00 | 520.48 | 49.66 | 53.86 | 0.00 |
| ippo-w10 | 80.23 [77.13, 84.00] | 16.18 [15.24, 16.99] | 624.00 | 532.60 | 49.90 | 41.50 | 0.00 |
| ippo-w3 | 81.17 [78.60, 84.00] | 16.64 [15.76, 17.53] | 624.00 | 528.68 | 52.18 | 43.14 | 0.00 |
| mappo-w0 | 76.94 [74.12, 79.92] | 20.07 [18.07, 22.11] | 624.00 | 510.26 | 51.52 | 62.22 | 0.00 |
| mappo-w10 | 82.73 [79.52, 86.42] | 17.22 [15.97, 18.45] | 624.00 | 520.72 | 57.06 | 46.22 | 0.00 |
| mappo-w3 | 81.57 [78.40, 84.64] | 16.67 [15.66, 17.68] | 624.00 | 526.72 | 53.90 | 43.38 | 0.00 |

### vehicle-overload

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 44.51 [43.98, 45.06] | 25.64 [24.84, 26.55] | 860.00 | 713.30 | 94.50 | 52.20 | 0.00 |
| actuated | 72.13 [69.08, 75.35] | 16.23 [15.82, 16.63] | 860.00 | 580.00 | 82.90 | 197.10 | 0.00 |
| pressure | 70.11 [67.91, 72.38] | 16.82 [16.12, 17.50] | 860.00 | 609.20 | 91.70 | 159.10 | 0.00 |
| ippo-w0 | 67.85 [62.14, 72.90] | 17.36 [15.94, 19.32] | 860.00 | 608.02 | 80.80 | 171.18 | 0.00 |
| ippo-w10 | 72.58 [69.68, 75.49] | 15.99 [15.45, 16.51] | 860.00 | 592.70 | 90.80 | 176.50 | 0.00 |
| ippo-w3 | 73.65 [71.00, 76.31] | 16.56 [15.78, 17.44] | 860.00 | 584.26 | 91.30 | 184.44 | 0.00 |
| mappo-w0 | 64.48 [59.86, 69.01] | 20.10 [18.00, 22.43] | 860.00 | 614.62 | 77.74 | 167.64 | 0.00 |
| mappo-w10 | 74.16 [71.71, 76.29] | 17.07 [15.86, 18.23] | 860.00 | 579.24 | 92.06 | 188.70 | 0.00 |
| mappo-w3 | 73.19 [70.81, 75.47] | 16.65 [15.82, 17.61] | 860.00 | 586.02 | 90.62 | 183.36 | 0.00 |

### vhigh-phigh

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 39.72 [38.77, 40.64] | 26.07 [25.08, 26.94] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| actuated | 39.94 [38.42, 41.61] | 16.18 [15.75, 16.60] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| pressure | 33.85 [31.92, 36.13] | 16.16 [15.68, 16.73] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 39.33 [37.30, 42.36] | 18.09 [16.48, 19.86] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 38.12 [36.36, 39.92] | 16.20 [15.53, 16.82] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 37.55 [35.15, 39.62] | 16.57 [15.71, 17.43] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 40.88 [38.41, 42.96] | 20.39 [18.45, 22.61] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 38.08 [36.52, 39.69] | 16.36 [15.68, 17.05] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 38.10 [36.33, 40.02] | 16.46 [15.71, 17.27] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |

### vhigh-plow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 40.02 [38.91, 41.01] | 25.91 [24.53, 27.38] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| actuated | 35.83 [34.13, 37.94] | 15.35 [14.02, 16.61] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| pressure | 30.13 [28.69, 31.60] | 16.76 [15.83, 17.68] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 34.26 [32.74, 36.44] | 16.96 [15.66, 18.14] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 32.99 [31.99, 33.90] | 16.75 [15.44, 17.83] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 32.75 [31.94, 33.66] | 16.72 [15.44, 17.77] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 36.30 [33.96, 38.88] | 18.93 [17.55, 20.50] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 33.33 [32.14, 34.49] | 16.79 [15.46, 17.87] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 32.76 [31.59, 33.98] | 16.55 [15.34, 17.55] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |

### vhigh-pmedium

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 39.48 [38.40, 40.67] | 25.52 [24.90, 26.12] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| actuated | 39.47 [37.52, 41.31] | 16.22 [15.45, 17.00] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| pressure | 33.49 [31.78, 35.06] | 16.41 [15.68, 17.03] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 38.35 [36.16, 40.98] | 17.54 [16.09, 19.27] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 37.09 [35.29, 38.85] | 16.04 [15.52, 16.52] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 36.37 [34.58, 38.40] | 16.16 [15.68, 16.64] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 40.01 [37.80, 42.04] | 19.57 [17.61, 21.80] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 37.18 [35.41, 38.83] | 16.12 [15.60, 16.59] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 36.91 [35.19, 38.69] | 16.27 [15.62, 16.95] | 288.00 | 288.00 | 0.00 | 0.00 | 0.00 |

### vlow-phigh

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 30.40 [29.25, 31.41] | 26.44 [25.73, 27.19] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| actuated | 31.38 [30.15, 32.59] | 15.77 [15.47, 16.14] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| pressure | 28.15 [26.85, 29.65] | 16.41 [15.92, 16.93] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 30.77 [29.30, 32.33] | 17.92 [16.93, 18.98] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 30.99 [29.17, 32.75] | 16.60 [15.96, 17.38] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 30.37 [28.56, 32.35] | 16.67 [16.03, 17.41] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 31.56 [30.33, 32.87] | 20.08 [18.14, 22.33] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 30.99 [29.17, 32.75] | 16.60 [15.96, 17.38] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 31.10 [29.34, 32.74] | 16.61 [15.97, 17.34] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |

### vlow-plow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 30.43 [29.21, 31.47] | 25.93 [23.58, 28.11] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| actuated | 29.73 [28.49, 31.08] | 16.00 [14.85, 17.15] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| pressure | 24.62 [23.57, 25.65] | 16.16 [14.93, 17.47] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 26.77 [25.43, 28.05] | 16.98 [15.45, 18.38] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 26.10 [24.81, 27.62] | 16.48 [14.90, 17.85] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 26.25 [24.90, 27.63] | 16.42 [14.89, 17.66] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 27.59 [26.55, 28.79] | 18.38 [16.97, 19.78] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 26.10 [24.81, 27.62] | 16.48 [14.90, 17.85] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 26.20 [24.87, 27.75] | 16.48 [14.91, 17.81] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |

### vlow-pmedium

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 30.36 [29.03, 31.69] | 25.85 [24.73, 26.87] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| actuated | 30.14 [28.80, 31.81] | 16.08 [15.17, 17.07] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| pressure | 26.95 [26.14, 27.79] | 16.57 [15.75, 17.40] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 30.03 [28.85, 31.33] | 17.50 [16.48, 18.56] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 30.20 [28.63, 32.12] | 16.57 [16.05, 17.01] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 29.69 [27.96, 31.62] | 16.59 [16.05, 17.06] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 30.31 [29.09, 31.65] | 18.94 [17.54, 20.35] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 30.20 [28.63, 32.12] | 16.57 [16.05, 17.01] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 30.00 [28.57, 31.84] | 16.51 [15.98, 16.96] | 96.00 | 96.00 | 0.00 | 0.00 | 0.00 |

### vmedium-phigh

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 35.94 [34.48, 37.30] | 25.71 [25.11, 26.30] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| actuated | 35.21 [34.25, 35.96] | 15.92 [15.16, 16.76] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| pressure | 30.02 [29.06, 30.97] | 16.18 [15.37, 17.01] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 32.81 [31.37, 34.26] | 17.68 [16.55, 18.86] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 32.43 [30.49, 34.47] | 15.97 [15.47, 16.46] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 31.81 [30.01, 33.83] | 16.29 [15.65, 17.00] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 34.03 [32.05, 35.91] | 20.07 [17.93, 22.41] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 32.46 [30.53, 34.50] | 15.99 [15.48, 16.50] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 32.14 [30.31, 34.06] | 16.11 [15.54, 16.66] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |

### vmedium-plow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 35.79 [34.87, 36.71] | 25.99 [24.45, 27.45] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| actuated | 32.30 [31.20, 33.39] | 16.17 [15.11, 17.20] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| pressure | 27.60 [26.55, 28.53] | 16.11 [14.98, 17.33] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 31.28 [29.80, 32.80] | 17.49 [16.29, 18.51] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 31.40 [29.38, 33.38] | 16.79 [15.79, 17.75] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 30.96 [29.01, 32.98] | 16.81 [15.85, 17.74] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 31.72 [30.11, 33.28] | 18.78 [17.73, 19.81] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 31.45 [29.53, 33.38] | 16.81 [15.89, 17.71] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 31.15 [29.18, 33.12] | 16.68 [15.74, 17.64] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |

### vmedium-pmedium

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 35.24 [34.02, 36.53] | 24.58 [23.50, 25.55] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| actuated | 34.70 [33.30, 36.45] | 16.22 [15.68, 16.69] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| pressure | 29.65 [28.60, 30.58] | 15.87 [15.24, 16.56] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 33.70 [32.01, 35.66] | 17.64 [16.33, 19.18] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 34.40 [31.99, 36.98] | 16.38 [15.72, 17.01] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 33.31 [30.71, 36.34] | 16.51 [15.87, 17.02] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 33.54 [31.48, 35.76] | 19.17 [17.68, 20.79] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 34.40 [31.99, 36.96] | 16.39 [15.72, 17.02] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 34.06 [31.72, 36.61] | 16.39 [15.72, 17.02] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |

### vmedium-pmedium-slow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 35.26 [34.04, 36.56] | 35.90 [33.99, 37.59] | 192.00 | 192.00 | 0.00 | 0.00 | 3.30 |
| actuated | 30.53 [29.11, 32.20] | 22.54 [21.42, 23.58] | 192.00 | 192.00 | 0.00 | 0.00 | 0.80 |
| pressure | 27.76 [26.72, 28.95] | 22.25 [21.51, 22.94] | 192.00 | 192.00 | 0.00 | 0.00 | 0.50 |
| ippo-w0 | 30.69 [29.43, 31.88] | 24.46 [22.85, 26.12] | 192.00 | 192.00 | 0.00 | 0.00 | 1.22 |
| ippo-w10 | 30.67 [28.89, 32.24] | 22.79 [21.76, 23.76] | 192.00 | 192.00 | 0.00 | 0.00 | 0.90 |
| ippo-w3 | 30.60 [29.16, 32.02] | 22.88 [21.90, 23.83] | 192.00 | 192.00 | 0.00 | 0.00 | 0.88 |
| mappo-w0 | 30.78 [29.29, 32.55] | 26.81 [23.89, 30.31] | 192.00 | 192.00 | 0.00 | 0.00 | 1.74 |
| mappo-w10 | 30.67 [28.90, 32.22] | 22.77 [21.75, 23.74] | 192.00 | 192.00 | 0.00 | 0.00 | 0.90 |
| mappo-w3 | 30.36 [28.75, 31.95] | 22.76 [21.79, 23.71] | 192.00 | 192.00 | 0.00 | 0.00 | 0.90 |

### Earlier primary crossing counts

| Controller | Crossings no worse / 12 (point estimates) | Original all-crossings descriptive target |
|---|---:|---|
| fixed | 0/12 | False |
| ippo-w0 | 3/12 | False |
| ippo-w10 | 9/12 | False |
| ippo-w3 | 8/12 | False |
| mappo-w0 | 0/12 | False |
| mappo-w10 | 9/12 | False |
| mappo-w3 | 8/12 | False |
| pressure | 9/12 | False |

Actuated is the reference: 12/12 by identity, not an improvement result.

## NUS-derived transfer: earlier controllers

- Three approximate candidate-site centroids and spacings are map-derived; site identities, geometry and crossing locations are not field-verified.
- Connecting links, side arms, sidewalks, protected crossings and all demand are synthetic project assumptions; no Singapore turning or pedestrian counts or deployed signal timing plans were used.
- The raw OSM import had no signal-controlled pedestrian crossings, so the protected derived network is not a reconstructed field junction.
- A sparse crossing p95 or incomplete bootstrap makes the every-crossing criterion inconclusive even if its point estimate appears favorable.

The raw OSM import contains 20 traffic-light program IDs; these may include separate signal nodes within a physical junction. That import is separate from the three-agent synthetic derived evaluation network. It is not evidence that the main experiment used twenty or four learning junctions.

Source: [results/singapore-transfer/summary.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/singapore-transfer/summary.json)

### nus-derived-synthetic-demand

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 36.47 [34.99, 38.06] | 15.14 [14.25, 16.05] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| fixed | 42.75 [41.25, 44.29] | 25.52 [24.46, 26.63] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w0 | 36.04 [33.41, 38.70] | 16.99 [15.41, 18.51] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w10 | 35.97 [33.36, 38.72] | 15.26 [14.37, 16.06] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-w3 | 35.50 [32.66, 38.12] | 15.52 [14.66, 16.35] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w0 | 37.21 [34.84, 39.55] | 19.16 [16.92, 21.39] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w10 | 35.98 [33.45, 38.75] | 15.34 [14.51, 16.13] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-w3 | 35.53 [32.83, 38.30] | 15.35 [14.54, 16.14] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |
| pressure | 29.37 [27.07, 31.94] | 16.29 [15.33, 17.29] | 192.00 | 192.00 | 0.00 | 0.00 | 0.00 |

## Initial exploratory study: historical only

One trained model per learned configuration and five evaluation seeds; sparse pedestrian samples. Keep separate from the later repeated-training and v2 studies. Scheduled counts were not included in this summary and remain unavailable here.

Source: [results/summary.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/summary.json)

### vhigh-phigh

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 37.01 [33.73, 40.67] | 17.55 [16.43, 19.19] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| fixed | 40.28 [38.37, 42.17] | 24.59 [22.11, 26.43] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 31.07 [28.16, 34.36] | 15.51 [13.86, 16.59] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 33.87 [30.84, 36.62] | 17.16 [14.23, 19.59] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 33.83 [30.73, 36.65] | 16.89 [14.07, 19.27] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 33.26 [30.27, 36.24] | 19.44 [15.90, 21.96] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |

### vhigh-plow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 31.07 [29.23, 33.03] | 13.89 [11.16, 16.38] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| fixed | 39.55 [38.06, 41.05] | 21.27 [16.20, 25.04] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 24.09 [22.85, 25.48] | 14.11 [11.69, 16.82] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 25.61 [24.55, 26.60] | 14.11 [11.69, 16.82] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 25.61 [24.55, 26.60] | 14.11 [11.69, 16.82] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 26.93 [25.94, 27.92] | 20.91 [17.07, 25.16] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |

### vhigh-pmedium

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 35.54 [32.98, 38.45] | 16.06 [14.18, 18.68] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| fixed | 39.75 [37.90, 41.61] | 26.39 [21.98, 30.67] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 27.09 [26.49, 27.69] | 15.12 [12.92, 17.49] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 29.91 [27.75, 32.05] | 19.11 [17.18, 20.61] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 29.80 [27.75, 31.72] | 19.02 [17.18, 20.51] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 29.08 [26.57, 31.59] | 18.85 [16.69, 21.10] | unavailable | 288.00 | 0.00 | 0.00 | 0.00 |

### vlow-phigh

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 28.03 [25.99, 29.71] | 14.77 [14.34, 15.26] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| fixed | 33.59 [31.94, 35.27] | 25.33 [22.66, 27.84] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 25.18 [22.60, 27.76] | 17.52 [14.97, 19.39] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 26.78 [24.20, 30.07] | 20.42 [16.73, 23.54] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 26.77 [24.20, 30.04] | 20.37 [16.68, 23.54] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 26.63 [25.82, 27.25] | 20.65 [18.54, 22.77] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |

### vlow-plow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 23.34 [22.07, 25.23] | 15.02 [11.65, 17.47] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| fixed | 32.94 [31.41, 34.63] | 19.44 [14.75, 24.47] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 19.41 [17.96, 20.67] | 16.56 [13.95, 19.69] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 21.26 [20.23, 22.55] | 17.44 [13.95, 20.93] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 21.26 [20.23, 22.55] | 17.44 [13.95, 20.93] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 21.69 [21.07, 22.20] | 16.05 [13.22, 18.84] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |

### vlow-pmedium

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 28.05 [24.99, 30.61] | 15.01 [13.36, 16.66] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| fixed | 33.35 [31.10, 35.46] | 22.52 [20.66, 23.99] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 23.54 [21.23, 25.67] | 16.92 [15.29, 18.37] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 25.01 [22.95, 26.87] | 19.64 [16.47, 22.98] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 25.01 [22.95, 26.87] | 19.64 [16.47, 22.98] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 24.94 [24.26, 26.03] | 19.10 [16.33, 22.13] | unavailable | 96.00 | 0.00 | 0.00 | 0.00 |

### vmedium-phigh

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 31.01 [29.36, 32.56] | 15.61 [13.40, 17.60] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| fixed | 36.23 [34.50, 38.09] | 24.02 [22.02, 26.86] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 27.36 [25.77, 29.14] | 14.35 [13.50, 15.54] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 29.39 [27.42, 31.39] | 17.58 [16.33, 18.64] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 29.31 [27.42, 31.23] | 17.51 [16.33, 18.57] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 28.51 [26.67, 30.34] | 20.01 [17.84, 22.34] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |

### vmedium-plow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 25.24 [22.70, 27.78] | 14.82 [11.93, 19.38] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| fixed | 35.58 [33.69, 37.47] | 26.00 [17.27, 35.60] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 20.82 [20.38, 21.15] | 16.16 [12.24, 20.42] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 22.75 [21.46, 24.12] | 16.78 [12.56, 20.71] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 22.75 [21.46, 24.12] | 16.78 [12.56, 20.71] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 22.67 [21.15, 24.65] | 18.24 [15.07, 21.53] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |

### vmedium-pmedium

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 29.28 [27.45, 30.75] | 16.30 [13.66, 18.93] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| fixed | 35.82 [34.26, 37.61] | 21.39 [19.30, 24.68] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-pedestrian | 24.91 [23.17, 26.91] | 16.77 [13.61, 20.24] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| ippo-vehicle | 27.48 [25.34, 29.51] | 18.18 [15.50, 20.87] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-pedestrian | 27.48 [25.34, 29.51] | 18.18 [15.50, 20.87] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | 26.39 [25.28, 27.50] | 20.76 [19.66, 22.37] | unavailable | 192.00 | 0.00 | 0.00 | 0.00 |

### vmedium-pmedium-slow

| Controller | Vehicle time loss: mean [95% CI], s | Pedestrian mean wait [95% CI], s | Scheduled | Completed | Unfinished | Uninserted | Unserved crossing requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| actuated | 26.81 [24.32, 30.15] | 20.11 [17.44, 22.26] | unavailable | 192.00 | 0.00 | 0.00 | 0.20 |
| fixed | 35.82 [34.26, 37.61] | 33.33 [30.22, 37.16] | unavailable | 192.00 | 0.00 | 0.00 | 0.80 |
| ippo-pedestrian | 25.00 [24.03, 26.04] | 20.58 [16.24, 23.30] | unavailable | 192.00 | 0.00 | 0.00 | 0.20 |
| ippo-vehicle | 26.55 [24.55, 28.79] | 22.96 [18.94, 26.30] | unavailable | 192.00 | 0.00 | 0.00 | 0.40 |
| mappo-pedestrian | 26.55 [24.55, 28.79] | 22.96 [18.94, 26.30] | unavailable | 192.00 | 0.00 | 0.00 | 0.40 |
| mappo-vehicle | 25.94 [25.07, 27.10] | 25.61 [20.98, 30.64] | unavailable | 192.00 | 0.00 | 0.00 | 0.60 |

## Raw data and development checks

- [runs/fixed-seed38/metrics.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/fixed-seed38/metrics.json)
- [runs/actuated-seed38/metrics.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/actuated-seed38/metrics.json)
- [runs/3j-fixed-seed38/metrics.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/3j-fixed-seed38/metrics.json)
- [runs/3j-fixed-seed38/connected.net.xml](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/3j-fixed-seed38/connected.net.xml)
- [results/final-study/runs.csv](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/final-study/runs.csv)
- [results/final-study/crossing_events.csv](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/final-study/crossing_events.csv)
- [results/final-study/crossing_comparison.csv](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/final-study/crossing_comparison.csv)
- [results/final-study/phase_history.csv](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/final-study/phase_history.csv)
- [results/measurement-resolution](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/measurement-resolution)
- [results/singapore-transfer](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/singapore-transfer)
- [results/service-priority/LOCAL_PILOT.md](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/results/service-priority/LOCAL_PILOT.md)
- [runs/service-priority-medium-pilot/paired_diagnostics.json](C:/Users/Shricharan/VSCodeProjects/trafficlightcontrol/runs/service-priority-medium-pilot/paired_diagnostics.json)
