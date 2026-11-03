# Service-priority cluster pilot review

The received archive passed CRC, source-hash, demand-pairing, completion, vehicle-count and crossing-quantile checks. Twelve models completed 32 episodes each (384 training episodes); 128 evaluation records cover four scenarios and two demand seeds. This is development evidence, not a final held-out comparison. The reserved 58001-58020 seeds were not used.

## Main finding

The service-priority intervention did not meet the combined target. Some crossing-tail point estimates improved relative to plain PPO, but the learned controllers still under-served vehicle movements. Low completed-vehicle time loss must be read beside completed and uninserted counts. Do not extend training unchanged merely because the software ran successfully.

## Pilot outcomes

Vehicle counts refer to trips scheduled in [300,2100) seconds and their status at the 2100-second cutoff. They are not whole-run completion counts at 2400 seconds. Crossing waits use requests reaching a crossing in [300,2100), including entries during follow-up to 2400. Unresolved requests are still waiting after that follow-up. Fractional counts below are averages across runs.

Each crossing p95 pools the two demand seeds within each training replica, then averages replica p95s; the reference p95 pools the two actuated runs. Counts of no-worse crossings are descriptive, with only machine-roundoff (1e-9 s) ignored. Several crossing samples are sparse. No confidence intervals or statistical superiority claims are made from this pilot.

### vmedium-pmedium

| Controller | Mean vehicle loss s | Delta vs actuated s | Completed | Unfinished | Uninserted | Crossing p95 no worse / 12 | Largest p95 delta s | Failed service runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| actuated | 51.64 | +0.00 | 933.5 | 41.0 | 0.0 | 12/12 | +0.00 | 0/2 |
| pressure | 41.07 | -10.58 | 942.0 | 32.5 | 0.0 | 1/12 | +14.00 | 0/2 |
| priority-pressure | 50.89 | -0.75 | 933.5 | 41.0 | 0.0 | 7/12 | +2.20 | 2/2 |
| age-pressure | 49.03 | -2.61 | 932.5 | 42.0 | 0.0 | 6/12 | +2.00 | 2/2 |
| plain-ippo | 34.98 | -16.67 | 682.2 | 51.8 | 240.5 | 5/12 | +1.40 | 6/6 |
| priority-ippo | 57.86 | +6.22 | 767.8 | 62.7 | 144.0 | 6/12 | +1.33 | 5/6 |
| plain-mappo | 32.62 | -19.03 | 596.3 | 77.3 | 300.8 | 4/12 | +1.40 | 6/6 |
| priority-mappo | 64.75 | +13.10 | 845.7 | 65.2 | 63.7 | 8/12 | +1.20 | 6/6 |

Minimum reported pooled crossing sample across these comparisons: 91 entries.

### vmedium-plow

| Controller | Mean vehicle loss s | Delta vs actuated s | Completed | Unfinished | Uninserted | Crossing p95 no worse / 12 | Largest p95 delta s | Failed service runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| actuated | 39.59 | +0.00 | 935.0 | 39.5 | 0.0 | 12/12 | +0.00 | 0/2 |
| pressure | 29.53 | -10.06 | 947.5 | 27.0 | 0.0 | 0/12 | +32.40 | 1/2 |
| priority-pressure | 40.15 | +0.55 | 933.5 | 41.0 | 0.0 | 3/12 | +4.80 | 1/2 |
| age-pressure | 38.92 | -0.68 | 936.0 | 38.5 | 0.0 | 3/12 | +5.20 | 1/2 |
| plain-ippo | 36.42 | -3.17 | 684.8 | 52.3 | 237.3 | 1/12 | +4.80 | 6/6 |
| priority-ippo | 69.18 | +29.58 | 884.5 | 50.3 | 39.7 | 5/12 | +4.00 | 4/6 |
| plain-mappo | 27.10 | -12.50 | 577.2 | 77.8 | 319.5 | 1/12 | +4.80 | 6/6 |
| priority-mappo | 64.17 | +24.58 | 923.7 | 49.7 | 1.2 | 3/12 | +3.80 | 5/6 |

Minimum reported pooled crossing sample across these comparisons: 27 entries.

### pedestrian-burst

| Controller | Mean vehicle loss s | Delta vs actuated s | Completed | Unfinished | Uninserted | Crossing p95 no worse / 12 | Largest p95 delta s | Failed service runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| actuated | 51.55 | +0.00 | 932.5 | 42.0 | 0.0 | 12/12 | +0.00 | 0/2 |
| pressure | 41.07 | -10.48 | 942.0 | 32.5 | 0.0 | 1/12 | +13.60 | 0/2 |
| priority-pressure | 50.69 | -0.87 | 933.5 | 41.0 | 0.0 | 6/12 | +2.60 | 2/2 |
| age-pressure | 50.28 | -1.27 | 938.0 | 36.5 | 0.0 | 6/12 | +1.60 | 1/2 |
| plain-ippo | 34.98 | -16.57 | 682.2 | 51.8 | 240.5 | 4/12 | +1.80 | 6/6 |
| priority-ippo | 57.96 | +6.41 | 760.8 | 63.7 | 150.0 | 4/12 | +1.73 | 5/6 |
| plain-mappo | 32.62 | -18.93 | 596.3 | 77.3 | 300.8 | 4/12 | +1.80 | 6/6 |
| priority-mappo | 72.90 | +21.35 | 867.3 | 59.8 | 47.3 | 6/12 | +1.53 | 5/6 |

Minimum reported pooled crossing sample across these comparisons: 91 entries.

### short-spacing

| Controller | Mean vehicle loss s | Delta vs actuated s | Completed | Unfinished | Uninserted | Crossing p95 no worse / 12 | Largest p95 delta s | Failed service runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| actuated | 56.60 | +0.00 | 760.0 | 61.5 | 1551.5 | 12/12 | +0.00 | 0/2 |
| pressure | 51.02 | -5.58 | 1904.5 | 79.0 | 389.5 | 0/12 | +30.80 | 2/2 |
| priority-pressure | 38.31 | -18.29 | 636.0 | 42.5 | 1694.5 | 10/12 | +0.60 | 2/2 |
| age-pressure | 32.13 | -24.47 | 702.5 | 43.0 | 1627.5 | 6/12 | +1.00 | 2/2 |
| plain-ippo | 24.44 | -32.16 | 654.8 | 63.2 | 1655.0 | 4/12 | +4.87 | 6/6 |
| priority-ippo | 47.75 | -8.85 | 781.3 | 55.2 | 1536.5 | 7/12 | +0.80 | 6/6 |
| plain-mappo | 26.63 | -29.97 | 640.5 | 55.0 | 1677.5 | 6/12 | +1.80 | 6/6 |
| priority-mappo | 41.42 | -15.18 | 786.7 | 51.5 | 1534.8 | 8/12 | +0.80 | 5/6 |

Minimum reported pooled crossing sample across these comparisons: 64 entries.

## A visible failure mechanism

In plain-mappo-r0, medium demand, seed 47001, C0 received only 12 seconds of EW green across the entire 2400-second simulation, versus 792 seconds NS green. The corresponding actuated C0 received 456 seconds EW and 396 seconds NS. These are actual recorded phase durations, not inferred from a mean performance metric.

The pilot used pedestrian reward weight 3 in every learned arm. Its medium-demand pedestrian trips all completed, but vehicle service suffered. This is different from the old vehicle-only models starving pedestrians. Protecting only pedestrians does not guarantee adequate service for every vehicle movement.

Priority-IPPO also has a short-spacing warning: one replica/seed has unresolved pedestrian requests. The exact phase/occupancy sequence is not present in this summary archive, so it is not yet possible to attribute that case to a particular blockage mechanism.

The training logs show high policy entropy; for example priority-ippo-r0 C0 at episode 32 is 0.688, close to log(2)=0.693. That average applies to sampled training states, not all evaluation states. Evaluation always takes argmax. A slight persistent probability preference can produce repetitive decisions under argmax. This is a testable explanation, not a demonstrated cause or a reason to silently change evaluation after inspecting test results.

The priority policy receives roughly 20 free actor decisions per junction in a 600-second training episode; forced transitions and required pedestrian service do not train an actor choice. Thirty-two episodes are a pilot, not evidence of convergence.

## What the original-study diagnosis now establishes

The nested archive recovered 420 original groups / 13,860 records, the full comparisons JSON and complete vehicle/service summaries. It does not retain individual wait arrays for every original crossing; the saved primary pooled inference is available unchanged.

At original primary medium demand, vehicle-only IPPO/MAPPO each leave an average 698.65 person trips unfinished at simulation end, versus zero for actuated. The original pedestrian-aware/constrained models have no unfinished person trips or unresolved measured crossing requests in that scenario. Their service flags therefore must not all be described as permanent pedestrian starvation; crossing-cutoff differences and some vehicle differences require separate reporting.

### Original primary absolute results (20 demand seeds; five replicas for learned methods)

| Controller | Mean vehicle loss s | Scheduled | Completed | Unfinished | Uninserted | Person trips unfinished at 2400 s |
|---|---:|---:|---:|---:|---:|---:|
| actuated | 51.68 | 952.7 | 917.4 | 35.0 | 0.35 | 0.00 |
| fixed | 33.14 | 952.7 | 919.2 | 33.1 | 0.35 | 0.00 |
| ippo-constrained | 53.63 | 952.7 | 920.3 | 32.0 | 0.35 | 0.00 |
| ippo-pedestrian | 53.21 | 952.7 | 920.3 | 32.0 | 0.35 | 0.00 |
| ippo-vehicle | 14.92 | 952.7 | 933.5 | 18.8 | 0.35 | 698.65 |
| mappo-constrained | 53.33 | 952.7 | 921.0 | 31.3 | 0.35 | 0.00 |
| mappo-pedestrian | 53.03 | 952.7 | 920.3 | 32.1 | 0.35 | 0.00 |
| mappo-vehicle | 15.19 | 952.7 | 933.7 | 18.7 | 0.35 | 698.65 |
| pressure | 40.36 | 952.7 | 921.2 | 31.1 | 0.35 | 0.00 |

## Recommended next step

Run a no-training diagnostic of the same saved pilot checkpoints: argmax versus seeded probability sampling, alongside uniform legal-action controls with and without the same priority rule. Use the existing development demand seeds; do not touch final test seeds. Record choice probabilities, movement service gaps, completion counts and critic diagnostics. This distinguishes evaluation collapse from a useful learned policy and from insufficient movement-service protection.

If sampling restores service but performs no better than uniform choices, the actor has not demonstrated useful learning. If movement neglect remains, introduce a demand-aware maximum-red/service rule covering vehicle movements as well as pedestrians, and compare a conventional version against a learned version. Validate the rule before another matched training pilot. Increasing pedestrian admission duration is a separate sensitivity experiment; it will not directly fix a vehicle direction that is almost never selected.

Do not launch another 512-episode study or consume the reserved final seeds yet. Preserve this negative pilot as evidence about the intervention.

## All original absolute scenario summaries

These descriptive means recover the previously missing fixed-time and actuated values. They are not confidence intervals. Every scenario/controller combination is also in summary.json.

### Original vlow-plow

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 34.92 | 462.9 | 16.1 | 0.20 | 0.0 |
| fixed | 29.08 | 459.9 | 19.1 | 0.20 | 0.0 |
| ippo-constrained | 46.59 | 461.4 | 17.6 | 0.20 | 0.0 |
| ippo-pedestrian | 45.88 | 462.2 | 16.8 | 0.20 | 0.0 |
| ippo-vehicle | 14.40 | 467.5 | 11.5 | 0.20 | 229.2 |
| mappo-constrained | 45.76 | 462.2 | 16.8 | 0.20 | 0.0 |
| mappo-pedestrian | 46.16 | 461.6 | 17.4 | 0.20 | 0.0 |
| mappo-vehicle | 14.78 | 467.6 | 11.3 | 0.20 | 229.2 |
| pressure | 29.45 | 463.6 | 15.3 | 0.20 | 0.0 |

### Original vlow-pmedium

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 43.06 | 459.9 | 19.1 | 0.20 | 0.0 |
| fixed | 29.08 | 459.9 | 19.1 | 0.20 | 0.0 |
| ippo-constrained | 46.30 | 461.3 | 17.7 | 0.20 | 0.0 |
| ippo-pedestrian | 46.05 | 461.5 | 17.5 | 0.20 | 0.0 |
| ippo-vehicle | 14.11 | 467.7 | 11.3 | 0.20 | 694.8 |
| mappo-constrained | 45.51 | 461.8 | 17.2 | 0.20 | 0.0 |
| mappo-pedestrian | 45.98 | 461.5 | 17.5 | 0.20 | 0.0 |
| mappo-vehicle | 14.51 | 467.6 | 11.4 | 0.20 | 694.8 |
| pressure | 40.17 | 460.1 | 18.9 | 0.20 | 0.0 |

### Original vlow-phigh

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 44.42 | 460.4 | 18.6 | 0.20 | 0.0 |
| fixed | 29.08 | 459.9 | 19.1 | 0.20 | 0.0 |
| ippo-constrained | 46.67 | 461.4 | 17.6 | 0.20 | 0.0 |
| ippo-pedestrian | 46.16 | 461.5 | 17.5 | 0.20 | 0.0 |
| ippo-vehicle | 14.38 | 467.9 | 11.1 | 0.20 | 1169.7 |
| mappo-constrained | 45.60 | 461.6 | 17.4 | 0.20 | 0.0 |
| mappo-pedestrian | 46.00 | 461.5 | 17.5 | 0.20 | 0.0 |
| mappo-vehicle | 15.00 | 467.6 | 11.4 | 0.20 | 1169.7 |
| pressure | 44.14 | 459.1 | 19.9 | 0.20 | 0.0 |

### Original vmedium-plow

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 39.97 | 923.0 | 29.3 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 0.0 |
| ippo-constrained | 53.33 | 920.7 | 31.7 | 0.35 | 0.0 |
| ippo-pedestrian | 52.35 | 920.8 | 31.6 | 0.35 | 0.0 |
| ippo-vehicle | 15.22 | 933.2 | 19.2 | 0.35 | 230.7 |
| mappo-constrained | 52.97 | 920.5 | 31.8 | 0.35 | 0.0 |
| mappo-pedestrian | 52.69 | 919.7 | 32.6 | 0.35 | 0.0 |
| mappo-vehicle | 15.56 | 933.1 | 19.2 | 0.35 | 230.7 |
| pressure | 29.60 | 928.5 | 23.9 | 0.35 | 0.0 |

### Original vmedium-pmedium

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 51.68 | 917.4 | 35.0 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 0.0 |
| ippo-constrained | 53.63 | 920.3 | 32.0 | 0.35 | 0.0 |
| ippo-pedestrian | 53.21 | 920.3 | 32.0 | 0.35 | 0.0 |
| ippo-vehicle | 14.92 | 933.5 | 18.8 | 0.35 | 698.6 |
| mappo-constrained | 53.33 | 921.0 | 31.3 | 0.35 | 0.0 |
| mappo-pedestrian | 53.03 | 920.3 | 32.1 | 0.35 | 0.0 |
| mappo-vehicle | 15.19 | 933.7 | 18.7 | 0.35 | 698.6 |
| pressure | 40.36 | 921.2 | 31.1 | 0.35 | 0.0 |

### Original vmedium-phigh

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 53.23 | 915.8 | 36.6 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 0.0 |
| ippo-constrained | 53.92 | 920.8 | 31.6 | 0.35 | 0.0 |
| ippo-pedestrian | 53.48 | 920.4 | 32.0 | 0.35 | 0.0 |
| ippo-vehicle | 15.19 | 934.2 | 18.2 | 0.35 | 1162.7 |
| mappo-constrained | 53.61 | 921.0 | 31.4 | 0.35 | 0.0 |
| mappo-pedestrian | 53.32 | 920.4 | 32.0 | 0.35 | 0.0 |
| mappo-vehicle | 15.70 | 932.8 | 19.5 | 0.35 | 1162.7 |
| pressure | 47.10 | 917.4 | 35.0 | 0.40 | 0.0 |

### Original vhigh-plow

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 53.09 | 1357.7 | 70.0 | 0.90 | 0.0 |
| fixed | 36.63 | 1373.2 | 54.9 | 0.50 | 0.0 |
| ippo-constrained | 90.43 | 1314.3 | 96.5 | 17.71 | 0.0 |
| ippo-pedestrian | 89.74 | 1315.0 | 95.2 | 18.35 | 0.0 |
| ippo-vehicle | 15.85 | 1395.1 | 33.1 | 0.41 | 234.2 |
| mappo-constrained | 86.96 | 1320.9 | 94.5 | 13.07 | 0.0 |
| mappo-pedestrian | 87.35 | 1318.8 | 95.2 | 14.55 | 0.0 |
| mappo-vehicle | 16.16 | 1395.0 | 33.1 | 0.44 | 234.2 |
| pressure | 32.15 | 1384.4 | 43.8 | 0.40 | 0.0 |

### Original vhigh-pmedium

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 97.60 | 1297.1 | 109.8 | 21.65 | 0.0 |
| fixed | 36.63 | 1373.2 | 54.9 | 0.50 | 0.0 |
| ippo-constrained | 95.12 | 1307.4 | 100.4 | 20.71 | 0.0 |
| ippo-pedestrian | 93.84 | 1308.1 | 99.4 | 21.10 | 0.0 |
| ippo-vehicle | 15.59 | 1394.5 | 33.6 | 0.42 | 695.6 |
| mappo-constrained | 93.14 | 1314.9 | 98.9 | 14.76 | 0.0 |
| mappo-pedestrian | 92.69 | 1312.5 | 98.8 | 17.26 | 0.0 |
| mappo-vehicle | 15.88 | 1395.2 | 32.9 | 0.44 | 695.6 |
| pressure | 44.35 | 1374.0 | 54.0 | 0.55 | 0.0 |

### Original vhigh-phigh

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 103.43 | 1290.3 | 114.2 | 24.00 | 0.0 |
| fixed | 36.63 | 1373.2 | 54.9 | 0.50 | 0.0 |
| ippo-constrained | 98.83 | 1300.4 | 104.2 | 23.88 | 0.0 |
| ippo-pedestrian | 97.58 | 1302.8 | 101.7 | 24.08 | 0.0 |
| ippo-vehicle | 15.98 | 1394.7 | 33.4 | 0.43 | 1162.9 |
| mappo-constrained | 97.35 | 1306.2 | 103.3 | 19.09 | 0.0 |
| mappo-pedestrian | 95.51 | 1308.7 | 101.1 | 18.69 | 0.0 |
| mappo-vehicle | 16.45 | 1393.8 | 34.3 | 0.43 | 1162.9 |
| pressure | 58.66 | 1360.8 | 67.2 | 0.50 | 0.0 |

### Original slow-walkers

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 52.62 | 915.3 | 37.0 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 2.8 |
| ippo-constrained | 53.85 | 920.9 | 31.5 | 0.35 | 0.0 |
| ippo-pedestrian | 53.43 | 921.0 | 31.4 | 0.35 | 0.0 |
| ippo-vehicle | 14.85 | 933.6 | 18.8 | 0.35 | 698.6 |
| mappo-constrained | 53.64 | 921.5 | 30.8 | 0.35 | 0.0 |
| mappo-pedestrian | 53.38 | 920.2 | 32.2 | 0.35 | 0.0 |
| mappo-vehicle | 15.22 | 933.5 | 18.8 | 0.35 | 698.6 |
| pressure | 44.19 | 917.9 | 34.5 | 0.35 | 0.0 |

### Original mixed-walkers

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 52.42 | 917.0 | 35.3 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 0.8 |
| ippo-constrained | 53.59 | 920.8 | 31.6 | 0.35 | 0.0 |
| ippo-pedestrian | 53.13 | 921.0 | 31.4 | 0.35 | 0.0 |
| ippo-vehicle | 14.91 | 933.4 | 18.9 | 0.35 | 698.6 |
| mappo-constrained | 53.48 | 921.5 | 30.9 | 0.35 | 0.0 |
| mappo-pedestrian | 53.17 | 920.6 | 31.7 | 0.35 | 0.0 |
| mappo-vehicle | 15.19 | 933.6 | 18.8 | 0.35 | 698.6 |
| pressure | 41.08 | 920.9 | 31.5 | 0.35 | 0.0 |

### Original pedestrian-burst

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 51.66 | 917.1 | 35.2 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 0.0 |
| ippo-constrained | 53.71 | 920.2 | 32.1 | 0.35 | 0.0 |
| ippo-pedestrian | 53.20 | 920.3 | 32.1 | 0.35 | 0.0 |
| ippo-vehicle | 14.91 | 933.5 | 18.8 | 0.35 | 718.6 |
| mappo-constrained | 53.33 | 921.0 | 31.3 | 0.35 | 0.0 |
| mappo-pedestrian | 53.03 | 920.2 | 32.2 | 0.35 | 0.0 |
| mappo-vehicle | 15.20 | 933.9 | 18.4 | 0.35 | 718.6 |
| pressure | 40.43 | 922.2 | 30.1 | 0.35 | 0.0 |

### Original extreme-burst

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 51.65 | 917.3 | 35.0 | 0.35 | 0.0 |
| fixed | 33.18 | 905.5 | 37.6 | 9.60 | 7.0 |
| ippo-constrained | 53.53 | 914.8 | 33.9 | 3.98 | 3.1 |
| ippo-pedestrian | 53.20 | 920.3 | 32.1 | 0.35 | 0.0 |
| ippo-vehicle | 14.91 | 933.4 | 19.0 | 0.35 | 778.6 |
| mappo-constrained | 53.33 | 921.0 | 31.3 | 0.35 | 0.0 |
| mappo-pedestrian | 53.03 | 920.1 | 32.2 | 0.35 | 0.0 |
| mappo-vehicle | 15.23 | 933.9 | 18.5 | 0.35 | 778.6 |
| pressure | 40.29 | 922.4 | 30.0 | 0.35 | 0.0 |

### Original vehicle-overload

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 140.98 | 1138.9 | 152.2 | 2704.75 | 0.0 |
| fixed | 66.33 | 2533.3 | 138.9 | 1323.50 | 0.0 |
| ippo-constrained | 110.93 | 1253.7 | 126.4 | 2615.73 | 0.0 |
| ippo-pedestrian | 124.04 | 1189.1 | 135.1 | 2671.61 | 0.0 |
| ippo-vehicle | 54.21 | 3155.8 | 142.8 | 697.24 | 695.8 |
| mappo-constrained | 118.23 | 1357.1 | 143.8 | 2494.91 | 0.0 |
| mappo-pedestrian | 113.81 | 1360.7 | 147.1 | 2487.99 | 0.0 |
| mappo-vehicle | 54.01 | 3163.5 | 142.4 | 689.88 | 695.8 |
| pressure | 78.26 | 2286.2 | 150.0 | 1559.55 | 1.4 |

### Original short-spacing

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 36.32 | 659.9 | 67.6 | 1667.30 | 9.6 |
| fixed | 38.69 | 2284.0 | 83.2 | 27.55 | 0.0 |
| ippo-constrained | 49.05 | 818.4 | 68.3 | 1508.10 | 5.1 |
| ippo-pedestrian | 46.09 | 842.8 | 68.4 | 1483.57 | 8.2 |
| ippo-vehicle | 23.19 | 2327.9 | 57.7 | 9.15 | 704.4 |
| mappo-constrained | 40.63 | 768.2 | 70.2 | 1556.41 | 6.3 |
| mappo-pedestrian | 45.85 | 816.7 | 67.1 | 1510.95 | 6.5 |
| mappo-vehicle | 23.61 | 2323.8 | 58.4 | 12.59 | 704.4 |
| pressure | 53.32 | 1888.7 | 83.5 | 422.65 | 0.3 |

### Original directional-imbalance

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 108.47 | 844.7 | 71.0 | 37.05 | 0.0 |
| fixed | 43.52 | 914.5 | 37.8 | 0.40 | 0.0 |
| ippo-constrained | 106.98 | 832.8 | 67.9 | 52.06 | 0.0 |
| ippo-pedestrian | 101.33 | 839.1 | 64.1 | 49.47 | 0.0 |
| ippo-vehicle | 17.80 | 930.1 | 22.2 | 0.40 | 698.6 |
| mappo-constrained | 104.44 | 832.7 | 65.3 | 54.76 | 0.0 |
| mappo-pedestrian | 104.07 | 827.4 | 65.1 | 60.16 | 0.0 |
| mappo-vehicle | 18.36 | 929.8 | 22.5 | 0.40 | 698.6 |
| pressure | 64.91 | 899.7 | 49.9 | 3.15 | 0.0 |

### Original sparse-pedestrian-route

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 51.67 | 916.9 | 35.5 | 0.35 | 0.0 |
| fixed | 33.14 | 919.2 | 33.1 | 0.35 | 0.0 |
| ippo-constrained | 53.72 | 920.8 | 31.6 | 0.35 | 0.0 |
| ippo-pedestrian | 53.18 | 921.0 | 31.4 | 0.35 | 0.0 |
| ippo-vehicle | 15.07 | 933.3 | 19.0 | 0.35 | 698.6 |
| mappo-constrained | 53.67 | 921.1 | 31.3 | 0.35 | 0.0 |
| mappo-pedestrian | 53.05 | 920.3 | 32.1 | 0.35 | 0.0 |
| mappo-vehicle | 15.60 | 933.5 | 18.8 | 0.35 | 698.6 |
| pressure | 40.71 | 924.2 | 28.1 | 0.35 | 0.0 |

### Original vehicle-platoons

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 52.43 | 930.1 | 26.9 | 0.00 | 0.0 |
| fixed | 32.98 | 929.1 | 27.9 | 0.00 | 0.0 |
| ippo-constrained | 54.77 | 932.4 | 24.6 | 0.00 | 0.0 |
| ippo-pedestrian | 54.47 | 932.8 | 24.2 | 0.00 | 0.0 |
| ippo-vehicle | 15.39 | 949.9 | 7.1 | 0.00 | 697.9 |
| mappo-constrained | 54.39 | 933.6 | 23.4 | 0.00 | 0.0 |
| mappo-pedestrian | 54.17 | 932.4 | 24.6 | 0.00 | 0.0 |
| mappo-vehicle | 15.47 | 949.8 | 7.2 | 0.00 | 697.9 |
| pressure | 40.05 | 936.0 | 21.1 | 0.00 | 0.0 |

### Original demand-ramp

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 62.79 | 995.5 | 19.1 | 0.05 | 0.0 |
| fixed | 34.31 | 995.8 | 18.9 | 0.05 | 0.0 |
| ippo-constrained | 64.71 | 997.4 | 17.3 | 0.00 | 0.0 |
| ippo-pedestrian | 64.22 | 997.2 | 17.4 | 0.00 | 0.0 |
| ippo-vehicle | 15.20 | 1004.3 | 10.3 | 0.02 | 691.5 |
| mappo-constrained | 65.12 | 997.3 | 17.3 | 0.00 | 0.0 |
| mappo-pedestrian | 64.47 | 996.8 | 17.9 | 0.00 | 0.0 |
| mappo-vehicle | 15.45 | 1004.1 | 10.5 | 0.00 | 691.5 |
| pressure | 41.93 | 997.9 | 16.8 | 0.00 | 0.0 |

### Original higher-turning-share

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 50.41 | 917.7 | 34.7 | 0.30 | 0.0 |
| fixed | 33.54 | 922.0 | 30.4 | 0.30 | 0.0 |
| ippo-constrained | 52.70 | 921.5 | 30.9 | 0.30 | 0.0 |
| ippo-pedestrian | 52.03 | 920.7 | 31.7 | 0.30 | 0.0 |
| ippo-vehicle | 14.27 | 934.8 | 17.6 | 0.30 | 698.6 |
| mappo-constrained | 52.50 | 921.4 | 31.0 | 0.30 | 0.0 |
| mappo-pedestrian | 52.19 | 920.6 | 31.8 | 0.30 | 0.0 |
| mappo-vehicle | 14.63 | 934.6 | 17.8 | 0.30 | 698.6 |
| pressure | 39.63 | 922.1 | 30.2 | 0.30 | 0.0 |

### Original heterogeneous-drivers

| Controller | Mean vehicle loss s | Completed | Unfinished | Uninserted | Person unfinished at end |
|---|---:|---:|---:|---:|---:|
| actuated | 55.96 | 915.6 | 44.2 | 0.20 | 0.0 |
| fixed | 37.88 | 921.1 | 38.6 | 0.20 | 0.0 |
| ippo-constrained | 57.66 | 919.0 | 40.8 | 0.20 | 0.0 |
| ippo-pedestrian | 57.05 | 918.8 | 41.0 | 0.22 | 0.0 |
| ippo-vehicle | 15.63 | 937.7 | 22.1 | 0.20 | 703.9 |
| mappo-constrained | 57.47 | 918.9 | 40.9 | 0.20 | 0.0 |
| mappo-pedestrian | 57.15 | 919.6 | 40.2 | 0.20 | 0.0 |
| mappo-vehicle | 16.14 | 937.7 | 22.1 | 0.20 | 703.9 |
| pressure | 43.24 | 923.0 | 36.8 | 0.20 | 0.0 |

