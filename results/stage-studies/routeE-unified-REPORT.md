# Stage 1: vehicle-axis service rule (frozen model)


New arm `mappo-early+vehrule` = continued-MAPPO + early-service pedestrian rule + larger-stopped-queue vehicle rule. No training. Compared vs. actuated and vs. the continued/early arms on matched demand. `descriptive_pass` = point estimates meet vehicle<0 AND all 12 crossings p95 no worse AND strict service gate clean.

## vlow-plow

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 36.32 | 0.73 | 0 | 23 | False |
| mappo-continue | 33.83 | -1.77 | 0 | 24 | False |
| mappo-tail | 36.65 | 1.05 | 0 | 27 | False |
| mappo-guided | 41.07 | 5.48 | 0 | 17 | False |
| mappo-early | 41.47 | 5.87 | 5 | 24 | False |
| mappo-early+vehrule | 36.28 | 0.68 | 5 | 24 | False |
| mappo-unified | 36.28 | 0.68 | 5 | 24 | False |

- **mappo-unified-vs-mappo-early+vehrule**: Δvehicle 0.00 s, crossings no-worse 12/12, service-fail-pairs 0
- **mappo-unified-vs-mappo-continue**: Δvehicle 2.45 s, crossings no-worse 12/12, service-fail-pairs 24

## vlow-pmedium

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 40.06 | -4.98 | 0 | 28 | False |
| mappo-continue | 38.74 | -6.29 | 0 | 27 | False |
| mappo-tail | 41.93 | -3.11 | 0 | 29 | False |
| mappo-guided | 45.47 | 0.43 | 0 | 29 | False |
| mappo-early | 48.24 | 3.20 | 8 | 29 | False |
| mappo-early+vehrule | 44.12 | -0.92 | 7 | 27 | False |
| mappo-unified | 44.12 | -0.92 | 7 | 27 | False |

- **mappo-unified-vs-mappo-early+vehrule**: Δvehicle 0.00 s, crossings no-worse 12/12, service-fail-pairs 0
- **mappo-unified-vs-mappo-continue**: Δvehicle 5.37 s, crossings no-worse 12/12, service-fail-pairs 23

## vmedium-plow

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 36.89 | -3.31 | 0 | 23 | False |
| mappo-continue | 34.06 | -6.14 | 0 | 23 | False |
| mappo-tail | 37.15 | -3.05 | 0 | 21 | False |
| mappo-guided | 41.06 | 0.86 | 0 | 24 | False |
| mappo-early | 45.02 | 4.82 | 4 | 20 | False |
| mappo-early+vehrule | 40.63 | 0.43 | 4 | 19 | False |
| mappo-unified | 40.63 | 0.43 | 4 | 19 | False |

- **mappo-unified-vs-mappo-early+vehrule**: Δvehicle 0.00 s, crossings no-worse 12/12, service-fail-pairs 0
- **mappo-unified-vs-mappo-continue**: Δvehicle 6.57 s, crossings no-worse 12/12, service-fail-pairs 22

## vmedium-pmedium

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 40.85 | -10.32 | 0 | 25 | False |
| mappo-continue | 38.43 | -12.74 | 0 | 24 | False |
| mappo-tail | 41.94 | -9.23 | 0 | 27 | False |
| mappo-guided | 46.91 | -4.26 | 0 | 21 | False |
| mappo-early | 54.97 | 3.80 | 6 | 26 | False |
| mappo-early+vehrule | 50.70 | -0.47 | 6 | 24 | False |
| mappo-unified | 50.70 | -0.47 | 6 | 24 | False |

- **mappo-unified-vs-mappo-early+vehrule**: Δvehicle 0.00 s, crossings no-worse 12/12, service-fail-pairs 0
- **mappo-unified-vs-mappo-continue**: Δvehicle 12.27 s, crossings no-worse 12/12, service-fail-pairs 26
