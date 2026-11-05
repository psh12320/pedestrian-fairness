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

- **mappo-early+vehrule-vs-mappo-continue**: Δvehicle 2.45 s, crossings no-worse 12/12, service-fail-pairs 24
- **mappo-early+vehrule-vs-mappo-early**: Δvehicle -5.19 s, crossings no-worse 8/12, service-fail-pairs 24
- **mappo-early-vs-mappo-continue**: Δvehicle 7.64 s, crossings no-worse 12/12, service-fail-pairs 27

## vlow-pmedium

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 40.06 | -4.98 | 0 | 28 | False |
| mappo-continue | 38.74 | -6.29 | 0 | 27 | False |
| mappo-tail | 41.93 | -3.11 | 0 | 29 | False |
| mappo-guided | 45.47 | 0.43 | 0 | 29 | False |
| mappo-early | 48.24 | 3.20 | 8 | 29 | False |
| mappo-early+vehrule | 44.12 | -0.92 | 7 | 27 | False |

- **mappo-early+vehrule-vs-mappo-continue**: Δvehicle 5.37 s, crossings no-worse 12/12, service-fail-pairs 23
- **mappo-early+vehrule-vs-mappo-early**: Δvehicle -4.12 s, crossings no-worse 3/12, service-fail-pairs 24
- **mappo-early-vs-mappo-continue**: Δvehicle 9.49 s, crossings no-worse 12/12, service-fail-pairs 28

## vmedium-plow

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 36.89 | -3.31 | 0 | 23 | False |
| mappo-continue | 34.06 | -6.14 | 0 | 23 | False |
| mappo-tail | 37.15 | -3.05 | 0 | 21 | False |
| mappo-guided | 41.06 | 0.86 | 0 | 24 | False |
| mappo-early | 45.02 | 4.82 | 4 | 20 | False |
| mappo-early+vehrule | 40.63 | 0.43 | 4 | 19 | False |

- **mappo-early+vehrule-vs-mappo-continue**: Δvehicle 6.57 s, crossings no-worse 12/12, service-fail-pairs 22
- **mappo-early+vehrule-vs-mappo-early**: Δvehicle -4.39 s, crossings no-worse 8/12, service-fail-pairs 18
- **mappo-early-vs-mappo-continue**: Δvehicle 10.95 s, crossings no-worse 12/12, service-fail-pairs 25

## vmedium-pmedium

| Arm | Vehicle loss s | Δ vs actuated s | Crossings no-worse /12 | Service fail pairs | descriptive_pass |
|---|---:|---:|---:|---:|---|
| mappo-original | 40.85 | -10.32 | 0 | 25 | False |
| mappo-continue | 38.43 | -12.74 | 0 | 24 | False |
| mappo-tail | 41.94 | -9.23 | 0 | 27 | False |
| mappo-guided | 46.91 | -4.26 | 0 | 21 | False |
| mappo-early | 54.97 | 3.80 | 6 | 26 | False |
| mappo-early+vehrule | 50.70 | -0.47 | 6 | 24 | False |

- **mappo-early+vehrule-vs-mappo-continue**: Δvehicle 12.27 s, crossings no-worse 12/12, service-fail-pairs 26
- **mappo-early+vehrule-vs-mappo-early**: Δvehicle -4.27 s, crossings no-worse 7/12, service-fail-pairs 22
- **mappo-early-vs-mappo-continue**: Δvehicle 16.54 s, crossings no-worse 12/12, service-fail-pairs 28
