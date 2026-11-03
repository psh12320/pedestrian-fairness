# PPO repair pilot review

All 18 models reached 64 episodes. The archive contains all 1,152 training-episode records, 108 validation runs, 432 development evaluation runs and 104 baseline runs. File coverage, source/protocol/manifest hashes, model-hash references, matched demand/network hashes, training normaliser counts, accounting, crossing statistics, phase-duration sums, and every aggregate/paired interval passed audit. Raw actor weights and simulator traces are absent, so the critic probes and physical conflicts cannot be replayed independently from this ZIP.

## Decision

No learned configuration meets the descriptive joint target in any of the four evaluated settings. The normalisation/clock repair improves critic calibration, but does not produce a reliable joint vehicle/pedestrian gain. Four-rollout batching is worse for pedestrian tails under this fixed-episode budget. Do not promote a winner or launch another unchanged large study.

A = original PPO. B = repaired critic (normalisation + clock), one episode/update. C = B with four episodes/update. These are all pedestrian-weight-3 follow-ups, not the original vehicle-versus-pedestrian reward comparison.

## Primary medium-demand traffic results

| Controller | Vehicle loss s | Delta vs actuated, 95% exploratory interval | Crossings no worse | Worst crossing p95 increase s |
|---|---:|---|---:|---:|
| actuated | 50.78 | +0.00 | 12/12 | +0.00 |
| fixed | 33.29 | -17.49 | 0/12 | +42.40 |
| pressure | 40.16 | -10.62 | 0/12 | +14.00 |
| uniform | 49.33 | -1.45 | 0/12 | +63.00 |
| ippo-normalized-clock | 52.19 | +1.41 [-1.40, +4.62] | 0/12 | +26.20 |
| ippo-normalized-clock-batch4 | 54.29 | +3.51 [-1.22, +9.16] | 0/12 | +41.13 |
| ippo-original | 54.20 | +3.42 [-0.87, +8.81] | 0/12 | +26.33 |
| mappo-normalized-clock | 55.14 | +4.36 [+0.91, +8.21] | 0/12 | +28.33 |
| mappo-normalized-clock-batch4 | 53.58 | +2.80 [-1.23, +8.25] | 0/12 | +39.20 |
| mappo-original | 54.24 | +3.46 [-1.53, +8.66] | 0/12 | +27.07 |

Every learned variant has higher p95 at all 12 crossings in the medium-demand comparison. Minimum pooled crossing counts are 272-274 across the compared replicas/reference, above the prespecified 100-entry diagnostic threshold. This failure cannot be dismissed as the earlier five-entry crossing problem. Six demand seeds and three training seeds still give limited uncertainty estimates.

The 15-18/18 failed primary service checks include pending requests at the measurement cutoff. All medium-demand person trips complete by the simulation end, with no unresolved measured-cohort crossing requests after followup. Across all evaluated settings, there are 78 unfinished person-trip records (75 in slow-walker runs and 3 under short-spacing pressure control), despite zero unresolved measured-cohort crossing requests and zero unknown/lost measured events. A person finishing their entire trip is distinct from a request that reached a crossing in the measurement window; later requests or walking after crossing require raw traces to classify. Individual vehicle and crossing cutoff differences remain in summary.json.

## What changed in the critic

| Controller | 600-second probe RMSE | Mean bias | Mean explained variance | Free-decision entropy |
|---|---:|---:|---:|---:|
| ippo-original | 113.53 | 104.96 | 0.001 | 0.888 |
| ippo-normalized-clock | 41.65 | 6.77 | 0.067 | 0.800 |
| ippo-normalized-clock-batch4 | 106.82 | 95.29 | 0.014 | 0.972 |
| mappo-original | 112.19 | 103.54 | 0.001 | 0.888 |
| mappo-normalized-clock | 42.77 | 5.21 | 0.032 | 0.823 |
| mappo-normalized-clock-batch4 | 97.88 | 85.04 | 0.011 | 0.975 |

B removes much of the positive value bias: roughly +105 to +7 reward units for IPPO, and +104 to +5 for MAPPO. However, explained variance remains low (about 0.067 and 0.032 on medium evaluation), so much of the gain is calibration rather than strong state-dependent prediction. Each policy generates different states/targets: these probe errors are diagnostics, not a matched-target regression comparison. They are not waiting-time improvements.

C remains biased (+95 IPPO, +85 MAPPO) and has higher action entropy. It receives only 16 update batches versus 64 for A/B, with four PPO passes per batch. This is consistent with slower learning under fewer updates, not proof that collecting independent trajectories is harmful. The offline 32-run check used 512 critic fitting steps on fixed data; this online training design does not replicate that update budget.

## Actual evidence of partial improvement

On short spacing, IPPO B lowers vehicle time loss against IPPO A by 7.63 seconds (exploratory paired interval -13.43 to -3.45 seconds), and improves 8/12 crossing p95 point estimates against A. The remaining crossings and service components prevent a joint pass. In slow walking, IPPO B improves all 12 crossing point estimates against A, but its vehicle interval includes no improvement, and it still loses every crossing against actuated. These are secondary exploratory findings, not a replacement primary target.

## Why service accounting matters

On short spacing, actuated control completes about 788/2,431 scheduled measurement-cohort vehicles, with about 1,569 uninserted. Its completed-vehicle time loss of 46.86 seconds therefore describes a selected served subset. IPPO B completes about 1,474 vehicles with 860 uninserted, at 76.56 seconds completed-vehicle time loss. Fixed time completes about 2,317 with 28 uninserted, at 39.28 seconds, but delays pedestrians. It would be misleading to rank this setting by completed-vehicle delay alone.

Pressure-inspired control is a useful strong benchmark at medium demand: 40.16 seconds vehicle loss versus 50.78 actuated, but its worst crossing p95 is 14.0 seconds higher and none of the crossings meets non-inferiority. Uniform choices have slightly lower vehicle loss than actuated but much worse pedestrian tails (worst +63.0 seconds). Randomness is not the overall winner. All four A/B configurations improve every crossing p95 point estimate against uniform control, while increasing vehicle loss. For example IPPO B costs 2.86 seconds more vehicle loss than uniform, but improves every crossing p95 by at least 4.8 seconds. This is a learned trade-off, not overall dominance or passage against actuated. C improves only 10/12 crossing point estimates against uniform.

## Recommended next decision

Preserve this as a completed controlled negative result and move the report forward. Do not rerun the same 64/512-episode configuration or claim that the earlier diagnostic solved the controller. The next technical step should be a bounded decision-learning audit: examine action advantages, actor versus critic gradient magnitudes and the cost of switching/PED service before another training campaign. Shared gradient clipping is a possible mechanism to inspect, not a proven cause from the total-gradient logs alone.

If testing another training intervention, isolate its change and keep a matched control; use the strong pressure/actuated references and retain all crossings and service counts. A baseline-guided actor is a possible future hypothesis, not a demonstrated improvement. Preserve the original IPPO/MAPPO proposal question and untouched final seeds.

Results are from synthetic three-junction settings, not calibrated Singapore operations. The burst and medium scenarios share much traffic and are not independent replications; short spacing also increases demand, combining two stressors. The four scenarios, three initialisations and single action-sampling stream per demand limit generalisation.
