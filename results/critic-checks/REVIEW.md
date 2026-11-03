# Local critic diagnosis, 3 November 2026

Eight 600-second SUMO trajectories and 72 critic fits completed locally. The first experiment took 89.1 seconds; the follow-up took 62.3 seconds. Twelve relevant unit checks and two additional short saved-policy collector smoke runs passed. No traffic actor was retrained, and no final test seed was used. Additional stored data is only a few megabytes.

## Finding 1: raw output units are difficult to fit at the current scale/budget

The existing critic outputs raw reward returns while scaling its squared error. On a known linear target around -300, its held-out RMSE remained 281.46 after 512 updates. The otherwise matched normalised-output model reached RMSE 0.475. This establishes a concrete output-scaling/optimisation weakness in that fixed-target test; it does not prove this is the sole cause of PPO's traffic results.

On real actuated-policy trajectories, normalisation greatly improved calibration but failed the predeclared constant-predictor gate:

| Inputs | Raw-output holdout MSE / constant | Normalised-output holdout MSE / constant |
|---|---:|---:|
| Local critic | 5.559 | 1.030 |
| Network-wide critic | 5.612 | 0.959 |

Values are at 512 matched updates, averaging three junctions and three initialisations, checked on demand seeds 49201/49202. Below 1 beats the constant training-mean predictor. Merely predicting the right mean is not enough. Increasing updates reduced training error while held-out error sometimes worsened: there is also a generalisation problem with these limited data.

## Finding 2: finite-horizon prediction needs remaining-time information

The code uses 600-second training episodes, zero terminal bootstrap and observations without remaining episode time. Phase age is included, but it is not the remaining horizon. Two otherwise identical traffic observations near the beginning and end of an episode can therefore have different target returns.

This hypothesis was checked with new demand seeds 49401/49402. Both models used normalised outputs, identical architecture and initial function, four fixed training trajectories and matched updates. Only the extra critic input differed: zero versus remaining-time / 600.

| Inputs | No-clock MSE / constant | Clock MSE / constant | Reduction vs matched no-clock |
|---|---:|---:|---:|
| Local critic | 0.979 | 0.586 | about 40% |
| Network-wide critic | 1.216 | 0.817 | about 33% |

These are 512-update results. Every method/initialisation passed the predeclared junction-average gate. All 18 junction/initialisation estimates pooled over the two check seeds beat the constant predictor, although two of the 36 individual junction/initialisation/check-seed cases did not (network-wide C1, seeds 49301/49302, demand 49402). Therefore this is promising evidence about value prediction, not uniform superiority across every individual case.

The two tables use different check seeds, so they must not be joined into a single raw-versus-normalised-with-clock effect estimate. The clock test was a disclosed follow-up to the first failed gate. Both seed pairs are development data. Temporally adjacent states are not independent observations for a confidence interval.

## Further statistical and modelling cautions

- A network-wide critic still has a substantial train/check gap. More available inputs do not guarantee better generalisation with only four trajectories. This is not an IPPO-versus-MAPPO traffic ranking.
- The original reward also includes due-but-uninserted vehicles, while the recorded actor/critic observation has no explicit count of those vehicles. Queues may provide indirect information, but this is an additional partial-observation issue to examine in overload/neglect scenarios. Its effect has not been isolated by these tests.
- Zero-terminal finite-horizon targets are a different design from continuing traffic with truncated rollout collection. Supplying a clock addresses the former. Correct bootstrapping would need a separate controlled test for the latter.
- A better critic can help PPO learn, but these supervised tests neither optimise an actor nor prove reduced vehicle delay or improved pedestrian p95.

## Next action

Run the prepared small CPU diagnostic on the cluster using the fixed saved `plain-mappo-r0` and `priority-mappo-r0` actors. The local test used actuated-generated trajectories; the cluster test checks whether output scaling and the clock also help on states visited by the learned policies. All models remain read-only. There are 16 short SUMO trajectories, plus fresh critic fits, and no full PPO training.

Instructions: [critic_checks.md](../../docs/critic_checks.md). Local detailed outputs: `runs/critic-fit-actuated/REPORT.md`, `runs/critic-horizon-actuated/REPORT.md`, and their JSON files. The true final evaluation seeds remain reserved.

If the learned-policy diagnostic confirms the finding, implement a finite-horizon PPO ablation separating normalisation and clock effects, with identical actor observations, rewards, safety rules, demand and training budgets. If it does not, retain that result and diagnose the state/return distribution before launching another large run.

## Implementation references and limits

The [official MAPPO trainer](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py) and [value normaliser](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/utils/valuenorm.py) distinguish normalised value outputs from raw-unit returns. Our fixed-statistics diagnostic is independently implemented and is not a reproduction of their complete online learning procedure.

The local Torch installation emitted its existing NumPy bridge warning. These tensor-only computations and all checks completed successfully; no packages were installed. The cluster job reuses the existing environment. No deployed traffic-light claim is supported by these synthetic tests.
