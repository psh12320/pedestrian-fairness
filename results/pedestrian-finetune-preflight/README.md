# Pedestrian fine-tuning preflight

36 unit/integration tests passed; 19 short SUMO simulations completed. A fresh same-seed repeat matched metrics, pedestrian events and phase histories exactly after JSON representation normalization. This verifies execution, accounting, learning updates and reproducibility, not traffic improvement.

The guided arm used 7 urgent teacher labels from 67 free choices across its two smoke episodes. All four learned variants still had the same aggregate outcome in the short evaluation. No claim of improvement follows from this smoke run.

The full development protocol has 876 simulations, three starting MAPPO replicas, three matched fine-tuning arms, frozen original parents and common conventional baselines. It is queued behind the existing CPU audit. See docs/pedestrian_finetune.md and runs/pedestrian-finetune/queue.json for details/status.
