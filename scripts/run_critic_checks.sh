#!/usr/bin/env bash
# One allocation, fixed saved actors, no full PPO retraining and no GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the existing service-priority project directory}"
source "$HOME/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SUMO_HOME
python -m unittest tests.test_critic_fit tests.test_fairness_learning -v
for model in plain-mappo-r0 priority-mappo-r0; do
  python -u scripts/critic_fit_check.py --pilot runs/service-priority-pilot \
    --model "$model" --output "runs/critic-fit-$model" \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
  python -u scripts/critic_horizon_check.py --parent "runs/critic-fit-$model" \
    --output "runs/critic-horizon-$model" --workers "${SLURM_CPUS_PER_TASK:-4}"
done
python scripts/package_critic_checks.py --results
