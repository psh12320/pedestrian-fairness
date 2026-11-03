#!/usr/bin/env bash
# One CPU allocation; no GPU, actor updates, or changes to previous results.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the existing service-priority project directory}"
source "$HOME/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SUMO_HOME
mkdir -p runs
exec 9>runs/critic-data-volume.lock
flock -n 9 || { echo 'Another critic-data-volume job is already writing these outputs.' >&2; exit 1; }
python -m unittest tests.test_critic_data_volume tests.test_critic_fit tests.test_fairness_learning -v
python -u scripts/package_critic_data_volume.py --preflight
for model in plain-mappo-r0 priority-mappo-r0; do
  python -u scripts/critic_data_volume_check.py --pilot runs/service-priority-pilot \
    --model "$model" --output "runs/critic-data-$model" \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
done
python -u scripts/package_critic_data_volume.py --results
echo 'Completed. Download runs/critic-data-volume-results.zip (a false scientific gate is still a completed test).'
