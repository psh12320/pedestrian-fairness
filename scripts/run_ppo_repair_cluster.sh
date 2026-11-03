#!/usr/bin/env bash
# Submit ONE CPU allocation from the separate ppo-repair checkout.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the new ppo-repair project directory}"
source "$HOME/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SUMO_HOME
mkdir -p runs
exec 9>runs/ppo-repair.lock
flock -n 9 || { echo 'Another repair-pilot job is writing these outputs.' >&2; exit 1; }
python -m unittest tests.test_ppo_repair tests.test_fairness_learning tests.test_fairness_analysis tests.test_fairness_v2.FairnessTests -v
python -u -m src.traffic_control.ppo_repair.pilot --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --output runs/ppo-repair-pilot
python scripts/package_ppo_repair.py --results
echo 'Completed. Download runs/ppo-repair-results.zip. A failed research target is still a completed experiment.'
