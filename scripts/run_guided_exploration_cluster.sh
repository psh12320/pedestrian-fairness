#!/usr/bin/env bash
# One allocation, no arrays/dispatchers; resubmit unchanged to resume.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the guided-exploration checkout}"
source "$HOME/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SUMO_HOME
mkdir -p runs
exec 9>runs/guided-exploration.lock
flock -n 9 || { echo 'Another guided-exploration process is writing this study.' >&2; exit 1; }
python -m unittest tests.test_guided_exploration tests.test_ppo_repair tests.test_fairness_learning tests.test_fairness_analysis tests.test_fairness_v2.FairnessTests -v
python -u -m src.traffic_control.guided_exploration.pilot \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --output runs/guided-exploration
python scripts/package_guided_exploration.py --results
echo 'Complete: runs/guided-exploration-results.zip. Scientific success is assessed separately.'
