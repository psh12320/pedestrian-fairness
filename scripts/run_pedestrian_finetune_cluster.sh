#!/usr/bin/env bash
# One CPU allocation; no arrays or follow-up job submissions. Resume by resubmitting.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pedestrian-finetune checkout}"
source "${CS4246_VENV:-$HOME/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset SUMO_HOME
mkdir -p runs
exec 9>runs/pedestrian-finetune-cluster.lock
flock -n 9 || { echo 'Another job is writing this study.' >&2; exit 1; }
python -m unittest tests.test_pedestrian_finetune_teacher tests.test_pedestrian_finetune_learning tests.test_pedestrian_finetune_reporting tests.test_pedestrian_finetune_pilot tests.test_pedestrian_finetune_cluster -v
python -u -m scripts.pedestrian_finetune_cluster --quick --workers 2 --output runs/pedestrian-finetune-cluster-smoke
python -u -m scripts.pedestrian_finetune_cluster --workers "${SLURM_CPUS_PER_TASK:-8}"
python scripts/package_pedestrian_finetune.py --results
echo 'Complete: runs/pedestrian-finetune-cluster-results.zip. Improvement is assessed separately.'
