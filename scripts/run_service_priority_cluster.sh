#!/usr/bin/env bash
# One allocation, no dispatcher chain or job arrays. Run from the NEW checkout.
# SBATCH options are supplied by the caller to respect local account policies.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the new project directory}"
old_study="${FAIRNESS_OLD_ROOT:-$HOME/cs4246-fairness-v2/trafficlightcontrol}"
source "$old_study/.cluster-venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SUMO_HOME
python -m unittest tests.test_service_priority tests.test_fairness_analysis -v
python scripts/diagnose_fairness_results.py --study "$old_study/runs/fairness-v2-study" \
    --output runs/fairness-diagnosis.zip
python -u -m src.traffic_control.service_priority.pilot --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --episodes 32 --replicas 3 --output runs/service-priority-pilot
python scripts/package_service_priority.py --results
