#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the service-priority project directory}"
source "$HOME/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SUMO_HOME
python -m unittest tests.test_policy_audit -v
python -u scripts/audit_service_policy.py --workers "${SLURM_CPUS_PER_TASK:-4}"
