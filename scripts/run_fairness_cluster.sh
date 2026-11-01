#!/usr/bin/env bash
# Submit CPU simulation jobs; do not run SUMO training on a login node.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Slurm executes a spool copy of this script; its dirname is not the project.
  cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
else
  cd "$(dirname "$0")/.."
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  command -v sbatch >/dev/null || { echo 'Slurm is required on the cluster.' >&2; exit 1; }
  mkdir -p runs/cluster-logs
  # Site account/partition may be supplied via ordinary SBATCH_* variables.
  sbatch --job-name=pedestrian-fairness --cpus-per-task="${FAIRNESS_WORKERS:-8}" --mem=16G --time="${FAIRNESS_TIME:-02:50:00}" \
    --output='runs/cluster-logs/%j.log' --signal=B:TERM@120 "$0" "$@"
  exit
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13), "Load Python 3.10-3.12 first"'
[[ -d .cluster-venv ]] || python3 -m venv .cluster-venv
source .cluster-venv/bin/activate
python -c 'import torch; assert torch.__version__.split("+")[0] == "2.3.0"' || {
  python -m pip install --no-cache-dir 'torch==2.3.0' --index-url https://download.pytorch.org/whl/cpu
}
python -c 'from importlib.metadata import version; assert version("numpy") == "1.26.4"; assert version("traci") == "1.27.1"; assert version("eclipse-sumo") == "1.27.1"; assert version("matplotlib") == "3.9.4"' || \
  python -m pip install --no-cache-dir 'numpy==1.26.4' 'traci==1.27.1' 'eclipse-sumo==1.27.1' 'matplotlib==3.9.4'
unset SUMO_HOME
sumo --version
if [[ "${FAIRNESS_ARRAY_STAGE:-}" != "train" && "${FAIRNESS_ARRAY_STAGE:-}" != "evaluate" ]]; then
  python -m unittest discover -s tests -p 'test_fairness*.py' -v
fi
if [[ "${1:-}" == audit ]]; then
  shift
  exec python -u -m src.traffic_control.fairness.audit --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
fi
# Checkpoints permit resubmitting this same command after the scheduler time limit.
if [[ "${FAIRNESS_ARRAY_STAGE:-}" == "train" ]]; then
  exec python -u -m src.traffic_control.fairness.pipeline --stage train \
    --task-index "$SLURM_ARRAY_TASK_ID" --train-until "$FAIRNESS_TRAIN_UNTIL" "$@"
elif [[ "${FAIRNESS_ARRAY_STAGE:-}" == "evaluate" ]]; then
  exec python -u -m src.traffic_control.fairness.pipeline --stage evaluate \
    --task-index "$SLURM_ARRAY_TASK_ID" "$@"
fi
# Do not change source files or protocol flags between resumptions.
python -u -m src.traffic_control.fairness.pipeline --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
