#!/usr/bin/env bash
# Submit the full 512-episode/30-replica study in restartable scheduler tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v sbatch >/dev/null || { echo 'Run this on the SoC Slurm login node.' >&2; exit 1; }
mkdir -p runs/cluster-logs
prepare_cpus="${FAIRNESS_PREPARE_CPUS:-8}"
batch="${FAIRNESS_BATCH_SIZE:-24}"
[[ "$batch" =~ ^[1-9][0-9]*$ && "$prepare_cpus" =~ ^[1-9][0-9]*$ ]] && (( batch <= 24 )) || {
  echo 'Use positive CPU/batch settings, with a batch size at most 24.' >&2
  exit 2
}
runner=scripts/run_fairness_cluster.sh
common=(--parsable --cpus-per-task=1 --mem=3G --time="${FAIRNESS_TIME:-02:50:00}" --output='runs/cluster-logs/%A_%a.log')
prepare=$(sbatch "${common[@]}" --cpus-per-task="$prepare_cpus" --mem=8G \
  --job-name=fairness-prepare "$runner" --stage prepare)
prepare="${prepare%%;*}"
printf 'Preparation job: %s\n' "$prepare"
bash scripts/queue_fairness_next.sh --after "$prepare" train 64 0
printf 'Further batches are submitted only after the preceding batch succeeds.\n'
printf 'Inspect: squeue -u "$USER"\nDo not start a second chain while this one is active.\n'
