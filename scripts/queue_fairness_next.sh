#!/usr/bin/env bash
# Queue one bounded batch and a short dependent dispatcher, never the whole study.
# Recovery: bash scripts/queue_fairness_next.sh --after JOBID train 128 0
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
else
  cd "$(dirname "$0")/.."
fi
mkdir -p runs/cluster-logs
submit() {
  if [[ "${FAIRNESS_DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY RUN sbatch ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    printf '99999\n'
  else
    sbatch "$@"
  fi
}
dispatch_options=(--parsable --cpus-per-task=1 --mem=256M --time=00:05:00
  --job-name=fairness-dispatch --output='runs/cluster-logs/dispatch-%j.log')
if [[ "${1:-}" == --after ]]; then
  after="${2:?Provide the existing successful prerequisite job ID}"
  [[ "$after" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric job ID.' >&2; exit 2; }
  shift 2
  [[ $# == 3 ]] || { echo 'Usage: --after JOBID train EPISODE FIRST_INDEX' >&2; exit 2; }
  job=$(submit "${dispatch_options[@]}" --dependency="afterok:$after" --kill-on-invalid-dep=yes \
    scripts/queue_fairness_next.sh "$@")
  printf 'Continuation dispatcher: %s (after %s succeeds)\n' "${job%%;*}" "$after"
  exit
fi
[[ -n "${SLURM_JOB_ID:-}" || "${FAIRNESS_DRY_RUN:-0}" == 1 ]] || {
  echo 'Use --after JOBID to attach to a completed or queued prerequisite.' >&2; exit 2;
}
[[ $# == 3 ]] || { echo 'Expected: train|evaluate|report EPISODE FIRST_INDEX' >&2; exit 2; }
stage="$1"; until="$2"; first="$3"
batch="${FAIRNESS_BATCH_SIZE:-24}"
[[ "$batch" =~ ^[1-9][0-9]*$ && "$first" =~ ^[0-9]+$ && "$until" =~ ^[0-9]+$ ]] || exit 2
(( batch <= 24 )) || { echo 'Batch size must be at most 24 to leave scheduler headroom.' >&2; exit 2; }
unset FAIRNESS_ARRAY_STAGE FAIRNESS_TRAIN_UNTIL
runner=scripts/run_fairness_cluster.sh
common=(--parsable --cpus-per-task=1 --mem=3G --time="${FAIRNESS_TIME:-02:50:00}"
  --output='runs/cluster-logs/%A_%a.log')
case "$stage" in
  train)
    (( until >= 64 && until <= 512 && until % 64 == 0 && first < 30 )) || exit 2
    total=30
    exports="ALL,FAIRNESS_ARRAY_STAGE=train,FAIRNESS_TRAIN_UNTIL=$until"
    ;;
  evaluate)
    (( first < 420 )) || exit 2
    total=420
    exports=ALL,FAIRNESS_ARRAY_STAGE=evaluate
    ;;
  report)
    job=$(submit "${common[@]}" --mem=8G --job-name=fairness-report "$runner" --stage report)
    printf 'Report job: %s\n' "${job%%;*}"
    exit
    ;;
  *) echo 'Unknown stage.' >&2; exit 2 ;;
esac
last=$((first + batch - 1))
if (( last >= total )); then last=$((total - 1)); fi
job=$(submit "${common[@]}" --array="$first-$last%$batch" --job-name="fairness-$stage-$until" \
  --export="$exports" "$runner")
job="${job%%;*}"
printf 'Submitted %s %s, indexes %s-%s: %s\n' "$stage" "$until" "$first" "$last" "$job"
next_stage="$stage"; next_until="$until"; next_first=$((last + 1))
if (( next_first == total )); then
  next_first=0
  if [[ "$stage" == train ]] && (( until < 512 )); then
    next_until=$((until + 64))
  elif [[ "$stage" == train ]]; then
    next_stage=evaluate
  else
    next_stage=report
  fi
fi
if ! continuation=$(submit "${dispatch_options[@]}" --dependency="afterok:$job" --kill-on-invalid-dep=yes \
  scripts/queue_fairness_next.sh "$next_stage" "$next_until" "$next_first"); then
  printf 'Batch %s was accepted, but its continuation was not. After a slot opens, run:\n' "$job" >&2
  printf 'bash scripts/queue_fairness_next.sh --after %s %s %s %s\n' "$job" "$next_stage" "$next_until" "$next_first" >&2
  exit 1
fi
printf 'Next dispatcher: %s\n' "${continuation%%;*}"
if [[ "${FAIRNESS_DRY_RUN:-0}" != 1 ]]; then
  printf '%s\t%s\t%s\t%s-%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$stage" "$until" "$first" "$last" "$job" "${continuation%%;*}" >> runs/cluster-logs/dispatch.tsv
fi
