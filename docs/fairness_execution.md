# Running the v2 study

The full v2 study is not yet complete. Short runs verify software operation, not
learning quality or the proposal target. Keep new results separate from the older
completed study. The branch is `pedestrian-fairness-v2`.

## Local

```powershell
$env:PYTHONPATH = (Resolve-Path .local/sumo-1.27.1/tools).Path
Remove-Item Env:SUMO_HOME -ErrorAction SilentlyContinue
python -m unittest discover -s tests -p 'test_fairness*.py' -v
python -m src.traffic_control.fairness.audit --workers 2
python -m src.traffic_control.fairness.pipeline --quick --workers 2 --output runs/my-engineering-check
```

Omit `--quick` for the full study: 512 episodes x 30 models; 20 calibration seeds,
10 validation seeds, 20 test seeds, and 21 scenarios. Training episodes are 1200 s.
Evaluation uses 300 s warm-up, 1800 s measurement, 300 s follow-up. There are 13,860
evaluation episodes. Resume with the same command; source/config/version changes
require a fresh output folder. Checkpoints retain model, optimizer and random state.

Calibration seeds: 22001–22020; validation: 23001–23010; test: 34001–34020;
initialisation: 25001–25005; training demand: 100000 + 10000*replica + episode.
Do not tune with test outputs. The 512-episode budget is common across methods,
not a convergence guarantee. Curriculum load factors cycle .75/1/1.25.
Engineering checks use separate seeds 19001/19002/19003. Earlier engineering
probes used 24001 under shortened demand; that seed block is now exploratory and
excluded from the new locked test set.

## Cluster

On Windows, from the repository:

```powershell
python scripts/package_fairness.py
scp -J shri@sjump.comp.nus.edu.sg runs/fairness-v2-source.tar.gz shri@xlogin.comp.nus.edu.sg:~/
ssh -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg
```

Then on the login node:

```bash
mkdir -p ~/cs4246-fairness-v2
tar -xzf ~/fairness-v2-source.tar.gz -C ~/cs4246-fairness-v2
cd ~/cs4246-fairness-v2/trafficlightcontrol
bash scripts/run_fairness_cluster.sh --quick --output runs/cluster-check
# After that job completes successfully:
bash scripts/submit_fairness_study.sh
squeue -u "$USER"
```

The scripts submit compute jobs through Slurm, not on the login node. Python
3.10–3.12 and package-download access are needed for the first job. They create an
isolated environment with CPU PyTorch 2.3.0, NumPy 1.26.4, SUMO/TraCI 1.27.1.
No GPU is requested. Linux package installation and authenticated cluster execution
remain unverified. The [NUS cluster guide](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/)
describes Slurm access and the default three-hour maximum.

The full launcher schedules preparation and one dependent dispatcher. Preparation
requests eight CPUs (`FAIRNESS_PREPARE_CPUS`). Each dispatcher then submits only
one batch of at most 24 single-CPU tasks (`FAIRNESS_BATCH_SIZE`) and a dependent
dispatcher for the next batch. This covers all 30 replicas at each 64-episode
checkpoint and all 420 evaluation groups without queuing the whole study at once.
There are at most 24 experiment tasks, one waiting dispatcher, and briefly the
current dispatcher from this chain. Actual concurrency also depends on account
limits, other jobs and available resources. A dispatcher consumes one CPU for a
short submission step; it does not wait on a compute node or keep a login shell open.
`FAIRNESS_TIME` defaults to 2:50:00;
`SBATCH_PARTITION` can select a site-permitted partition. A failed/time-limited task
stops dependent jobs. Resubmit to resume completed work, but never run overlapping
chains against the same folder. Preparation caches individual seeds; evaluation
caches individual controllers within each matched-demand group.

The initial all-at-once launcher hit `AssocMaxSubmitJobLimit`: preparation 1049572
and the first 30-task training array 1049573 were accepted, but the next array was
rejected. Pending array tasks count toward the submission limit; a `%` concurrency
cap alone cannot solve this. The numeric account limit has not been established.
After uploading the corrected scripts, attach the remaining stages **once**:

```bash
bash scripts/queue_fairness_next.sh --after 1049573 train 128 0
```

This waits for all tasks in the existing 64-episode stage to succeed. It does not
restart preparation or discard checkpoints. If that attachment itself hits the
submission limit, retry after preparation finishes and frees a slot. If a batch
fails, dependent dispatchers are cancelled and saved checkpoints remain; diagnose
the failed batch before submitting a recovery chain. Successful submissions are
recorded in `runs/cluster-logs/dispatch.tsv` and dispatcher logs.

Recovery commands (submit each only when its prerequisites are complete):

```bash
bash scripts/run_fairness_cluster.sh --stage prepare
bash scripts/run_fairness_cluster.sh --stage train --task-index 0 --train-until 64
bash scripts/run_fairness_cluster.sh --stage evaluate --task-index 0
bash scripts/run_fairness_cluster.sh --stage report
```

After jobs disappear from `squeue`, inspect their `sacct` states and exit codes.
An empty queue does not distinguish success from failure/time limit. The standalone
`scripts/collect_fairness_status.py` produces `runs/fairness-cluster-status.txt`
and a compact ZIP with model progress, scheduler accounting, selected results and
log tails. It excludes checkpoints and dependencies and performs no simulation.
Upload that helper separately if the cluster checkout predates it, then run:

```bash
python3 scripts/collect_fairness_status.py
```

If only the first 64-episode stage completed successfully for all 30 models and
there is no active continuation chain, submit a fresh dispatcher for episode 128:

```bash
sbatch --cpus-per-task=1 --mem=256M --time=00:05:00 --job-name=fairness-dispatch \
  --output='runs/cluster-logs/dispatch-%j.log' scripts/queue_fairness_next.sh train 128 0
```

This avoids referencing an old completed job ID which Slurm may have purged.
Do not infer success from missing jobs or absent `complete.json` files: those
markers are written only after the full 512-episode training budget. If accounting
shows a failure, inspect the bundled logs before scheduling more experiments.

Source archives exclude dependencies, old results and models. Episode XML files
are temporary; evaluation waits are compressed. A minimum-free-space check stops
episodes below 256 MiB. Monitor overall output size; this is not a hard total quota.

## What reports mean

`REPORT.md` and `comparisons.json` contain diagnostic paired results.
`primary_inference.json` pools crossing waits over demand seeds within each replica,
then averages replica p95 values. It does not pool all replicas into one distribution.
MAPPO-constrained alone is designated for the stronger joint inference. It requires
20 demand seeds, five replicas, at least 300 entries per crossing per replica and
baseline, and no service/measurement failures. A one-sided vehicle bound and a
simultaneous worst-crossing bound must both support the target. Other comparisons
remain descriptive. Sampling/geometric uncertainty still limits tiny differences.

The software remains a development protocol pending review of the audit and
assumptions. It cannot guarantee a positive finding. Final report/presentation
refresh requires completed experiments and factual team contributions.
