# Running the next fine-tuning study on cluster CPUs

**Completed:** all 876 cluster simulations finished; [review](../results/pedestrian-finetune-cluster-review/REVIEW.md). The laptop duplicate was stopped at 42/876, and the separate local audit also finished. The commands below document the original launch and transfer; no new submission is needed to read results.

Finish the existing laptop audit locally. The already verified supplementary
fine-tuning study can run in one cluster allocation with eight CPU workers.
It does not request a GPU, install packages or change the algorithm/budget.
The new orchestrator reuses the same independent jobs as the laptop runner;
its own source hash is included in a separate cluster protocol. Never combine
cluster and laptop replicas as independent evidence from different seeds.

The planned study has 876 simulations plus a 19-simulation engineering smoke
check. Nine model continuations are independent; parallelism helps without
changing their seeds. Speedup depends on queue time, CPU speed, memory and
stage barriers. Eight workers is not a promise of eightfold speedup.

The existing laptop queue is left intact as a fallback. Once the cluster job
shows `R` (running) in `squeue` and you move the work there, run this on the laptop:

```powershell
New-Item -ItemType File -Force "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\pedestrian-finetune\STOP"
```

This affects only the supplementary fine-tuning study, not the current audit.
If fine-tuning has already started it stops at a simulation boundary, preserving
its checkpoints. The cluster starts its own complete study in a separate folder.
Keep local partial outputs; do not pool them with the cluster study. Do not run
both full studies unnecessarily. To restore the fallback after a failed transfer,
remove only this STOP marker and run the documented local pilot command; the
waiting helper may already have exited and is not automatically restarted.

Upload from PowerShell (absolute path works from any current directory):

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\pedestrian-finetune-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/pedestrian-finetune-source.tar.gz
```

First submission, in the cluster terminal:

```bash
mkdir -p ~/cs4246-pedestrian-finetune
tar -xzf ~/pedestrian-finetune-source.tar.gz -C ~/cs4246-pedestrian-finetune
cd ~/cs4246-pedestrian-finetune/trafficlightcontrol
mkdir -p runs
sbatch --partition=normal --cpus-per-task=8 --mem=16G --time=02:50:00 \
  --job-name=ped-finetune --output='runs/ped-finetune-%j.log' \
  scripts/run_pedestrian_finetune_cluster.sh
```

This reuses `~/cs4246-fairness-v2/trafficlightcontrol/.cluster-venv`. Set
`CS4246_VENV` to another existing compatible environment if necessary. Python,
Torch, TraCI and SUMO versions are recorded. The smoke run must succeed before
the full run starts. Nothing computes on the login node except submission and
lightweight status reads. There are no job arrays or chained submissions that
could reproduce the previous submission-count failure; resource/QOS limits can
still reject an allocation.

The 2h50 allocation is a resource request, not a completion estimate. If time
runs out, resubmit the same `sbatch` command from the same directory. Do not
unpack changed code into the study. Atomic optimizer/model/RNG checkpoints,
cached baselines and evaluations resume; an interrupted episode may repeat.
Two locks prevent overlapping writers. Keep the source and runtime environment
unchanged when resuming. A source mismatch stops rather than mixing studies.

Check progress without starting another job:

```bash
python3 -m scripts.pedestrian_finetune_cluster --status
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 25 runs/ped-finetune-JOB_ID.log
```

Completion requires successful status and an audited results archive, not simply
absence from `squeue`. The archive retains final models, paired outcomes,
crossing events, teacher labels, source and a hash manifest. Resume checkpoints
stay on the cluster. Download the completed results in PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-pedestrian-finetune/trafficlightcontrol/runs/pedestrian-finetune-cluster-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\pedestrian-finetune-cluster-results.zip"
```

After these results, assess the observation/clipping audit and the targeted
fine-tuning contrasts before choosing a larger confirmation or new algorithm.
Longer-horizon training and isolated junction-coupling tests remain hypotheses,
not studies that have already run. Do not automatically spend the remaining
budget on every previously mentioned algorithm or choose only favorable results.
