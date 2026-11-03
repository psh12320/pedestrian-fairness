# Four versus 32 independent training trajectories

This development diagnostic follows the critic scaling and remaining-time checks. On saved learned-policy trajectories, the network-wide critic still generalised poorly despite low training error. The hypothesis is that broader training coverage improves prediction on new traffic demand. This is not a new traffic-controller evaluation or a claim that more data will solve the proposal criterion.

## Frozen comparison

- Two existing, hash-verified actors: `plain-mappo-r0` and `priority-mappo-r0`. Sample actions reproducibly. Do not update these actors or select another replica using results.
- Three connected junctions, medium demand (1,920 vehicles/hour and 1,200 pedestrians/hour network-wide), original safe control, 600-second trajectories, no warmup. No real Singapore calibration is claimed.
- Training demand seeds 49101–49132. Small arm is the first four, exactly the previous diagnostic's training seed set. Large arm contains those same four plus 28 new seeds.
- Eight fresh development check demand seeds 49501–49508. They cannot affect fitting, scaling, stopping, or model selection. Final evaluation seeds 58001–58020 remain unused.
- Three critic initialisations 49301–49303, local and network-wide inputs, separate critic per junction, 64×64 tanh networks. Both arms receive remaining-time/600 and normalised outputs.
- Identical initial weights within each pair. One shared mean/scale transform fitted only on the four-trajectory subset. The larger dataset therefore cannot change the output parameterisation or initial predictions.
- Full-batch Adam updates: 128 and 512; learning rate 0.0003, epsilon 0.00001, gradient clipping 0.5, discount 0.99. **Equal updates are not equal compute:** the larger arm processes eight times as many states at every update. The experiment evaluates broader data at a fixed update budget, not data diversity isolated from sample exposure.

Total: 80 short SUMO trajectories and 72 fresh critic fits across both actors. This does not resume PPO. The original PPO study had many training episodes; the four-trajectory dataset belongs only to this small regression diagnostic.

## Outcomes and interpretation

Keep 128/512-update results, all initialisations and all junctions. Report training/check RMSE and bias in reward units, MSE relative to a constant four-trajectory training mean, and relative to each arm's own training mean. Compare arms by absolute MSE, avoiding denominator changes that could manufacture an apparent improvement.

Predeclared gate for each frozen actor: at 512 updates, every input-method/initialisation pair must improve over four trajectories and beat both constant references, pooling the three junctions. Preserve individual junction/demand/initialisation cases even if the pooled gate passes. A gate failure is a valid completed experiment.

An exploratory paired 95% interval resamples critic initialisations and whole check demand seeds, keeping junctions together. It is conditional on these fixed training datasets, excludes variation from sampling other training datasets, and has only three initialisations. Adjacent simulator observations are not treated as independent samples. Do not interpret these intervals as final confirmation.

If the larger dataset helps, the next test is a bounded on-policy training comparison with the verified critic changes, multiple rollout seeds per update, matched training budgets and sampled actor evaluation. It must measure completed-vehicle time loss, completed/unfinished/uninserted service, crossing-specific p95 and unresolved pedestrian requests. Better critic prediction alone does not establish better vehicle or pedestrian outcomes. If it fails, retain the failure and inspect observation sufficiency/target noise before another expensive training campaign.

## Run on the existing cluster project

Upload from Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\critic-data-volume-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/critic-data-volume-source.tar.gz
```

Run in the cluster terminal:

```bash
cd ~/cs4246-service-priority/trafficlightcontrol
tar -xzf ~/critic-data-volume-source.tar.gz
mkdir -p runs
sbatch --partition=normal --cpus-per-task=4 --mem=8G --time=01:00:00 \
  --job-name=critic-data --output='runs/critic-data-%j.log' \
  scripts/run_critic_data_volume.sh
```

This is one CPU allocation, no GPU or new package installation. The script uses the existing fairness-v2 virtual environment. Saved models, pilot source hashes and unit tests are checked before collection. A file lock prevents simultaneous writers. Previous actor weights and result directories are untouched; new data go to `runs/critic-data-plain-mappo-r0` and `runs/critic-data-priority-mappo-r0`.

Check the job number returned by `sbatch` (replace `JOB_ID`):

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 25 runs/critic-data-JOB_ID.log
ls -lh runs/critic-data-volume-results.zip
```

Successful completion requires `COMPLETED`, `0:0`, and the combined results ZIP, not just an empty queue. If interrupted, resubmit the unchanged command after the previous job has stopped: completed trajectories and critic fits are reused. A fit interrupted before its atomic write restarts. Source/config changes require a new output folder and protocol.

Download after completion:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-service-priority/trafficlightcontrol/runs/critic-data-volume-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\critic-data-volume-results.zip"
```

The result ZIP includes both reports, all scalar fit results, protocols, checkpoint/source hashes and a file manifest. Compact trajectory tensors remain on the cluster for further audit. No checkpoints, full simulator traces or large dependencies are downloaded. The source package is a small patch for this existing project, not a standalone repository.

## Local validation before delivery

Fourteen targeted unit/integration tests passed, including a synthetic-tensor fit/cache/provenance round trip. Bash syntax validation passed. The known local Torch/NumPy bridge warning did not prevent these tensor-only checks. The full saved-actor SUMO experiment is intentionally left for the cluster; no new traffic outcome or critic improvement is reported before those results arrive.
