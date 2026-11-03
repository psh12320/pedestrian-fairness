# Check the policy before another training study

The completed 32-episode service-priority pilot has 12 models and 128 evaluation records. None meets the joint proposal target. See `results/service-priority-cluster/REVIEW.md` for all four scenarios and the recovered original-study results.

## Evidence motivating this diagnostic

In plain MAPPO replica 0, development demand seed 47001, C0 has only its initial 12 seconds of EW green during the 2400-second medium-demand run. Actuated control gives the same movement 456 seconds. The new pilot improves some crossing-tail point estimates with pedestrian priority but still leaves many scheduled vehicles uninserted at the measurement cutoff. This is different from the original vehicle-only models leaving pedestrians unserved.

The actor samples a categorical action during training; evaluation takes its highest-probability action. Near-equal probabilities can therefore produce varied training decisions but repetitive evaluation choices. High training entropy makes this worth testing, but does not prove it caused the failure. The pilot has only 32 training episodes and cannot establish convergence or the general performance of PPO.

## Frozen diagnostic comparison

Use each of the 12 existing selected `best.pt` checkpoints without retraining or changing physical rules. Use the same two development demand seeds, 47001 and 47002, on medium vehicle/medium pedestrian demand. Do not use the reserved final test seeds.

- Reproduce the original deterministic evaluation and actuated reference. Compare vehicle, person and crossing outcomes exactly; investigate mismatches first.
- Evaluate categorical sampling with policy seeds 48001, 48002 and 48003. Each junction has a separate random stream. These policy seeds are distinct from SUMO demand seeds. Python's categorical sampler draws from the same probability distribution as the actor; it does not reproduce Torch's training random sequence.
- Include uniform choices among currently allowed actions, both with and without the same pedestrian-priority restriction. This tests whether random allocation alone explains restored service.
- Record local action confidence, queued approaches lacking green, vehicle completion/unfinished/uninserted counts, crossing waits and unresolved requests. Retain all individual crossing waits in the diagnostic JSON.
- Inspect critic predictions against discounted Monte Carlo and GAE targets on these trajectories. These are evaluation-trajectory diagnostics, not proof of training convergence or a complete PPO correctness audit.

Total: 96 learned-policy evaluations, 12 uniform-control evaluations and two actuated replays = 110 runs. No training or GPU is needed. Policy-seed repeats are not independent trained models. The same source/configuration hashes are required, and old models/results are read-only.

## Decision after the diagnostic

1. If deterministic replays differ, resolve simulator/version or reproducibility differences before interpreting sampling effects.
2. If sampling restores service but learned decisions do no better than uniform choices, report an evaluation-mode problem and no demonstrated benefit from learning. Inspect effective decision counts, reward scales and critic fit before retraining.
3. If movement neglect persists, prototype a demand-aware service rule that covers both vehicle directions and pedestrian requests. Apply the same rule to conventional and learned controllers, and test safety and service before training. A finite service bound may be infeasible under persistent blockage; do not promise a hard waiting-time guarantee.
4. Only after a development pilot shows a credible improvement should the team freeze a final budget, five training seeds and the fresh test protocol. Keep the original and unsuccessful pilot results in the report. Do not select a favourable evaluation mode on the final tests.

Increasing the 12-second pedestrian admission window is a separate duration-sensitivity experiment. It cannot by itself correct a vehicle direction that is never selected.

## Run on the cluster

The small upload contains only two new scripts, their tests and this document. It does not replace the simulation, training code, checkpoints or pilot outputs. Extract into the existing **service-priority** checkout, where the 12 trained checkpoints remain. The downloaded results ZIP does not include those checkpoints.

Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\service-policy-audit-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/service-policy-audit-source.tar.gz
```

Cluster:

```bash
cd ~/cs4246-service-priority/trafficlightcontrol
tar -xzf ~/service-policy-audit-source.tar.gz
mkdir -p runs
sbatch --partition=normal --cpus-per-task=4 --mem=8G --time=02:50:00 \
  --job-name=policy-audit --output='runs/policy-audit-%j.log' \
  scripts/run_service_policy_audit.sh
```

This submits one CPU allocation. It reuses the existing virtual environment. The requested wall time is a limit, not a runtime prediction. If the job times out, resubmit the same command with unchanged files: completed evaluation JSONs are reused. Do not run two copies against the same output directory.

Check the job ID printed by `sbatch`:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 20 runs/policy-audit-JOB_ID.log
cat runs/service-policy-audit/REPORT.md
```

Replace `JOB_ID` with the actual number. Successful completion produces 110/110 records, the report and the following ZIP. A disappearing queue entry alone is not success.

Download from Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-service-priority/trafficlightcontrol/runs/service-policy-audit-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\service-policy-audit-results.zip"
```

## Local verification

Observed verification: 13 unit tests passed, seven short SUMO evaluations completed, and all three deterministic replays (actuated, plain IPPO and priority IPPO) matched the saved vehicle/person/crossing outcomes exactly. Vehicle counts reconciled in all seven runs. The full 110-run cluster diagnostic has not been run yet.

The local Torch installation emitted a NumPy compatibility warning; all checks still completed without using the NumPy bridge. The cluster runner reuses its existing working environment rather than installing packages.

The diagnostic unit tests cover masked, reproducible categorical sampling, deterministic near-tie choices and explained variance. The SUMO smoke mode uses the existing two-episode engineering checkpoints, one short demand seed and both plain and priority IPPO; it is not performance evidence. Reproduce with:

```powershell
Set-Location "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol"
$env:PYTHONPATH = (Resolve-Path .local/sumo-1.27.1/tools).Path
python -m unittest tests.test_policy_audit tests.test_service_priority tests.test_fairness_analysis -v
python -u scripts/audit_service_policy.py --pilot runs/service-priority-verified --output runs/service-policy-audit-verified --smoke --workers 1
```
