# Critic diagnostics before another PPO study

The policy audit found poorly calibrated values, but a regression diagnostic is needed before attributing this to a particular implementation choice. These checks train prediction models on fixed trajectories. They do not update traffic actors or establish better traffic control.

## Check 1: output scaling

Six 600-second SUMO trajectories at medium vehicle/pedestrian demand, using all original physical rules. Four complete demand seeds (49101-49104) supply the training observations and Monte Carlo targets. Two separate development seeds (49201-49202) supply check data. The true final seeds 58001-58020 remain untouched.

Compare the existing raw-unit output/scaled-error formula with a normalised output that is converted back into raw units for metrics. Same 64x64 tanh architecture, Adam settings, initial network tensors and full-batch update budgets (128 and 512); three critic initialisations (49301-49303), separately for local-input and network-input critics. Statistics are learned from training targets only. A constant training-mean predictor is the essential reference: centring the target alone is not evidence that state-dependent prediction was learned.

Report bias, RMSE, explained variance and MSE divided by the constant predictor's MSE. The predeclared gate requires a ratio below 1 for every method/initialisation at 512 steps, averaging junctions. Both budgets and individual junctions remain in the results. We do not promote the best checkpoint after inspecting these check data.

## Check 2: remaining episode time

The first local check failed its gate despite a large scaling improvement. The code defines finite 600-second training episodes with zero terminal bootstrap, but observations contain phase age rather than episode time remaining. Even the same traffic state can have a different finite-horizon return near the end of the episode.

The follow-up keeps the same four training trajectories and uses fresh check seeds 49401/49402. Both critics use normalised outputs. Both have the same network shape with an extra input, and its initial weight is zero, giving identical initial predictions. One receives zero in that input; the other receives remaining-time / 600. Budgets, seeds and optimiser are matched. The predeclared gate requires the clock version to beat both the constant predictor and its no-clock control for every method/initialisation at 512 steps, averaging junctions.

This is a finite-horizon critic test. An alternative continuing-traffic objective would need correct truncation bootstrapping; that is a separate formulation. Neither adding time nor correcting output units demonstrates better policy decisions. Actor observations remain local and unchanged in these diagnostics.

## Local and cluster scope

Local results are recorded in [results/critic-checks/REVIEW.md](../results/critic-checks/REVIEW.md). Output normalisation alone failed the constant-predictor gate; the separately controlled remaining-time follow-up passed on fresh development check seeds. These findings justify the saved-policy diagnostic below, not a claim that traffic performance has improved.

The first local pair uses frozen actuated control. The cluster pair uses the existing selected actors `plain-mappo-r0` and `priority-mappo-r0`, with reproducible categorical sampling, to check whether the finding carries over to learned-policy trajectories. Replica 0 is fixed in advance, not chosen for winning results. Fresh critics are fitted with both local and network inputs; this is not a performance ranking of IPPO and MAPPO.

The archive downloaded previously contains summaries, not those saved actor weights. The cluster therefore runs these two additional policy cases where the checkpoints already exist. It performs 16 short SUMO runs in total plus small CPU regression fits. It does not resume PPO training. No GPU or additional installation is needed. The bundle excludes model weights and compact trajectory tensors; they remain available for inspection on the cluster.

Checks include hand-calculated returns, normalisation round trips, raw-unit GAE targets, exact model/optimizer save-and-resume, equal initial weights, exclusion of check targets from fitting, and the clock control's identical initial predictions. The local Python environment emits a known Torch/NumPy bridge warning; these checks use Torch tensors directly and have not required that bridge.

## Commands

PowerShell upload:

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\critic-check-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/critic-check-source.tar.gz
```

Cluster:

```bash
cd ~/cs4246-service-priority/trafficlightcontrol
tar -xzf ~/critic-check-source.tar.gz
sbatch --partition=normal --cpus-per-task=4 --mem=8G --time=01:00:00 \
  --job-name=critic-check --output='runs/critic-check-%j.log' \
  scripts/run_critic_checks.sh
```

One allocation avoids the previous array-submission limit. Check the actual job number printed by `sbatch`:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 20 runs/critic-check-JOB_ID.log
```

Replace `JOB_ID` with the number. Completion produces `runs/critic-check-results.zip`, including all four reports. A false scientific gate is a valid completed result, not a scheduler failure. If interrupted, resubmit the unchanged command; complete trajectories and fits are reused. Do not submit two writers to the same output folders.

Download from PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-service-priority/trafficlightcontrol/runs/critic-check-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\critic-check-results.zip"
```

## Interpretation

Value-output scaling is motivated by the [official MAPPO trainer](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py) and [value normaliser](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/utils/valuenorm.py). Our fixed-statistics supervised experiment is not an exact reproduction of their online normaliser or PPO training.

If fitting improves, the next justified experiment is an isolated on-policy training ablation, with normalisation and remaining-time effects separated. Preserve the old trained models and negative results. These development seeds may guide implementation choices; they must not be reused as untouched final evaluation data. A predictive critic is only one ingredient and does not guarantee the proposal's vehicle/pedestrian target.
