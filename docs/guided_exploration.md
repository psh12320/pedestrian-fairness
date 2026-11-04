# Guided exploration experiment

**Completed:** [review and compact archive](../results/guided-exploration-cluster-review/REVIEW.md). The [later full table](../results/metrics-guide/guided_all_settings.md) corrects crossing pass counts affected by arithmetic roundoff. Subsequent fine-tuning and failure-replay studies are indexed in the [catalogue](experiments/README.md).

This is a separate development study after the negative PPO-repair result. It preserves the three-junction network, 12 crossings, physical action mask, minimum greens, yellow/all-red transitions, protected pedestrian admission/clearance, event measurement and service accounting. It does not replace the original four-controller proposal comparison or overwrite existing results. No simulation settings are changed to favour a method.

## Recorded engineering verification (3 November 2026)

39 unit/regression tests passed. The four-arm smoke completed 38 short SUMO simulations, including actual imitation fitting, PPO/RND updates, baselines, validation and final evaluation. Runtime safety assertions passed. A separate repeated evaluation reproduced demand, metrics and action diagnostics exactly. Restarting the completed pipeline left all training/model/demonstration checkpoints unchanged. RND model/optimiser resume was also checked exactly in the unit suite. The Bash launch script passed syntax checking, and the source/results archives passed content/completeness checks.

The smoke caught and fixed a teacher tie-break bug: normalised floating-point queue sums could disagree when the underlying integer vehicle counts were equal. Labels now recover integer counts before comparing, with a regression test. An initial test-suite failure came from Windows temporary-folder permissions; rerunning with the temporary directory inside the workspace passed.

Evidence: `runs/guided-unit-tests.log`, `runs/guided-exploration-smoke-verified/REPORT.md`, and `runs/guided-exploration-smoke-verification.json`. The 180-second sparse smoke is **not a research result**. Full 128-episode cluster training has not been started by this local verification. The cluster requires the user's interactive password login; enter it only in their terminal.

## Frozen comparison

The default is **four MAPPO arms × three initialisations × 128 episodes = 12 trained models and 1,536 training simulations**. All use the repaired critic, local actors, network-wide critic plus remaining training time, gamma .99, GAE .95, four full-batch PPO passes per episode, Adam .0003, gradient clip .5, entropy .01 and value coefficient .5. Each episode is 600 seconds. Same demand seeds, episode counts and PPO passes across arms. Final checkpoints only; no best-run selection.

| Arm | Changed relative to control |
|---|---|
| control | Nothing: numerically checked against previous repaired PPO |
| age | Dense bounded quadratic penalty for oldest waiting request at each crossing |
| guided | Age reward plus actuated actor pretraining and decaying teacher cross entropy |
| rnd | Age reward plus local random-network-distillation novelty during training only |

The age penalty is `sum(min(oldest_wait / 60, 5)^2)` over crossings with pending requests, subtracted from the existing `-(vehicle_queue + uninserted_due + 3 * pending_count) / 10`. It penalises a long wait even at a low-volume crossing. It is a bounded quadratic adaptation inspired by long-wait-sensitive reward research, **not EI-MWAQ reproduction, a p95 constraint, or a fairness guarantee**. The count/age information already existed in the observations. Test age versus control to isolate this reward change.

Guidance: collect 12 independent actuated demonstrations, evenly cycling through the four training settings. Check that labels reconstructed using only local observations and masks exactly match actual baseline actions. Fit actors with 1,000 minibatches/junction (up to 512 free-decision states each, Adam .001); leave critics unchanged. During PPO, add teacher cross entropy weighted by `.1 * max(0, 1 - episode_index / 128)`. Inference is the sampled actor, with no teacher override. The imitation-only actors are evaluated independently, and their closed-loop validation outcomes are reported before RL. High classification accuracy is insufficient evidence of successful imitation. Poor fidelity is retained as a diagnostic, not used to discard a replica or silently select another model. The guided arm bundles pretraining and online guidance; it does not isolate those two components.

RND: a fixed random local-observation network supplies feature targets to a predictor. Update input moments once per training rollout, standardise/clamp input to [-5,5], divide squared prediction error by its cumulative mean, cap novelty at 5 and add it with weight .05. Fit the predictor with four passes per rollout, Adam .001. Target parameters are frozen. Moment statistics and predictor optimiser are checkpointed. No curiosity module, intrinsic bonus, or normaliser updates run during evaluation. This is an independent RND adaptation, not a reproduction of the Helsinki paper. RND rewards novelty, which may include undesirable congestion; traffic metrics remain the outcome.

Guidance adds demonstrations and supervised updates; RND adds predictor work. Thus equal PPO experience/updates do not mean equal total compute. Log wall time and additional sample/update counts. A separate optional `--methods ippo mappo` run supports both algorithms and doubles PPO model count; use a different output directory. The first bounded run uses MAPPO; do not claim an IPPO intervention result from it.

## Demand, seeds and inference

Training cycles medium, low pedestrian demand, pedestrian burst and short spacing. Evaluation uses medium, burst, short spacing and slow walkers. Rates are synthetic: medium 1,920 vehicles/h and 1,200 pedestrians/h network-wide; low pedestrians 400/h; burst adds 20 arrivals; short spacing combines 65-m separation with 4,800 vehicles/h, so it is not a pure geometry test. Slow walkers use .8 m/s and are held out from training. Standard evaluation is 2,400 seconds, with a 300–2,100-second measurement cohort and followup through 2,400.

- Initialisations: 85001–85003.
- Demonstrations: 860001–860012; no validation/evaluation samples used for fitting.
- Training demand: 870001 + 1000 × replica + episode index.
- Validation: 880001/880002, medium demand, after episodes 32/64/128.
- Development evaluation: 890001–890006, all four settings.
- Reserved final seeds 58001–58020 remain unused.
- Shared categorical action streams follow the previous pilot; one stream per demand/junction, not independently replicated action noise.

Baselines: fixed, actuated, pressure-inspired and uniform, 104 simulations. Teacher demonstrations: 12. Teacher validation: 6. Learned validation: 72. Evaluation: (12 learned + 3 imitation-only) × 4 scenarios × 6 demand seeds = 360. **Total: 2,090 simulations** including 1,536 training runs. Teacher data are shared across replicas; report that dependence.

Primary target remains lower completed-vehicle time loss, no higher p95 at every crossing, and no service-count deterioration versus matched actuated control. Include all scheduled/completed/unfinished/uninserted counts; per-crossing mean/p95/entries; pending at cutoff and unresolved after followup; complete person trips separately from measured crossing requests. Report paired exploratory intervals over whole training replicas and demand seeds, plus the secondary count of crossings helped. P95 is pooled across demand seeds within a replica, then averaged across replicas. Fewer than 100 entries at any crossing/replica/reference makes the tail comparison inconclusive. Three initialisations, six demand seeds, synthetic demand and one small corridor limit claims.

## Other MARL methods considered

- [QMIX, ICML 2018](https://proceedings.mlr.press/v80/rashid18a.html): value-based central training with local action-value networks and monotonic mixing. Replay offers a different learning mechanism from PPO. A credible next implementation must handle joint transitions, physical masks, forced-phase intervals, target networks and replay consistently. Monotonic factorisation is a modelling restriction and does not ensure pedestrian fairness. Not implemented or trained by this pilot.
- [COMA, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794): counterfactual central critic credits an individual agent's action while holding other actions fixed. Relevant to coordination, but it introduces a harder action-value estimation problem when our current value estimates are already weak. Not implemented or tested here.
- [RND-MAPPO, IET ITS](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.70247): reports gains in average vehicle and pedestrian waiting against pedestrian-aware MAPPO. Its averages do not establish every-crossing tail non-inferiority; its table also contains a competing controller with lower vehicle waiting.
- [Pedestrian–vehicle PPO](https://arxiv.org/abs/2504.05018): maximum-wait-sensitive EI-MWAQ reward, joint central control on a different corridor, improvements against fixed-time control. We borrow a testable reward principle, not performance guarantees or its stopped-person waiting definition.
- [FitLight](https://arxiv.org/abs/2502.11937): imitation alongside RL for traffic signals motivates guidance, but our local actuated teacher, pretraining and pedestrian objective are our own adaptation. We do not claim to reproduce its federated system.

The evidence is not yet a joint win. This study tests learning explanations rather than expanding into many unverified algorithms. QMIX is the next distinct algorithm candidate if a further study is justified; preserve all negative results.

## Run on the cluster

CPU workers are appropriate: SUMO dominates this workload. No GPU or new installation is required. A single allocation avoids the previous array submission limit. Use the existing cluster environment in the fairness-v2 checkout.

PowerShell upload from any directory:

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\guided-exploration-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/guided-exploration-source.tar.gz
```

Cluster terminal:

```bash
mkdir -p ~/cs4246-guided-exploration
tar -xzf ~/guided-exploration-source.tar.gz -C ~/cs4246-guided-exploration
cd ~/cs4246-guided-exploration/trafficlightcontrol
mkdir -p runs
sbatch --partition=normal --cpus-per-task=8 --mem=8G --time=02:50:00 \
  --job-name=guided-pilot --output='runs/guided-exploration-%j.log' \
  scripts/run_guided_exploration_cluster.sh
```

The time request is not a runtime promise. If the job times out, resubmit the same command without changing source/configuration: completed baselines, teacher models, atomic model/optimizer/RNG states and evaluations resume. An interrupted demonstration-collection stage is rebuilt from its fixed seeds. A lock prevents two writers. Do not repeatedly unpack newer code into an active study.

```bash
python3 scripts/collect_guided_exploration_status.py
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 30 runs/guided-exploration-JOB_ID.log
```

Successful execution requires COMPLETED/0:0 and the audited report/archive, irrespective of whether the scientific target passes. Source/protocol hashes, paired demand/network hashes, sample counts, model hashes and coverage are checked. The results archive includes final weights and matching source for replay, but excludes optimizer histories, demonstration tensors and bulky simulator traces. Keep them in the cluster checkout.

PowerShell download:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-guided-exploration/trafficlightcontrol/runs/guided-exploration-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\guided-exploration-results.zip"
```

Local engineering smoke (not scientific evaluation):

```powershell
$env:PYTHONPATH = "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\.local\sumo-1.27.1\tools"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m unittest tests.test_guided_exploration tests.test_ppo_repair -v
python -m src.traffic_control.guided_exploration.pilot --quick --workers 2 --output runs/guided-exploration-smoke
python scripts/package_guided_exploration.py --results --root runs/guided-exploration-smoke
```

Quick mode uses four 180-second training episodes/arm, one replica, four demonstrations, 60 imitation minibatches/junction and sparse development traffic. It checks execution, measurement and safety assertions, not whether training works.
