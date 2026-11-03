# On-policy PPO repair pilot

The 4-versus-32 offline diagnostic improved held-out value prediction but did not update traffic actors. This separate experiment tests actual controller learning. It retains the original network, safety, event measurement and completed/unfinished/uninserted accounting. It neither overwrites old models nor replaces the proposal's four-controller vehicle/pedestrian reward comparison.

## Frozen design

| Arm | Value estimator | Independent episodes before each PPO update |
|---|---|---:|
| A: original | Original raw output and scaled loss; no clock | 1 |
| B: normalized-clock | Normalised output; remaining time for critic only | 1 |
| C: normalized-clock-batch4 | Same repaired estimator as B | 4 |

Both IPPO and MAPPO, three independent training initialisations per arm: **18 models × 64 episodes = 1,152 training simulations**. Each training simulation lasts 600 seconds, giving 38,400 simulated seconds/model (599 transitions/junction/episode). All arms see matched demand seeds and scenarios per replica. A/B use 64 update batches, C 16; all use four full-batch PPO epochs per update. Equal simulated experience does not equal the number of policy updates or trajectories visited. C tests batching plus reduced update frequency, not diversity in isolation. B combines normalisation and time inputs and does not isolate their individual effects.

Actors always see local observations. MAPPO's critic sees all three junctions. IPPO's critic sees only its own junction. Only repaired critics get remaining-time/horizon; actors never receive that extra input. Both repaired arms begin with exactly the same actor parameters and raw critic function as A; the extra clock column starts at zero.

Pedestrian reward weight is fixed at 3 for this diagnostic. Rewards, discount 0.99, GAE lambda 0.95, Adam learning rate 0.0003/epsilon 0.00001, PPO clipping 0.2, entropy weight 0.01, value weight 0.5, four epochs and gradient limit 0.5 match the existing implementation. No service-priority restriction is applied in any arm. Forced-phase states train the critic; only states with a real choice train actors. No actor update, including Adam momentum, is allowed for an entirely forced batch.

## Online normalisation and boundaries

Offline fixed target statistics cannot simply be copied into evolving on-policy training. Here each repaired critic maintains cumulative mean/variance of **training GAE targets**, updated once per rollout batch. The final linear layer is rescaled in place so the moment update preserves predictions in raw reward units. GAE and its terminal-zero bootstrap always use raw-unit values. The value loss uses normalised targets and outputs; evaluation never updates moments. This is a PopArt-style adaptation with cumulative sample-weighted statistics and a minimum scale of 1, not an exact reproduction of the reference implementation. Existing Adam state remains attached to the same parameter objects.

Reference implementations inspected: [MAPPO trainer](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py) and [PopArt output preservation](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/popart.py). We implement the operations independently. Normaliser moments, model parameters, optimiser and completed history are checkpointed together.

Four-run batches are collected under an unchanged actor. Old action log probabilities are checked against the collection distribution. Returns/GAE are computed separately for each complete episode before concatenation. No return or advantage propagates into the next episode. Finite 600-second training is the declared objective; this is not an implementation of continuing-task truncation bootstrapping.

## Demand and evaluation

Training cycles through medium demand, low pedestrian demand, a pedestrian burst and short junction spacing, matching the earlier pilot's scenario mix. Mean rates and profiles are explicitly synthetic. Medium = 1,920 vehicles/hour and 1,200 pedestrians/hour network-wide; low pedestrian = 400/hour. Burst adds 20 arrivals at one route. Short spacing changes the separation from 140 to 65 metres and vehicle demand to 4,800/hour, so it is a combined queue-coupling stress case, not a geometry-only causal experiment.

Evaluation uses medium, burst, short-spacing and slow-walker settings. Slow walkers all travel at 0.8 m/s; the common clearance already protects that speed. Slow walking is outside the training scenario mix. These cases probe flow, concentrated pedestrian demand, downstream blocking and accessibility. They are a bounded development subset of the wider study, not sufficient evidence about all traffic conditions or real Singapore operations.

- Initialisation seeds: 72001–72003.
- Training demands: 810001 + 1,000 × replica index + episode index (zero-based).
- Validation demands: 820001/820002 at 16, 32 and 64 episodes, medium demand only.
- Development evaluation demands: 830001–830006, all four scenarios.
- Prospective final test seeds 58001–58020 are unused.
- Action streams: 900000 + demand seed + 1000003 × junction index. Categorical sampling is used in training and evaluation; uniform control samples the same physical legal mask. There is one fixed action stream per demand, not a separate replicated action-seed study.

Use the final 64-episode checkpoint for every model. Validation is a learning curve, **not model selection**. Evaluation runs use the existing 2,400-second horizon, 300–2,100-second measurement cohort and followup until 2,400. Compare fixed-time, actuated, pressure-inspired and uniform controls on identical scheduled demand and network hashes. The baseline count is 26 demand groups × 4 controllers = 104 simulations. Model validation uses 108 simulations; final development evaluation uses 432. Total planned simulations = **1,796**.

Critic diagnostics on evaluation runs use their first 600 seconds, with a matching finite-horizon return, and are labelled a prediction probe. Full-horizon traffic outcomes remain separate. Reports preserve action frequencies, phase time, queued movements without green, entropy, approximate KL, gradient norms, value bias/error and normaliser statistics. No critic improvement is treated as a traffic improvement.

## Interpretation

Report vehicle mean time loss with scheduled/completed/unfinished/uninserted counts. For each crossing, pool waits across demand seeds separately for each training replica; then average replica p95s. Include crossing mean waits, entries per replica, pending at cutoff, unresolved after followup, and retained raw per-run waits/speed groups. Do not pool replicas as if they were independent pedestrians. List crossings with p95 no worse than actuated out of 12 as a secondary measure.

The descriptive target still requires lower vehicle loss, all crossing p95 differences nonpositive, and no per-pair service/measurement deterioration. Sparse estimates (<100 pooled entries at any crossing for any replica or reference) are labelled inconclusive. Exploratory paired intervals resample whole training replicas and demand seeds, preserving paired controls, for actuated comparisons and B–A/C–B ablations. Three replicas, six demands and one action stream do not justify a strong confirmatory claim. Other baseline comparisons retain point estimates. Pure random control is an explanatory comparator, not a deployment recommendation.

If the pilot shows repeatable actual traffic improvement, freeze a larger protocol before using final seeds. If prediction improves without traffic gains, preserve that result and investigate rewards/actions rather than silently increasing budgets or choosing a winning run. Original negative results remain part of the paper. Synthetic simulations cannot establish deployed safety or benefit, or predict a grade.

## Run

This is a small full-source archive for a **new directory**, using the existing cluster environment. It needs CPU workers; no GPU or new package installation.

Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\ppo-repair-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/ppo-repair-source.tar.gz
```

Cluster terminal:

```bash
mkdir -p ~/cs4246-ppo-repair
tar -xzf ~/ppo-repair-source.tar.gz -C ~/cs4246-ppo-repair
cd ~/cs4246-ppo-repair/trafficlightcontrol
mkdir -p runs
sbatch --partition=normal --cpus-per-task=8 --mem=8G --time=02:50:00 \
  --job-name=ppo-repair --output='runs/ppo-repair-%j.log' \
  scripts/run_ppo_repair_cluster.sh
```

This requests one allocation, avoiding array/dispatcher submission limits. Runtime depends on available CPUs and congestion; the time limit is not a completion promise. On timeout, resubmit the same `sbatch` command from the same directory. Completed groups/episodes and optimiser state resume automatically. An interrupted four-episode batch restarts from its last atomic checkpoint; no completed batch is applied twice. Source/protocol changes require a new output. Never run two writers concurrently; a file lock checks this.

Progress (replace JOB_ID):

```bash
python3 scripts/collect_ppo_repair_status.py
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 30 runs/ppo-repair-JOB_ID.log
```

Successful completion requires scheduler `COMPLETED`, exit code `0:0`, report and results ZIP. Failed scientific criteria do not mean failed execution. No weights or bulky temporary simulator traces are put in the results ZIP; weights remain on the cluster. Keep the source/protocol unchanged until completion.

Download in PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-ppo-repair/trafficlightcontrol/runs/ppo-repair-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\ppo-repair-results.zip"
```

Local engineering smoke (four short episodes × six models, one replica, sparse demand; no research conclusions):

```powershell
$env:PYTHONPATH = "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\.local\sumo-1.27.1\tools"
python -m unittest tests.test_ppo_repair tests.test_fairness_learning tests.test_fairness_analysis -v
python -m src.traffic_control.ppo_repair.pilot --quick --workers 2 --output runs/ppo-repair-smoke
```

## Validation before delivery

Thirty targeted unit tests passed, covering the old-update equivalence, equal initial actor/value functions, raw-output preservation when normaliser moments change, episode-boundary returns, stale-policy rejection, forced-action masking (including Adam momentum), exact optimiser/normaliser restoration, missing export recovery, paired inference and existing phase/event rules. The weighted-CDF bootstrap matches explicit resampling on hand-generated varied datasets.

The final local engineering run in `runs/ppo-repair-verified` completed all six method/variant models, 24 training episodes, six validation and six evaluation runs, plus eight baseline runs: 44 SUMO simulations. Every run uses the existing conflict/occupancy assertions. A separately replayed saved MAPPO batch-four controller matched all vehicle/person/crossing results and diagnostics exactly, and evaluation left all parameters and normaliser buffers unchanged. Resubmission preserved every checkpoint and the summary byte-for-byte. Bash syntax and source archive checks passed. The full 18-model scientific pilot has **not** been run locally. Known local Torch/NumPy bridge and optional SUMO XML-data warnings did not prevent execution; no new packages were installed.
