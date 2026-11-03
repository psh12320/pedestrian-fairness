# Service priority follow up

## Evidence and scope

The completed fairness-v2 evaluation includes 420 groups, 30 trained replicas and 512 episodes per replica. No learned configuration met the joint primary target. Pedestrian-aware/constrained vehicle-loss differences were +1.34 to +1.95 seconds against actuated. Their selected checkpoints were from episodes 64-192, despite continuing training to 512. Vehicle-only models had zero entries for some crossings and failed the service gate in every primary run. The outcome is evidence about this implementation and these simulated conditions, not about the impossibility of useful traffic RL.

The status bundle does not contain complete raw records. Its comparisons JSON was tail-truncated by the old collector. Do not parse that fragment as a full dataset. The original cluster files are not known to be corrupt. Use the diagnostic collector below; it retains whole JSON files and validates matched demand and protocol hashes.

The combined service score is not unique unserved people: completed/unfinished differences and pending crossing counts overlap. The new diagnostic export separates each signed component, unknown events, lost events and crossing sample counts. A count discrepancy at a finite cutoff must be distinguished from continued non-service after follow-up. Preserve the original pass criterion; do not relax it to manufacture success.

## Mechanistic hypothesis

The v2 safety machine allows repeated alternating vehicle phases without pedestrian admission. Minimum greens and conflict prevention guarantee neither timely service nor fairness. This is a permitted behavioural failure, not in itself a collision. Sparse episode-normalised crossing costs compete with dense queue rewards; small multipliers alone do not demonstrate that the policy is constrained. Examine raw reward/cost/advantage scales before attributing all failure to optimisation.

Hypothesis: constrain pedestrian admission decisions explicitly, and let RL optimise vehicle-phase allocation. This reduces the decision space and removes an avenue for pedestrian starvation. It does not guarantee p95 non-inferiority or the entire proposal target. A persistent physical blockage can still prevent entry even when admission is green.

The service-priority wrapper forces PED only when a pending crossing request exists AND PED is already legal in the original safety mask. It forbids empty pedestrian admissions. It never shortens minimum greens, clearance, yellow or all-red, and never invents a legal phase. Both actor sampling and PPO likelihood evaluation use the same effective mask; forced actions do not receive actor updates. Observations remain local, with the MAPPO critic receiving the original global state.

## Controlled pilot

Four conventional arms: unchanged actuated; unchanged pressure; priority-pressure (same immediate service rule); age-pressure (20-second oldest-request trigger, plus age-weighted pressure). The latter is an exploratory comparison, not a formal maximum-wait guarantee. Pressure is an aggregate queue heuristic and does not inherit published max-pressure stability proofs.

Four learned arms: plain IPPO, priority IPPO, plain MAPPO, priority MAPPO. All use the same existing neural architecture, optimiser, reward weight 3, initialisation seed per replica, scheduled demand and episode budget. This isolates the service rule without simultaneously changing reward scaling or adding imitation learning. The stronger new conventional baseline tests whether RL adds benefit beyond the rule.

Default pilot: 3 independent initialisations per learned arm, 32 episodes each, 600 simulated seconds per training episode, cycling medium demand, low pedestrian demand, pedestrian burst and short spacing. Initialisations 46001-46003; training demand starts 60001 with a 10000 offset per replica. Validation uses 47001-47002 and full original 2400-second horizons. Select checkpoints every eight episodes by the existing lexicographic service/crossing/vehicle criterion on medium and low-pedestrian validation only. Other scenarios test development robustness, not held-out generalisation. Compare all policies on identical per-seed demand hashes.

The quick mode uses 2 episodes, one replica, seed 45001, 180-second evaluation, 120-second cutoff and demand ending at 90 seconds. This is SOFTWARE VALIDATION ONLY, with sparse crossings; no superiority or convergence claim is valid from it.

Prospective fresh seeds 58001-58020 are reserved and are NOT used by this runner. After reviewing the pilot, freeze the selected method, budgets and final protocol before using them. The original 34001-34020 tests are now inspected and cannot serve as untouched tests for the new method. Use five training replicas and paired hierarchical uncertainty for a final method comparison. Even with 20 demand seeds, do not treat 100 replica/seed pairs as 100 independent training runs.

## Interpretation and course report

Report the negative result, mechanism, controlled intervention and uncertainty. Do not claim that the proposed intervention works before its results support that claim. If only the service rule helps and learning adds nothing, that is an informative outcome. If the strict target remains unmet, explain the trade-off and retain the negative results. Simulation cannot establish field safety or deployment benefit. A grade cannot be inferred from whether a hypothesis was supported; clarity, valid methodology and defensible insight address the supplied evaluation criteria.

## Commands

Local engineering check (PowerShell):

```powershell
$env:PYTHONPATH = (Resolve-Path .local/sumo-1.27.1/tools).Path
python -m unittest tests.test_service_priority tests.test_fairness_analysis -v
python -m src.traffic_control.service_priority.pilot --quick --workers 2 --output runs/service-priority-smoke
python scripts/package_service_priority.py
```

Keep a fresh checkout/output: v2 fingerprints cover all source files, so adding modules to the old checkout will invalidate its resume fingerprint. The original source/configuration files are unmodified, but the new pilot belongs in a separate directory on the cluster.

Upload from PowerShell:

```powershell
Set-Location "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol"
python scripts/package_service_priority.py
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\service-priority-source.tar.gz" shri@xlogin.comp.nus.edu.sg:~/service-priority-source.tar.gz
```

On the cluster:

```bash
mkdir -p ~/cs4246-service-priority
tar -xzf ~/service-priority-source.tar.gz -C ~/cs4246-service-priority
cd ~/cs4246-service-priority/trafficlightcontrol
mkdir -p runs
sbatch --partition=normal --cpus-per-task=4 --mem=8G --time=02:50:00 \
  --job-name=service-pilot --output='runs/service-pilot-%j.log' \
  scripts/run_service_priority_cluster.sh
```

This uses the existing cluster virtual environment without installing another copy. One CPU allocation avoids a chain of dependent array submissions; no GPU is required. If it times out, rerun the same command with unchanged source/protocol to resume atomic model checkpoints and completed baseline groups. Intermediate validation JSON may be recomputed after interruption. Do not run concurrent copies against the same output directory.

Check completion with `sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed`, replacing JOB_ID with the number printed by sbatch. A job disappearing from squeue does not establish success. The successful job creates `runs/service-priority-pilot/REPORT.md` and `runs/service-priority-results.zip`.

After success, download from Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-service-priority/trafficlightcontrol/runs/service-priority-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\service-priority-results.zip"
```

The results bundle includes the original-study component diagnosis and new pilot summaries, excluding weights and raw SUMO traces. The pilot is exploratory: four learned arms x three replicas x 32 episodes, with all four using the pedestrian reward weight 3. It supplements the original vehicle/pedestrian reward comparison and does not replace its four controllers or final evaluation.

## Research basis

- Pedestrian queues in max-pressure: https://arxiv.org/abs/2406.19305. Motivation for a stronger responsive comparator, not an exact reproduction.
- Residual reinforcement learning: https://arxiv.org/abs/1812.03201. General motivation for combining a conventional controller with learning. Our discrete action restriction is not that paper's additive continuous residual controller.
