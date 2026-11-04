# Pedestrian-focused continuation of vehicle-efficient MAPPO

**Completed on the cluster, 4 November 2026:** all 876 simulations finished. [Results](../results/pedestrian-finetune-cluster-review/REVIEW.md). The local duplicate stopped at 42/876 after transfer; it is not pooled with the cluster study. Queue/launch instructions below are historical reproduction guidance.

This supplementary experiment asks whether a model that already moves vehicles
well can learn earlier pedestrian service without losing its vehicle advantage.
It does not change the original IPPO/MAPPO research question or replace previous
negative results. Results are exploratory simulation findings.

## Why this experiment

The archived guided-exploration study found that MAPPO-control saved vehicle
time but worsened all crossing tails in the primary medium-demand setting.
Imitation and teacher-guided models protected more crossings while surrendering
much of the vehicle gain. Those models learned imitation first. Here we reverse
that order: start from each vehicle-efficient MAPPO-control checkpoint and add
targeted pedestrian learning. MAPPO-control already used pedestrian reward
weight 3; it must not be described as a vehicle-only-reward model.

The three original training replicas are all included. The complete actor,
critic and value-normalisation state is retained. Since the archived final
checkpoint contains no optimizer, every arm starts the same fresh Adam.

| Arm | Additional learning |
| --- | --- |
| Original | Frozen parent, no additional learning; evaluated again on new matched demand |
| Continue | Original queue/pending reward, common conservative fine-tuning procedure |
| Tail | Continue plus a bounded penalty for each crossing's oldest pending age beyond 20 seconds |
| Guided | Tail plus local actuated advice when a pedestrian request is at least 20 seconds old and pedestrian service is legal |

All three trained arms share learning rate 0.0001, four PPO epochs, separate
actor/critic gradient clipping at 0.5, and a small 0.02 penalty for departure
from the frozen parent's action distribution. This is a matched conservative
fine-tuning comparison, not exact continuation of the original optimizer.
Parent anchoring is a regularizer, not a guarantee of preserving vehicle flow.

The tail cost per crossing is
`min(max((oldest_wait - 20) / 20, 0), 5)^2`, summed over pending crossings.
Guidance has fixed coefficient 0.5. Its selected cross-entropy losses are summed
and divided by **all free decisions**, so a rare urgent label cannot dominate
an entire episode. The 20-second threshold is a predeclared engineering choice,
not a direct p95 constraint or a guaranteed acceptable pedestrian wait.

## Where the expert data comes from

During each training episode the student acts in our own SUMO environment.
The local actuated controller labels every state at which the student had a
real choice. We retain observations, legal masks, teacher/requested actions,
student actions, time and selection metadata in compressed JSON datasets.
Only urgent labels contribute guidance in the guided arm. Labels are also saved
for the other two arms for comparable fidelity diagnostics.

This resembles online dataset aggregation, but the teacher does not override
the rollout action. PPO still uses the exact probabilities of the actions the
student actually took. Forced transition states and the terminal action with
no successor are excluded from the label dataset. Poor outcomes and jams are
retained. No external demonstration data or old CS3263 artifacts are used.

## Frozen development protocol

- Three parents, three trained arms, 64 additional 600-second episodes each:
  **576 training simulations**.
- The same four training settings are cycled in every arm: medium, low
  pedestrian demand, concentrated pedestrian burst, and short spacing.
  Seeds are `990001 + replica*1000 + zero_based_episode`.
- Diagnostic validation at episodes 16, 32 and 64, on seeds 970001/970002:
  **54 validation simulations**. Validation does not select checkpoints.
- Four evaluation settings: medium, pedestrian burst, short spacing and slow
  walkers. Four new seeds 980001–980004, 2400-second horizon. Nine fine-tuned
  models and three frozen parents: **192 evaluation simulations**.
- Fixed, pedestrian-actuated and pressure-inspired baselines on the same
  evaluation demand and two validation seeds: **54 baseline simulations**.
- **876 simulations total.** Final checkpoints only; all replicas included.
  Reserved final-study seeds 58001–58020 remain unused.

All controllers use the current common signal-safety implementation and collision
checks. The original models are reevaluated because historical aggregate results
are not paired comparisons under the current simulator code and new seeds.
Local actors keep the original 42 inputs. Only MAPPO's critic sees global state.
This experiment does not silently combine the separate controller-memory audit.

Report completed-vehicle time loss alongside scheduled/completed/unfinished/
uninserted service and insertion delay. For every crossing report wait mean,
p95, entries and pending/unresolved requests. Retain pedestrian completion,
entered-without-observed-exit counts, phase histories and compressed entry/exit
events. Entry waiting alone does not describe a person stalled on a crossing.

Compare each arm against actuated and its corresponding frozen parent. Compare
Tail–Continue and Guided–Tail to isolate the new reward and teacher effects.
Use paired exploratory intervals over whole training replicas and demand seeds.
Crossing intervals are pointwise; selecting a favourable subset after observing
them does not establish simultaneous non-inferiority for that subset.

Vehicle improvement with 6/12 crossing p95 point estimates no higher is useful
**partial crossing protection**, provided service counts and uncertainty are
also shown. It does not satisfy the original every-crossing criterion. Estimates
with fewer than 100 entries per pooled replica/reference crossing are flagged
as sparse. No failed seed or less successful replica is removed.

## Run and resume

Run from the repository root with the existing Python/SUMO installation:

```powershell
python -m pytest tests/test_pedestrian_finetune_teacher.py tests/test_pedestrian_finetune_learning.py tests/test_pedestrian_finetune_reporting.py tests/test_pedestrian_finetune_pilot.py -q
python -u -m scripts.pedestrian_finetune.pilot --quick --workers 1 --output runs/pedestrian-finetune-smoke
python -u -m scripts.pedestrian_finetune.pilot --workers 1 --output runs/pedestrian-finetune
python -m scripts.pedestrian_finetune.pilot --status --output runs/pedestrian-finetune
```

The quick run has 19 short simulations and is only an integration test.
It cannot establish traffic improvement. The full run is CPU/SUMO-bound; no
large GPU package or external dataset download is needed. One worker avoids an
extra Torch interpreter on the laptop. Two workers are available after other
experiments finish and if memory permits.

On 3 November 2026, all 19 smoke simulations completed. The 36 unit/integration
tests passed (9 teacher, 11 learner, 10 reporting, 6 orchestration). A fresh
same-seed SUMO replay reproduced results, event timestamps and phase history
exactly after normalising tuple/list serialization. The live guided arm received
7 urgent labels among 67 free training choices and applied a nonzero guidance
loss. Two smoke updates are too few to assess performance; the four learned
variants had identical aggregate evaluation outcomes in that one short run.

The full protocol is frozen at fingerprint
`bfa5d776ad0f45628de0374d95375d4a0bfd3d0de0f646cc34a29d24cf47105b`.
The lightweight `scripts/queue_pedestrian_finetune.py` launcher waits for the
existing audit process to exit before starting this study with one worker.
It verifies that sources and the parent archive still match the successful
smoke test. `runs/pedestrian-finetune/queue.json` shows queue state; after launch,
`status.json` shows experiment progress, with logs in `runner.log`/`errors.log`.
The computer must remain awake. These files distinguish queued, running,
complete and failed states; absence from a process list alone is not success.

The default parent archive is `runs/guided-exploration-results.zip`;
`--parent-archive PATH` overrides its location. Its manifest, protocol and all
selected parent hashes are verified; only the three needed model files are
copied. Archived weights are loaded with `weights_only=True`.

Repeating the identical command resumes atomic model/optimizer/RNG checkpoints.
Protocol/source hashes prevent resuming against changed experiment code. Keep
the generated source snapshot for reproduction. Create `OUTPUT/STOP` to stop at
the next simulation boundary; remove that specific marker before resuming.
The writer lock prevents overlapping writers. A 512 MiB output cap and 512 MiB
free-disk floor stop the run before unbounded storage growth. No old result is
deleted automatically.

## Limits of the conclusions

This is a three-replica, four-test-seed development study; it is not the final
five-by-ten confirmation proposed earlier. Synthetic demand assumptions remain,
the short-spacing case also changes demand, and 600-second training from empty
conditions differs from the longer evaluation. An actuated teacher cannot
guarantee a better-than-actuated policy. Physical pedestrian stalls may remain
even when signal colours are legal. The shared safety mechanism must continue
holding clearance while occupied; learning cannot waive that rule. The report
must describe both these failures and any improvement, without a deployed-road
claim or a promise that the original target will be achieved.
