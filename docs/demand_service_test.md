# Demand sensitivity and early pedestrian service

This is a supplementary, frozen-model development experiment. It tests whether
lower demand or earlier pedestrian decisions improve the vehicle/pedestrian
trade-off. It does not replace the previous negative results or the original
IPPO/MAPPO comparison, and does not train or select a new model.

**Completed 4 November 2026:** all 720 evaluations finished. The independently
checked [review and figures](../results/demand-service-review/REVIEW.md) explain
the outcomes. Early service improved every crossing p95 point estimate relative
to continued MAPPO, but had higher vehicle time loss than actuated in all four
settings. No learned/hybrid configuration met the joint target. Saved run output
uses approximately 116 MiB.

The protocol is saved before evaluation in `runs/demand-service-test/protocol.json`.
Source snapshots, model hashes, demand/network identities, compressed pedestrian
events, free decision observations, and individual run results are retained.

## Fixed experiment

| Setting | Network vehicles/hour | Network pedestrians/hour |
|---|---:|---:|
| vlow-plow | 960 | 400 |
| vlow-pmedium | 960 | 1200 |
| vmedium-plow | 1920 | 400 |
| vmedium-pmedium | 1920 | 1200 |

These are assumed Poisson demand rates, not real Singapore counts. The network
remains the existing three-junction, 12-crossing, 140 m corridor. Normal walkers
are used here; previous slow-walker and congestion results remain part of the
project. This experiment does not isolate geometry effects.

Ten new development seeds, 1010001–1010010, are paired across all controllers.
The reserved final seeds 58001–58020 remain unused. Changing vehicle demand also
changes the shared random generator's pedestrian realizations across demand
cells, so only within-cell controller comparisons share identical traffic.

Controllers:

- Fixed time, pedestrian-actuated, and pressure-inspired: 120 runs in total.
- Original, continued, tail-penalty and guided MAPPO, each with all three saved
  training replicas: 480 runs in total.
- Early-service hybrid using the same three continued MAPPO checkpoints: 120 runs.

Total: **720 simulations; zero training episodes**. All use the existing 2400 s
horizon, vehicle measurement cohort scheduled in [300,2100), and pedestrian
arrival cohort in the same interval with follow-up to 2400 s.

The early-service hybrid requests PED whenever at least one local crossing
request is pending and PED is legal. Otherwise it keeps the saved policy's
categorical action distribution. It does not change minimum greens, yellow,
all-red, admission duration, clearance or occupancy protection. Empty PED choices
are deliberately retained from the original policy to isolate early service.
This is a hybrid rule plus MAPPO, not a newly trained algorithm or a guarantee
of pedestrian p95 non-inferiority. The primary intervention comparison is
early-service versus continued MAPPO, using identical weights and paired seeds.

## Run and inspect

From the repository root, with the existing Python, Torch and SUMO installation:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -m unittest tests.test_demand_service_test -v
python -m scripts.demand_service_test --quick --workers 1 --verify-replay --output runs/demand-service-preflight-new
python -m scripts.demand_service_test --workers 2
python -m scripts.demand_service_test --status
```

The complete reviewed archive must exist at
`runs/pedestrian-finetune-cluster-results.zip`. Only its verified checkpoint
members are read, with Torch's weights-only loader; the archive is not extracted
or its source executed. A changed source fingerprint requires a new output
directory. The runner is resumable and exclusively locked; do not start a second
copy against an active directory. `--workers 1` runs in the main process to
reduce memory use. GPU acceleration is unnecessary for these SUMO evaluations.

To stop gracefully, create `runs/demand-service-test/STOP`. Completed outcomes
remain saved. Resume only after removing that specific STOP file, with unchanged
source and the same command. A 512 MiB output cap and minimum free-space check
bound storage. Temporary SUMO XML files are removed after each completed run.

Successful completion creates `REPORT.md` and `summary.json` in the output
directory. Reports require every planned run and verify event/decision hashes,
model identities, demand/network pairing, and reconstructed crossing waits.
Do not treat disappearance of a process or a partial result count as completion.

## Interpretation decided before results

- Lower completed-vehicle time loss alone is insufficient: include insertion
  delay, scheduled/completed/unfinished/uninserted vehicle counts, whole-run
  person completion, pending crossing requests, and requests still unresolved.
- Each crossing p95 pools the ten demand seeds within a training replica; report
  the mean of three replica p95s and the individual replica estimates.
- Paired bootstrap intervals resample whole seeds and training replicas. These
  are exploratory intervals, not simultaneous guarantees across crossings.
- Fewer than 100 entries in any pooled replica/reference crossing is sparse and
  inconclusive. Missing or unserved requests are never assigned zero wait.
- The original target requires vehicle improvement, all 12 crossing p95 point
  estimates no higher, and no strict service deterioration. Cutoff-only failures
  are reported separately from requests still unresolved after follow-up.
- Report all four demand settings. A favorable low-demand setting establishes
  a limited condition for benefit, not general superiority.

## Preflight observed on 4 November 2026

Five policy/coverage unit tests passed. The verified preflight at
`runs/demand-service-preflight-v3` completed all eight reduced-horizon controller
runs plus two identical replays (actuated and early-service). Metrics, events
and free decision records reproduced exactly. All 116 recorded crossing entries
in the eight saved runs occurred under the admission state active during the
movement step. This small preflight validates execution, not algorithm performance.

Two earlier preflights are retained for traceability: the first stalled at the
restricted local simulator connection and was stopped before producing outcomes;
the second exposed a replay-check representation issue (in-memory tuples versus
saved JSON lists). The replay checker now compares canonical JSON structures.
No simulator, trained weights, or pedestrian metric was changed by that fix.
