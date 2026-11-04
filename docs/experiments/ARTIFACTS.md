# Reading, reproducing and sharing the evidence

## What a GitHub clone contains

The [catalogue](README.md) and [registry](registry.json) identify completed studies,
their code, checked reviews and compact results. The active research branch is
`fairness-service-priority`; `master` and `pedestrian-fairness-v2` are older
snapshots. Select the research branch to see the latest material.

```bash
git clone --branch fairness-service-priority https://github.com/psh12320/pedestrian-fairness.git
cd pedestrian-fairness
python scripts/experiment_inventory.py --check
```

The check uses only Python's standard library. It validates registered paths,
exported JSON/gzip, integrity hashes and catalogue links. It does not run SUMO,
train models or certify the scientific correctness of every statistic.

Shared evidence includes original reports, complete aggregate tables (including
unfavorable settings), protocols, status snapshots, reviews and plots. E12's
large aggregate summary is losslessly gzip-compressed; nothing is filtered out.
For example, read it without creating another copy:

```python
import gzip, json
with gzip.open('results/audit-repair-review/archived_summary.json.gz', 'rt') as f:
    summary = json.load(f)
print(summary.keys())
```

The [public manifest](../../results/experiment-index/manifest.json) records file
sizes and SHA-256 hashes, plus the exact source of exported compact artifacts.
Historical text hashes normalize CRLF to LF for portable checkout verification;
export payload hashes and binary/gzip hashes preserve the exact original bytes.
Hashes identify bytes; they do not independently establish correct measurement.

## What remains outside Git

`runs/`, portable SUMO, Python environments, newer model directories and personal
LTA application documents are ignored. A folder being inside this local project
does **not** mean it is on GitHub. Do not force-add those folders or personal
documents. No full raw dataset or cloud release has been published by this update.

| Evidence | Availability and replay limit |
|---|---|
| E01–E03 legacy outputs | Compact results, many per-run/event tables and audits are tracked. Ignored training/model folders are needed for exact model replay. |
| E04 fairness-v2 | Compact protocol/report/primary inference are tracked. Full raw outputs/checkpoints remain in the original cluster study. The downloaded 2 MiB `comparisons.json` is truncated and invalid; do not analyze it. Recovered absolute aggregates are in `results/service-priority-cluster/summary.json`. |
| E05–E06 priority and action audit | Reviews, summaries and exported archive protocols are tracked. Downloaded result ZIPs remain local. The action audit archive lacks actors. |
| E07–E09 critic and PPO repair | Reviews, summaries and protocols are tracked; ZIPs lack raw critic trajectory tensors and/or actor weights. Exact simulator replay is not possible from those bundles alone. |
| E10 guided exploration | Compact results are tracked; the local ZIP includes fifteen model checkpoints but omits raw simulator traces and demonstration tensors. |
| E11 local timing | Compact results are tracked; raw local study files are ignored. |
| E12 repair trial | Full aggregate summary, protocol, report, completion status and review are tracked; raw runs, checkpoints and source snapshot remain local. |
| E13 fine-tuning | Compact results are tracked. The local 21 MB ZIP includes twelve parent/continuation checkpoints and detailed event evidence. The local duplicate was stopped and is not pooled. |
| E14 failure replays | Summary/review/protocol are tracked; detailed instrumented local traces and the E10 checkpoint archive are needed for exact replay. |
| E15 demand/early service | Full aggregate summary, protocol, report, completion status, review and plots are tracked. Raw event/decision traces occupy about 116 MiB locally; they depend on E13's checkpoint ZIP. |
| E16 external friend's work | Our review and verified comparison tables only. The friend's code/archive is not incorporated or redistributed. |

The [local artifact inventory](../../results/experiment-index/local_artifacts.json)
lists available result/source archives with byte sizes, hashes and ZIP member
counts, and every top-level local `runs/` directory with file count/size.
Unclassified directories are explicitly marked: many are engineering probes,
retries or temporary outputs, not undocumented independent research results.
This is a dated snapshot of one machine, not proof that a file exists on yours.
Historical absolute paths inside archived metadata describe the original run.

## Getting additional artifacts from the team

Ask the teammate holding the workspace/cluster outputs for the exact archive
named in `local_artifacts.json`; verify its SHA-256 before use. There is currently
no public download URL for these bundles. Do not claim the repository is a
self-contained raw-data/checkpoint release. Some old bundles never included
the missing weights or trajectories; obtaining the original cluster directory
is necessary in those cases.

For E15 replay, place `pedestrian-finetune-cluster-results.zip` under `runs/`.
Its size is **21,368,736 bytes (20.38 MiB)**, not 116 MiB; the latter refers to
the separate E15 evaluation-output folder. Keep the ZIP intact. Check the
archive, twelve checkpoint hashes and frozen source compatibility without
starting any simulations:

```bash
python scripts/check_frozen_checkpoints.py
# If only source line endings differ after checkout:
python scripts/check_frozen_checkpoints.py --repair-line-endings
```

The optional repair restores LF/CRLF differences only and refuses actual code
changes. Do not bypass source checks to make an altered experiment appear frozen.

The expected SHA-256 is
`255b82536b3a341c8964ef2338b5bd0a5e54fbe73dfd8e89fd9315b18c46354b`.
For E10/E14, use `guided-exploration-results.zip`, SHA-256
`ea17b162e3329e777abe0f98122ffae42d2642453bedd704fc5e8eef75447725`.
Use `Get-FileHash -Algorithm SHA256 <file>` in PowerShell or `sha256sum <file>`
on Linux. Verify actual values against the manifest rather than trusting a name.

For the team member with NUS cluster access, these commands download the existing
fine-tuning bundle. They do not submit new jobs. Run in your own local terminal:

```powershell
New-Item -ItemType Directory -Force runs
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/cs4246-pedestrian-finetune/trafficlightcontrol/runs/pedestrian-finetune-cluster-results.zip runs/
```

The earlier large study lives under
`~/cs4246-fairness-v2/trafficlightcontrol/runs/fairness-v2-study` on the original
cluster account. Check sizes before transferring; the full folder can exceed
the laptop storage budget. Preserve source snapshots, protocols, checkpoints
and raw outputs there. Avoid downloading/re-extracting entire archives to read
a few JSON members. Do not share credentials or personal contact forms.

## Reproduction has three different meanings

1. **Read the evidence:** fresh clone, catalogue, reports and public integrity
   check. No PyTorch, SUMO or large downloads needed.
2. **Recalculate or replay saved experiments:** obtain the study's specified
   artifacts and frozen source/runtime. Review scripts often expect ignored
   `runs/` paths; see their study guide and command-line help. An aggregate
   summary can reproduce tables but cannot reconstruct every pedestrian trace.
3. **Train again:** substantial CPU/SUMO work, independent random replicas and
   exact frozen settings. Do not rerun merely to read results. Different source,
   simulator or libraries constitute a new protocol; do not overwrite old runs.

The original standard-library linear learner uses `requirements.txt` (TraCI).
The newer neural studies additionally need PyTorch. The historical cluster
environment pins CPU PyTorch 2.3.0, NumPy 1.26.4 and SUMO/TraCI 1.27.1;
see [setup](../fairness_execution.md). Portable SUMO and a Python environment
are deliberately not vendored. GPU availability does not remove SUMO's CPU cost.
Run expensive jobs on compute nodes, not cluster login nodes.

Use the guide for the intended study rather than one generic pipeline command:

| Work | Guide / entry point |
|---|---|
| Single-junction/baseline smoke | [Root README](../../README.md), `src.traffic_control.cli` |
| Legacy repeated comparison | [Historical study](../../results/final-study/REPORT.md), `scripts/finish_study.py` |
| Neural full study | [Execution](../fairness_execution.md), `src.traffic_control.fairness.pipeline` |
| Priority / policy choice | [Priority](../service_priority_followup.md), [action audit](../service_policy_audit.md) |
| Critic diagnostics | [Critic checks](../critic_checks.md), [data volume](../critic_data_volume.md) |
| PPO repair / exploration | [Repair](../ppo_repair_pilot.md), [exploration](../guided_exploration.md) |
| Memory/clipping factorial | [Audit trial](../audit_trial.md), `scripts/run_audit_trial_windows.py` |
| Fine-tuning | [Cluster guide](../pedestrian_finetune_cluster.md), `scripts.pedestrian_finetune_cluster` |
| Latest demand/early service | [Design and exact commands](../demand_service_test.md), `scripts.demand_service_test` |

Modern sources evolved after older experiments. Saved `protocol.json` source
hashes and source ZIPs identify the code that actually ran. The current Git
version alone does not guarantee a byte-identical rerun of E01–E15.

## Maintaining the documentation

When a new study finishes, add a registry entry and explain its hypothesis,
complete coverage, budgets, seed roles, measurement/safety version, results and
limitations. Preserve failures and partial attempts. Record evidence rather
than an unsupported claim that a process ended successfully.

On the owner's workspace, after reviewing any new outputs:

```powershell
python scripts/experiment_inventory.py --export-local
python scripts/experiment_inventory.py --check
```

Export copies only named compact records and selected protocol members; it does
not change frozen code, run simulations, delete files or include model binaries.
The compressed E12 summary retains the original uncompressed hash. Inspect the
Git diff before sharing. Public-manifest hashes cover result files, not the
registry/docs themselves, to avoid recursive checksums.

Keep AI assistance disclosed and contribution descriptions factual. Personal
application documents and local assistant instructions remain ignored; their
absence from Git is not a substitute for disclosure in the course report.
