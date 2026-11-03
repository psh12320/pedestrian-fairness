# Final-week evidence, experiments and presentation

Review date: 2 November 2026. This is a proposed work schedule, not a claim that the follow-up experiments have run.

## Keep the original question and expose partial progress

The proposal asks whether IPPO or MAPPO reduces vehicle time loss while keeping pedestrian p95 no higher than pedestrian-actuated control at every crossing. Retain this primary criterion.

Add a secondary descriptive outcome: how many of the 12 crossings have a p95 point estimate no higher than the matched actuated baseline? Always show the largest increase, all 12 differences, vehicle service counts and pending requests beside this count. An undefined p95 is not a successful zero wait. Keep the denominator at 12 and identify crossings that are unevaluable. Distinguish point-estimate counts from statistically supported non-inferiority. Do not change the comparison tolerance after seeing results.

The latest primary summaries give 3/12 for IPPO-pedestrian and 3/12 for MAPPO-pedestrian. The two vehicle-only configurations do not have an evaluable complete crossing vector in the downloaded summary. The extra constrained configurations give 4/12 for IPPO and 3/12 for MAPPO. These are not claims of a statistically established partial success.

## What is actually simulated

The main experiment has three signal agents, C0/C1/C2, and four crossing edges per signal, giving 12 crossings. The cluster protocol's observation schema and the network-generation code agree. The local network, environment, demand, safety and configuration hashes checked during this review match the cluster protocol.

The latest study contains 21 scenario configurations: a 3 x 3 independent vehicle/pedestrian demand grid plus 12 sensitivity/stress cases. This is not 21 independent road layouts. The main topology is one synthetic three-junction corridor, with a shorter-spacing variant. Demand rates are assumptions, not measured Singapore traffic or calibrated levels of saturation. Slower walkers, bursts, platoons, unequal routes and high demand test distinct mechanisms; the labels low/medium/high alone do not justify real-world realism.

The earlier full study used different demand and linear hold/change actors. The latest study uses neural three-choice actors and revised measurement. Never combine absolute old baselines with new paired differences. The NUS-derived extension uses approximate map-derived site centroids and spacings, but synthetic connecting roads, crossings, demand and signal plans. It is not calibrated Singapore control.

## Proposal reference audit

Reading depth is stated explicitly: reading an abstract or relevant methods/results sections is not a full replication.

| Ref. | Primary source and reading basis | What it supports | Boundary |
|---|---|---|---|
| 1 | [Chu et al., MA2C](https://arxiv.org/abs/1903.04527), primary abstract checked | Decentralised multi-agent actor-critic control can improve vehicle traffic in simulated networks | Uses A2C; does not establish IPPO/MAPPO every-crossing pedestrian fairness |
| 2 | [Wei et al., CoLight](https://arxiv.org/abs/1905.05717), primary abstract checked | Learned neighbour communication can improve network traffic control | Different algorithm and information design; no guarantee for our local actors or pedestrian target |
| 3 | [Ault and Sharon, RESCO](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/f0935e4cd5920aa6c7c996a5ee53a70f-Paper-round1.pdf), methods/results checked | Rankings change with sensing/layout; best-episode IPPO results can conceal instability | Supports careful evaluation rather than a universal RL advantage |
| 4 | [Ault and Sharon, cooperative MARL workshop study](https://people.engr.tamu.edu/guni/papers/CIKM22-MATCS.pdf), full four-page text checked | Cooperation is difficult; none of the tested algorithms beat fixed time on their Cologne Corridor task | Different implementations and vehicle-only objectives; useful negative evidence, not a diagnosis of our specific code |
| 5 | [Xu et al., PV-TSC](https://huangjunqin.com/assets/pdf/XuTITS2023PVTSC.pdf), problem, agent design and experimental setup checked | Pedestrians can be included in distributed signal RL; movement grouping, pressure and cycle design matter | Uses DQN and different sensing/phase assumptions; not the same crossing-tail target |
| 6 | [Kumarasamy et al., pedestrian-benefit optimisation](https://impact.ornl.gov/en/publications/optimizing-traffic-signal-control-to-enhance-transportation-effic/), author-institution abstract checked; full PDF retrieval timed out | Reports vehicle and pedestrian average improvements in a simulated real corridor | Comparison includes automated detection vs baseline pedestrian recall, so it does not isolate exactly our algorithm change or baseline |
| 7 | [Chen et al., Helsinki RND-MAPPO](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.70247), relevant methods/results and Table 10 checked | Pedestrian-aware MAPPO and curiosity exploration; explicit vehicle-pedestrian trade-off | Average pedestrian metrics, not every-crossing p95 vs our actuated controller |
| 8 | [Yu et al., PPO in cooperative games](https://arxiv.org/abs/2103.01955), primary abstract checked | Foundation for IPPO/MAPPO as competitive cooperative-learning baselines | Benchmarks are games, not evidence of traffic or pedestrian fairness |

Together these justify the research question and evaluation. They do not establish novelty for pedestrian-aware RL itself, predict a guaranteed win, or prove that our implementation reproduces their methods.

## Two priority hypotheses and one conditional extension

1. **Service decisions explain much of the apparent vehicle gain.** Compare plain IPPO/MAPPO with the prepared service-priority versions. Keep the reward, network, training budget, demand and physical safety rules matched. Include actuated and priority-pressure baselines. Measure time between pedestrian services, empty pedestrian phases, pending-request age, crossing counts and actual vehicle/pedestrian outcomes. Priority is an experimental service rule, not a proof that p95 must match actuated.
2. **The central critic or constraint signal is not learning useful predictions.** Log explained variance, value/return scale, reward and cost advantage magnitudes, update size and policy entropy. Examine the same development seeds across replicas. If a clear problem is observed, change one implementation choice and rerun both PPO methods with matched controls. A shared-reward ablation is a separate hypothesis; do not combine it with critic normalisation and claim to know which change helped.
3. **Conditional extension: the available phases restrict the trade-off.** Only if the core experiments are stable by Day 2, test a small set of compatible pedestrian/vehicle phases on the same geometry, after conflict-matrix and slow-walker clearance validation. Rerun conventional comparators under the same rules. Otherwise document the hypothesis and leave it as future work. A new road topology or a 16-junction network is not the priority in a one-week finish.

## Seven-day schedule

| Day | Deliverable and decision |
|---|---|
| 1 | Retrieve the complete cluster evaluation records. Reconcile all six core controllers, per-crossing counts, completion counts and scenario inventory. Run the already prepared 32-episode x three-replica pilot for four learned arms, with development-only seeds. |
| 2 | Read service traces and learning diagnostics. Choose one intervention using validation results and a fixed selection rule. Record why other hypotheses are deferred. Decide whether any phase extension is feasible; exclude it if safety validation is incomplete. |
| 3 | Run matched training for the selected intervention and its controls, retaining both IPPO and MAPPO. Prefer five training replicas. Freeze the common budget in advance; if extending, extend all compared arms together. Do not assume equal episode counts from different horizons mean equal experience. |
| 4 | Freeze methods/checkpoints and run fresh held-out evaluation. Primary: medium/medium. Prespecify a compact robustness set: high/medium, medium/low, slow walkers and short spacing. Keep all 21 original-study scenarios in the paper/appendix data. Do not choose final settings after observing gains. |
| 5 | Finish analysis: paired vehicle uncertainty, all 12 crossing differences, crossing-count outcome, worst-crossing result and service counts. Any incomplete training/evaluation stays labelled preliminary. Produce reproducible figures and matched-demand GUI clips. Stop introducing new methods. |
| 6 | Finish the strict ten-page report. Check claim-to-figure consistency, simulation assumptions, literature attribution, limitations, factual team contributions and the required LLM-use disclosure. |
| 7 | Rehearse the presentation and questions; inspect every number and source. Verify commands and submitted artifacts, and keep a fallback recording of the GUI demonstration. |

This plan prioritises interpretable evidence rather than promising to settle every hypothesis in seven days. Cluster queue times may require reducing the follow-up scope; preserve independent seeds and honest completeness reporting instead of claiming an incomplete experiment is final. The original completed study remains usable regardless of the extension's outcome.

Suggested role allocation follows the proposal, not a claim of completed contributions: Jason/Jarrell handle phase and simulator evidence; Shricharan/Jewi handle learning diagnostics and matched training; Reet/Huan Kiat handle crossing analysis and figures. Everyone checks the story, report and Q&A.

## Presentation story

Use the example deck's structure: visible failure, diagnosis, targeted change, evidence and remaining limitations. All examples, graphics, simulations and findings must be newly produced for CS4246; do not reuse CS3263 code, models, results or figures.

1. **The question.** Show a car queue and a waiting pedestrian. Can we improve vehicle flow without worsening long waits at any crossing?
2. **A fair test.** Show the three-junction/12-crossing map, two PPO methods x two rewards, and the fixed/actuated baselines. Explain the common safety wrapper.
3. **What counts as success.** Explain p95 with one crossing's event timeline. Show why completed vehicles and unresolved pedestrian requests must accompany means.
4. **The first result.** Present the four-controller medium-demand table. Vehicle-only gains cannot count as success when pedestrian service is missing. Pedestrian-aware variants miss the combined target.
5. **Who is affected?** Show all 12 crossing differences and the secondary count. Present magnitude and uncertainty, not just red/green pass labels.
6. **Our explanation.** Show a matched trace: serving pedestrians interrupts vehicles; delaying that service can improve cars while worsening a crossing. Label untested mechanisms as hypotheses.
7. **The targeted change.** Explain the service-priority experiment in one diagram. Compare with priority-pressure to show whether learning contributes beyond the service rule.
8. **What the new experiment established.** Use this slide only after results exist. If it helps, report where and at what cost. If it does not, report the tested mechanism and remaining uncertainty. Do not script a victory in advance.
9. **Scope and learning.** Distinguish average efficiency, crossing-level tails and guaranteed service. State the synthetic-demand and topology limitations. Finish with a concrete answer supported by the completed experiments.

Use the allotted presentation time from the course schedule; this outline does not assume a duration. Keep detailed ablations and the full 21-scenario table available for questions.

## Retrieve all latest results without rerunning SUMO

The latest status ZIP omitted the compressed evaluation groups and truncated comparisons.json. The following preserves the original complete files without copying model weights or SUMO installations.

Run on the cluster:

```bash
cd ~/cs4246-fairness-v2/trafficlightcontrol
python3 - <<'PY'
from pathlib import Path
import json, zipfile
root = Path('runs/fairness-v2-study')
protocol = json.loads((root / 'protocol.json').read_text())
groups = sorted((root / 'evaluation').glob('*.json.gz'))
expected = len(protocol['scenarios']) * len(protocol['test_seeds'])
assert len(groups) == expected, (len(groups), expected)
output = Path.home() / 'fairness-complete-results.zip'
assert not output.exists(), f'{output} already exists; use a new output name or reuse the existing archive'
with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
    for name in ('protocol.json', 'REPORT.md', 'primary_inference.json', 'comparisons.json'):
        archive.write(root / name, name)
    for path in groups:
        archive.write(path, 'evaluation/' + path.name, compress_type=zipfile.ZIP_STORED)
print(f'{len(groups)} groups; {output}; {output.stat().st_size / 2**20:.1f} MiB')
PY
```

Then run in Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:~/fairness-complete-results.zip "C:\Users\Shricharan\VSCodeProjects\trafficlightcontrol\runs\fairness-complete-results.zip"
```

The group count check is a completeness screen, not a substitute for validating fingerprints and matched demand after download. Keep the complete archive; do not delete the original cluster results.
