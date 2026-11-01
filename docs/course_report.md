# Course report draft: ten-page allocation

**Working title:** When Does Multi-Agent PPO Reduce Vehicle Delay Without Increasing Long Pedestrian Waits? A Controlled Three-Junction SUMO Study

**Status:** Archived writing scaffold. The completed, checked ten-page paper is [the final PDF](../output/pdf/cs4246-team38-final-report.pdf), generated from 4,290 main-study and 330 separate uncalibrated transfer runs. The material below is retained as a planning reference. Do not submit the bracketed placeholders. Populate results only from the validation-selected, complete `models/v3/training_manifest.json` (150 episodes) or `models/v4/training_manifest.json` (300 episodes) and the locked evaluation in `results/final-study/`. The earlier `results/REPORT.md` uses a different crossing-wait definition and five inspected seeds; its estimates are exploratory. Version-2 models also belong to a superseded phase-safety protocol and must not be mixed with the revised final study. The supplied **draft** guideline says AY2026/27 and was issued 26 August 2026, but its report line reads **“2359 Sunday, 15 Nov 2025.”** That year conflicts with the academic year and issue date. Confirm the actual deadline in Canvas or with course staff; this scaffold does not infer a corrected due date.

The public [NUS Computing course schedule](https://www.comp.nus.edu.sg/cug/soc-sched/), checked on 30 September 2026, lists CS4246 teaching arrangements but does not resolve the project submission deadline. The Canvas/course-staff confirmation remains outstanding.

| Page | Content and evidence to fit | Suggested figure/table |
| --- | --- | --- |
| 1 | Title, abstract, motivation, precise question, contributions and outcome, including a failed target if applicable | One-sentence result with uncertainty only after full evaluation |
| 2 | Related work and gap: pedestrian-aware control exists; explain why crossing-specific p95 and unserved service matter | Compact literature comparison |
| 3 | New corridor, seeded demand, local observations, PPO methods and reward weights | Corridor and controller diagram |
| 4 | Common phase safety, audited crossing events, vehicle accounting, 30 training replicas, separate validation, ten locked seeds, 13 scenarios and paired criterion | Compact design and measurement description |
| 5 | Primary vehicle outcome, all service counts, worst-crossing differences and criterion results for every comparator | Outcome tables with paired intervals |
| 6 | All 12 crossing p95 differences for the prespecified IPPO weight-3 illustration, entry counts, pending requests and worst-crossing test; complete comparator tables stay in the supplement | One 12-crossing plot |
| 7 | Vehicle-versus-worst-crossing trade-off for all nine controllers and the weight-0/3/10 comparison | Trade-off plot with uncertainty |
| 8 | Independent training variation, validation-only checkpoint selection, frozen budget rule and policy-class limitations | Validation learning curves |
| 9 | Singapore geometry audit and, only if completed, the separate ten-seed simplified-corridor sensitivity test; limitations, responsible interpretation, factual team contribution and LLM disclosure | Small source-attributed map or compact transfer table, only if space permits |
| 10 | References and, if permitted, a short reproducibility pointer | References |

The report itself must be **at most ten pages** after typesetting, as stated in the supplied guideline; Appendix A says the limit is strictly enforced. This page allocation is a layout target, not a claim that Markdown enforces length. Keep detailed per-seed tables and raw traces in the reproducibility package; the report should carry the central evidence. Give every plotted difference its direction and uncertainty, and show the entry count behind each p95.

The supplied guideline specifies single spacing, one column, at least 12-point type, one-inch margins, Vancouver references, and PDF submission through Canvas. It excludes the title page and references from the ten-page count; verify whether any appendix is also excluded before relying on one.

## Report preparation

The experimental report and PDF generators have been removed. Prepare the report from the saved evaluation outputs and verify the selected training budget, replica counts, held-out seeds and crossing coverage before making final claims. Existing generated artifacts are historical snapshots. Fill individual contribution statements with verifiable completed work.

## Abstract template (complete only after locked evaluation)

Traffic signals trade vehicle delay against pedestrian service, and network averages can conceal long waits at one crossing. We built an original three-junction SUMO corridor with protected crossings and compared fixed-time, pedestrian-actuated and pressure-inspired controllers with local-actor IPPO and MAPPO under a shared safe phase machine. Each PPO method used pedestrian reward weights 0, 3 and 10; independent training seeds, separate validation seeds, and locked matched-demand test seeds supported the comparison. We measured vehicle time loss together with completed and unserved trips, and recorded crossing-arrival and crossing-entry events for each person. **[Insert paired vehicle result, worst-crossing p95 result, service counts, interval, and whether the every-crossing criterion passed. If it failed or was inconclusive, say so here.]** These are simulation findings on constructed geometry and synthetic demand, not evidence about deployed Singapore signals.

## Problem, motivation, and contribution

A controller can appear to improve average travel time while withholding a crossing phase or leaving vehicles uninserted. The question is therefore not simply whether PPO is faster, but **when, if ever, it reduces completed-vehicle time loss without increasing the p95 wait at any of twelve protected crossings relative to pedestrian-actuated control**. The paper contributes a common safe action interface, explicit event-based crossing waits and end-of-run service accounting, paired repeated-training comparisons, and analysis of the worst affected crossing. Pedestrian-aware reinforcement learning and MAPPO traffic control are established ideas; do not claim either as a new algorithm. See [the source-checked literature review](literature.md) for IVPL, pedestrian queue max-pressure, Helsinki MAPPO, real-corridor PPO and evaluation-protocol references.

## Method and measurement text to retain

The main network is an original, straight three-junction SUMO corridor with sidewalks and four protected crossings per junction. The default demand and signal layout are synthetic. Seeded route generation and the same SUMO seed give every controller identical scheduled traffic within a scenario. Turning traffic appears in the stress cases where the generated movements pass the network conflict test. Each actor observes only its junction's queues, pending crossing requests, current phase and phase age. Both PPO methods train from the same per-junction reward at each pedestrian weight; IPPO's value estimator uses local observations, while MAPPO's per-junction value estimator additionally receives all junction observations during training. The three conventional controllers and learned actors request phase changes through one state machine. The queue-pressure controller is inspired by pressure principles and should not be described as an exact reproduction of [published pedestrian queue max-pressure](https://www.sciencedirect.com/science/article/pii/S0968090X24003863).

Vehicle phases have a 12-second minimum green, three-second yellow and two-second all-red transition. The pedestrian group has a 12-second green followed by all-red clearance based on the longest crossing and 0.8 m/s design walking speed plus three seconds. Clearance extends if someone remains on a crossing and requires three consecutive empty crossing observations before release. Every generated phase is checked against SUMO's junction `foes` matrix; sampled applied states are checked in the simulation. These are experiment-specific legal movement and clearance rules, not a certification against an actual Singapore timing plan.

For each person–crossing pair, arrival is the first one-second observation inside a 1.2 m zone at the crossing-lane endpoint, or the start of the first stopped approach interval for a queue farther back. Entry is the first observation on the crossing edge; their difference is the recorded wait. Direct entries without an approach observation are flagged and imputed zero. Reached crossings without entry by the 600-second horizon are counted as unserved. The report must expose these edge cases, the one-second sampling resolution, and any source breakdown relevant to the primary comparison. TraCI `waitingTime` helps detect a stopped approach but is not the final crossing-wait measure. Vehicle TripInfo gives completed-trip time loss; scheduled, departed, arrived, unfinished and uninserted trip IDs reconcile service.

## Experimental protocol and decision rule

The primary factors are IPPO or MAPPO and reward weight 0, 3 or 10. Five independent training seeds per configuration yield 30 replicas; each starts with 150 episodes and uses only separate validation seeds for checkpoint selection. A decision to extend learning must apply the same 300-episode budget to all configurations and be made from validation curves. Ten new seeds are locked for final evaluation. The 3×3 vehicle/pedestrian demand grid plus slow-walker, pedestrian-burst, overload and short-spacing scenarios gives 13 conditions. The primary test is medium vehicle and medium pedestrian demand. Report paired controller-minus-actuated differences and confidence intervals that include training and demand variation; retain the complete crossing table and per-crossing entry counts. Target at least 100 pooled entries at every crossing per training replica in the primary setting, and label shorter tails inconclusive.

A controller **descriptively meets** the proposal target only when its completed-vehicle time loss is lower, every crossing p95 point estimate is no higher than pedestrian-actuated control, completed, unfinished, uninserted and total unserved service do not deteriorate, and no individual crossing has more pending requests. A statistically stronger statement also needs paired intervals supporting the vehicle gain and the maximum crossing-level difference, plus adequate entries; the service constraints remain descriptive. A controller that fails any crossing must be described as failing the *every-crossing* criterion even if its network-average pedestrian wait improves. The largest crossing difference is the main tail outcome; do not select the best seed.

## Results slots and analysis questions

1. **Service and flow.** Fill the primary table from `results/final-study/summary.json` with paired vehicle time loss, scheduled/completed/unfinished/uninserted counts, pedestrian unserved requests, and intervals. Show the fixed-time overload fraction across locked seeds; the one-seed development pilot is not an estimate of method performance.
2. **Who waits?** Plot all twelve primary crossing p95 differences from `crossing_comparison.csv`, preserving entry and unserved counts. Give the largest difference and whether it is supported by its paired interval. Explain any crossing with sparse entries or direct-entry imputation.
3. **Why?** Compare reward weights 0, 3 and 10 within each PPO method, inspect `action_decisions.csv` and `phase_history.csv`, and separate a decision difference from an outcome difference. If the central critic does not change observed decisions under coupling, report the null result.
4. **Stress and transfer.** Show slow-walker, concentrated burst, overload and short-spacing outcomes without turning them into post hoc primary tests. In the Singapore pilot, the uncorrected OSM import has zero signal-controlled crossing edges; the safe NUS-derived corridor is a simplified geometry with synthetic demand. If the separate 9101–9110 transfer run with all 30 selected-budget replicas completes, report its paired uncertainty and sparse-crossing limits as a **geometry sensitivity test**, not an observed-data Singapore validation. Only add an observed-data Singapore comparison after geometry, protected phases, turning counts and independent calibration are verified.

## Limitations, contributions and AI disclosure

The corridor is synthetic; a linear PPO policy and finite training budget limit generality. Mean completed-trip time loss is subject to survivor bias if vehicles remain unfinished. One-second event sampling and the operational 1.2 m entrance rule can shift individual wait estimates, so per-crossing counts and outstanding requests are necessary. The NUS-area OSM/signal snapshots do not provide actual phase plans, and the project has no observed pedestrian arrivals or junction-turn calibration yet. A simulation gain must not be called an improvement to deployed traffic lights. A paper beyond the course would need multiple independently calibrated sites, observed pedestrian demand and legal phase plans, and a faithful pedestrian-aware max-pressure comparator.

The team's stated lead assignments are below. They establish responsibility, **not evidence that a task was completed by that individual**. Replace every placeholder with factual completed work, evidence, and writing/presentation input before submission.

| Member | Stated lead area | Actual completed contribution and evidence |
| --- | --- | --- |
| Jason | Simulation and baselines | [Team to fill: specific work, verification, report/presentation contribution] |
| Jarrell | Simulation and baselines | [Team to fill: specific work, verification, report/presentation contribution] |
| Shricharan | MARL implementation and training | [Team to fill: specific work, verification, report/presentation contribution] |
| Jewi | MARL implementation and training | [Team to fill: specific work, verification, report/presentation contribution] |
| Reet | Pedestrian evaluation and analysis | [Team to fill: specific work, verification, report/presentation contribution] |
| Huan Kiat | Pedestrian evaluation and analysis | [Team to fill: specific work, verification, report/presentation contribution] |

All six members are assigned to contribute to the report and presentation. Do not infer individual effort from code authorship or these planned roles.

Suggested disclosure to verify and adapt: “OpenAI Codex assisted with drafting portions of the source code, tests, experiment documentation and report text. The team [describe its actual review, source checking, execution and corrections]. The team is responsible for the study design, reported results and final claims.” The bracketed review clause must be replaced with what the team really did; check the course's current disclosure requirements before submission.
