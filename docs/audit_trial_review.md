# Independent review of the local repair trial

This review concerns the new CPU experiment testing the audit findings in SUMO. It does not treat a successful software test as a traffic improvement. Final observations below will distinguish completed verification from intended checks.

## Frozen comparison reviewed before performance results

The proposed design is IPPO and MAPPO, crossed with controller-memory observations on/off and separate actor/critic clipping on/off. Two initialization seeds give 16 models. Every arm uses the normalized-clock critic, pedestrian reward weight 3, 64 training episodes of 600 seconds, identical physical phases, and matched scheduled demand. Thus `legacy` means the current repaired learner with legacy observations, not the earlier linear controller or original unnormalized critic.

The design includes 1,024 training simulations, 48 medium-demand validation simulations at episodes 16/32/64, 256 evaluation simulations (four scenarios and four held-out demand seeds for each model), and 68 conventional/uniform baseline simulations, totaling 1,396. Validation is a learning diagnostic, not checkpoint selection. Initializations 94001/94002, training demands 930001 + 1000 * replica + episode_index, validation 950001 and evaluation 960001–960004 are distinct; reserved final seeds 58001–58020 remain unused.

The common collision checks and new insertion-delay reporting apply to all arms. They improve verification/reporting, but must not be attributed to a learned-policy performance effect. All runs retain vehicle service counts, all 12 crossing tails and unresolved requests. The original every-crossing target remains unchanged.

## Key implementation checks

- The extra committed-phase and clearance timer features are nonzero only during forced transitions. At free actor decisions, their values are zero. The observation factor therefore primarily tests critic information for assigning credit through transitions. Initial actor and critic functions must match across the 42/47-input schemas; random initialization with the same integer seed is insufficient when layer dimensions change.
- Global critic input columns must be mapped by junction and feature, with the finite-horizon clock kept in its proper position. Added columns start at zero. All existing hidden/output weights must be copied from the same canonical initialization.
- A singleton action mask needs no neural forward pass: its legal action has probability 1 and log probability 0. This saves CPU while preserving behavior. Nontrivial action samples must use independently seeded per-junction streams, matched across arms.
- Every rollout retains forced transition costs. Only free choices contribute actor loss; all states can train the critic. Episode boundaries stop GAE according to the declared finite-horizon objective. No trial arm silently changes reward, demand, rollout horizon or episode-reset behavior.
- Completed source/protocol fingerprints, exact checkpoint hashes and paired demand/network hashes must be checked on reuse. Atomic checkpoints retain model, optimizer, normalization state and episode history. Interrupted validation/evaluation needs safe recovery rather than duplicate training.
- Two training replicas and four evaluation seeds support a bounded development study. Paired intervals are exploratory; report individual replica outcomes, missing/sparse crossing estimates and any failures. No arm is promoted because it wins on inspected test results.

## Resources and limits

Two CPU workers with one Torch thread each are appropriate for the laptop's limited spare RAM. Based on preceding run times, the full experiment is likely to take hours, not minutes; actual benchmark timings should determine the estimate. A disk floor, compact saved records, explicit status, progress logs, output lock and resumability are needed. Do not edit fingerprinted source while the experiment runs.

This experiment isolates two implementation changes. It does not test the other audit hypotheses—critic backlog inputs, continuing-task credit, greater training exposure, shared rewards or stronger junction coupling. Lack of improvement here would not validate those alternatives; improvement here would not establish deployment benefit.

## Verification status

The reviewer inspected the new `learning.py`, `pilot.py`, and `reporting.py`, and completed six independent tests in `tests/test_audit_trial_integration.py`:

1. Five short SUMO simulations (one conventional reference plus initial legacy/memory IPPO and MAPPO) established identical initial sampled actions, phase histories, vehicle/person/crossing outcomes and demand/network hashes across observation schemas. Action probabilities agreed within numerical tolerance. Model parameters and normalization buffers remained unchanged by evaluation. The five added memory features were zero whenever an actor was called and nonzero during forced transitions.
2. Three further short SUMO simulations (a separately initialized reference plus two actual runner evaluations) verified exact repeatability of demand, metrics, diagnostics, events and rollout traces. Recorded log probabilities passed the learner's on-policy check. Reported free-choice counts equaled the number of samples actually used by PPO.
3. The protocol produced the independently calculated 1,396 simulations and disjoint training/validation/evaluation/reserved demand seed sets. All arms retained reward weight 3 and fixed final-checkpoint selection.
4. A second writer was rejected by the study lock, and a changed/missing frozen source file was rejected.
5. A completed atomic checkpoint regenerated missing final exports without rerunning a training episode.
6. A small on-policy fixture run as two episodes followed by a resume for two more produced bitwise-identical model weights, optimizer moments, collected rows and update statistics to uninterrupted four-episode training. Wall-clock durations were appropriately excluded from equality.

The reader should not add these verification simulations to the research budget: they use distinct engineering seeds and short horizons, and do not support a performance claim. When both live tests run together, their common reference is generated only once (seven simulations); the recorded separate invocations generated it twice (eight simulations).

Two review findings were addressed before the long run: terminal observations no longer inflate the diagnostic count of trained actor choices, and the report now checks prescribed validation coverage and matched validation demand/network hashes. Raw saved pedestrian event files are also checked against their recorded hashes before final reporting.

No launch-blocking implementation defect remained in this review. This is approval of the bounded experiment's engineering/design, not a conclusion that corrections improve SUMO outcomes. The parent agent separately runs the complete quick pipeline, completed-run resume, and saved-model replay before launching the full study. Runtime source files were not edited by this reviewer.
