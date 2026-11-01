# V2 verification record — 31 October 2026

This records engineering verification, not a positive research result.

- **22 distinct tests passed:** the 20-test suite in
  `runs/fairness-v2-current-tests.log` (59.5 s) plus the two additional joint-inference
  tests in `runs/fairness-v2-inference-tests.log`. These include live SUMO baselines,
  repeated-seed equality, masked policy updates, slow walkers, an 80-person burst,
  short links, turns, vehicle accounting and inference rejection of worse crossings.
- **Eight full-length fixed-time load runs completed** at 960/1920/2880/8000
  vehicles/hour with two exploratory seeds each. Their measurement cohorts contain
  zero unknown waits and zero lost requests. The overload service deficits are
  37.8% and 36.2%, exceeding the planned 5% screen. See
  `runs/fairness-v2-current-audit/summary.json` and its load records.
- **Same-trajectory measurement audit:** 38 matched crossing visits; zero missing
  approaches, unmatched visits or non-overlapping conservative intervals. Maximum
  observed difference between 0.1 s and 0.2 s sampling was 0.1 s. This is evidence
  for this trace, not a universal error bound. Positions, entrance endpoints and
  event timestamps are retained in `runs/fairness-v2-current-audit/sampling.json`.
- **All six configurations completed the engineering pipeline:** two training
  episodes each, one replica, separate calibration/validation/test seeds, nine
  matched evaluation runs including the three conventional controllers, CSV/JSON
  reporting and figure generation. Results are in `runs/fairness-v2-current-check`.
  The short sample is insufficient for complete 12-crossing estimates; empty
  comparison plots are missing estimates, not zero delay or a successful target.
- **Resume verified:** all 12 best/latest checkpoint files and the compressed
  evaluation file retained identical hashes. No extra training updates were added.
  See `runs/fairness-v2-resume-check.json`. A previous staged check also verified
  resuming training from episode 1 to episode 2.
- **Compute evidence:** median short-rollout time was 2.81 s and neural update time
  0.16 s. Full audit episodes had median wall time 69.6 s while other checks were
  also running. This supports CPU parallelism; it is not a cluster runtime promise.
- **Storage:** development outputs occupied about 38 MiB at the check. The transfer
  archive contains only source/configuration/tests/docs, approximately 132 KiB.
  No GPU package or large dataset was downloaded locally.
- Both Bash scripts passed syntax checks. SUMO 1.27.1 Linux wheels were verified in
  PyPI metadata; authenticated cluster execution remains unverified.

The local full study was started on four CPU workers. Its process record is
`runs/fairness-v2-local-job.json`, logs are `runs/fairness-v2-local-study.stdout.log`
and `.stderr.log`, and output directory is `runs/fairness-v2-local-study`. A process
record documents launch, not completion. Final study results have not yet been
produced. The code automatically writes reports only after all evaluations finish.

Resume an interrupted local study with the unchanged source and command in the
process record. Do not start a second writer against the same output directory.
The CPU-cluster launcher is an alternative execution location; its output is kept
separate. Do not combine Windows/Linux runs into one statistical study merely to
increase sample size.

Earlier `first-milestone`, `quick`, `verified`, `final-check`, `release-check` and
interrupted audit folders are development probes. They include superseded entrance
geometry, implicit speed variation or different seed allocation. Do not pool them
with current results. One source-change guard deliberately rejected a report step
after code changed; that incomplete probe is excluded.

Remaining research limitations: synthetic demand, one corridor family, sparse-OD
rather than verified sparse-crossing intervention, no new LTA calibration, and no
guarantee that training will satisfy the every-crossing target. The existing course
report describes the older completed study until the new experiments are analysed.
