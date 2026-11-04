# Documentation and sharing verification — 5 November 2026

This update documents existing results; it does not modify frozen experiment
source or rerun SUMO training/evaluation.

- The public catalogue covers sixteen study/review entries. The manifest checks
  164 public artifact content hashes and 24 lossless metadata exports. All
  exported JSON parses, and the catalogue's local links resolve.
- The same integrity check passed in a temporary clean checkout made solely
  from staged Git files, with no ignored `runs/` artifacts. Git's staged
  whitespace check passed; frozen CRLF records and generated SVG formatting are
  deliberately preserved. The shared update adds about 9 MiB of files.
- The three latest completion snapshots show 720/720 demand evaluations,
  1,396/1,396 repair-trial simulations and 876/876 cluster fine-tuning simulations.
  The separate local fine-tuning duplicate remains explicitly stopped at 42/876.
- Twenty-eight newly shared Python files parsed successfully. A scan of newly
  shared text found no private-key or common access-token patterns; this is a
  bounded check, not a guarantee of every possible sensitive-data format.
- **47 targeted regression tests passed in 27.641 seconds**, covering early-service
  masks, complete scheduling, replay comparison, local teacher labels, guidance,
  optimizer/RNG resume, provenance and preservation of service failures in reports.
  Orchestration fixtures use synthetic traces; these are not new traffic experiments.

Command used (from repository root, one CPU thread per numerical library):

```bash
python -m unittest tests.test_demand_service_test tests.test_guided_failure_replay tests.test_pedestrian_finetune_teacher tests.test_pedestrian_finetune_learning tests.test_pedestrian_finetune_reporting tests.test_pedestrian_finetune_pilot tests.test_pedestrian_finetune_cluster -v
python scripts/experiment_inventory.py --check
```

The existing laptop runtime emitted a PyTorch 2.3.0 / NumPy 2.2.3 bridge
compatibility warning; the selected tests passed without using that bridge.
No packages were changed during this documentation task. Use the study's pinned
environment for reproduction rather than assuming this warning is harmless for
other NumPy-dependent code.

The earlier integration and saved-trace audits remain attributed to their own
study reviews. Passing these new focused tests does not independently revalidate
every historical safety assertion or scientific confidence interval.
