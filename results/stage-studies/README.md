# Stage studies — headline evidence

Small, tracked summaries of the local experiments (full raw traces live under
`runs/`, which is gitignored). Every number here reproduces via the scripts in
`scripts/` on a SUMO-equipped machine; see `docs/final_writeup.md` for the account.

| File | Study |
|---|---|
| `stage0-reproduction-REPORT.md` | Stage 0: local reproduction of the reviewed frontier |
| `stage1-vehicle-rule-REPORT.md` | Stage 1: `early+vehrule` — the frontier-moving controller |
| `stage15-critic-gate.json` | Stage 1.5: critic fits held-out returns (~0.49 EV) |
| `stage2-constrained-eval.json` | Stage 2: constrained MAPPO reverts to vehicle-max |
| `routeE-unified-REPORT.md` | Route E: per-crossing deadline — no-op (admission saturated) |
| `routeD-timing-search.json` | Route D: green-hold search — no-op (phase-cycle floor) |

Conclusion: the joint win is infeasible under the current safety machine; the
residual pedestrian p95 is an irreducible phase-cycle floor (~36 s). Dev seeds
only; reserved final seeds 58001-58020 untouched.
