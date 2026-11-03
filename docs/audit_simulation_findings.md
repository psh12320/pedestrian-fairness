# Deep simulation audit, 3 November 2026

This report records the independent pre-repair inspection. For the subsequently
implemented fixes and final verification, see the [consolidated audit](deep_audit_2026_11_03.md).

This audit reviews the current `fairness` environment used by fairness-v2,
service-priority, PPO-repair, guided-exploration and local timing experiments.
Earlier legacy runs use a different tracker and phase controller and should not
be combined with these results. No production source, checkpoint, or completed
experiment record was changed by this audit.

## What was checked

- All 144 saved local timing runs, including every recorded crossing-entry time
  and every completed phase interval.
- Four fresh 600-second SUMO runs with explicit junction collision detection:
  fixed, actuated and pressure on the 65 m high-demand corridor, plus actuated at
  medium demand, seed 920101. All are development diagnostics.
- An independent repeat of the short-spacing fixed run without the extra collision
  instrumentation; its complete vehicle summary matched exactly.
- A separate 600-second medium actuated trace (seed 920102), with 1,800 junction
  observations/actions, sent to the independent RL implementation audit.

Reproduction: `python scripts/deep_audit_simulation.py --live`. The default command
audits existing records only. Evidence is in `runs/deep-audit-simulation/audit.json`;
the explicit installed-default configuration is `defaults.sumocfg` in that folder.
The extra medium/repeat cases are retained in that JSON; the command's standard
live mode runs the three short-spacing cases.

## Confirmed verification gap: junction collision detection was disabled

`fairness/environment.py:62` starts SUMO without `--collision.check-junctions`.
The installed SUMO 1.27.1 configuration template confirms its default is `false`.
The current runner checks signal conflict bits and the colliding-vehicle counter,
but that counter is not a complete junction-overlap check with this option off.
SUMO's official [Safety documentation](https://sumo.dlr.de/docs/Simulation/Safety.html)
explains this distinction. Enable explicit junction checks in a new version and
retain typed `getCollisions()` records, including intermodal events. Do not rewrite
the provenance of existing experiments to imply this check was already enabled.

The existing vehicle-collision counter does include a vehicle involved in a
detected intermodal collision in SUMO 1.27.1. This was checked in the versioned
[Simulation implementation](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/libsumo/Simulation.cpp)
and [intermodal collision handler](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_27_1/src/microsim/MSLane.cpp).
It is therefore sufficient to retain the existing per-tick subscribed counter
and fetch typed collision details when it is nonzero. The missing detection flag,
not the counter's name, is the confirmed gap.

This is a gap in validation, **not evidence that an unsafe collision occurred**.
All four fresh checked runs completed with zero collision events and zero internal
vehicles observed during pedestrian green at one-second occupancy sampling. The
sampling alone is not a proof that no vehicle was present between observations.
The fresh fixed run's vehicle summary was identical with and without the extra
collision instrumentation. One emergency-braking warning occurred in that case;
an emergency brake is not itself a collision or a detected phase conflict.

## Current pedestrian measurement and phase execution passed the inspected checks

Across the 144 records, all **135,627 crossing entries** occurred during their
junction's PED admission phase. There were no unknown approaches, lost events, or
completed phase-duration violations. Timestamps at a transition are interpreted
using the state before that change, because the environment samples movement
before it commands the next signal state.

The records show 12-second vehicle minimums, 3-second yellows, 2-second all-reds,
the requested 12/18/24-second pedestrian admission, and at least 11-second
pedestrian clearance. The existing live runner checks the actual signal state
against the command and the SUMO conflict matrix each 0.2 seconds. Clearance
waits for crossing occupancy to clear. These checks do not validate pedestrian
geometry against a real site or certify physical road safety.

`events.py` uses a full-width entrance region plus a connected stopped queue,
records actual edge entry, and keeps unknown waits rather than imputing zero.
The legacy tracker in `metrics.py` did impute zero for missing approaches; that
legacy defect is not present in the inspected current local timing records.

## Confirmed structural issue: the critic cannot observe a pending transition target

`SafeSignal.target` distinguishes an EW-to-NS transition from an EW-to-PED
transition. During yellow/all-red, the environment exposes the current phase and
age but omits this target (`environment.py:149`). Two otherwise equal observations
can therefore lead to different next services. During pedestrian clearance, the
elapsed empty-crossing timer is also omitted. Actors are masked during these
transitions, but the value function is trained on them, so this can increase value
prediction error. This is an observation-design defect, not proof that it explains
the entire empirical failure. Add explicit target and clearance-timer features in
an opt-in, schema-versioned experiment and retrain; old checkpoints cannot safely
consume a changed feature layout. The parent audit is implementing a discriminating
observation test.

## Observed mechanism: the conventional baselines prioritize different users

At medium demand, averaged over the four new seeds with 12-second admission:

| Controller | Vehicle-green share | Pedestrian admission share | Pedestrian clearance share |
|---|---:|---:|---:|
| Fixed-time | 60.42% | 14.50% | 13.00% |
| Pedestrian-actuated | 35.27% | 26.17% | 23.99% |
| Pressure-inspired | 41.85% | 22.96% | 21.05% |

The actuated rule (`fairness/signals.py:102`) serves a pending pedestrian request
as soon as PED becomes legal. It is therefore a very responsive pedestrian
benchmark, with substantial lost vehicle service through frequent pedestrian
stages and their mandatory clearance. Fixed-time gives substantially more time to
vehicles and begins all three signals in the same phase; it is a synchronized
baseline, not an independently randomized timing plan. A learned policy needs to
improve allocation enough to compensate for both effects. PPO being a standard
algorithm does not ensure that it dominates both baselines.

This supports testing vehicle-phase allocation while preserving pedestrian
priority, reducing unnecessary switching, and including transition memory. It
does not support shortening mandatory clearance or declaring a new policy better
before matched evaluation.

There are also pedestrian trips crossing two arms. In medium seed 910001, 254 of
739 pedestrians have two crossing visits and 485 have one. Under actuated control,
113 of the two-crossing trips enter their crossings in different PED periods;
under fixed-time, 67 do. An arrival near the end of admission may miss the second
crossing and create a new request later. Longer admission and more frequent
admission are different mechanisms; increasing duration alone can delay the next
service and vehicles. These descriptive counts are not a causal ablation.

## Congestion is real; completed-trip averages require their service counts

In the new short-spacing 600-second diagnostic, all three controllers had 677
scheduled vehicles:

| Controller | Completed | Unfinished | Uninserted |
|---|---:|---:|---:|
| Fixed-time | 670 | 7 | 0 |
| Actuated | 411 | 86 | 180 |
| Pressure-inspired | 586 | 33 | 58 |

No collision was needed to produce the large flow difference. Under short spacing,
queue lengths on the short corridor links are physically capped while longer
side-road queues can be larger. Raw queue comparison can favor side roads even
when substantial corridor demand remains outside insertion. The current actor
does not observe uninserted demand, although training traces record it for a
penalty. A backlog-aware diagnostic must state whether such a feature would be
measurable in practice. Do not claim this mechanism is established as the sole
cause without a matched counterfactual.

`metrics.py` reports completed-trip `timeLoss`, which does not include the separate
TripInfo `departDelay` field. Add insertion-delay and all-scheduled demand service
diagnostics alongside the required time loss. Official
[TripInfo documentation](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html)
defines these fields separately. The independent protocol audit covers the mixed
measurement/follow-up horizons and their implications.

## Limits

Four short live diagnostics and 144 saved conventional runs do not validate every
learned policy, extreme demand, or real site. No existing learning result was
discarded. The evidence rules out some simple suspected bugs (in the inspected
cases, pedestrians entered legally and phase durations were correct), identifies
a genuine safety-verification gap and missing value-state information, and gives
specific mechanisms to test. It does not guarantee that correcting these issues
will meet the every-crossing non-deterioration target.
