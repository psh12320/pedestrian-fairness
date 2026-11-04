"""Local actuated advice on student-visited decision states.

This module does not control SUMO or change its common safety machine. Labels
are requested service actions, not the phase currently executed during a
yellow/clearance transition. All free decisions are retained for fidelity
analysis; a predeclared age threshold marks the subset eligible for targeted
guidance. The weight is a selection indicator, not a calibrated confidence.

No rollout is rejected for poor reward, unserved requests or a simulator jam.
Such outcomes belong in evaluation; they must not be hidden by data filtering.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from numbers import Integral

from src.traffic_control.connected import incoming_groups

PHASES = ('EW', 'NS', 'PED')
SCHEMA = 'local-actuated-urgency-labels-v1'


def crossing_tail_cost(observation, names, *, threshold_seconds=20., scale_seconds=20., cap=5.):
    """Sum bounded squared excess-oldest-wait costs over pending local crossings.

    Formula per pending crossing: min(max((age-threshold)/scale, 0), cap)^2.
    This is a dense proxy for long waits, not a direct p95 constraint or bound.
    Counts and ages use the existing observation scales (count/5, oldest/60).
    """
    if (not all(math.isfinite(v) for v in (threshold_seconds, scale_seconds, cap))
            or threshold_seconds < 0 or scale_seconds <= 0 or cap < 0):
        raise ValueError('Invalid tail-cost threshold, scale or cap')
    if len(names) != len(observation) or len(set(names)) != len(names):
        raise ValueError('Observation feature names must be unique and aligned')
    values = dict(zip(names, map(float, observation)))
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite tail-cost observation')
    crossings = sorted(n[:-6] for n in names if n.startswith(':') and n.endswith('/count'))
    if not crossings or len({c.split('_')[0] for c in crossings}) != 1:
        raise ValueError('Tail cost requires one junction\'s crossing observations')
    cost = 0.
    for crossing in crossings:
        if crossing + '/oldest' not in values:
            raise ValueError('Missing oldest waiting-age observation')
        count, age = values[crossing + '/count'], values[crossing + '/oldest'] * 60
        if count < 0 or age < 0:
            raise ValueError('Negative crossing count or waiting age')
        if count > 0:
            excess = min(max((age - threshold_seconds) / scale_seconds, 0.), cap)
            cost += excess * excess
    return cost


def _validated_state(junction, observation, names, mask):
    if len(names) != len(observation) or len(set(names)) != len(names):
        raise ValueError('Observation feature names must be unique and aligned')
    if len(mask) != len(PHASES) or any(x not in (False, True, 0, 1) for x in mask):
        raise ValueError('Expected three binary legal-action indicators')
    values = dict(zip(names, map(float, observation)))
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite teacher observation')
    groups = incoming_groups(3)
    if junction not in groups:
        raise ValueError('Unknown corridor junction')
    required = {'EW', 'NS'} | {e + '/queue' for edges in groups[junction].values() for e in edges}
    if not required <= values.keys():
        raise ValueError('Missing local actuated-controller features')
    for phase in ('EW', 'NS'):
        if values[phase] not in (0., 1.):
            raise ValueError('Phase flags must be binary')
    if values['EW'] + values['NS'] > 1:
        raise ValueError('Conflicting vehicle phase flags')
    # Critic-wide inputs are deliberately not accepted as actor teaching data.
    crossings = sorted(n[:-6] for n in names if n.startswith(':') and n.endswith('/count'))
    if not crossings or any(not c.startswith(':' + junction + '_c') for c in crossings):
        raise ValueError('Teacher needs this junction\'s crossing features only')
    for crossing in crossings:
        if crossing + '/oldest' not in values:
            raise ValueError('Missing oldest waiting-age observation')
        if values[crossing + '/count'] < 0 or values[crossing + '/oldest'] < 0:
            raise ValueError('Negative pedestrian count or waiting age')
    for phase, legal in zip(PHASES, mask):
        key = 'legal_' + phase
        if key in values and values[key] != float(legal):
            raise ValueError('Trace mask differs from observation legal mask')
    queues = []
    for axis in ('EW', 'NS'):
        components = [values[e + '/queue'] for e in groups[junction][axis]]
        if any(v < 0 for v in components):
            raise ValueError('Negative vehicle queue')
        # Match the baseline's integer counts rather than unequal floating sums.
        queues.append(sum(round(v * 10) for v in components))
    return values, crossings, queues, tuple(map(bool, mask))


def query_teacher(junction, observation, names, mask, *, urgency_seconds):
    """Return actuated advice and urgency metadata for one local observation.

    ``urgency_seconds`` is mandatory so that each experiment must freeze it.
    It is a training-selection threshold, not a promised p95 bound. A forced
    mask can be queried for audit, but is never selected for actor guidance.
    """
    if not math.isfinite(urgency_seconds) or urgency_seconds < 0:
        raise ValueError('Urgency threshold must be finite and nonnegative')
    values, crossings, queues, mask = _validated_state(junction, observation, names, mask)
    legal = [i for i, yes in enumerate(mask) if yes]
    if not legal:
        raise ValueError('No teacher action exists during a forced transition')
    pending = {c: round(values[c + '/count'] * 5) for c in crossings}
    ages = {c: values[c + '/oldest'] * 60 for c in crossings if pending[c] > 0}
    urgent = [c for c, age in ages.items() if age >= urgency_seconds]
    if sum(pending.values()) > 0 and mask[2]:
        desired, reason = 2, 'pending_pedestrian_service'
    elif values['EW'] or values['NS']:
        current = 0 if values['EW'] else 1
        desired = current if queues[current] > 0 else 1 - current
        reason = 'retain_queued_vehicle_phase' if desired == current else 'serve_other_vehicle_axis'
    else:
        desired = max((0, 1), key=lambda i: (queues[i], -i))
        reason = 'largest_vehicle_queue_after_clearance'
    if desired not in legal:
        desired, reason = legal[0], 'same_legal_fallback_as_actuated'
    selected = len(legal) > 1 and mask[2] and bool(urgent)
    return {
        'action': desired,
        'phase': PHASES[desired],
        'label_source': 'local-actuated',
        'reason': reason,
        'free_decision': len(legal) > 1,
        'guidance_selected': bool(selected),
        'guidance_weight': float(selected),
        'weight_is_confidence': False,
        'urgency_seconds': float(urgency_seconds),
        'oldest_pending_seconds': max(ages.values(), default=0.),
        'pending_count': sum(pending.values()),
        'urgent_crossings': urgent,
    }


def labelled_rollout(trace, features, *, horizon, urgency_seconds):
    """Generate JSON-safe demonstrations from a complete student rollout.

    Exclude the terminal action (it has no observed successor) and all forced
    ticks. Keep both urgent and nonurgent decisions, with selection metadata.
    The caller may preserve outcome records separately; no outcome score is an
    input to this function and no evaluation run is selected or discarded.
    """
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Positive finite rollout horizon required')
    if not math.isfinite(urgency_seconds) or urgency_seconds < 0:
        raise ValueError('Urgency threshold must be finite and nonnegative')
    data = {tls: {'observations': [], 'masks': [], 'labels': [], 'times': [],
                  'student_actions': [], 'guidance_weights': [], 'metadata': []}
            for tls in features}
    counts = Counter()
    last_times = {}
    for row in trace:
        tls, now = row['junction'], float(row['time'])
        if tls not in data or not math.isfinite(now) or now < 0 or now > horizon:
            raise ValueError('Trace row lies outside its declared junction/horizon')
        if now <= last_times.get(tls, -math.inf):
            raise ValueError('Trace decision times must increase within each junction')
        last_times[tls] = now
        if now == horizon:
            counts['terminal_rows_excluded'] += 1
            continue
        mask = row['mask']
        if len(mask) != 3 or any(x not in (False, True, 0, 1) for x in mask):
            raise ValueError('Invalid trace legal-action mask')
        if sum(mask) < 2:
            counts['forced_rows_excluded'] += 1
            continue
        action = row['action']
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in range(3) or not mask[action]:
            raise ValueError('Free student decision must have a legal requested action')
        advice = query_teacher(tls, row['observation'], features[tls], mask,
                               urgency_seconds=urgency_seconds)
        local = data[tls]
        local['observations'].append(list(map(float, row['observation'])))
        local['masks'].append(list(map(bool, mask)))
        local['labels'].append(advice['action'])
        local['times'].append(now)
        local['student_actions'].append(int(action))
        local['guidance_weights'].append(advice['guidance_weight'])
        local['metadata'].append(advice)
        counts['free_decisions'] += 1
        counts['selected_guidance_decisions'] += int(advice['guidance_selected'])
        counts['student_teacher_disagreements'] += int(action != advice['action'])
    result = {'schema': SCHEMA, 'horizon_seconds': float(horizon),
              'urgency_seconds': float(urgency_seconds),
              'features': {tls: list(names) for tls, names in features.items()},
              'data': data, 'counts': dict(counts),
              'outcome_filtering': 'none; all free decisions in the supplied rollout are retained',
              'target': 'requested legal service action, not currently executed phase'}
    result['data_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result
