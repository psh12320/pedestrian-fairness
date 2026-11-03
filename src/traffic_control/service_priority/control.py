"""Local pedestrian-priority masks and pressure-inspired vehicle allocation.

These are proposed adaptations, not a reproduction of published max-pressure.
The wrapper can only remove actions from the existing physical safety mask.
"""
from ..connected import incoming_groups


def service_mask(observation, names, legal, deadline=0.):
    if deadline < 0:
        raise ValueError('Deadline must be nonnegative')
    result = list(legal)
    counts = [observation[i] * 5 for i, n in enumerate(names) if n.startswith(':') and n.endswith('/count')]
    ages = [observation[i] * 60 for i, n in enumerate(names) if n.startswith(':') and n.endswith('/oldest')]
    pending = sum(counts)
    if legal[2] and pending and max(ages, default=0) >= deadline:
        return (False, False, True)
    if not pending:
        result[2] = False
    # Never invent a legal phase during clearance/minimum green.
    return tuple(result)


def pressure_action(tls, observation, names, mask, age_weight=1.):
    values = dict(zip(names, observation))
    incoming = incoming_groups(3)[tls]
    scores = [sum(values[e + '/queue'] * 10 for e in incoming[axis])
              - values['downstream_' + axis] * 10 for axis in ('EW', 'NS')]
    counts = sum(v * 5 for n, v in values.items() if n.startswith(':') and n.endswith('/count'))
    oldest = max((v * 60 for n, v in values.items() if n.startswith(':') and n.endswith('/oldest')), default=0)
    scores.append(2 * counts + age_weight * oldest)
    available = [i for i, allowed in enumerate(mask) if allowed]
    if not available:
        raise ValueError('No legal decision')
    # Hold current vehicle green on exact score ties to avoid gratuitous changes.
    current = 0 if values['EW'] else 1 if values['NS'] else -1
    return max(available, key=lambda i: (scores[i], i == current, -i))


class ProtectedPolicy:
    def __init__(self, features, model=None, deterministic=True, deadline=0.):
        self.features, self.model = features, model
        self.deterministic, self.deadline = deterministic, deadline
        self.calls = self.restricted = self.forced_pedestrian = 0

    def __call__(self, tls, observation, global_state, legal):
        mask = service_mask(observation, self.features[tls], legal, self.deadline)
        self.calls += 1
        self.restricted += tuple(legal) != mask
        self.forced_pedestrian += mask == (False, False, True) and sum(legal) > 1
        if self.model is None:
            return pressure_action(tls, observation, self.features[tls], mask)
        return self.model.act(tls, observation, mask, self.deterministic)[0]

    def align_training_trace(self, trace):
        # PPO log probabilities must use the distribution that sampled the action.
        # The forced service rows are critic-only, not fictitious actor decisions.
        for row in trace:
            mask = service_mask(row['observation'], self.features[row['junction']], row['mask'], self.deadline)
            if row['action'] is not None and not mask[row['action']]:
                raise AssertionError('Executed action is absent from effective mask')
            row['safety_mask'] = row['mask']
            row['mask'] = mask


def service_components(result, baseline):
    """Raw signed differences; deliberately do not sum/double-count people."""
    v, b = result['vehicles'], baseline['vehicles']
    out = {'vehicle_' + k + '_delta': v[k] - b[k] for k in
           ('completed_count', 'unfinished_count', 'uninserted_count')}
    for k in ('completed', 'unfinished', 'uninserted'):
        out['person_' + k + '_delta'] = result.get('persons', {}).get(k, 0) - baseline.get('persons', {}).get(k, 0)
    for k in ('pending_at_cutoff', 'unresolved_after_followup'):
        out[k + '_delta'] = sum(c[k] for c in result['crossings'].values()) - sum(c[k] for c in baseline['crossings'].values())
    for k in ('unknown_cohort_entries', 'lost'):
        out[k] = sum(c[k] for c in result['crossings'].values())
        out['baseline_' + k] = sum(c[k] for c in baseline['crossings'].values())
    out['zero_entry_crossings'] = [c for c, r in result['crossings'].items() if r['entries'] == 0]
    return out
