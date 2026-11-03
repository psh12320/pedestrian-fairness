"""Repeat a saved SUMO policy and verify completed-study resumption is idempotent."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.traffic_control.audit_trial.pilot import episode, digest, verify_source
from src.traffic_control.audit_trial.learning import TrialNetworks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/audit-repair-smoke')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if json.loads((root / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Verify only a completed engineering study')
    verify_source(protocol)
    torch.set_num_threads(1)
    label = 'mappo-memory-separate-r0'
    seed = protocol['evaluation_seeds'][0]
    scenario = protocol['evaluation_scenarios'][0]
    saved = torch.load(root / 'models' / label / 'final.pt', map_location='cpu', weights_only=True)
    model = TrialNetworks.from_snapshot(saved['model'])
    before = copy.deepcopy(model.state_dict())
    reference = json.loads((root / 'evaluation' / f'{label}-{scenario}-{seed}.json').read_text())
    replay = episode(root, protocol, scenario, seed, 'policy', model)
    for field in ('result', 'diagnostics', 'phase_history', 'vehicles_end', 'network_sha256', 'features'):
        # JSON stores phase-history tuples as lists; compare serialized values.
        if json.loads(json.dumps(replay[field])) != reference[field]:
            raise AssertionError('Replay changed ' + field)
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items()):
        raise AssertionError('Evaluation changed model parameters or normalization')
    for junction, diagnostics in replay['diagnostics'].items():
        rows = [r for r in replay['rollout']['trace'] if r['junction'] == junction]
        actual = sum(r['action'] is not None and sum(r['mask']) > 1 for r in rows[:-1])
        if diagnostics['free_decisions'] != actual:
            raise AssertionError('Free choices count a terminal action with no successor')
    files = [p for p in root.rglob('*') if p.is_file() and p.name not in ('status.json', 'writer.lock')]
    hashes = {p: digest(p) for p in files}
    command = [sys.executable, '-m', 'src.traffic_control.audit_trial.pilot', '--output', str(root), '--workers', '2']
    if protocol['quick']:
        command.append('--quick')
    with (ROOT / 'runs/audit-trial-resume-verification.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(digest(p) != sha for p, sha in hashes.items()):
        raise AssertionError('Resuming a completed study changed a saved scientific artifact')
    result = dict(replay_exact=True, evaluation_read_only=True, free_choice_count_correct=True,
                  completed_resume_unchanged_files=len(hashes), fingerprint=protocol['fingerprint'],
                  scope='Engineering smoke verification, not a performance result')
    (ROOT / 'runs/audit-trial-engineering-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
