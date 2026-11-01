"""Collect compact project diagnostics without model files or dependencies.

Run on the cluster login node: python3 scripts/collect_fairness_status.py
This only reads small project files and scheduler status; it does not run SUMO.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 24 * 1024 * 1024


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=25)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "error": str(error)}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"read_error": str(error)}


def progress(study):
    protocol = read_json(study / 'protocol.json')
    rows = []
    for method in ('ippo', 'mappo'):
        for objective in ('vehicle', 'pedestrian', 'constrained'):
            for replica in range(protocol.get('replicas', 5)):
                name = f'{method}-{objective}-r{replica}'
                folder = study / name
                last, malformed = {}, 0
                history = folder / 'training.jsonl'
                if history.exists():
                    with history.open() as stream:
                        for line in stream:
                            try:
                                last = json.loads(line)
                            except json.JSONDecodeError:
                                malformed += 1
                checkpoint = folder / 'latest.pt'
                rows.append({"model": name, "last_logged_episode": last.get('episode', 0),
                    "checkpoint_present": checkpoint.is_file(),
                    "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
                    "complete": read_json(folder / 'complete.json') if (folder / 'complete.json').exists() else None,
                    "selection": read_json(folder / 'selected.json') if (folder / 'selected.json').exists() else None,
                    "last_validation_score": last.get('validation_score'),
                    "malformed_history_lines": malformed})
    return {"protocol": protocol, "models": rows,
        "calibration_files": len(list(study.glob('calibration-*.json.gz'))),
        "validation_files": len(list(study.glob('validation-*.json.gz'))),
        "evaluation_groups": len(list((study / 'evaluation').glob('*.json.gz'))),
        "evaluation_groups_expected": len(protocol.get('scenarios', [])) * len(protocol.get('test_seeds', [])),
        "report_present": (study / 'REPORT.md').is_file(),
        "failure_files": [str(p.relative_to(study)) for p in study.rglob('failure*.json')],
        "note": "Logged episodes can be ahead of the last atomic checkpoint after interruption. A job disappearing from squeue is not evidence of success."}


def collect(study, output, job_ids):
    study, output = study.resolve(), output.resolve()
    for path in (study, output):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Use paths inside this project')
    state = progress(study)
    state['collected_utc'] = datetime.now(timezone.utc).isoformat()
    logs = ROOT / 'runs/cluster-logs'
    dispatch = logs / 'dispatch.tsv'
    ids = set(job_ids)
    if dispatch.exists():
        for line in dispatch.read_text().splitlines():
            columns = line.split('\t')
            if len(columns) == 6:
                ids.update(c for c in columns[4:6] if c.isdigit())
    state['accounting'] = command(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(ids)),
                                  '--format=JobIDRaw,JobName%80,State%30,ExitCode,Elapsed'])
    state['queue'] = command(['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%i|%j|%T|%R'])
    # Retain only this project's jobs if other work is running under the account.
    if 'stdout' in state['queue']:
        state['queue']['stdout'] = '\n'.join(line for line in state['queue']['stdout'].splitlines()
                                            if '|fairness' in line or '|pedestrian-fairness' in line)
    lines = [f"Collected: {state['collected_utc']}",
             f"Calibration: {state['calibration_files']}; validation baselines: {state['validation_files']}",
             f"Evaluation groups: {state['evaluation_groups']}/{state['evaluation_groups_expected']}",
             f"Report present: {state['report_present']}", '', 'Model | last logged episode | latest checkpoint | full-budget completion marker']
    for row in state['models']:
        lines.append(f"{row['model']} | {row['last_logged_episode']} | {row['checkpoint_present']} | {row['complete'] is not None}")
    lines += ['', 'Scheduler accounting:', state['accounting'].get('stdout', ''),
              state['accounting'].get('stderr', state['accounting'].get('error', '')),
              '', 'Active project jobs:', state['queue'].get('stdout', ''),
              state['queue'].get('stderr', state['queue'].get('error', '')), '', state['note']]
    summary = '\n'.join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.txt').write_text(summary)
    files = [study / n for n in ('protocol.json', 'validation_baselines.json', 'REPORT.md',
                                'primary_inference.json', 'comparisons.json')]
    for name in ('training.jsonl', 'selected.json', 'complete.json'):
        files.extend(study.glob('*/' + name))
    files.extend(sorted(study.rglob('failure*.json'))[-30:])
    recent_logs = sorted(logs.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)[:80]
    # Include original preparation/training logs even after subsequent jobs finish.
    for job in job_ids:
        recent_logs.extend(logs.glob(job + '_*.log'))
    files.extend(recent_logs)
    files.append(dispatch)
    included, skipped, used = [], [], 0
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys(files):
            if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
                continue
            limit = 32768 if path.suffix == '.log' else MAX_FILE
            size = path.stat().st_size
            count = min(size, limit)
            if used + count > MAX_TOTAL:
                skipped.append(str(path.relative_to(ROOT)))
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, size - limit))
                data = stream.read(limit)
            name = str(path.relative_to(ROOT)).replace('\\', '/')
            archive.writestr(name, data)
            included.append({"path": name, "tail_only": size > limit, "bytes": len(data)})
            used += len(data)
        archive.writestr('SUMMARY.txt', summary)
        archive.writestr('status.json', json.dumps(state, indent=2))
        archive.writestr('bundle_manifest.json', json.dumps({"included": included, "skipped": skipped,
            "excludes": ['model checkpoints', 'dependencies', 'raw simulation files', 'raw evaluation event groups']}, indent=2))
    print(summary)
    print(f'\nBundle: {output}\nSize: {output.stat().st_size / 1024:.1f} KiB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'runs/fairness-v2-study')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/fairness-cluster-status.zip')
    parser.add_argument('--jobs', default='1049572,1049573', help='Comma-separated job IDs; dispatcher history is also included')
    args = parser.parse_args()
    ids = args.jobs.split(',')
    if not ids or not all(i.isdigit() for i in ids):
        parser.error('jobs must be numeric IDs')
    collect(args.study, args.output, ids)


if __name__ == '__main__':
    main()
