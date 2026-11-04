"""Resume the frozen study with bounded retries for Windows file replacement.

Windows can deny an atomic rename while the progress reader or a scanner has
the destination open. Retry the identical operation; never delete the target,
change data, suppress a persistent failure, or alter the learner/simulator.
This wrapper is imported by spawned workers as well as the parent.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_original_replace = Path.replace


def replace_with_retry(path, target):
    for attempt in range(51):
        try:
            return _original_replace(path, target)
        except PermissionError:
            if os.name != 'nt' or attempt == 50:
                raise
            if attempt == 0:
                print(f'Retrying temporarily denied atomic replacement: {path.name}', flush=True)
            time.sleep(.1)


if os.name == 'nt':
    Path.replace = replace_with_retry


if __name__ == '__main__':
    from src.traffic_control.audit_trial.pilot import main
    # Sidecar records the operational difference outside the frozen runtime.
    args = sys.argv[1:]
    folder = Path(args[args.index('--output') + 1]) if '--output' in args else ROOT / 'runs/audit-repair-trial'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'windows-io-retry.json').write_text(json.dumps({
        'wrapper': 'scripts/run_audit_trial_windows.py',
        'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'change': 'Retry identical Path.replace after PermissionError, at most 5 seconds',
        'learner_simulator_config_seeds_unchanged': True,
        'does_not_delete_destination': True,
    }, indent=2) + '\n')
    main()
