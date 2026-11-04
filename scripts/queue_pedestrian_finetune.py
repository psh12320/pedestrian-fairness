"""Wait for an existing CPU experiment, then run the fixed fine-tuning study.

This lightweight launcher imports no Torch. On Windows it retains a process
handle so PID reuse cannot mistake a later process for the dependency.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--verified-smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pedestrian-finetune')
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This wait helper is Windows-specific; run the pilot directly elsewhere')
    root = args.output.resolve()
    root.mkdir(exist_ok=True, parents=True)
    smoke = json.loads((args.verified_smoke / 'status.json').read_text())
    p = json.loads((args.verified_smoke / 'protocol.json').read_text())
    if not p['quick'] or smoke['state'] != 'complete' or smoke['completed_simulations'] != 19:
        raise ValueError('A completed 19-simulation smoke test is required')
    # Holding this handle is read-only and prevents confusion from PID reuse.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, args.after_pid)
    if not handle and ctypes.get_last_error() != 87:
        raise ctypes.WinError(ctypes.get_last_error())
    status = dict(launcher_pid=os.getpid(), dependency_pid=args.after_pid,
                  verified_smoke_fingerprint=p['fingerprint'], planned_simulations=876,
                  output=str(root), workers=1)

    def update(state, **extra):
        status.update(state=state, updated=datetime.now(timezone.utc).isoformat(), **extra)
        temp = root / 'queue.json.tmp'
        temp.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        for attempt in range(51):
            try:
                temp.replace(root / 'queue.json')
                break
            except PermissionError:
                if attempt == 50:
                    raise
                time.sleep(.1)

    try:
        update('waiting_for_existing_experiment')
        while handle:
            wait_result = kernel.WaitForSingleObject(handle, 0)
            if wait_result == 0:
                break
            if wait_result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            if (root / 'STOP').exists():
                update('stopped_before_launch')
                return
            time.sleep(20)
            update('waiting_for_existing_experiment')
        for name, expected in p['source'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Source changed since verified smoke: ' + name)
        archive = ROOT / 'runs/guided-exploration-results.zip'
        if sha(archive) != p['starting_models']['archive_sha256']:
            raise ValueError('Parent archive changed since verified smoke')
        if (root / 'STOP').exists():
            update('stopped_before_launch')
            return
        command = [sys.executable, '-u', '-m', 'scripts.pedestrian_finetune.pilot',
                   '--workers', '1', '--output', str(root)]
        update('launching')
        with (root / 'runner.log').open('a', encoding='utf-8') as log, (root / 'errors.log').open('a', encoding='utf-8') as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            update('running', experiment_pid=process.pid)
            code = process.wait()
        final = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        update('complete' if code == 0 and final.get('state') == 'complete' else 'failed',
               exit_code=code, experiment_state=final.get('state'))
    except BaseException as exc:
        update('failed', error=repr(exc))
        raise
    finally:
        if handle:
            kernel.CloseHandle(handle)


if __name__ == '__main__':
    main()
