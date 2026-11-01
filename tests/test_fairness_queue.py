"""Exercise scheduler command construction without submitting cluster jobs."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = str(Path('C:/Program Files/Git/bin/bash.exe')) if os.name == 'nt' else shutil.which('bash')


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash required for launcher tests')
class QueueTests(unittest.TestCase):
    def invoke(self, *args, batch='24'):
        env = dict(os.environ)
        for key in ('SLURM_JOB_ID', 'SLURM_SUBMIT_DIR'):
            env.pop(key, None)
        env.update(FAIRNESS_DRY_RUN='1', FAIRNESS_BATCH_SIZE=batch)
        return subprocess.run([BASH, 'scripts/queue_fairness_next.sh', *map(str, args)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)

    def test_all_stages_covered_once_with_bounded_submissions(self):
        stage, until, first = 'train', 64, 0
        covered = {}
        for _ in range(40):
            result = self.invoke(stage, until, first)
            self.assertEqual(result.returncode, 0, result.stderr)
            if stage == 'report':
                self.assertIn('--stage report', result.stderr)
                break
            matches = re.findall(r'--array=(\d+)-(\d+)%(\d+)', result.stderr)
            self.assertEqual(len(matches), 1)
            low, high, cap = map(int, matches[0])
            self.assertLessEqual(high - low + 1, 24)
            self.assertEqual(cap, 24)
            self.assertEqual(result.stderr.count('DRY RUN sbatch'), 2)
            covered.setdefault((stage, until), []).extend(range(low, high + 1))
            next_args = re.search(r'scripts/queue_fairness_next.sh (train|evaluate|report) (\d+) (\d+)', result.stderr)
            self.assertIsNotNone(next_args)
            stage, until, first = next_args.groups()
            until, first = int(until), int(first)
        else:
            self.fail('Dispatcher did not terminate at report')
        for until in range(64, 513, 64):
            self.assertEqual(covered[('train', until)], list(range(30)))
        self.assertEqual(covered[('evaluate', 512)], list(range(420)))

    def test_existing_array_recovery_submits_only_one_continuation(self):
        result = self.invoke('--after', 1049573, 'train', 128, 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count('DRY RUN sbatch'), 1)
        self.assertIn('--dependency=afterok:1049573', result.stderr)
        self.assertIn('scripts/queue_fairness_next.sh train 128 0', result.stderr)
        self.assertNotIn('--array=', result.stderr)

    def test_oversized_batch_rejected_before_submission(self):
        result = self.invoke('train', 128, 0, batch='32')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('DRY RUN sbatch', result.stderr)
