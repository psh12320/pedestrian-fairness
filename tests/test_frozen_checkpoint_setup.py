"""Line-ending portability must not relax frozen-source integrity."""
from pathlib import Path
import tempfile
import unittest

from scripts.check_frozen_checkpoints import source_repairs


class FrozenSourceSetupTests(unittest.TestCase):
    def test_exact_bytes_need_no_repair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'a.py').write_bytes(b'x = 1\r\n')
            self.assertEqual(source_repairs(root, {'a.py': b'x = 1\r\n'}), [])

    def test_newline_difference_is_planned_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'a.py').write_bytes(b'x = 1\n')
            plan = source_repairs(root, {'a.py': b'x = 1\r\n'})
            self.assertEqual(plan, [((root / 'a.py').resolve(), b'x = 1\r\n')])
            self.assertEqual((root / 'a.py').read_bytes(), b'x = 1\n')

    def test_actual_change_rejected_before_any_repair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'a.py').write_bytes(b'x = 1\n')
            (root / 'b.py').write_bytes(b'y = 99\n')
            with self.assertRaisesRegex(ValueError, 'Actual source difference'):
                source_repairs(root, {'a.py': b'x = 1\r\n', 'b.py': b'y = 2\n'})
            self.assertEqual((root / 'a.py').read_bytes(), b'x = 1\n')

    def test_missing_path_is_not_created(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'Missing or unsafe'):
                source_repairs(Path(d), {'missing.py': b'x=1\n'})
            self.assertFalse((Path(d) / 'missing.py').exists())

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'Missing or unsafe'):
                source_repairs(Path(d), {'../outside.py': b'x=1\n'})


if __name__ == '__main__':
    unittest.main()
