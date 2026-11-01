"""Check that presentation captures cannot reuse stale protocol frames."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.traffic_control.capture_clip import frame_directory, prepare_frame_directory
from src.traffic_control.scenario import ROOT


class GUICaptureCacheTests(unittest.TestCase):
    def test_protocol_and_model_have_distinct_frame_directories(self) -> None:
        source_root = ROOT / "runs" / "gui" / "source"
        old = source_root / "old-rule" / "vmedium-p120" / "seed-6001" / "fixed" / "baseline"
        new = source_root / "new-rule" / "vmedium-p120" / "seed-6001" / "fixed" / "baseline"
        other_model = (source_root / "new-rule" / "vmedium-p120" / "seed-6001" /
                       "ippo-vehicle" / "different-model-sha")
        self.assertNotEqual(frame_directory(old, 138, 20),
                            frame_directory(new, 138, 20))
        self.assertNotEqual(frame_directory(new, 138, 20),
                            frame_directory(other_model, 138, 20))

    def test_prior_frames_cannot_satisfy_new_capture(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "frame-000.png").write_bytes(b"stale")
            (directory / "frame-001.png").write_bytes(b"stale")
            (directory / "notes.txt").write_text("preserve", encoding="utf-8")
            prepare_frame_directory(directory)
            self.assertEqual(list(directory.glob("frame-*.png")), [])
            self.assertEqual((directory / "notes.txt").read_text(encoding="utf-8"),
                             "preserve")


if __name__ == "__main__":
    unittest.main()
