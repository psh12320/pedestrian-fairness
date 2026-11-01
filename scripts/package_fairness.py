"""Create a small source-only archive, excluding data, models and dependencies."""
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
destination = root / "runs/fairness-v2-source.tar.gz"
destination.parent.mkdir(exist_ok=True)
paths = [root / "AGENTS.md", root / "README.md", root / "requirements.txt"]
for folder, pattern in (("src", "*.py"), ("config", "*.json"), ("tests", "test_fairness*.py"),
                        ("scenario", "single.*.xml"), ("scripts", "*fairness*.sh"),
                        ("scripts", "collect_fairness_status.py"),
                        ("docs", "fairness*.md")):
    paths.extend((root / folder).rglob(pattern))
with tarfile.open(destination, "w:gz") as archive:
    for path in sorted(set(paths)):
        if path.is_file():
            archive.add(path, arcname=str(Path("trafficlightcontrol") / path.relative_to(root)))
print(destination)
print(f"{destination.stat().st_size / 1024:.1f} KiB")
