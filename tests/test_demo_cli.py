import subprocess
import sys
from pathlib import Path


def test_committed_demo_runs_exact_format(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "answers.txt"
    completed = subprocess.run([
        sys.executable,
        str(root / "scripts" / "answer_questions.py"),
        "--recording", str(root / "demo" / "recording.csv"),
        "--questions", str(root / "demo" / "questions.txt"),
        "--out", str(out),
        "--no-slm",
    ], cwd=root, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    blocks = out.read_text(encoding="utf-8").strip().split("\n\n")
    assert len(blocks) == 8
    for block in blocks:
        lines = block.splitlines()
        assert lines[0].startswith("Answer: ")
        assert lines[1].startswith("Activity/Event: ")
        assert lines[2] == "Evidence:"
        assert lines[3].startswith("  Timestamp(s): ")
        assert lines[4].startswith("  Sensor Modality: ")
        assert lines[5].startswith("  Sensor Channel(s): ")
        assert lines[6].startswith("Explanation: ")
