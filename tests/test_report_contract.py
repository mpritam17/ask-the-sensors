import importlib.util
from pathlib import Path


def test_committed_preliminary_pdf_contract():
    script = Path(__file__).resolve().parents[1] / "report" / "check_report.py"
    spec = importlib.util.spec_from_file_location("check_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
