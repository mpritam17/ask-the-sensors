import importlib.util
import io
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_extrasensory.py"
SPEC = importlib.util.spec_from_file_location("fetch_extrasensory", SCRIPT)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class FakeResponse(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_resume_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    destination = tmp_path / "file.zip"
    destination.write_bytes(b"old-partial")
    monkeypatch.setattr(
        fetch.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b"complete", 200, {"Content-Length": "8"}),
    )
    fetch.download("https://example.invalid/file.zip", destination)
    assert destination.read_bytes() == b"complete"


def test_resume_appends_only_on_partial_content(tmp_path, monkeypatch):
    destination = tmp_path / "file.zip"
    destination.write_bytes(b"abc")
    monkeypatch.setattr(
        fetch.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b"def", 206, {"Content-Range": "bytes 3-5/6"}),
    )
    fetch.download("https://example.invalid/file.zip", destination)
    assert destination.read_bytes() == b"abcdef"


def test_zip_verification_and_traversal_guard(tmp_path):
    valid = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid, "w") as archive:
        archive.writestr("inside/file.txt", "ok")
    fetch.verify_zip(valid)
    fetch.unpack(valid, tmp_path / "out")
    assert (tmp_path / "out" / "inside" / "file.txt").read_text() == "ok"

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.txt", "no")
    with pytest.raises(RuntimeError, match="unsafe ZIP member"):
        fetch.unpack(unsafe, tmp_path / "unsafe-out")
