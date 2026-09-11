#!/usr/bin/env python3
"""Fail if the preliminary report violates its auditable submission contract."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

PDF = Path(__file__).resolve().parent / "technical_report.pdf"
PENDING = "Pending real-data experiment — no result available in this preliminary revision."


def main() -> int:
    reader = PdfReader(PDF)
    if not 10 <= len(reader.pages) <= 12:
        raise SystemExit(f"expected 10-12 pages, found {len(reader.pages)}")
    page_text = [page.extract_text() or "" for page in reader.pages]
    blanks = [index + 1 for index, text in enumerate(page_text) if not text.strip()]
    if blanks:
        raise SystemExit(f"blank PDF pages: {blanks}")
    text = "\n".join(page_text)
    if text.count(PENDING) != 5:
        raise SystemExit(f"expected five exact pending figure notices, found {text.count(PENDING)}")
    test_counts = {int(value) for value in re.findall(r"(\d+) (?:committed )?tests passed", text)}
    if test_counts != {20}:
        raise SystemExit(f"unexpected reported test results: {sorted(test_counts)}")
    for placeholder in ("MEMBER_2_ROLL", "MEMBER_3_ROLL"):
        if placeholder not in text:
            raise SystemExit(f"missing visible team placeholder: {placeholder}")
    print(f"OK: {len(reader.pages)} pages; five pending figures; reported test result 20/20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
