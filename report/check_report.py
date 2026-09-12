#!/usr/bin/env python3
"""Fail if the final report violates its auditable submission contract."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

PDF = Path(__file__).resolve().parent / "technical_report.pdf"


def main() -> int:
    reader = PdfReader(PDF)
    if not 10 <= len(reader.pages) <= 12:
        raise SystemExit(f"expected 10-12 pages, found {len(reader.pages)}")
    page_text = [page.extract_text() or "" for page in reader.pages]
    blanks = [index + 1 for index, text in enumerate(page_text) if not text.strip()]
    if blanks:
        raise SystemExit(f"blank PDF pages: {blanks}")
    text = "\n".join(page_text)
    for stale in ("PENDING REAL-DATA", "PRELIMINARY STATUS", "preliminary revision"):
        if stale.lower() in text.lower():
            raise SystemExit(f"stale preliminary marker remains: {stale}")
    for figure in range(2, 7):
        if f"Figure {figure}." not in text:
            raise SystemExit(f"missing required result Figure {figure}")
    test_counts = {
        int(value)
        for value in re.findall(r"(\d+)\s+(?:automated\s+)?tests\s+pass(?:ed|ing)?", text)
    }
    if test_counts != {60}:
        raise SystemExit(f"unexpected reported test results: {sorted(test_counts)}")
    for placeholder in ("MEMBER_2_ROLL", "MEMBER_3_ROLL"):
        if placeholder not in text:
            raise SystemExit(f"missing visible team placeholder: {placeholder}")
    title = str(reader.metadata.title or "")
    if title != "Ask the Sensors - Technical Report":
        raise SystemExit(f"unexpected PDF title: {title!r}")
    required_claims = (
        "35.99%",
        "13.5%",
        "3,314.03 ms",
        "all 36 generations were rejected",
    )
    for claim in required_claims:
        if claim.lower() not in text.lower():
            raise SystemExit(f"missing measured/disclosed result: {claim}")
    print(f"OK: {len(reader.pages)} pages; five measured figures; reported test result 60/60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
