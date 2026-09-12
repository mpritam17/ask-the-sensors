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
    if "STATUS:" in text:
        raise SystemExit("status box text remains in final report")
    for figure in range(2, 7):
        if f"Figure {figure}." not in text:
            raise SystemExit(f"missing required result Figure {figure}")
    test_counts = {
        int(value)
        for value in re.findall(r"(\d+)\s+(?:automated\s+)?tests\s+pass(?:ed|ing)?", text)
    }
    if test_counts != {60}:
        raise SystemExit(f"unexpected reported test results: {sorted(test_counts)}")
    required_members = (
        "22CS30069 - Ritabrata Bharati",
        "22CS30077 - Vishv Magarvadia",
        "23CS30041 - Pritam Mondal",
    )
    for member in required_members:
        if member not in text:
            raise SystemExit(f"missing team member: {member}")
    for allocation in (
        "Data engineering",
        "Recognition and evaluation",
        "QA and integration",
        "shared equally by all three members",
    ):
        if allocation.lower() not in text.lower():
            raise SystemExit(f"missing equal work allocation: {allocation}")
    if "MEMBER_2_ROLL" in text or "MEMBER_3_ROLL" in text:
        raise SystemExit("stale team placeholder remains")
    title = str(reader.metadata.title or "")
    if title != "Ask the Sensors - Technical Report":
        raise SystemExit(f"unexpected PDF title: {title!r}")
    required_claims = (
        "35.99%",
        "20.00%",
        "9.67%",
        "3.56/5",
        "3,314.03 ms",
        "all 36 generations were rejected",
        "76.00% specificity",
        "Overall QA macro accuracy",
    )
    for claim in required_claims:
        if claim.lower() not in text.lower():
            raise SystemExit(f"missing measured/disclosed result: {claim}")
    author = str(reader.metadata.author or "")
    for name in ("Ritabrata Bharati", "Vishv Magarvadia", "Pritam Mondal"):
        if name not in author:
            raise SystemExit(f"PDF metadata author missing: {name}")
    print(
        f"OK: {len(reader.pages)} pages; three named members; five measured figures; "
        "reported test result 60/60"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
