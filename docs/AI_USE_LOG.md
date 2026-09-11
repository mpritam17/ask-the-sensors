# AI-use log

Maintained continuously, as required by the CS60055 AI-use policy. Each entry records
what an AI tool produced, and what the team did to verify or change it. The summary in
the report is derived from this file.

**Tool:** Claude (Anthropic), used through the chat interface as a pair-programming and
design assistant.

| Phase | AI contribution | Team action |
|---|---|---|
| 1 — repo scaffold | Proposed the folder layout, README skeleton, Makefile, and the two config files with default parameter values and their stated rationale. | *(to fill: reviewed / adjusted X / rejected Y)* |
| 2 — data pipeline | Wrote `resample.py`, `raw_io.py`, `labels.py`, `windowing.py`, `build_dataset.py`, `synthetic.py`, the three scripts, and 20 unit tests. Identified from the ExtraSensory documentation that the 7 challenge classes correspond to the dataset's mutually-exclusive "main activity" labels, available only in the original-label release. | *(to fill: verified the raw file layout against the real download with `inspect_raw_layout.py`; confirmed / corrected the parser branch that fired)* |

## Verification obligations carried forward

These are assumptions an AI assistant could not check and the team must:

1. **Raw file layout.** `raw_io.py` auto-detects a 3-column (x,y,z) or 4-column
   (t,x,y,z) layout because the exact ExtraSensory raw format could not be confirmed
   without downloading the 15 GB archives. Run `scripts/inspect_raw_layout.py` after the
   first download and record which branch fires.
2. **Accelerometer units.** The builder rescales to g when the median magnitude looks
   like m/s². Confirm this fires (or does not) on real users, and report how many.
3. **Original-label reliability.** The original labels are pre-cleaning. Report how many
   examples the consistency filter drops, per class — a high drop rate on one class is a
   finding, not a nuisance.
