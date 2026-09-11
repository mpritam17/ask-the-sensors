# ExtraSensory data inspection record

This file distinguishes observations made from downloaded official files from
assumptions that still require the large sensor archives.

## Original-label metadata - verified 2026-09-11

- Archive endpoint: official ExtraSensory `per_uuid_original_labels` release.
- Extracted layout: 60 flat `UUID.original_labels.csv.gz` files.
- Observed columns use the prefix `original_label:` (not `label:`).
- Example file: `098A72A5-E3E5-4F54-A152-BBDA0DF7B694.original_labels.csv.gz`.
- Shape: 6,813 rows and 118 columns.
- Timestamp range: 1,438,708,458 through 1,441,139,452 Unix seconds.
- Seven-class parse without the cleaned-label consistency filter:

| Class | Kept examples |
|---|---:|
| lying_down | 2,724 |
| sitting | 2,246 |
| walking | 187 |
| bicycling | 120 |
| standing_in_place | 80 |
| standing_and_moving | 78 |
| running | 33 |
| no main activity (dropped) | 1,345 |

These are inspection counts for one user, not recognition or QA results. The
inspection revealed and prompted the `original_label:` normalization fix.

## Full original-label archive - verified 2026-09-12

`scripts/summarize_official_labels.py` parsed all 60 official per-user files.
Before the cleaned-label contradiction filter is available, the archive has
377,346 examples, of which 307,220 (81.4%) contain exactly one target main
activity. The remaining 70,126 have no target main activity; no multi-main
examples were observed.

| Class | Original-label examples |
|---|---:|
| lying_down | 104,210 |
| sitting | 136,356 |
| standing_in_place | 8,028 |
| standing_and_moving | 29,754 |
| walking | 22,517 |
| running | 1,335 |
| bicycling | 5,020 |

These are real metadata counts, not recognition results. They show a roughly
102:1 sitting-to-running imbalance and motivate macro-F1, balanced accuracy,
class weighting, and per-user sampling caps.

## Still pending

- Cleaned-label contradiction counts; the aggregate JSON currently records
  `users_with_cleaned_consistency_filter: 0` and will be regenerated once
  `features_labels` finishes.
- Raw accelerometer/gyroscope nesting, column count, timestamp units, sample
  rate, accelerometer units, missing-modality rate, and coverage statistics.
- Full per-user/class build manifest and the official-fold mapping.
