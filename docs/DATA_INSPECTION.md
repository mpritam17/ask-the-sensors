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

## Still pending

- Cleaned-label contradiction counts, once `features_labels` finishes.
- Raw accelerometer/gyroscope nesting, column count, timestamp units, sample
  rate, accelerometer units, missing-modality rate, and coverage statistics.
- Full per-user/class build manifest and the official-fold mapping.
