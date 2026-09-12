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

## Feature/label metadata - verified 2026-09-12

- Extracted layout: 60 flat `UUID.features_labels.csv.gz` files, matching all
  60 original-label users.
- The inspected file has 278 columns: 26 `raw_acc` features, 26 `proc_gyro`
  features, 51 cleaned label columns, timestamp, and other sensor/context
  features excluded from this challenge baseline.
- Relevant cleaned labels contain explicit 0/1 values; unlabeled rows use NaN.
- The consistency filter matched all 60 users and found zero contradictions in
  the 307,220 retained single-main-activity examples.
- The official five-fold archive contains 20 UUID-list files under
  `cv_5_folds/` (train/test lists for Android and iPhone in each fold).

## Still pending

- Gyroscope nesting, sampling characteristics, and missing-modality/coverage
  statistics once the second raw archive is available.

## Raw accelerometer - verified 2026-09-12

- Verified ZIP extraction occupies about 29 GB and contains 377,056 files for
  all 60 users under `raw_acc/raw_acc/<UUID>/<timestamp>.m_raw_acc.dat`.
- Files have four numeric columns: sensor time followed by x, y, and z.
- Three inspected iPhone-user sessions contain 800 samples, span 23.13-23.16
  seconds, have a median instantaneous rate of 33.9 Hz, and a 95th-percentile
  inter-sample gap of 0.030 seconds. Median vector magnitude is 0.997, so these
  values are already in `g`.
- An inspected Android-user session contains 800 samples over 16.09 seconds at
  a 49.65 Hz median rate. Its median magnitude is 10.02, indicating `m/s²`.
- This cross-device heterogeneity validates timestamp-based resampling to the
  required 25 Hz and per-session accelerometer unit normalization; a fixed
  source-rate assumption would be incorrect.
- An accelerometer-only smoke build for official user
  `00EABED2-271D-49D8-B599-1D4A09240601` retained 2,118 sessions and produced
  29,643 valid 2.56-second windows. The count is preprocessing evidence, not a
  recognition metric. Its artifact records `accelerometer` as the sole
  available modality and downstream timeline evidence is constrained to the
  three accelerometer channels.
- Full per-user/class build manifest and raw-window recognition metrics across
  a broader user-disjoint subset remain pending.
