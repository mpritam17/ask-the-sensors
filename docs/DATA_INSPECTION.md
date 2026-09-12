# ExtraSensory data inspection record

This file records observations made directly from the downloaded official
metadata and raw sensor archives. Counts here are preprocessing evidence unless
an entry explicitly names a recognition or QA result artifact.

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
- The completed 60-user accelerometer-only build retained 294,772 sessions and
  produced 3,809,155 valid windows. Of 307,220 labeled examples, 14 had no
  indexed accelerometer file and 12,434 failed the 75% coverage threshold.
- A real fold-0 accelerometer recognizer was then trained and evaluated; its
  saved metrics and figures are scoped as accelerometer-only, not as the final
  dual-sensor result.

## Processed gyroscope - verified 2026-09-12

- Verified ZIP extraction occupies about 27 GB and contains 359,912 files for
  57 users under `proc_gyro/proc_gyro/<UUID>/<timestamp>.m_proc_gyro.dat`.
  Thus, three of the 60 accelerometer users have no released gyroscope tree.
- Files have four numeric columns: sensor time followed by x, y, and z.
- Three inspected sessions contain 800 samples over 19.97 seconds, with a
  40.0 Hz median and effective rate, a 0.025-second 95th-percentile gap, and a
  median vector magnitude of about 0.002.

## Full dual-sensor build - verified 2026-09-12

- The indexed build produced reports for 60 users; 57 users have usable dual
  windows and three have no compatible gyroscope data.
- The label files contain 307,220 retained target examples. The build visited
  307,206 examples, used 270,871, dropped 16,437 for a missing required sensor,
  and dropped 19,912 below the 75% coverage threshold.
- Resampling/windowing produced 3,792,194 candidate windows and retained
  3,535,086 valid windows: 1,207,591 lying, 1,567,707 sitting, 94,731 standing
  in place, 335,506 standing and moving, 258,786 walking, 16,207 running, and
  54,558 bicycling.
- The split manifest contains 32 train, 8 validation, 12 official fold-0
  recognition-test, and 5 QA users, mutually exclusive by UUID.
- `artifacts/raw_both_build_summary.json` records the manifest checksum and
  aggregate counts without committing any raw sample arrays.

## Measured raw dual-sensor results - verified 2026-09-12

- The selected balanced logistic model uses 98 features and is 8,276 bytes.
- On 144,056 untouched-user windows it reached 35.99% accuracy, 37.36%
  balanced accuracy, and 29.78% macro-F1.
- The end-to-end QA set contains 25 short held-out recordings and 600 balanced
  questions. Macro answer accuracy across the eight types is 20.00%; requiring
  correct cited evidence, modality, and channels gives 9.67%. Detailed outcomes
  are retained in `artifacts/raw_both_evaluation_results.json`.
- Five report figures were generated only from that saved real-data JSON under
  `report/figures/real_raw_both/`.
