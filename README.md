# Ask the Sensors

**Submission report:** [preliminary technical report (PDF)](report/technical_report.pdf)
with its reproducible [Markdown source](report/technical_report.md) and
[ReportLab builder](report/build_report.py). Pending experiments are labeled explicitly.

Grounded, explainable activity question answering from wearable accelerometer and
gyroscope signals.

CS60055 Ubiquitous Computing, IIT Kharagpur — Hackathon Challenge 1.

---

## What this system does

Given (a) a triaxial accelerometer + gyroscope recording and (b) a natural-language
question about it, the system returns one structured answer whose every claim is
traceable to a specific stretch of signal:

```
Answer:            <direct answer, or N/A>
Activity/Event:    <activity or event, or N/A>
Evidence:
  Timestamp(s):    <time range(s), or N/A>
  Sensor Modality: <accelerometer, gyroscope, both, or N/A>
  Sensor Channel(s): <Acc X/Y/Z, Gyro X/Y/Z, All, or N/A>
Explanation:       <reasoning grounded in the observed signal, or N/A>
```

**Time base.** Every timestamp this system emits is **seconds from the start of the
recording**, where second 0 is the first retained example. The absolute Unix time of
second 0 is stored alongside each processed recording (`t0_unix`) so answers can be
converted to clock time on request, but the reported convention never changes.

## Architecture: a two-speed system

Most questions in Tasks 1–3 are deterministic computations over a detected activity
timeline, not open-ended reasoning. Asking a language model "how many seconds of
walking are there" invites it to approximate an arithmetic sum we can compute exactly.
So we compute them exactly, and reserve the language model for the one tier that
genuinely needs semantic reasoning.

```
  raw acc + gyro          ┌──────────────┐   ┌─────────────┐   ┌────────────┐
  (40 Hz, 20 s sessions)→ │ Preprocessing │ → │ Recognition │ → │ Aggregation│
                          │ resample 25Hz │   │ 7-class per │   │ windows →  │
                          │ + validity    │   │ window      │   │ intervals  │
                          └──────────────┘   └─────────────┘   └─────┬──────┘
                                                                     │ timeline
   question ─────────────→ ┌──────────────┐                          │
                           │ Query router │ ─── fast path (T1–3) ────┤
                           │ intent class │      deterministic        │
                           └──────┬───────┘                          │
                                  └──────── slow path (T4) ──────────┤
                                            SLM over serialised      │
                                            interval evidence        │
                                                                     ▼
                                                        ┌────────────────────┐
                                                        │ Output formatter   │
                                                        └────────────────────┘
```

Three consequences worth stating up front, because they are the design argument:

1. **Fast-path answers are provably correct given the timeline.** A duration is a sum
   over intervals, not a generation. Errors can only come from the recognition layer,
   which is measurable on its own.
2. **Evidence citation is free and exact.** The evidence *is* the data structure the
   answer was computed from, so cited intervals cannot drift from the answer.
3. **Most queries cost near zero.** The SLM runs on the open-world tier only, which is
   what makes the accuracy-versus-overhead curve interesting rather than flat.

## Setup

```bash
git clone https://github.com/mpritam17/ask-the-sensors.git
cd ask-the-sensors
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock  # exact environment used for verification
pip install -e .            # installs the `ats` package from src/
```

Python 3.9+ is supported. The recorded verification environment is WSL Ubuntu with
Python 3.14.4 because Python 3.12 was not installed on the available machine. PyTorch,
Transformers, and Accelerate are optional and needed only for the Qwen Task 4 path.

## Current implementation status

- Preprocessing, feature extraction, three recognizers, timeline aggregation, Tasks
  1-3, grounded Task 4 fallback/Qwen validation, evaluation, figures, and efficiency
  benchmarking are implemented.
- `pytest -q` currently passes 45 tests, including end-to-end CLI, downloader-recovery,
  official-schema, preliminary-report-contract, and grounded-output checks.
- Saved synthetic smoke-test outputs are under `artifacts/` and
  `report/figures/synthetic/`; all are explicitly stamped synthetic.
- A real, user-disjoint baseline using the official precomputed accelerometer and
  gyroscope features is complete: the compact RF reached 54.77% accuracy and 36.21%
  macro-F1 on 54,323 examples from the 12 official fold-0 test users. This is explicitly a
  sensor-feature recognition baseline, not raw-window or end-to-end QA evaluation.
- Raw-archive end-to-end evaluation and the Qwen weight run remain pending. The
  preliminary PDF therefore retains honest pending-result panels for those results.

## Reproducing our results

### Option A — smoke test, no download (about 2 minutes)

Verifies the whole pipeline runs end to end on synthetic recordings shaped exactly like
ExtraSensory. **Numbers produced this way are not results**; every figure generated from
synthetic input is stamped SYNTHETIC.

```bash
python scripts/make_synthetic_data.py --users 8 --scale 0.06 --seed 41
python scripts/build_dataset.py --raw-root data/raw/synthetic --synthetic
python scripts/train_recognizer.py \
  --processed data/processed/synthetic \
  --splits artifacts/synthetic_split_manifest.json \
  --model artifacts/synthetic_recognizer.joblib \
  --results artifacts/synthetic_recognition_results.json --synthetic
python scripts/evaluate_system.py \
  --processed data/processed/synthetic \
  --splits artifacts/synthetic_split_manifest.json \
  --model artifacts/synthetic_recognizer.joblib \
  --recognition-results artifacts/synthetic_recognition_results.json \
  --out artifacts/synthetic_evaluation_results.json
python scripts/generate_figures.py \
  --results artifacts/synthetic_evaluation_results.json \
  --out report/figures/synthetic
python scripts/benchmark_efficiency.py \
  --model artifacts/synthetic_recognizer.joblib \
  --recording-npz data/processed/synthetic/00000004-0000-4000-8000-000000000004.npz \
  --out artifacts/synthetic_efficiency.json --runs 30
pytest -q
```

### Option B — the real thing

```bash
# 1. Fetch. Small files first; the raw archives are ~15 GB and are fetched
#    sequentially because the dataset site asks for one download at a time.
DATA_ROOT=/home/$USER/datasets/extrasensory
python scripts/fetch_extrasensory.py --root "$DATA_ROOT" \
  --only original_labels features_labels cv_folds
python scripts/summarize_official_labels.py --root "$DATA_ROOT"
python scripts/train_precomputed_baseline.py --root "$DATA_ROOT" \
  --modalities both \
  --model artifacts/precomputed_real_both_recognizer.joblib \
  --splits artifacts/precomputed_real_split_manifest.json \
  --results artifacts/precomputed_real_both_results.json

# The next two archives are needed for the end-to-end raw-window experiment.
python scripts/fetch_extrasensory.py --root "$DATA_ROOT" --only raw_acc
python scripts/fetch_extrasensory.py --root "$DATA_ROOT" --only proc_gyro

# 2. Confirm the on-disk layout (run once; paste the output into the report)
python scripts/inspect_raw_layout.py --root "$DATA_ROOT"

# 3. Build windowed, labelled arrays
python scripts/build_dataset.py --raw-root "$DATA_ROOT"
python scripts/make_splits.py --cv-folds "$DATA_ROOT/cv_folds"
python scripts/train_recognizer.py
```

The downloader resumes safely, detects servers that ignore HTTP Range, verifies every
ZIP before extraction, and reports actionable recovery commands.

### Running on a fresh recording

```bash
python scripts/answer_questions.py \
  --recording demo/recording.csv \
  --questions demo/questions.txt \
  --out answers.txt \
  --no-slm
```

Accepts a CSV with columns `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`
(any sampling rate; it is resampled to 25 Hz) and one question per line. Writes answers
in the required format. Pass `--model artifacts/recognizer.joblib` after real-data
training. If no recognizer exists, the CLI emits a conspicuous warning and uses a
transparent demo heuristic; that output is never a reported accuracy result.

Omit `--no-slm` only after installing the optional model dependencies. Qwen receives
structured intervals/features rather than raw samples, and its JSON is rejected if a
timestamp lies outside the supplied evidence.

## Repository layout

```
configs/            every tunable parameter, with its rationale inline
  data.yaml           sampling, windowing, split
  labels.yaml         the 7-class mapping and imbalance strategy
src/ats/
  config.py           config loading
  data/               preprocessing: download, raw I/O, resampling, labels,
                      windowing, dataset building, synthetic generator
  features/           engineered feature extraction        (Phase 3)
  recognition/        per-window 7-class classifier         (Phase 3)
  timeline/           window → interval aggregation         (Phase 4)
  qa/                 query router + deterministic answers  (Phase 5)
  slm/                open-world reasoning path             (Phase 6)
  eval/               metrics and the five required figures (Phase 7)
  efficiency/         size, memory, latency, energy         (Phase 8)
scripts/            runnable entry points
tests/              unit tests; `pytest -q` from the repo root
docs/               design decisions, AI-use log
report/             the technical report and its figures
data/               git-ignored; raw data is never committed
```

## Data

ExtraSensory (Vaizman, Ellis & Lanckriet, *IEEE Pervasive Computing* 16(4), 2017),
<http://extrasensory.ucsd.edu/>. Downloaded by `scripts/fetch_extrasensory.py`; never
committed to this repository.

Two properties of the dataset shape everything downstream, and are documented in
`docs/DESIGN_DECISIONS.md`:

- **Recordings are not continuous.** The collection app recorded a 20-second session
  once per minute, so a "recording" is a sequence of 20 s observations with 40 s gaps.
  Durations and interval boundaries are therefore estimated at minute resolution, and
  the aggregation layer must say so rather than pretend to continuous coverage.
- **The seven challenge classes are ExtraSensory's own mutually-exclusive "main
  activity" category**, which survives intact only in the *original* label release. The
  cleaned release merges "standing in place" and "standing and moving" into a single
  `OR_standing` label and so cannot express two of our seven classes.

## Academic integrity

External resources are cited at the point of use in the source, and collectively in the
report. AI assistance is logged continuously in `docs/AI_USE_LOG.md` and disclosed in
the report as required by the CS60055 policy.
