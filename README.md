# Ask the Sensors

Grounded, explainable activity question answering from wearable accelerometer and gyroscope signals.

CS60055 Ubiquitous Computing, IIT Kharagpur - Hackathon Challenge 1.

Submission artifact: [technical report PDF](report/technical_report.pdf), with its reproducible [Markdown source](report/technical_report.md) and [ReportLab builder](report/build_report.py).

## Status

The repository implements the full pipeline:

- official ExtraSensory download, inspection, and validity-aware 25 Hz preprocessing;
- 98-feature raw dual-sensor and 49-feature accelerometer-only recognition;
- user-disjoint train, validation, recognition-test, and five-user QA splits;
- gap-aware activity intervals and deterministic Tasks 1-3;
- local Qwen Task 4 with strict JSON/timestamp/channel validation and deterministic fallback;
- exact-format CLI, evaluation metrics, five required figures, and efficiency benchmarks.

`pytest -q` passes **60 tests** in the recorded WSL/Python 3.14.4 environment.

### Measured real-data results

| Scope | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Raw accelerometer + gyroscope, 144,056 test windows | 35.99% | 37.36% | 29.78% |
| Raw accelerometer-only fallback | 34.88% | 40.99% | 28.13% |
| Official precomputed dual-feature baseline | 54.77% | - | 36.21% |

The balanced end-to-end QA score is 13.5% over 600 held-out questions. These modest scores and all zero-valued temporal/grounding results are reported directly in the report; software completeness is not presented as model accuracy.

Qwen2.5-1.5B-Instruct loaded in FP16 on the RTX 5050. In the recorded 30-run benchmark its generated answers violated the evidence contract, so all 36 calls (one initial, five warm-ups, 30 measured) were rejected and the grounded fallback answered. This is the intended safety behavior, not a successful SLM-quality claim.

## Output contract

Every answer uses this field order:

```text
Answer:              <direct answer to the query, or N/A>
Activity/Event:      <activity or event, or N/A>
Evidence:
    Timestamp(s):      <time range or ranges, or N/A>
    Sensor Modality:   <accelerometer, gyroscope, both, or N/A>
    Sensor Channel(s): <Acc X/Y/Z, Gyro X/Y/Z, All, or N/A>
Explanation:         <reasoning grounded in the observed signal, or N/A>
```

Timestamps are seconds from the first retained sample. A supported answer cites the interval used to compute it. Unsupported questions return `Inconclusive`/`N/A`; rejected model output is never formatted for the user.

## Architecture

```text
raw acc + gyro -> 25 Hz preprocessing -> 7-class recognition -> activity timeline
                                                                    |
question -> intent router -> Tasks 1-3 deterministic operations ----+-> validator -> answer
                         \-> Task 4 Qwen over interval summaries ----/
                                      \-> grounded fallback on rejection
```

Tasks 1-3 perform duration, count, onset, verification, identification, comparison, and grounding operations directly over intervals. Qwen receives structured interval/feature summaries rather than raw samples.

## Quick setup

```bash
git clone https://github.com/mpritam17/ask-the-sensors.git
cd ask-the-sensors
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
pytest -q
```

The core pipeline works without PyTorch. For the optional local Qwen path:

```bash
python -m pip install -r requirements-slm.txt
```

## Run the required CLI

The recording CSV must contain `timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z`; the questions file has one question per line.

```bash
python scripts/answer_questions.py \
  --recording demo/recording.csv \
  --questions demo/questions.txt \
  --model artifacts/raw_both_recognizer.joblib \
  --out answers.txt \
  --no-slm
```

Remove `--no-slm` only after installing the optional dependencies and downloading/caching Qwen. If no model is supplied, the command emits a conspicuous warning and uses a transparent demonstration heuristic; that path is not used for reported accuracy.

## Reproduce the official raw-data results

Keep the roughly 56 GB of extracted sensor data outside Git/OneDrive.

```bash
DATA_ROOT=/path/to/extrasensory
PROCESSED_ROOT=/path/to/processed_both

python scripts/fetch_extrasensory.py --root "$DATA_ROOT" \
  --only original_labels features_labels cv_folds
python scripts/fetch_extrasensory.py --root "$DATA_ROOT" --only raw_acc
python scripts/fetch_extrasensory.py --root "$DATA_ROOT" \
  --timeout-seconds 180 --only proc_gyro
python scripts/inspect_raw_layout.py --root "$DATA_ROOT"

python scripts/build_dataset.py --raw-root "$DATA_ROOT" \
  --out "$PROCESSED_ROOT" --modalities both
python scripts/make_splits.py --processed "$PROCESSED_ROOT" \
  --cv-folds "$DATA_ROOT/cv_folds" \
  --out artifacts/raw_both_split_manifest.json
python scripts/train_recognizer.py --processed "$PROCESSED_ROOT" \
  --splits artifacts/raw_both_split_manifest.json \
  --model artifacts/raw_both_recognizer.joblib \
  --results artifacts/raw_both_recognition_results.json
python scripts/evaluate_system.py --processed "$PROCESSED_ROOT" \
  --splits artifacts/raw_both_split_manifest.json \
  --model artifacts/raw_both_recognizer.joblib \
  --recognition-results artifacts/raw_both_recognition_results.json \
  --out artifacts/raw_both_evaluation_results.json
python scripts/generate_figures.py \
  --results artifacts/raw_both_evaluation_results.json \
  --out report/figures/real_raw_both
python scripts/benchmark_efficiency.py \
  --model artifacts/raw_both_recognizer.joblib \
  --recording-npz "$PROCESSED_ROOT/00EABED2-271D-49D8-B599-1D4A09240601.npz" \
  --out artifacts/raw_both_efficiency.json --runs 30
```

The downloader resumes safely, detects a server that ignores HTTP Range, verifies ZIP integrity, prevents path traversal during extraction, and writes completion markers only after success.

### Optional Qwen runtime benchmark

```bash
HF_HOME=/path/to/huggingface-cache \
python scripts/benchmark_slm.py --out artifacts/slm_efficiency.json --runs 30
```

## Synthetic smoke workflow

This verifies the entire pipeline without downloading ExtraSensory. Synthetic artifacts are stamped `synthetic` and must not be cited as real results.

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
```

## Repository layout

```text
artifacts/          committed models, split manifests, measurements, result JSON
configs/            sampling/window/label configuration
demo/               small recording and questions
docs/               design decisions, data inspection, and AI-use log
report/             report source/PDF and real/synthetic figures
scripts/            download, build, train, evaluate, benchmark, and CLI entry points
src/ats/            data, features, recognition, timeline, QA, SLM, eval, efficiency
tests/              unit and integration tests
```

Raw data, expanded archives, virtual environments, downloaded SLM weights, and oversized experimental models are excluded from Git.

## Data and integrity

ExtraSensory: Vaizman, Ellis, and Lanckriet, *IEEE Pervasive Computing* 16(4), 2017, <http://extrasensory.ucsd.edu/>. The original label release is used because the cleaned release merges standing in place and standing/moving into `OR_standing`.

Claude (Anthropic) assisted with the initial scaffold and preprocessing foundation. OpenAI Codex assisted with audit, implementation, testing, experiments, and the report. Details and verification actions are recorded in [docs/AI_USE_LOG.md](docs/AI_USE_LOG.md).
