.PHONY: help install test smoke data demo report clean lint

help:
	@echo "install  - install deps and the ats package"
	@echo "test     - run the unit test suite"
	@echo "smoke    - synthetic data -> full pipeline, no download needed"
	@echo "data     - build the dataset from real ExtraSensory raw files"
	@echo "demo     - answer the committed synthetic demonstration questions"
	@echo "report   - rebuild and validate the final technical report PDF"
	@echo "clean    - remove generated data and caches (keeps raw downloads)"

install:
	pip install -r requirements.lock
	pip install -e .

test:
	PYTHONPATH=src pytest -q

smoke:
	PYTHONPATH=src python scripts/make_synthetic_data.py --users 8 --scale 0.06 --seed 41
	PYTHONPATH=src python scripts/build_dataset.py --raw-root data/raw/synthetic --synthetic
	PYTHONPATH=src python scripts/train_recognizer.py --processed data/processed/synthetic --splits artifacts/synthetic_split_manifest.json --model artifacts/synthetic_recognizer.joblib --results artifacts/synthetic_recognition_results.json --synthetic
	PYTHONPATH=src python scripts/evaluate_system.py --processed data/processed/synthetic --splits artifacts/synthetic_split_manifest.json --model artifacts/synthetic_recognizer.joblib --recognition-results artifacts/synthetic_recognition_results.json --out artifacts/synthetic_evaluation_results.json
	PYTHONPATH=src python scripts/generate_figures.py --results artifacts/synthetic_evaluation_results.json --out report/figures/synthetic
	PYTHONPATH=src pytest -q

data:
	PYTHONPATH=src python scripts/build_dataset.py

demo:
	PYTHONPATH=src python scripts/answer_questions.py --recording demo/recording.csv --questions demo/questions.txt --out demo/answers.txt --no-slm

report:
	PYTHONPATH=src python report/build_report.py
	PYTHONPATH=src python report/check_report.py

clean:
	rm -rf data/interim/* data/processed/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
