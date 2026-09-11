.PHONY: help install test smoke data clean lint

help:
	@echo "install  - install deps and the ats package"
	@echo "test     - run the unit test suite"
	@echo "smoke    - synthetic data -> full pipeline, no download needed"
	@echo "data     - build the dataset from real ExtraSensory raw files"
	@echo "clean    - remove generated data and caches (keeps raw downloads)"

install:
	pip install -r requirements.txt && pip install -e .

test:
	PYTHONPATH=src pytest -q

smoke:
	PYTHONPATH=src python scripts/make_synthetic_data.py --users 3
	PYTHONPATH=src python scripts/build_dataset.py --raw-root data/raw/synthetic --synthetic
	PYTHONPATH=src pytest -q

data:
	PYTHONPATH=src python scripts/build_dataset.py

clean:
	rm -rf data/interim/* data/processed/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
