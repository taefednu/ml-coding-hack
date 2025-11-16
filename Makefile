PYTHON ?= python3
VENV := .venv
ACTIVATE := . $(VENV)/bin/activate;
CONFIG := configs/default.yaml
INPUT ?= data/sample_applications.csv
OUTPUT ?= artifacts/predictions.csv

.PHONY: venv inventory eda train master_table evaluate predict api tests

venv:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) pip install -U pip
	$(ACTIVATE) pip install -r requirements.txt

inventory:
	$(ACTIVATE) $(PYTHON) src/data_loading.py data/train --output artifacts/data_inventory.json

eda:
	$(ACTIVATE) $(PYTHON) scripts/eda_report.py --config $(CONFIG)

master_table:
	$(ACTIVATE) $(PYTHON) scripts/build_master_table.py --data-dir data/train --output artifacts/master_table.parquet

train:
	$(ACTIVATE) $(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(ACTIVATE) $(PYTHON) scripts/evaluate.py --config $(CONFIG)

predict:
	$(ACTIVATE) $(PYTHON) scripts/predict.py --config $(CONFIG) --input $(INPUT) --output $(OUTPUT)

predict_evaluation:
	$(ACTIVATE) $(PYTHON) scripts/predict.py --config $(CONFIG) --evaluation --output artifacts/evaluation_predictions.csv

api:
	$(ACTIVATE) uvicorn api.app:app --reload --port 8080

tests:
	$(ACTIVATE) pytest -q
