PYTHON ?= python

.PHONY: data stage1 stage2 stage3 all test smoke clean

data:
	$(PYTHON) scripts/01_download_inspect.py
	$(PYTHON) scripts/02_preprocess.py

stage1: data test

stage2:
	$(PYTHON) scripts/03_train.py --config configs/cinn_2d.yaml
	$(PYTHON) scripts/04_evaluate.py --config configs/cinn_2d.yaml
	$(PYTHON) scripts/05_prior_shift.py --config configs/cinn_2d.yaml

stage3:
	$(PYTHON) scripts/03_train.py --config configs/cinn_6d.yaml
	$(PYTHON) scripts/04_evaluate.py --config configs/cinn_6d.yaml
	$(PYTHON) scripts/05_prior_shift.py --config configs/cinn_6d.yaml

all: data stage2 stage3 test

test:
	pytest tests/ -v

smoke:
	$(PYTHON) scripts/01_download_inspect.py --smoke
	$(PYTHON) scripts/02_preprocess.py --smoke
	$(PYTHON) scripts/03_train.py --config configs/cinn_2d.yaml --smoke
	$(PYTHON) scripts/04_evaluate.py --config configs/cinn_2d.yaml --smoke
	$(PYTHON) scripts/05_prior_shift.py --config configs/cinn_2d.yaml --smoke
	$(PYTHON) scripts/03_train.py --config configs/cinn_6d.yaml --smoke
	$(PYTHON) scripts/04_evaluate.py --config configs/cinn_6d.yaml --smoke
	$(PYTHON) scripts/05_prior_shift.py --config configs/cinn_6d.yaml --smoke
	pytest tests/ -v
