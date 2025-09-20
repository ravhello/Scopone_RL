SHELL := /bin/bash

.PHONY: test install run benchmark

test:
	conda run -n pt_ipex pytest -q

install:
	conda run -n pt_ipex python -m pip install -U pip setuptools wheel
	conda run -n pt_ipex python -m pip install -r requirements.txt || true

run:
	conda run -n pt_ipex python main.py

benchmark:
	conda run -n pt_ipex python benchmark.py


