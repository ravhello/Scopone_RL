SHELL := /bin/bash

.PHONY: test install run benchmark

test:
	conda run -n gym_tf pytest -q

install:
	conda run -n gym_tf python -m pip install -U pip setuptools wheel
	conda run -n gym_tf python -m pip install -r requirements.txt || true

run:
	conda run -n gym_tf python main.py

benchmark:
	conda run -n gym_tf python benchmark.py


