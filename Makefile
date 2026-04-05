# Makefile for dev tasks

PYTHON := uv run
SRC := src

.PHONY: lint format test 

lint:
	$(PYTHON) ruff check --fix $(SRC)

lint-check:
	$(PYTHON) ruff check $(SRC)

format:
	$(PYTHON) ruff format $(SRC)

format-check:
	$(PYTHON) ruff format --check $(SRC)

test:
	$(PYTHON) pytest tests/
