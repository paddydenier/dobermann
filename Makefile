# Makefile for dev tasks

PYTHON := uv run
SRC := src

.PHONY: lint format test 

lint:
	$(PYTHON) ruff check $(SRC)

format:
	$(PYTHON) ruff format --check $(SRC)

test:
	$(PYTHON) pytest
