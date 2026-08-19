PYTHON := uv run
SRC := .
PORT ?= 8000

.PHONY: lint format test docs

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

docs-readme:
	$(PYTHON) dev/update_readme.py

docs:
	$(MAKE) -C docs html

docs-clean:
	$(MAKE) -C docs clean

docs-api:
	$(PYTHON) sphinx-apidoc -o docs/source/api src/dobermann

docs-api-clean:
	rm -rf docs/source/api

backend:
	uv run uvicorn api.main:app --reload --port $(PORT)
