PYTHON := uv run
SRC := .
PORT ?= 8000

.PHONY: lint lint-check format format-check test docs docs-readme docs-clean docs-api docs-api-clean backend benchmark release

lint:
	$(PYTHON) ruff check --fix $(SRC)

lint-check:
	$(PYTHON) ruff check $(SRC)

format:
	$(PYTHON) ruff format $(SRC)

format-check:
	$(PYTHON) ruff format --check $(SRC)

test:
	$(PYTHON) pytest

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

benchmark:
	uv run python -m benchmark.main

# auto tag last commit based on current uv version
# build: bump version to x.x.x
release:
	@VERSION=$$(uv version --short); \
	TAG="v$$VERSION"; \
	echo "Creating tag $$TAG"; \
	git tag -a "$$TAG" -m "Release $$VERSION"; \
	git push origin "$$TAG"
