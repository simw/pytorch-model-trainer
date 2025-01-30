.DEFAULT_TARGET: help
sources = src tests


.PHONY: prepare
prepare:
	uv sync --frozen --group test --group lint


.PHONY: lintable
lintable: prepare
	uv run black $(sources)
	uv run ruff check --fix $(sources)


.PHONY: lint
lint: prepare
	uv run black --check --diff $(sources)
	uv run ruff check $(sources)
	uv run mypy $(sources)


.PHONY: test
test: prepare
	uv run coverage run -m pytest
	uv run coverage report


.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -rf coverage.xml


.PHONY: docker
docker: prepare
	docker build -f Dockerfile-cpu -t pytorch-model-trainer-cpu .
	docker build -f Dockerfile-cuda -t pytorch-model-trainer-cuda .
