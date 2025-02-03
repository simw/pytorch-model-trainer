include .env

.DEFAULT_TARGET: help
sources = src tests


.PHONY: prepare
prepare:
	uv sync --extra cpu --frozen --group test --group lint


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
	docker build --platform linux/amd64 -f Dockerfile-cpu -t pytorch-model-trainer-cpu .
	docker build --platform linux/amd64 -f Dockerfile-cuda -t pytorch-model-trainer-cuda .


.PHONY: docker-push
# Note: prerequisites are:
# 1) a gcp project with an artifact registry setup
# 2) gcloud auth login
# 3) gcloud auth configure-docker <region>-docker.pkg.dev (or other registry)
# 4) a .env file with DOCKER_REPO=<region>-docker.pkg.dev/<project-id>/<repo-name> (or similar)
docker-push:
	docker tag pytorch-model-trainer-cpu ${DOCKER_REPO}/pytorch-model-trainer-cpu
	docker push ${DOCKER_REPO}/pytorch-model-trainer-cpu
	docker tag pytorch-model-trainer-cuda ${DOCKER_REPO}/pytorch-model-trainer-cuda
	docker push ${DOCKER_REPO}/pytorch-model-trainer-cuda
