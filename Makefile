.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")


.PHONY: clean
clean:             ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '.ipynb_checkpoints' -exec rm -rf {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

.PHONY: generate_datasets
generate_datasets:
	@echo "--- Generating datasets ---"
	@$(ENV_PREFIX)python experiments/generate_datasets.py


.PHONY: install     ## Install the package in the current environment
install:
	@echo "Installing bno"
	@echo "--- Warning: make sure that you've activated the virtual environment before running this command ---"
	@pip install .


.PHONY: install_dev  ## Install the package in editable mode
install_dev:
	@echo "Installing bno in editable mode"
	@echo "--- Warning: make sure that you've activated the virtual environment before running this command ---"
	@pip install -e .


.PHONY: jaxgpu
jaxgpu:              ## Installs jax for *nix systems with CUDA
	@echo "Installing jax..."
	@$(ENV_PREFIX)pip install --upgrade pip
	@$(ENV_PREFIX)pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


.PHONY: virtualenv
virtualenv:          ## Create a virtual environment. Checks that python > 3.8
	@echo "creating virtual environment ..."
	@python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8 or higher is required'" || exit 1
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip

	@echo "Installing JAX with GPU support"
	@make jaxgpu

	@echo "Instaling JaxDF and jwave"
	@./.venv/bin/pip install git+https://github.com/ucl-bug/jaxdf.git
	@mkdir .venv/raw
	@git clone git@github.com:ucl-bug/jwave.git .venv/raw/jwave
	@./.venv/bin/pip install .venv/raw/jwave

	@echo "Installing bno"
	@./.venv/bin/pip  install -e .

	@echo "Installing pre-commit"
	@$(ENV_PREFIX)pre-commit install

	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment before using BNO!!!"

.PHONY: test
test:             ## Run tests with pytest (discover mode)
	$(ENV_PREFIX)pytest
