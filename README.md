# FIRM_CE
FIRM_CE is a business rules-based long-term planning model (BR-LTP) for electricity systems. The electricity network is defined as a network of spatial nodes connected by major transmission lines. Generators and energy storage systems are located at each node. 

The capacity of generators, storage systems, and transmission lines is optimised through an evolutionary algorithm in an attempt to minimise cost while retaining system reliability. For each candidate solution, assets are dispatched in accordance with a set of deterministic business rules to determine whether load can be balanced by generation in every time interval over the entire model horizon. Business rules attempt to capture the most important dispatch behaviour of each asset, while remaining simple enough to evaluate each candidate solution in a matter of seconds. No temporal aggregation is required.

## Installation
For usage:

`pip install .`

For development:

`pip install -e .[dev]`

## Model Configuration
Example input data is provided in the `inputs` folder. 

Time-independent inputs are stored in `inputs/config` and are used to define the network topology, electricity generation and storage assets, fules, scenarios, and overall model configuration. Each row represents a separate instance for the corresponding class of objects.

Time-series data is stored in `inputs/data`. Data includes loads at each node; availability traces for solar, wind, and baseload generators; and annual generation limits for flexible generators.

## Example Models
The `examples` folder contains two scripts that demonstrate how to build and run the evolutionary optimisation for a new model (`model_build_and_solve.py`) and how to build a model and generate results for a set of candidate solutions stored as initial guesses (`model_build_and_statistics.py`).

## Australian National Electricity Market (NEM)
Inputs for a model of the NEM, based on the Australian Energy Market Operator's publicly available 2024 ISP Model for PLEXOS, are available on the ANU Data Commons: TBD

## Python Tools

These are the python tools conducting code checks.

| Name | Configuration file | Purpose | URL |
| --- | --- | --- | --- |
| flake8 | setup.cfg |  Linter | https://flake8.pycqa.org/en/latest/ |
| black | pyproject.toml | Formatter | https://github.com/psf/black |
| isort | pyproject.toml | Formats import statements | https://pypi.org/project/isort/ |
| safety | - | Checks for known security vulnerabilities in imported modules | https://pypi.org/project/safety/ |
| codespell | - | Spellchecker | https://pypi.org/project/codespell/ |
| bandit | pyproject.toml | Checks your code for security issues | https://github.com/PyCQA/bandit |
| pytest | pytest.ini | Testing Framework | https://docs.pytest.org/ |

Some tools have a configuration settings. The table above indicates in which configuration file to look for settings for that tool. For example, the flake8 settings can be found in the `[flake8]` section of setup.cfg. Others can be find in pyproject.toml

In order to get all the tools to work well together the chosen line length must be consistent. You will see
a value of 120 appearing in multiple places in `pyproject.toml`, `setup.cfg` and also in the editor settings, for example, settings.json for vscode.

We also recommend using [mypy](http://www.mypy-lang.org/). The `pyproject.toml` contains a default configuration and mypy can be enabled in your editors settings (see below)

## .gitignore

The .gitignore is based off [Github standard python .gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore) with a few extra exclusions for files associated with code editors and operating systems.

### vscode

The file `vscode/settings.json` is an example configuration for vscode. To use these setting copy this file to `.vscode/settings,json`

The main features of this settings file are:
    - Enabling flake8 and disabling pylint
    - Autoformat on save (using the black and isort formatters)

Settings that you may want to change:
- Set the python path to your python in your venv with `python.defaultInterpreterPath`.
- Enable mypy by setting `python.linting.mypyEnabled` to true in settings.json.

