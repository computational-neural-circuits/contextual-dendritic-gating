# Assembly-based computations through contextual dendritic gating of plasticity 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-Ruff-261230.svg)](https://github.com/astral-sh/ruff)

Brian2 simulations that accompany the paper "Assembly-based computations through contextual dendritic gating of plasticity". The repository contains the simulation code used to generate the paper figures and the accompanying model specifications.

## Repository layout

- `src/` — Brian2 network models, analysis helpers, and model specifications.
- `scripts/` — scripts for Figures 2–8 and supplementary figures.
- `scripts/matlab/` — MATLAB plotting and supplementary simulation scripts.
- `results/figures/` — versioned figure PDFs, including both the panels
  produced by the figure scripts and the composite figures used in the
  publication. See [`results/figures/README.md`](results/figures/README.md) for
  a per-figure description of the Python and MATLAB scripts and the PDFs they
  generate.
- `results/` — simulation caches, source data, and intermediates that are
  downloaded separately or regenerated locally.
- `results/matlab/` — local MATLAB inputs organized by figure.
- `stored_networks/` — precomputed network states used by figure scripts in
  load-only mode.
- `plots_style.txt` — shared Matplotlib style used by the figure scripts.

## Simulation infrastructure

All network models inherit from the `HandleParametersAndResults` class in
`src/handle_parameters_and_results.py`, the core of the codebase. It loads the
model parameters and equations from `src/model_specs/`, applies per-run
overrides, and hashes the full parameter and equation set into a unique key.
Simulation results are cached as HDF5 files in `results/sim_files/` under that
key, so a script reuses existing results whenever the parameters match and only
re-simulates when they change; with `only_load_results` set, it loads the cached
results directly instead of building a network.

Full Brian2 network states are also saved in `stored_networks/`. Each snapshot
is identified by a filename derived from the same parameter hash, so the network
matching a given parameter set can be located and restored later. This supports a
common workflow in which an expensive learning phase (for example, imprinting
assemblies) is run once and its complete network state is saved; later stages
such as recall, pattern completion, or additional imprints then look up that
snapshot by its key and continue from it, rather than repeating the training
every time. The stored-network archive on Zenodo provides these snapshots so that
such analyses can be reproduced without re-running the learning phase.

## Requirements

- Python 3.10
- A C++ compiler for Brian2 code generation:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Debian/Ubuntu: `sudo apt install build-essential`
- Conda or a Python virtual environment

The optional Figure 6 dependencies include PyTorch and Torchvision. Install
them only when reproducing that figure, using wheels appropriate for your
platform. On first use, Figure 6 downloads EMNIST to
`results/datasets/EMNIST/`; this local cache is excluded from Git.

## Installation

### Conda

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
conda env create -f environment.yml
conda activate contextual-dendritic-gating
python -m pip install -e .
```

Install dependencies required only by selected figures:

```bash
python -m pip install -e ".[figures]"
```

### Pip

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproducing figures

Run figure scripts from `scripts/`; their relative imports, style path, and
output paths depend on that working directory:

```bash
cd scripts
python Fig_2.py
```

The available main scripts are `Fig_2.py` through `Fig_8.py`; supplementary
scripts use the `Fig_S*.py` naming pattern.

Many scripts default to loading cached simulation results. Final figure PDFs in
`results/figures/` are versioned in Git; simulation caches, MATLAB inputs,
EMNIST data, intermediate outputs, and stored network states are not.

We recommend downloading the cached results from Zenodo:

```text
https://doi.org/10.5281/zenodo.21299329
```

Using the cached results lets you regenerate every figure directly from the
published data without re-running the simulations, which are computationally
expensive and can take many hours.

The Zenodo record provides two archives: a zipped copy of the `results/` folder
and a zipped copy of the `stored_networks/` folder. Download both and extract
them at the repository root so that they recreate the `results/` and
`stored_networks/` folders in place (merging with the existing directory
structure). After extraction, `results/` holds the cached simulation outputs
(`sim_files/`), MATLAB inputs (`matlab/`), and figure PDFs, while
`stored_networks/` holds the precomputed network states in their per-figure
subdirectories (for example, `stored_networks/Fig_3/`).

Without these files, configure the relevant script to re-run its simulation
from scratch. See [`results/figures/README.md`](results/figures/README.md) for
the full figure-PDF inventory and a per-figure description of each script.

### MATLAB figure scripts

Selected figure panels and the MATLAB-only simulations for Figures S4 and S5
are available in `scripts/matlab/`. Their corresponding local inputs are in
`results/matlab/`, grouped by figure. MATLAB plotting scripts locate those
inputs from their own file locations and can therefore run from any working
directory.

## Development

Install development tools:

```bash
python -m pip install -e ".[dev]"
```

Format and lint the maintained Python code:

```bash
black src scripts
ruff check src scripts
```

Ruff intentionally ignores `F403` and `F405`: Brian2 unit names are provided
through `from brian2.units import *`, which is retained to preserve simulation
behavior. The committed Ruff configuration checks unused imports and bindings
(`F401` and `F841`) without reporting those known wildcard-import diagnostics.

For a reproducible environment, use the committed `environment.yml` rather
than exporting a local Conda environment. Maintainers can generate an exact
lock file from a clean environment with `pip freeze`, excluding the editable
local project entry.

## License

Copyright (C) 2026 Sebastian Onasch, Christoph Miehl & M. Maurycy Miekus.

This project is licensed under the [GNU General Public License v3.0](LICENSE).
