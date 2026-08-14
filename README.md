# Event Extraction

## Environment

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.

Python version: **3.11**

## Setup

After cloning the repository:

```bash
uv sync
```

Download the official BKEE processed dataset (CC BY-NC 4.0):

```bash
bash data/raw/BKEE/download.sh
```

This creates `train.json`, `dev.json`, and `test.json` under
`data/raw/BKEE/`. The upstream files use JSON Lines format even though their
extension is `.json`; the project loaders handle this automatically.

This will create the virtual environment and install all dependencies from pyproject.toml and uv.lock.

```bash
Run Python

```

Run Python inside the project environment with:

```bash
uv run python
```

Run a Python module:

```bash
uv run python -m src.adapter.MAVEN_adapter
```

## Import rules

Use absolute imports for code inside this project, beginning with `src.`. This
ensures imports work consistently when the code is run as a module.

```python
# Correct
from src.unified_format.event import Event

# Do not use relative imports for project modules
from .event import Event
```

Always run project code from the repository root with `uv run python -m`, using
the module path (file path without `.py`, with `/` replaced by `.`):

```bash
# src/adapter/MAVEN_adapter.py
uv run python -m src.adapter.MAVEN_adapter

# main.py
uv run python -m main
```
Add a dependency

If you need to install a new package:

```bash
uv add <package-name>
```

Example:

```bash
uv add pandas
```

Do not manually modify uv.lock.

After adding dependencies, commit both:

```bash
pyproject.toml
uv.lock
```
so other members can run:

```bash
uv sync
```

to update their environment.
