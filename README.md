# k-center

A selection of implementations for the k-center (and k-supplier) problem.

This is a mixed Rust/Python project: the algorithms are implemented in Rust and
exposed to Python through [PyO3](https://pyo3.rs) and
[maturin](https://maturin.rs).

> **Rust is only needed to build the project, not to use it.** The published
> PyPI wheel bundles the pre-compiled Rust code together with the Python
> package, so end users just run `pip install k-center` — no Rust toolchain
> required. Rust (and `cargo`) is only needed by contributors/CI to compile the
> project from source.

## Project layout

```
├── Cargo.toml          # Rust crate (compiled to k_center._k_center)
├── pyproject.toml      # Python package metadata + maturin config
├── src/                # Rust source
│   ├── lib.rs          # crate root; registers the Python module
│   └── algorithms/     # per-algorithm Rust modules
│       ├── mod.rs
│       └── gonzalez.rs
├── python/k_center/    # pure-Python package (sklearn estimators etc.)
│   ├── __init__.py
│   └── core.py
└── tests/              # Python tests (pytest)
```

## Prerequisites

- Rust toolchain (`cargo`), e.g. via [rustup](https://rustup.rs)
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (for the Python dev environment)

## Setup

```bash
uv sync                 # installs the dev group (maturin, pytest) into .venv
```

> `uv sync` installs packages into `.venv/` but does not activate it. Run
> maturin/pytest through `uv run` (e.g. `uv run maturin build`) rather than
> relying on PATH.

## Build and install

Build the wheel:

```bash
uv run maturin build
```

Or install directly into the current venv for development:

```bash
uv run maturin develop
```

## Run the tests

Rust unit tests (the `#[cfg(test)]` blocks inside `src/`):

```bash
cargo test
```

Python tests (pytest):

```bash
uv run pytest tests/
```

> Rust and Python developer dependencies are managed separately: Cargo
> dependencies live in `Cargo.toml` `[dependencies]`, while Python dev tools
> (maturin, pytest) live in the `[dependency-groups]` `dev` group of
> `pyproject.toml`. Runtime Python dependencies (e.g. numpy, scikit-learn) go
> into `[project] dependencies` and are only recorded in the wheel metadata.


