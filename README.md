# k-center Clustering

This is a mixed Rust/Python project: the algorithms are implemented in Rust and
exposed to Python through [PyO3](https://pyo3.rs) and
[maturin](https://maturin.rs).


**Rust is only needed to build the project, not to use it.** The published
PyPI wheel bundles the pre-compiled Rust code together with the Python
package, so end users just run
```
pip install k-center
```


## Algorithm

The library solves the k-center clustering problem: given $n$ points,
pick $k$ cluster centers (chosen from the input points) so as to minimize
the objective radius, i.e. the maximum distance between any point and
its nearest center. This is an NP-hard problem in general, so the library
provides approximation algorithms.

Currently only the greedy Gonzalez algorithm is implemented
(`algorithm="gonzalez"`). The Gonzalez algorithm runs in $\mathcal{O}(k n d)$ time for $n$ points with $d$ dimensions and guarantees a 2-approximation: the resulting objective radius is at most twice the optimal value.

Supported distance metrics are `euclidean`, `manhattan`, and `chebyshev`.

## Usage

The `KCenter` estimator follows the scikit-learn API (`fit` / `predict`).

```python
import numpy as np
from k_center import KCenter

X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])

model = KCenter(n_clusters=2, distance_metric="euclidean", random_state=42)
model.fit(X)

model.labels_               # array([0, 0, 1, 1]) - cluster of each point
model.cluster_centers_      # coordinates of the two chosen centers
model.cluster_radii_        # radius of each cluster
model.objective_radius_     # the k-center objective (largest cluster radius)
model.center_indices_       # row indices of the chosen centers in X

# Assign new points to the nearest previously chosen center
model.predict([[5.0, 5.0]])
```


## Project layout

```
├── Cargo.toml          # Rust crate (compiled to k_center._k_center)
│                       #   also the single source of truth for the package version
├── pyproject.toml      # Python package metadata + maturin config
│                       #   (`dynamic = ["version"]` pulls the version from Cargo.toml)
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
## Development
### Prerequisites

- Rust toolchain (`cargo`), e.g. via [rustup](https://rustup.rs)
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (for the Python dev environment)

### Setup

```bash
uv sync
```

`uv sync` installs packages into `.venv/` but does not activate it. Run
maturin/pytest through `uv run` (e.g. `uv run maturin build`) rather than
relying on PATH.

### Build and install

Build the wheel:

```bash
uv run maturin build
```

Or install directly into the current venv for development:

```bash
uv run maturin develop
```

### Run the tests

Rust unit tests (the `#[cfg(test)]` blocks inside `src/`):

```bash
cargo test
```

Python tests (pytest):

```bash
uv run pytest tests/
```

Rust and Python developer dependencies are managed separately: Cargo
dependencies live in `Cargo.toml` `[dependencies]`, while Python dev tools
(maturin, pytest) live in the `[dependency-groups]` `dev` group of
`pyproject.toml`. Runtime Python dependencies (e.g. numpy, scikit-learn) go
into `[project] dependencies` and are only recorded in the wheel metadata.
