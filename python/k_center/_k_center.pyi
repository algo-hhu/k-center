"""Type stubs for the compiled ``k_center._k_center`` Rust extension module."""

from typing import List, Optional, Tuple

import numpy as np

def fit(
    points: np.ndarray,
    n_clusters: int,
    distance_metric: str,
    random_state: Optional[int],
) -> Tuple[float, List[int], List[List[float]], List[float], List[int]]: ...
def predict(
    points: np.ndarray,
    centers: np.ndarray,
    distance_metric: str,
) -> List[int]: ...
