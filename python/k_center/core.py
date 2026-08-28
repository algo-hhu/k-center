"""k-center clustering estimator with a Rust (pyo3) backend.

The :class:`KCenter` class implements k-center clustering using the greedy
Gonzalez algorithm.  Computations are done by the compiled ``gonzalez``
Rust module, while this module provides a scikit-learn-compatible API.
"""

import numpy as np

from k_center import _k_center
from numbers import Integral
from typing import Optional, Sequence
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, validate_data


_DISTANCE_METRICS = ("euclidean", "manhattan", "chebyshev")


class KCenter(ClusterMixin, BaseEstimator):
    """k-center clustering using the greedy Gonzalez algorithm.

    The Gonzalez algorithm selects ``n_clusters`` centers one at a time.  The
    first center is chosen at random (seeded by ``random_state``) and each
    subsequent center is the point furthest from the centers chosen so far.
    This is an approximation with a 2-approximation guarantee for the k-center
    objective (minimize the maximum distance between a point and its nearest
    center).

    Parameters
    ----------
    algorithm : {'gonzalez'}, default='gonzalez'
        The clustering algorithm to run.  Only the greedy Gonzalez algorithm
        is currently implemented.

    n_clusters : int, default=3
        The number of clusters to form, and the number of centers to select.

    distance_metric : {'euclidean', 'manhattan', 'chebyshev'}, default='euclidean'
        The metric used to compute the distance between two points.

    random_state : int or None, default=42
        Seed used to select the first (random) center.  Pass ``None`` to
        deterministically start with the first sample instead.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Label of each sample, in ``range(n_clusters)``.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of the selected cluster centers.

    cluster_radii_ : ndarray of shape (n_clusters,)
        Radius of each cluster, i.e. the maximum distance from its center to
        any sample assigned to it.

    center_indices_ : ndarray of shape (n_clusters,)
        Row indices (into the input points passed to :meth:`fit`) of the
        selected centers.

    objective_radius_ : float
        The k-center objective value, i.e. the largest cluster radius.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> from k_center import KCenter
    >>> X = [[0.0], [1.0], [10.0], [11.0]]
    >>> model = KCenter(n_clusters=2, random_state=0)
    >>> model.fit(X)
    KCenter(n_clusters=2, random_state=0)
    >>> model.cluster_radii_
    array([1., 1.])
    """

    def __init__(
        self,
        algorithm: str = "gonzalez",
        n_clusters: int = 3,
        distance_metric: str = "euclidean",
        random_state: Optional[int] = 42,
    ) -> None:
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.random_state = random_state

    def fit(self, X: Sequence[Sequence[float]], y: Optional[Sequence] = None) -> "KCenter":
        """Run the Gonzalez algorithm on the given points.

        Selects ``n_clusters`` centers from the input points, assigns each
        point to its nearest center, and stores the outcome in the fitted
        attributes (``labels_``, ``cluster_centers_``, ``cluster_radii_``,
        ``center_indices_``, ``objective_radius_``).  After fitting,
        :meth:`predict` can assign new points to the nearest selected
        center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The points to cluster.  Sparse input is not supported.

        y : array-like of shape (n_samples,), default=None
            Ignored.  Present for API consistency with scikit-learn.

        Returns
        -------
        self : KCenter
            The fitted estimator.
        """
        data = self.validate_params(X)
        (
            objective_radius,
            labels,
            cluster_centers,
            cluster_radii,
            center_indices,
        ) = _k_center.fit(
            data,
            self.n_clusters,
            self.distance_metric,
            self.random_state,
        )

        self.objective_radius_ = float(objective_radius)
        self.labels_ = np.asarray(labels, dtype=np.int64)
        self.cluster_centers_ = np.asarray(cluster_centers, dtype=float)
        self.cluster_radii_ = np.asarray(cluster_radii, dtype=float)
        self.center_indices_ = np.asarray(center_indices, dtype=np.int64)
        self.n_features_in_ = data.shape[1]
        return self

    def fit_predict(self, X: Sequence[Sequence[float]], y: Optional[Sequence] = None) -> np.ndarray:
        """Run the Gonzalez algorithm and return the label of each input point.

        Equivalent to calling :meth:`fit` and then reading the ``labels_``
        attribute.  The label of a point is the index of the center that
        the point was assigned to during the run, so ``fit_predict(X)``
        returns one label per point in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The points to cluster.  Sparse input is not supported.

        y : array-like of shape (n_samples,), default=None
            Ignored.  Present for API consistency with scikit-learn.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            One label per input point: the index of the center each point
            was assigned to during the run.
        """
        return self.fit(X, y).labels_

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Assign each point to the closest of the previously selected centers.

        Intended for points that were not part of the :meth:`fit` input.
        Each point is assigned to the center in ``cluster_centers_`` that
        is closest under the configured ``distance_metric``.  Calling this
        on the points used in :meth:`fit` reproduces the labels returned by
        :meth:`fit_predict` (the training labels are recomputed here
        rather than reused from ``labels_``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The points to assign to cluster centers.  Sparse input is not
            supported.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            One label per input point: the index of the closest center.
        """
        check_is_fitted(self, attributes=["cluster_centers_"])
        data = self._to_2d_array(X, reset=False)
        labels = _k_center.predict(data, self.cluster_centers_, self.distance_metric)
        return np.asarray(labels, dtype=np.int64)

    def validate_params(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Validate hyper-parameters and input points.

        Checks the hyper-parameters against their allowed values, converts
        ``X`` to a validated float64 array and enforces that enough points
        are provided for the requested number of clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Candidate input points.

        Returns
        -------
        data : ndarray of shape (n_samples, n_features)
            Validated input points as a C-contiguous float64 array.

        Raises
        ------
        ValueError
            If a hyper-parameter is invalid, or ``X`` does not have at least
            ``n_clusters`` samples.
        """
        if self.algorithm != "gonzalez":
            raise ValueError("algorithm must be 'gonzalez'")
        if not isinstance(self.n_clusters, Integral) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if self.distance_metric not in _DISTANCE_METRICS:
            raise ValueError(
                "distance_metric must be 'euclidean', 'manhattan', or 'chebyshev'"
            )

        data = self._to_2d_array(X, reset=True)
        if data.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of input points")
        return data

    def _to_2d_array(
        self, X: Sequence[Sequence[float]], *, reset: bool
    ) -> np.ndarray:
        """Validate and convert input to a dense, finite, C-contiguous float64 array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Candidate input data.

        reset : bool
            If True (e.g. from :meth:`fit`), set the ``n_features_in_``
            attribute.  If False (e.g. from :meth:`predict`), check that the
            number of features is consistent with ``n_features_in_``.
        """
        return np.ascontiguousarray(
            validate_data(
                self,
                X,
                dtype=np.float64,
                accept_sparse=False,
                ensure_2d=True,
                ensure_all_finite=True,
                reset=reset,
            )
        )