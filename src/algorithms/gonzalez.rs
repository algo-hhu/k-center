use pyo3::prelude::*;
use distances::vectors::{chebyshev, euclidean, manhattan};
use ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

#[derive(Clone, Copy)]
enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

impl DistanceMetric {
    fn parse(metric: &str) -> PyResult<Self> {
        match metric {
            "euclidean" => Ok(Self::Euclidean),
            "manhattan" => Ok(Self::Manhattan),
            "chebyshev" => Ok(Self::Chebyshev),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "distance_metric must be 'euclidean', 'manhattan', or 'chebyshev'",
            )),
        }
    }

    fn distance(self, left: &[f64], right: &[f64]) -> f64 {
        match self {
            Self::Euclidean => euclidean(left, right),
            Self::Manhattan => manhattan(left, right),
            Self::Chebyshev => chebyshev(left, right),
        }
    }
}

#[pyfunction]
pub fn fit(
    _py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    n_clusters: usize,
    distance_metric: &str,
    random_state: Option<i64>,
) -> PyResult<(f64, Vec<usize>, Vec<Vec<f64>>, Vec<f64>, Vec<usize>)> {
    let metric = DistanceMetric::parse(distance_metric)?;
    let points = points.as_array();

    gonzalez_impl(points, n_clusters, metric, random_state)
}

#[pyfunction]
pub fn predict(
    _py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    centers: PyReadonlyArray2<'_, f64>,
    distance_metric: &str,
) -> PyResult<Vec<usize>> {
    let metric = DistanceMetric::parse(distance_metric)?;
    let points = points.as_array();
    let centers = centers.as_array();

    if points.nrows() == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data must contain at least one point",
        ));
    }
    if centers.nrows() == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "centers must contain at least one cluster center",
        ));
    }
    if points.ncols() != centers.ncols() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "points and centers must have the same number of features",
        ));
    }

    Ok(assign_to_nearest_centers(points, centers, metric))
}

fn row_distance(
    metric: DistanceMetric,
    left: ndarray::ArrayView1<'_, f64>,
    right: ndarray::ArrayView1<'_, f64>,
) -> f64 {
    metric.distance(
        left.as_slice().expect("contiguous row"),
        right.as_slice().expect("contiguous row"),
    )
}

fn initial_center_index(random_state: Option<i64>, n_points: usize) -> usize {
    match random_state {
        None => 0usize, // default to the first point when no seed is given
        Some(seed) => {
            // Uses PCG random number generator (XSL RR 128/64 (LCG) variant).
            let mut rng = Pcg64::seed_from_u64(seed as u64);
            rng.random_range(0..n_points)
        }
    }
}

fn assign_to_nearest_centers(
    points: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    metric: DistanceMetric,
) -> Vec<usize> {
    let n_centers = centers.nrows();
    let mut labels = Vec::with_capacity(points.nrows());

    for point_index in 0..points.nrows() {
        let point = points.row(point_index);
        let mut best_center = 0usize;
        let mut best_distance = f64::INFINITY;
        for center_index in 0..n_centers {
            let distance = row_distance(metric, point, centers.row(center_index));
            if distance < best_distance {
                best_distance = distance;
                best_center = center_index;
            }
        }
        labels.push(best_center);
    }
    labels
}

fn gonzalez_impl(
    points: ArrayView2<'_, f64>,
    n_clusters: usize,
    metric: DistanceMetric,
    random_state: Option<i64>,
) -> PyResult<(f64, Vec<usize>, Vec<Vec<f64>>, Vec<f64>, Vec<usize>)> {
    let n_points = points.nrows();
    let n_features = points.ncols();

    if n_points == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data must contain at least one point",
        ));
    }
    if n_clusters == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_clusters must be a positive integer",
        ));
    }
    if n_clusters > n_points {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_clusters cannot exceed the number of input points",
        ));
    }

    let initial_center = initial_center_index(random_state, n_points);

    // Initialize
    let mut center_indices = Vec::with_capacity(n_clusters);
    let mut is_center = vec![false; n_points];
    let mut nearest_distances = vec![f64::INFINITY; n_points];
    let mut labels = vec![0usize; n_points];

    center_indices.push(initial_center);
    is_center[initial_center] = true;

    // Compute initial distances from the first center
    let initial_center_row = points.row(initial_center);
    for point_index in 0..n_points {
        nearest_distances[point_index] = row_distance(
            metric,
            points.row(point_index),
            initial_center_row,
        );
    }

    while center_indices.len() < n_clusters {

        let mut next_center = 0usize;
        let mut next_center_distance = f64::NEG_INFINITY;
        for point_index in 0..n_points {
            if is_center[point_index] {
                continue;
            }
            let distance = nearest_distances[point_index];
            if distance > next_center_distance {
                next_center = point_index;
                next_center_distance = distance;
            }
        }

        center_indices.push(next_center);
        is_center[next_center] = true;

        let current_center_id = center_indices.len() - 1;
        nearest_distances[next_center] = 0.0f64;
        labels[next_center] = current_center_id;

        let next_center_row = points.row(next_center);
        for point_index in 0..n_points {
            if is_center[point_index] {
                continue;
            }
            let candidate_distance = row_distance(
                metric,
                points.row(point_index),
                next_center_row,
            );
            if candidate_distance < nearest_distances[point_index] {
                nearest_distances[point_index] = candidate_distance;
                labels[point_index] = current_center_id;
            }
        }
    }

    let mut cluster_radii = vec![0.0f64; n_clusters];
    for (point_index, &label) in labels.iter().enumerate() {
        let radius = nearest_distances[point_index];
        if radius > cluster_radii[label] {
            cluster_radii[label] = radius;
        }
    }

    let objective_radius = cluster_radii.iter().copied().fold(0.0f64, f64::max);
    let mut cluster_centers = Vec::with_capacity(n_clusters);
    for &center_index in &center_indices {
        let mut row = Vec::with_capacity(n_features);
        row.extend(points.row(center_index).iter().copied());
        cluster_centers.push(row);
    }

    Ok((objective_radius, labels, cluster_centers, cluster_radii, center_indices))
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayView2;

    use super::{
        DistanceMetric, assign_to_nearest_centers, gonzalez_impl, initial_center_index,
    };

    fn view_2d(rows: usize, cols: usize, data: &[f64]) -> ArrayView2<'_, f64> {
        ArrayView2::from_shape((rows, cols), data).unwrap()
    }

    #[test]
    fn parse_accepts_supported_metrics() {
        for metric in ["euclidean", "manhattan", "chebyshev"] {
            assert!(DistanceMetric::parse(metric).is_ok());
        }
    }

    #[test]
    fn parse_rejects_unknown_metric() {
        assert!(DistanceMetric::parse("minkowski").is_err());
    }

    #[test]
    fn euclidean_distance() {
        let metric = DistanceMetric::Euclidean;
        assert!(metric.distance(&[0.0, 0.0], &[3.0, 4.0]) == 5.0);
        assert!(metric.distance(&[1.0, 2.0], &[1.0, 2.0]) == 0.0);
    }

    #[test]
    fn manhattan_distance() {
        let metric = DistanceMetric::Manhattan;
        assert!(metric.distance(&[0.0, 0.0], &[3.0, 4.0]) == 7.0);
    }

    #[test]
    fn chebyshev_distance() {
        let metric = DistanceMetric::Chebyshev;
        assert!(metric.distance(&[0.0, 0.0], &[3.0, 4.0]) == 4.0);
    }

    #[test]
    fn initial_center_defaults_to_first_point() {
        assert!(initial_center_index(None, 5) == 0);
    }

    #[test]
    fn initial_center_with_seed_is_in_range() {
        for _ in 0..20 {
            let index = initial_center_index(Some(42), 5);
            assert!(index < 5);
        }
    }

    #[test]
    fn gonzalez_deterministic_run() {
        let points = view_2d(3, 1, &[0.0, 10.0, 11.0]);
        let (objective, labels, centers, radii, center_indices) =
            gonzalez_impl(points, 2, DistanceMetric::Euclidean, None).unwrap();

        assert!(objective == 1.0);
        assert!(labels == vec![0, 1, 1]);
        assert!(centers == vec![vec![0.0], vec![11.0]]);
        assert!(radii == vec![0.0, 1.0]);
        assert!(center_indices == vec![0, 2]);
    }

    #[test]
    fn assign_to_nearest_centers_euclidean() {
        let points = view_2d(2, 1, &[0.0, 10.4]);
        let centers = view_2d(2, 1, &[0.0, 10.0]);
        let labels = assign_to_nearest_centers(
            points, centers, DistanceMetric::Euclidean,
        );
        assert!(labels == vec![0, 1]);
    }

    #[test]
    fn assign_to_nearest_center_breaks_ties_to_first() {
        let points = view_2d(1, 1, &[1.0]);
        let centers = view_2d(2, 1, &[0.0, 2.0]);
        let labels = assign_to_nearest_centers(
            points, centers, DistanceMetric::Euclidean,
        );
        assert!(labels == vec![0]);
    }

    #[test]
    fn assign_to_nearest_center_manhattan() {
        let points = view_2d(2, 2, &[9.0, 9.0, 0.0, 0.0]);
        let centers = view_2d(2, 2, &[0.0, 0.0, 10.0, 0.0]);
        let labels = assign_to_nearest_centers(
            points, centers, DistanceMetric::Manhattan,
        );
        assert!(labels == vec![1, 0]);
    }

    #[test]
    fn assign_to_nearest_center_chebyshev() {
        let points = view_2d(2, 2, &[9.0, 9.0, 0.0, 0.0]);
        let centers = view_2d(2, 2, &[0.0, 0.0, 10.0, 10.0]);
        let labels = assign_to_nearest_centers(
            points, centers, DistanceMetric::Chebyshev,
        );
        assert!(labels == vec![1, 0]);
    }
}
