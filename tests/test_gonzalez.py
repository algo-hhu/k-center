import unittest

import numpy as np
from k_center import KCenter


class GonzalezTests(unittest.TestCase):
    def test_fit_returns_expected_attributes(self) -> None:
        model = KCenter(n_clusters=2, random_state=None)
        model.fit([[0.0], [10.0], [11.0]])

        self.assertEqual(model.labels_.shape, (3,))
        self.assertEqual(model.cluster_centers_.shape, (2, 1))
        self.assertEqual(model.cluster_radii_.shape, (2,))
        self.assertAlmostEqual(model.objective_radius_, 1.0)
        self.assertTrue(np.array_equal(model.center_indices_, np.array([0, 2])))

    def test_fit_supports_chebyshev_distance(self) -> None:
        model = KCenter(n_clusters=2, distance_metric="chebyshev", random_state=None)
        model.fit([[0.0], [10.0], [11.0]])

        self.assertEqual(model.labels_.shape, (3,))
        self.assertAlmostEqual(model.objective_radius_, 1.0)
        self.assertTrue(np.array_equal(model.center_indices_, np.array([0, 2])))

    def test_fit_rejects_invalid_distance_metric(self) -> None:
        model = KCenter(n_clusters=2, distance_metric="minkowski")

        with self.assertRaises(ValueError):
            model.fit([[0.0], [10.0], [11.0]])

    def test_predict_uses_fitted_centers(self) -> None:
        model = KCenter(n_clusters=2, random_state=None)
        model.fit([[0.0], [10.0], [11.0]])

        predictions = model.predict([[0.2], [10.4]])
        self.assertTrue(np.array_equal(predictions, np.array([0, 1])))

    def test_predict_uses_chebyshev_distance(self) -> None:
        model = KCenter(n_clusters=2, distance_metric="chebyshev", random_state=None)
        model.fit([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0]])

        predictions = model.predict([[9.0, 9.0], [0.0, 0.0]])
        self.assertTrue(np.array_equal(predictions, np.array([1, 0])))

    def test_predict_uses_manhattan_distance(self) -> None:
        model = KCenter(n_clusters=2, distance_metric="manhattan", random_state=None)
        model.fit([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0]])

        predictions = model.predict([[9.0, 9.0], [0.0, 0.0]])
        self.assertTrue(np.array_equal(predictions, np.array([1, 0])))

    def test_predict_on_training_points_matches_labels(self) -> None:
        model = KCenter(n_clusters=2, random_state=None)
        X = [[0.0], [10.0], [11.0]]
        model.fit(X)

        predictions = model.predict(X)
        self.assertTrue(np.array_equal(predictions, model.labels_))

    def test_predict_rejects_feature_mismatch(self) -> None:
        model = KCenter(n_clusters=2, random_state=None)
        model.fit([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0]])

        with self.assertRaises(ValueError):
            model.predict([[0.0]])

    def test_random_state_selects_initial_center(self) -> None:
        model = KCenter(n_clusters=2, random_state=42)
        model.fit([[0.0], [10.0], [11.0]])

        first_center = model.center_indices_[0]
        self.assertIn(first_center, [0, 1, 2])

    def test_same_random_state_is_reproducible(self) -> None:
        first = KCenter(n_clusters=2, random_state=42)
        first.fit([[0.0], [10.0], [11.0]])
        second = KCenter(n_clusters=2, random_state=42)
        second.fit([[0.0], [10.0], [11.0]])

        self.assertTrue(np.array_equal(first.center_indices_, second.center_indices_))
        self.assertAlmostEqual(first.objective_radius_, second.objective_radius_)

    def test_validation_rejects_invalid_cluster_count(self) -> None:
        model = KCenter(n_clusters=4)
        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
