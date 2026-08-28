"""scikit-learn estimator compliance tests.

These tests run the official scikit-learn ``check_estimator`` suite against
``KCenter`` so that any future API or validation regression is caught early.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks
from k_center import KCenter


@parametrize_with_checks([KCenter()])
def test_sklearn_compliance(estimator, check):
    """Run the full scikit-learn estimator checks on KCenter."""
    check(estimator)
