"""Stage 4 — meta-feature extraction.

The emphasis here is on the properties the similarity search depends on, rather than on
re-asserting arithmetic that pandas already guarantees:

* every PRD-named feature is present, and the vector's order is fixed;
* the same frame produces the same vector, twice, and column order does not change it;
* degenerate frames return a vector rather than raising, because a corpus builder that
  died on one odd dataset out of a hundred would be useless;
* the values are right on a frame whose answers can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import metafeatures as mf


@pytest.fixture
def simple() -> pd.DataFrame:
    """A frame whose every meta-feature can be computed on paper.

    6 rows. Features: ``num`` (numeric, one blank), ``other`` (numeric),
    ``cat`` (3 distinct), ``flag`` (2 distinct). Target ``y``: four 0s, two 1s.
    """
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "other": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "cat": ["a", "b", "c", "a", "b", "c"],
            "flag": ["y", "n", "y", "n", "y", "n"],
            "y": [0, 0, 0, 0, 1, 1],
        }
    )


# --------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------


def test_every_prd_named_feature_is_present(simple):
    """PRD.md §10 names eight meta-features. All eight are produced."""
    features = mf.extract(simple, "y", "binary classification")
    missing = [name for name in mf.PRD_FIELDS if name not in features]
    assert missing == [], f"PRD-named features absent: {missing}"


def test_vector_length_matches_the_declared_field_order(simple):
    features = mf.extract(simple, "y", "binary classification")
    assert mf.as_vector(features).shape == (len(mf.VECTOR_FIELDS),)


def test_every_vector_field_is_finite(simple):
    """A NaN coordinate makes every distance involving the record NaN."""
    vector = mf.as_vector(mf.extract(simple, "y", "binary classification"))
    assert np.isfinite(vector).all()


def test_version_is_stamped_on_every_extraction(simple):
    features = mf.extract(simple, "y", "binary classification")
    assert features["metafeature_version"] == mf.METAFEATURE_VERSION


def test_as_vector_refuses_an_incomplete_record():
    """A zero-filled coordinate is a wrong distance that looks like a right one."""
    with pytest.raises(ValueError, match="missing"):
        mf.as_vector({"log_n_rows": 1.0})


# --------------------------------------------------------------------------------------
# Determinism — the property the whole stage rests on
# --------------------------------------------------------------------------------------


def test_extraction_is_deterministic(simple):
    first = mf.as_vector(mf.extract(simple, "y", "binary classification"))
    second = mf.as_vector(mf.extract(simple, "y", "binary classification"))
    assert np.array_equal(first, second)


def test_column_order_does_not_change_the_vector(simple):
    """Two people exporting the same data with columns in a different order must land
    in the same place in meta-feature space."""
    shuffled = simple[["flag", "y", "other", "cat", "num"]]
    assert np.allclose(
        mf.as_vector(mf.extract(simple, "y", "binary classification")),
        mf.as_vector(mf.extract(shuffled, "y", "binary classification")),
    )


def test_row_order_does_not_change_the_vector(simple):
    reversed_rows = simple.iloc[::-1].reset_index(drop=True)
    assert np.allclose(
        mf.as_vector(mf.extract(simple, "y", "binary classification")),
        mf.as_vector(mf.extract(reversed_rows, "y", "binary classification")),
    )


# --------------------------------------------------------------------------------------
# Values, computed by hand
# --------------------------------------------------------------------------------------


def test_counts_exclude_the_target(simple):
    features = mf.extract(simple, "y", "binary classification")
    assert features["n_rows"] == 6
    assert features["n_cols"] == 4          # num, other, cat, flag — not y
    assert features["n_numerical"] == 2     # num, other
    assert features["n_categorical"] == 2   # cat, flag


def test_sparsity_is_measured_over_feature_cells(simple):
    """One blank among 6 rows x 4 feature columns."""
    features = mf.extract(simple, "y", "binary classification")
    assert features["sparsity"] == pytest.approx(1 / 24)


def test_imbalance_ratio_is_majority_over_minority(simple):
    """Four 0s and two 1s."""
    features = mf.extract(simple, "y", "binary classification")
    assert features["imbalance_ratio"] == pytest.approx(2.0)


def test_n_classes_counts_distinct_target_values(simple):
    assert mf.extract(simple, "y", "binary classification")["n_classes"] == 2


def test_cardinality_covers_non_numeric_columns_only(simple):
    """cat has 3 distinct, flag has 2 — mean 2.5, max 3. The numeric columns, whose
    distinct count is a property of precision rather than shape, are not counted."""
    features = mf.extract(simple, "y", "binary classification")
    assert features["mean_cardinality"] == pytest.approx(2.5)
    assert features["max_cardinality"] == pytest.approx(3.0)


def test_mean_abs_correlation_excludes_the_diagonal():
    """Two perfectly correlated columns give 1.0, not the 1.0-inflated-by-the-diagonal
    value that including self-correlation would produce."""
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "y": [0, 1, 0, 1]})
    features = mf.extract(frame, "y", "binary classification")
    assert features["mean_abs_correlation"] == pytest.approx(1.0)


def test_log_dimensions_are_base_ten(simple):
    features = mf.extract(simple, "y", "binary classification")
    assert features["log_n_rows"] == pytest.approx(np.log10(6))
    assert features["log_n_cols"] == pytest.approx(np.log10(4))


def test_duplicate_share_is_measured(simple):
    doubled = pd.concat([simple, simple], ignore_index=True)
    assert mf.extract(doubled, "y", "binary classification")["duplicate_share"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Task handling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task", ["binary classification", "multiclass classification", "classification"]
)
def test_classification_tasks_are_recognised(simple, task):
    features = mf.extract(simple, "y", task)
    assert features["is_classification"] == 1.0
    assert features["is_regression"] == 0.0


def test_regression_reports_no_classes_and_even_balance():
    """A continuous target has no classes to count and no imbalance to report. 1.0 is
    the value meaning "even", not a sentinel the distance metric would misread."""
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "y": [1.5, 2.5, 3.5, 4.5, 5.5]})
    features = mf.extract(frame, "y", "regression")
    assert features["is_regression"] == 1.0
    assert features["n_classes"] == 0
    assert features["imbalance_ratio"] == pytest.approx(1.0)


def test_task_label_is_carried_through_verbatim(simple):
    """`target_type` is one of the eight PRD features and is the product's own label."""
    assert mf.extract(simple, "y", "binary classification")["target_type"] == "binary classification"


# --------------------------------------------------------------------------------------
# Degenerate frames — a corpus builder must survive these
# --------------------------------------------------------------------------------------


def test_single_feature_column():
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "y": [0, 1, 0, 1]})
    features = mf.extract(frame, "y", "binary classification")
    assert features["n_cols"] == 1
    # One column means no pair to correlate. 0.0 is "nothing measured", not a claim.
    assert features["mean_abs_correlation"] == 0.0
    assert np.isfinite(mf.as_vector(features)).all()


def test_all_categorical_features():
    frame = pd.DataFrame({"a": list("abab"), "b": list("ccdd"), "y": [0, 1, 0, 1]})
    features = mf.extract(frame, "y", "binary classification")
    assert features["n_numerical"] == 0
    assert features["categorical_ratio"] == pytest.approx(1.0)
    assert features["mean_abs_skew"] == 0.0
    assert np.isfinite(mf.as_vector(features)).all()


def test_constant_column_does_not_poison_correlation():
    """pandas returns NaN for a constant column's correlations. Those pairs are dropped
    rather than counted as zero, which would be an unmeasured claim of independence."""
    frame = pd.DataFrame(
        {"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "k": [7.0, 7, 7, 7], "y": [0, 1, 0, 1]}
    )
    features = mf.extract(frame, "y", "binary classification")
    assert features["mean_abs_correlation"] == pytest.approx(1.0)
    assert np.isfinite(mf.as_vector(features)).all()


def test_fully_blank_column():
    frame = pd.DataFrame(
        {"a": [1.0, 2, 3, 4], "empty": [None] * 4, "y": [0, 1, 0, 1]}
    )
    features = mf.extract(frame, "y", "binary classification")
    assert features["sparsity"] == pytest.approx(0.5)
    assert np.isfinite(mf.as_vector(features)).all()


def test_single_class_target_reports_even_balance():
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "y": [1, 1, 1, 1]})
    assert mf.extract(frame, "y", "binary classification")["imbalance_ratio"] == pytest.approx(1.0)


def test_target_absent_from_the_frame():
    """The corpus builder can meet a record whose named target was dropped upstream."""
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8]})
    features = mf.extract(frame, "missing_target", "binary classification")
    assert features["n_cols"] == 2
    assert features["n_classes"] == 0
    assert np.isfinite(mf.as_vector(features)).all()
