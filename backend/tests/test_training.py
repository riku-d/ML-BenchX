"""Stages 5 and 6 — the guarantees the leaderboard and the explanation rest on.

Each test here pins a claim the interface makes to the user. The claims are about
honesty as much as correctness: that a score was measured on rows no transformer saw,
that a model which did not run has a reason rather than a zero, that concurrency is
measured rather than asserted, and that permutation importance is never called SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import explainability, training


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _classification(n: int = 300, features: int = 6, seed: int = 0):
    """A separable-ish binary problem, already encoded — the shape a run stores."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, features)),
        columns=[f"f{i}" for i in range(features)],
    )
    y = pd.Series((X["f0"] + X["f1"] * 0.5 + rng.normal(0, 0.3, n) > 0).astype(int), name="y")
    return X, y


def _regression(n: int = 300, features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, features)),
        columns=[f"f{i}" for i in range(features)],
    )
    y = pd.Series(X["f0"] * 3 + X["f1"] + rng.normal(0, 0.4, n), name="y")
    return X, y


def _split(X, y, at: int = 220):
    return X.iloc[:at], y.iloc[:at], X.iloc[at:], y.iloc[at:]


# --------------------------------------------------------------------------------------
# Six families, really trained
# --------------------------------------------------------------------------------------


def test_all_six_families_are_submitted_and_scored() -> None:
    """Six rows, one per family, whatever happened to each.

    The leaderboard must account for every candidate. A table that silently omits the
    family that failed lets a reader believe only five were ever considered.
    """
    X, y = _classification()
    X_train, y_train, X_test, y_test = _split(X, y)

    report = training.train_all(X_train, y_train, X_test, y_test, "binary classification")

    assert len(report.models) == 6
    assert {row["family"] for row in report.models} == set(training.FAMILIES)
    assert report.best_family is not None
    assert all(row["status"] in {"succeeded", "failed", "skipped"} for row in report.models)


def test_scores_are_measured_on_the_held_out_fold_not_the_training_fold() -> None:
    """The metric must change when the held-out rows change.

    This is the test that would fail if someone scored on `X_train` for convenience.
    Two different test folds against one training fold produce two different numbers;
    a model scored on its own training data would produce the same number twice.
    """
    X, y = _classification(n=400)
    X_train, y_train = X.iloc[:240], y.iloc[:240]

    first = training.train_all(
        X_train, y_train, X.iloc[240:320], y.iloc[240:320], "binary classification"
    )
    second = training.train_all(
        X_train, y_train, X.iloc[320:], y.iloc[320:], "binary classification"
    )

    def value(report, family):
        for row in report.models:
            if row["family"] == family and row["status"] == "succeeded":
                return row["primary_metric_value"]
        return None

    differences = [
        (value(first, f), value(second, f))
        for f in training.FAMILIES
        if value(first, f) is not None and value(second, f) is not None
    ]
    assert differences, "no family scored on either fold"
    assert any(a != b for a, b in differences), (
        "every family scored identically on two different held-out folds, which means "
        "the metric is not being measured on the held-out fold at all"
    )


def test_a_dummy_baseline_is_scored_on_the_same_fold() -> None:
    """Without it the leaderboard has a winner and no sense of whether winning meant
    anything."""
    X, y = _classification()
    X_train, y_train, X_test, y_test = _split(X, y)

    report = training.train_all(X_train, y_train, X_test, y_test, "binary classification")

    assert report.baseline["status"] == "succeeded"
    assert report.primary_metric in report.baseline["metrics"]
    winners = [r for r in report.models if r["status"] == "succeeded"]
    assert all("beats_baseline" in row for row in winners)


def test_regression_is_scored_with_regression_metrics() -> None:
    X, y = _regression()
    X_train, y_train, X_test, y_test = _split(X, y)

    report = training.train_all(X_train, y_train, X_test, y_test, "regression")

    assert report.primary_metric == "r2"
    succeeded = [r for r in report.models if r["status"] == "succeeded"]
    assert succeeded
    for row in succeeded:
        assert {"rmse", "mae", "r2"} <= set(row["metrics"])
        assert "accuracy" not in row["metrics"]


# --------------------------------------------------------------------------------------
# Never a fake number
# --------------------------------------------------------------------------------------


def test_a_family_that_cannot_run_carries_a_reason_and_no_score() -> None:
    """A zero would read as "tried and terrible", which is a different and untrue claim.

    Driven through the feature cap, which is the reachable way to make a family decline
    without corrupting the data underneath it.
    """
    X, y = _classification(n=120, features=40)
    X_train, y_train, X_test, y_test = _split(X, y, at=90)

    # Force the two distance/kernel families to decline.
    original = training.CAPS["svm_rbf"]["features"]
    training.CAPS["svm_rbf"]["features"] = 5
    try:
        report = training.train_all(X_train, y_train, X_test, y_test, "binary classification")
    finally:
        training.CAPS["svm_rbf"]["features"] = original

    svm = next(row for row in report.models if row["family"] == "svm_rbf")
    assert svm["status"] == "skipped"
    assert svm["primary_metric_value"] is None
    assert svm["rank"] is None
    assert svm["reason"] and "5" in svm["reason"]


def test_sampling_is_disclosed_on_every_row_that_was_sampled() -> None:
    """A leaderboard where one model saw 8,000 rows and another saw 20,000 without
    saying so is not a comparison."""
    X, y = _classification(n=400)
    X_train, y_train, X_test, y_test = _split(X, y, at=320)

    original = training.CAPS["knn"]["rows"]
    training.CAPS["knn"]["rows"] = 100
    try:
        report = training.train_all(X_train, y_train, X_test, y_test, "binary classification")
    finally:
        training.CAPS["knn"]["rows"] = original

    knn = next(row for row in report.models if row["family"] == "knn")
    assert knn["status"] == "succeeded"
    assert knn["sampled"] is True
    assert knn["n_train_rows_used"] == 100
    assert knn["n_train_rows_available"] == 320
    assert "100" in knn["sampling_note"]

    tree = next(row for row in report.models if row["family"] == "decision_tree")
    assert tree["sampled"] is False
    assert tree["n_train_rows_used"] == 320


# --------------------------------------------------------------------------------------
# Concurrency, measured
# --------------------------------------------------------------------------------------


def test_overlap_is_computed_from_the_spans_not_from_the_worker_count() -> None:
    """`_max_overlap` is the only evidence the payload offers for parallelism, so it
    has to be right about both the overlapping and the serial case."""
    assert training._max_overlap([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]) == 1
    assert training._max_overlap([(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)]) == 3
    assert training._max_overlap([(0.0, 2.0), (1.0, 3.0)]) == 2
    # Touching spans are not overlapping: one ended exactly as the next began.
    assert training._max_overlap([(0.0, 1.0), (1.0, 2.0)]) == 1
    assert training._max_overlap([]) == 0


def test_a_single_worker_pool_reports_one_not_three() -> None:
    """The payload must never claim concurrency the executor did not deliver."""
    X, y = _classification()
    X_train, y_train, X_test, y_test = _split(X, y)

    report = training.train_all(
        X_train, y_train, X_test, y_test, "binary classification", max_workers=1
    )

    assert report.concurrency["workers"] == 1
    assert report.concurrency["max_observed_overlap"] == 1


def test_three_workers_actually_overlap() -> None:
    """The product claim is parallel execution. This asserts it happened."""
    X, y = _classification(n=900, features=12)
    X_train, y_train, X_test, y_test = _split(X, y, at=700)

    report = training.train_all(
        X_train, y_train, X_test, y_test, "binary classification", max_workers=3
    )

    assert report.concurrency["max_observed_overlap"] >= 2, (
        "six jobs on three workers never ran two at once — the pool is serialising, "
        f"payload: {report.concurrency}"
    )
    assert report.concurrency["threads_per_model"] == 1


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_the_same_data_and_seed_produce_the_same_leaderboard() -> None:
    X, y = _classification()
    X_train, y_train, X_test, y_test = _split(X, y)

    first = training.train_all(X_train, y_train, X_test, y_test, "binary classification")
    second = training.train_all(X_train, y_train, X_test, y_test, "binary classification")

    assert first.best_family == second.best_family
    assert [r["family"] for r in first.models] == [r["family"] for r in second.models]
    assert [r["primary_metric_value"] for r in first.models] == [
        r["primary_metric_value"] for r in second.models
    ]


# --------------------------------------------------------------------------------------
# Explainability — the labelling rule
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["random_forest", "decision_tree"])
def test_a_tree_model_is_explained_with_genuine_shap(family: str) -> None:
    """When a SHAP explainer ran, and only then, the payload says SHAP."""
    if not explainability.SHAP_AVAILABLE:
        pytest.skip("shap is not installed in this environment")

    X, y = _classification()
    X_train, y_train, X_test, y_test = _split(X, y)
    model = training._estimators(True, 42)[family]
    model.fit(X_train, y_train)

    out = explainability.explain_model(
        model, family, "Whatever", X_test, y_test, background=X_train
    )

    assert out["status"] == "succeeded"
    assert out["is_shap"] is True
    assert out["method"] == "shap_tree"
    assert "SHAP" in out["method_label"]
    assert out["fallback_reason"] is None
    assert len(out["features"]) > 0


@pytest.mark.parametrize("family", ["svm_rbf", "knn"])
def test_a_non_tree_model_falls_back_and_is_never_labelled_shap(family: str) -> None:
    """The rule the whole module exists for: permutation importance is not SHAP, and
    no field in the payload may suggest that it is."""
    X, y = _classification(n=200)
    X_train, y_train, X_test, y_test = _split(X, y, at=150)
    model = training._estimators(True, 42)[family]
    model.fit(X_train, y_train)

    out = explainability.explain_model(
        model, family, "Whatever", X_test, y_test, background=X_train
    )

    assert out["status"] == "succeeded"
    assert out["is_shap"] is False
    assert out["method"] == "permutation"
    assert "not SHAP" in out["method_label"]
    assert "SHAP" not in out["method_label"].replace("not SHAP", "")
    assert out["fallback_reason"], "a fallback must say why it fell back"


def test_the_explanation_states_its_sampling() -> None:
    """PRD.md §10 asks for 1,000 rows *with the sampling stated*."""
    X, y = _classification(n=1600)
    X_train, y_train, X_test, y_test = _split(X, y, at=400)
    model = training._estimators(True, 42)["decision_tree"]
    model.fit(X_train, y_train)

    out = explainability.explain_model(
        model, "decision_tree", "Decision Tree", X_test, y_test, background=X_train
    )

    assert out["n_rows_available"] == 1200
    assert out["n_rows_used"] == explainability.EXPLAIN_ROWS
    assert out["sampled"] is True
    assert "1,000" in out["sampling_note"]


def test_importances_are_ranked_and_finite() -> None:
    X, y = _classification()
    X_train, y_train, X_test, y_test = _split(X, y)
    model = training._estimators(True, 42)["random_forest"]
    model.fit(X_train, y_train)

    out = explainability.explain_model(
        model, "random_forest", "Random Forest", X_test, y_test, background=X_train
    )
    values = [f["value"] for f in out["features"]]

    assert values == sorted(values, reverse=True)
    assert all(np.isfinite(v) for v in values)
    # f0 and f1 are the columns the label was built from; one of them should lead.
    assert out["features"][0]["feature"] in {"f0", "f1"}
