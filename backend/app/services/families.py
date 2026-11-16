"""The candidate algorithm families Stage 4 ranks.

One list, imported by both the offline corpus builder and the recommendation engine. If
the builder labelled records with ``"forest"`` and the engine ranked ``"random_forest"``,
every recommendation would silently rank a set of families that no record had ever been
scored on — and the output would look completely normal.

The six are the ones PRD.md §10 names for Stage 5: Logistic Regression, Random Forest,
XGBoost, SVM (RBF), KNN and Decision Tree.

On XGBoost
----------
``gradient_boosting`` is represented by scikit-learn's ``HistGradientBoosting*``, not by
XGBoost. NFR-018 scores the best *family*, not the best library, and adding a compiled
dependency in order to label a corpus is not a trade worth making at this stage.
``HistGradientBoosting`` is the same family of model — boosted trees on binned features
— and is already available in the pinned scikit-learn.

Recorded here, and in STAGE4_PLAN.md §7, so that nobody later reads a stored
``gradient_boosting`` label as a claim that XGBoost was run. Stage 5 adds XGBoost within
this family; the label does not change when it does.
"""

from __future__ import annotations

#: Stable identifiers. These are written into ``experiment_records.family_scores`` and
#: read back by the engine, so renaming one is a data migration, not a refactor.
FAMILIES: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
    "knn",
    "decision_tree",
)

#: What to call each one in a sentence a user reads. Stage 4's whole point is a ranking
#: with a plain-English rationale, and ``svm_rbf`` is not English.
FAMILY_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm_rbf": "SVM (RBF kernel)",
    "knn": "K-Nearest Neighbours",
    "decision_tree": "Decision Tree",
}

#: For a regression task the linear family is Ridge rather than Logistic Regression.
#: The identifier stays the same so a record's scores are comparable across tasks by
#: position; what changed is the estimator, and the record's ``task`` says which.
FAMILY_LABELS_REGRESSION: dict[str, str] = {
    **FAMILY_LABELS,
    "logistic_regression": "Ridge Regression",
}


def label(family: str, task: str) -> str:
    """The display name for a family, given the task it was scored on."""
    if task.strip().lower().startswith(("binary", "multiclass", "class")):
        return FAMILY_LABELS.get(family, family)
    return FAMILY_LABELS_REGRESSION.get(family, family)
