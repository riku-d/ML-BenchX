"""Stage 4 — describing a dataset as a fixed numeric vector.

A meta-learner ranks algorithms for a new dataset by finding past datasets that look
like it. "Look like it" has to mean something precise, and this module is that meaning:
one function turning a DataFrame and its target into a fixed, ordered vector of numbers.

The one invariant
-----------------
**The corpus and the query must be described by this function, at this version.**

A cosine similarity between a vector built one way and a vector built another is not a
weak signal — it is a meaningless number that will still render as a confident
percentage on a screen. Three things guard against that:

* This is the only place a vector is produced. The offline corpus builder and the live
  recommendation path both call ``extract``; neither computes anything itself.
* Every vector carries ``METAFEATURE_VERSION``. Stored records at a different version
  are excluded from the search rather than silently compared.
* ``VECTOR_FIELDS`` fixes the order. ``as_vector`` reads it, so a feature cannot be
  added in one place and forgotten in another — the order is data, not convention.

Measured on the raw frame, not the prepared one
-----------------------------------------------
After Stage 3 every column is numeric, so ``n_categorical`` would be zero for every
dataset ever recorded and the feature would be dead weight. The raw frame is also the
only thing an OpenML record and a user's upload genuinely have in common: the prepared
frame depends on plan choices, and two people preparing the same dataset differently
would otherwise look like two different datasets.

What this will not do
---------------------
Nothing here is sampled, estimated or capped. Every number is computed over the whole
frame, because a meta-feature that changed depending on how big the dataset was would
make two records incomparable for the worst possible reason.

Missing values are not imputed before measuring. ``sparsity`` is the feature that
records them, and filling them first would erase it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Bumped whenever a field is added, removed, or computed differently.
#:
#: This is not a formality. A corpus seeded at version 1 and queried by version 2 would
#: compare vectors of different lengths, or the same length with different meanings —
#: and the second failure is silent. ``metalearning`` filters on this value.
METAFEATURE_VERSION = 1

#: The vector's field order. ``as_vector`` reads this, so adding a field here is the
#: only step needed to put it in the vector — and forgetting to bump the version above
#: is caught by a test rather than by a wrong recommendation.
VECTOR_FIELDS: tuple[str, ...] = (
    "log_n_rows",
    "log_n_cols",
    "n_numerical",
    "n_categorical",
    "categorical_ratio",
    "sparsity",
    "duplicate_share",
    "imbalance_ratio",
    "mean_abs_correlation",
    "mean_cardinality",
    "max_cardinality",
    "mean_abs_skew",
    "n_classes",
    "is_classification",
    "is_regression",
)

#: The eight named in PRD.md §10. Asserted by a test, so the contract cannot be
#: narrowed by a later refactor without someone noticing.
PRD_FIELDS: tuple[str, ...] = (
    "n_rows",
    "n_cols",
    "n_numerical",
    "n_categorical",
    "sparsity",
    "imbalance_ratio",
    "mean_abs_correlation",
    "target_type",
)


def _feature_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    """Everything except the target.

    The target is described separately — by ``n_classes``, ``imbalance_ratio`` and the
    task flags — and counting it among the features would make a binary target look like
    one more categorical column.
    """
    return [c for c in frame.columns if c != target]


def _is_classification(task: str) -> bool:
    """Whether this task label names a classification.

    Matches ``target.detect_task``'s vocabulary — "binary classification",
    "multiclass classification", "regression" — by prefix rather than by equality, so a
    future "ordinal classification" is classified rather than silently treated as a
    regression.
    """
    return task.strip().lower().startswith(("binary", "multiclass", "class"))


def _mean_abs_correlation(frame: pd.DataFrame, columns: list[str]) -> float:
    """Mean absolute Pearson correlation between numeric feature columns.

    Off-diagonal only: a column's correlation with itself is 1 by definition and
    including it would pull every dataset towards the same number, with the pull being
    strongest for the datasets that have the fewest columns.

    Returns 0.0 for fewer than two numeric columns — there is no pair to correlate, and
    0.0 is the honest reading of "no redundancy measured" for a distance metric.
    """
    numeric = frame[columns].select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return 0.0

    matrix = numeric.corr(numeric_only=True).to_numpy(dtype="float64")
    if matrix.size == 0:
        return 0.0

    # A constant column correlates with nothing and pandas returns NaN for it. Those
    # pairs are dropped rather than treated as zero correlation, which would be a claim
    # about independence that was never measured.
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    values = np.abs(matrix[off_diagonal])
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def _imbalance_ratio(series: pd.Series, classification: bool) -> float:
    """Majority class count divided by minority class count.

    1.0 for a regression and for a single-class target: there is no imbalance to report,
    and 1.0 is the value that means "even" rather than a sentinel a distance metric
    would treat as a real coordinate.
    """
    if not classification:
        return 1.0
    counts = series.dropna().value_counts()
    if len(counts) < 2:
        return 1.0
    smallest = int(counts.iloc[-1])
    return float(counts.iloc[0] / smallest) if smallest else 1.0


def _cardinalities(frame: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    """Mean and maximum distinct-value count over non-numeric feature columns.

    Cardinality is what decides how a category is encoded, and a dataset of ten
    two-value columns behaves very differently from one with a 900-value identifier.
    Numeric columns are excluded: their distinct count is a property of measurement
    precision rather than of the data's shape.
    """
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if not categorical:
        return 0.0, 0.0
    counts = [float(frame[c].nunique(dropna=True)) for c in categorical]
    return float(np.mean(counts)), float(max(counts))


def _mean_abs_skew(frame: pd.DataFrame, columns: list[str]) -> float:
    """Mean absolute skew over numeric feature columns.

    Absolute because a long left tail and a long right tail are the same problem for an
    algorithm choice; the sign would make two equally lopsided datasets look opposite.
    """
    numeric = frame[columns].select_dtypes(include="number")
    if numeric.empty:
        return 0.0
    skews = numeric.skew(numeric_only=True).to_numpy(dtype="float64")
    skews = np.abs(skews[np.isfinite(skews)])
    return float(skews.mean()) if skews.size else 0.0


def extract(frame: pd.DataFrame, target: str | None, task: str) -> dict[str, Any]:
    """Describe this dataset as meta-features.

    Pure and deterministic: no I/O, no randomness, no dependence on anything outside the
    arguments. The same frame and target produce the same vector on any machine, which
    is what lets a record seeded on burst compute be compared with a query computed in a
    web request.

    Returns a flat dict of every named feature plus ``metafeature_version``. Use
    ``as_vector`` to get the ordered numeric form the similarity search consumes.
    """
    columns = _feature_columns(frame, target)
    n_rows = int(frame.shape[0])
    n_cols = len(columns)

    numeric_columns = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    n_numerical = len(numeric_columns)
    n_categorical = n_cols - n_numerical

    classification = _is_classification(task)
    target_series = frame[target] if target and target in frame.columns else pd.Series(dtype="float64")

    # Cells of the *features*, not of the whole frame: a missing target is a different
    # problem (those rows are dropped before fitting) and folding it in here would make
    # sparsity mean two things at once.
    total_cells = max(n_rows * n_cols, 1)
    missing_cells = int(frame[columns].isna().sum().sum()) if columns else 0

    mean_cardinality, max_cardinality = _cardinalities(frame, columns)
    n_classes = int(target_series.dropna().nunique()) if classification else 0

    return {
        "metafeature_version": METAFEATURE_VERSION,
        # --- the eight named in PRD.md §10 ------------------------------------------
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numerical": n_numerical,
        "n_categorical": n_categorical,
        "sparsity": missing_cells / total_cells,
        "imbalance_ratio": _imbalance_ratio(target_series, classification),
        "mean_abs_correlation": _mean_abs_correlation(frame, columns),
        "target_type": task,
        # --- derived, and the reason each earns its place ---------------------------
        # Row counts span 10^2 to 10^6 across any real corpus. Unlogged, this single
        # coordinate sets the scale for the whole vector and every other feature
        # becomes rounding error.
        "log_n_rows": float(np.log10(max(n_rows, 1))),
        "log_n_cols": float(np.log10(max(n_cols, 1))),
        # Scale-free shape: a 10-column and a 100-column dataset can be the same kind
        # of dataset, and the ratio is what says so.
        "categorical_ratio": (n_categorical / n_cols) if n_cols else 0.0,
        "duplicate_share": float(frame.duplicated().mean()) if n_rows else 0.0,
        "mean_cardinality": mean_cardinality,
        "max_cardinality": max_cardinality,
        "mean_abs_skew": _mean_abs_skew(frame, columns),
        "n_classes": n_classes,
        # One-hot rather than an integer code. A single `task` column with 0/1/2 would
        # tell the distance metric that multiclass sits between binary and regression,
        # which is not a fact about anything.
        "is_classification": 1.0 if classification else 0.0,
        "is_regression": 0.0 if classification else 1.0,
    }


def as_vector(features: dict[str, Any]) -> np.ndarray:
    """The ordered numeric vector, for the similarity search.

    Raises on a missing field rather than substituting a default: a silently-zeroed
    coordinate is a wrong distance, and a wrong distance is a wrong recommendation that
    looks exactly like a right one.
    """
    missing = [name for name in VECTOR_FIELDS if name not in features]
    if missing:
        raise ValueError(f"meta-features are missing {', '.join(missing)}")
    return np.array([float(features[name]) for name in VECTOR_FIELDS], dtype="float64")
