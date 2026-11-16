"""Stage 6 — explaining the winning model, and saying which method did the explaining.

One global explanation per run, for the family that topped the leaderboard. PRD.md §10
asks for SHAP on the top model, subsampled to 1,000 rows **with the sampling stated**;
both halves of that are honoured here.

The labelling rule, which is the whole point of this module
-----------------------------------------------------------
**SHAP is only ever called SHAP when a SHAP explainer actually ran.** Shapley values and
permutation importance answer different questions. A Shapley value decomposes one
prediction into additive per-feature contributions with a uniqueness guarantee;
permutation importance measures how much a metric degrades when one column is shuffled.
They frequently rank features similarly, which is exactly why mislabelling one as the
other is tempting and dishonest.

So every payload carries three fields a caller cannot avoid reading:

``method``    ``shap_tree`` | ``shap_linear`` | ``permutation``
``is_shap``   a boolean, so a UI cannot accidentally print the wrong heading
``method_label``  the exact words to show the user

When a SHAP explainer cannot read a fitted model — scikit-learn's histogram booster is
the common case, and kernel-based and neighbour-based models are never tree-shaped —
this falls back to permutation importance and **says so in all three fields**. It does
not fall back silently, and it does not fall back to
:class:`shap.KernelExplainer`: that is exponential in the feature count and on a wide
matrix would run for hours to produce an approximation nobody asked for.

What is deliberately not claimed
--------------------------------
These are **global** importances — the mean absolute contribution of each feature across
the sampled rows. A per-prediction waterfall is a different artifact and is not produced
here; the payload does not imply one exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.config import SEED

__all__ = ["EXPLAIN_ROWS", "explain_model", "SHAP_AVAILABLE"]

#: Rows the explanation is computed over, before sampling is reported.
EXPLAIN_ROWS = 1000

#: How many times each column is shuffled for permutation importance. Five is enough to
#: separate signal from shuffle noise without multiplying the cost by twenty.
PERMUTATION_REPEATS = 5

#: Families whose fitted estimator a tree explainer can read exactly. The histogram
#: booster is **not** here: it is a boosted-tree model, but its internal representation
#: is not one `shap.TreeExplainer` reads, and attempting it raises. It therefore takes
#: the permutation path and is labelled accordingly.
TREE_FAMILIES = frozenset({"random_forest", "decision_tree"})

#: Families with a linear decision function a linear explainer can read.
LINEAR_FAMILIES = frozenset({"logistic_regression"})

try:  # pragma: no cover - import availability is environmental, not logical
    import shap as _shap

    SHAP_AVAILABLE = True
    SHAP_VERSION = getattr(_shap, "__version__", "unknown")
except Exception:  # pragma: no cover
    _shap = None
    SHAP_AVAILABLE = False
    SHAP_VERSION = None


def _sample(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, bool]:
    """At most :data:`EXPLAIN_ROWS` rows, and whether sampling happened."""
    if len(frame) <= EXPLAIN_ROWS:
        return frame, False
    return frame.sample(n=EXPLAIN_ROWS, random_state=seed), True


def _collapse(values: np.ndarray) -> np.ndarray:
    """Reduce a SHAP value array to one mean absolute importance per feature.

    The array arrives as ``(rows, features)`` for a regressor or a binary model, and as
    ``(rows, features, classes)`` for a multiclass one. Taking the absolute value first
    and the mean afterwards is the order that matters: contributions that push in
    opposite directions across rows are both evidence that a feature matters, and
    averaging them signed would cancel them to nothing.
    """
    collapsed = np.abs(np.asarray(values, dtype=float))
    while collapsed.ndim > 2:
        collapsed = collapsed.mean(axis=-1)
    return collapsed.mean(axis=0)


def _ranked(names: list[str], importances: np.ndarray, limit: int = 25) -> list[dict[str, Any]]:
    """Features by descending importance, the top ``limit`` of them.

    Truncated because a chart of 400 bars communicates nothing; the count is reported
    alongside so the truncation is visible rather than implied.
    """
    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": str(names[i]), "value": float(importances[i])}
        for i in order
        if np.isfinite(importances[i])
    ]


def _shap_explanation(
    model: Any, family: str, sample: pd.DataFrame, background: pd.DataFrame
) -> dict[str, Any] | None:
    """A genuine SHAP explanation, or ``None`` if no explainer can read this model.

    ``None`` rather than an approximation. The caller's fallback is labelled honestly,
    which is a better outcome than a kernel explainer running for an hour.
    """
    if not SHAP_AVAILABLE:
        return None

    if family in TREE_FAMILIES:
        try:
            explainer = _shap.TreeExplainer(model)
            values = explainer(sample).values
            return {
                "method": "shap_tree",
                "is_shap": True,
                "method_label": "SHAP values (TreeExplainer)",
                "importances": _collapse(values),
                "explainer": "shap.TreeExplainer",
            }
        except Exception:
            return None

    if family in LINEAR_FAMILIES:
        try:
            explainer = _shap.LinearExplainer(model, background)
            values = explainer(sample).values
            return {
                "method": "shap_linear",
                "is_shap": True,
                "method_label": "SHAP values (LinearExplainer)",
                "importances": _collapse(values),
                "explainer": "shap.LinearExplainer",
            }
        except Exception:
            return None

    return None


def _permutation_explanation(
    model: Any, sample: pd.DataFrame, y_sample: pd.Series, seed: int
) -> dict[str, Any]:
    """Permutation importance, labelled as permutation importance.

    Measured on held-out rows, which is the only place it means anything: permuting a
    column in the training fold measures how much the model memorised it.
    """
    result = permutation_importance(
        model,
        sample,
        y_sample,
        n_repeats=PERMUTATION_REPEATS,
        random_state=seed,
        n_jobs=1,
    )
    return {
        "method": "permutation",
        "is_shap": False,
        "method_label": "Permutation importance (not SHAP)",
        "importances": np.asarray(result.importances_mean, dtype=float),
        "explainer": "sklearn.inspection.permutation_importance",
    }


def explain_model(
    model: Any,
    family: str,
    family_label: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    background: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Explain one fitted model globally, on held-out rows, and name the method used.

    ``background`` is the training matrix the model was actually fitted on — the linear
    explainer needs a reference distribution, and the right reference is the data the
    coefficients were learned from, not the data being explained.
    """
    sample, sampled = _sample(X_test, seed)
    y_sample = y_test.loc[sample.index]
    names = [str(c) for c in X_test.columns]

    attempted_shap = SHAP_AVAILABLE and (
        family in TREE_FAMILIES or family in LINEAR_FAMILIES
    )
    outcome = _shap_explanation(model, family, sample, background)

    if outcome is None:
        try:
            outcome = _permutation_explanation(model, sample, y_sample, seed)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "family": family,
                "label": family_label,
                "reason": (
                    f"No explanation could be produced: {type(exc).__name__}: {exc}"[:300]
                ),
                "is_shap": False,
                "method": None,
            }
        if not SHAP_AVAILABLE:
            fallback_reason = (
                "shap is not installed in this environment, so permutation importance "
                "was used instead. It is a different measurement and is labelled as such."
            )
        elif attempted_shap:
            fallback_reason = (
                f"A SHAP explainer was attempted for {family_label} and could not read "
                "the fitted model, so permutation importance was used instead. It is a "
                "different measurement and is labelled as such."
            )
        else:
            fallback_reason = (
                f"{family_label} is neither tree-shaped nor linear, so no exact SHAP "
                "explainer applies. Permutation importance was used instead — a "
                "different measurement, labelled as such. A kernel SHAP approximation "
                "was deliberately not run: it is exponential in the feature count."
            )
    else:
        fallback_reason = None

    importances = outcome.pop("importances")

    return {
        "status": "succeeded",
        "family": family,
        "label": family_label,
        "n_rows_used": int(len(sample)),
        "n_rows_available": int(len(X_test)),
        "sampled": sampled,
        "sampling_note": (
            f"Computed over a random sample of {len(sample):,} of {len(X_test):,} "
            f"held-out rows (cap: {EXPLAIN_ROWS:,})."
            if sampled
            else f"Computed over all {len(sample):,} held-out rows."
        ),
        "n_features": len(names),
        "n_features_shown": min(25, len(names)),
        "shap_available": SHAP_AVAILABLE,
        "shap_version": SHAP_VERSION,
        "fallback_reason": fallback_reason,
        "features": _ranked(names, importances),
        **outcome,
    }
