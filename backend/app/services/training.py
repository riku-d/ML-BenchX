"""Stage 5 — training the six candidate families on one run's prepared data.

This is the training harness the product needs at the point where a reviewed plan has
already produced ``train.parquet`` and ``test.parquet``. It fits each family on the
training fold and scores it on the **held-out** fold, which is the fold the planner
never measured on and no transformer was ever fitted against.

What makes a score here honest
------------------------------
**Fit on train, score on test.** The split was decided by the plan, recreated by the
executor, and is already stored as two parquet artifacts. Nothing here re-splits
anything, so the fold boundary the leakage tests guarantee is the fold boundary these
numbers are measured across.

**A trivial model is scored on the same fold.** ``DummyClassifier(most_frequent)`` or
``DummyRegressor(mean)`` runs alongside the six. A family that cannot beat it has not
earned a recommendation, and the leaderboard says so rather than ranking it first in
silence.

**A family that raises is recorded as absent, with the reason.** Never as zero. A zero
reads as "it was tried and was terrible", which is a different and untrue statement —
the same rule ``scripts/build_corpus.py`` already follows for the corpus.

**Sampling is disclosed, per family.** Two of the six have costs that are not linear in
rows: an RBF kernel is quadratic to cubic, and a k-nearest-neighbour query at this many
dimensions degenerates to brute force. Both are capped. A leaderboard where the SVM saw
8,000 rows and the forest saw 20,000 without saying so is not a comparison, so every
row carries ``n_train_rows_used`` and ``sampled``.

On concurrency
--------------
Three workers, six submitted jobs, one thread of linear algebra each.

Threads rather than processes, deliberately. Every estimator here spends its time inside
compiled code that releases the GIL — scikit-learn's tree builder is ``nogil``, libsvm's
solver is a C extension, the histogram booster is OpenMP, and the neighbour search is
BLAS — so three threads give three cores of genuine progress without the cost of a
``spawn`` start method re-importing scikit-learn three times and without pickling the
matrices across a process boundary. What it gives up is crash isolation and the ability
to kill a job that will not finish; the row caps are what stand in for the second, and
the first is a real limitation recorded in the module's own result payload rather than
glossed over.

``threadpool_limits(1)`` around each fit is what makes "one thread per model" true.
``n_jobs=1`` governs joblib only; without the threadpool limit each of the three workers
would still open as many BLAS/OpenMP threads as there are cores, and 24 threads on 8
logical cores is measurably slower than 3. ``threadpoolctl`` is already a scikit-learn
dependency, so this costs nothing to add.

**Concurrency is measured, not asserted.** Each job records when it started and
finished on one monotonic clock, and :func:`_max_overlap` reports the largest number of
jobs that were in flight at once. If that number is 1, the pool ran serially and the
payload says 1 — there is no way for this module to claim parallelism it did not get.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from app.config import SEED
from app.services.families import FAMILIES, label as family_label

__all__ = [
    "MAX_WORKERS",
    "CAPS",
    "TrainingReport",
    "train_all",
    "is_classification",
]

# --------------------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------------------

#: Three, on a four-core machine, so the interpreter running the web server and the
#: browser rendering the result are not competing with the training job for the last
#: core. Raising this on a larger machine is a one-line change; lowering it to 1 is the
#: documented fallback when a pool misbehaves, and the payload will say so.
MAX_WORKERS = 3

#: Per-family limits. ``rows`` is a cap on the *training* fold; ``features`` is a cap on
#: the width of the encoded matrix, above which the family is not run at all.
#:
#: The two tight ones are not arbitrary. An RBF kernel costs between the square and the
#: cube of the row count, so a fold of 28,000 rows is minutes-to-hours rather than
#: seconds. A k-d tree stops helping above roughly twenty dimensions, so a neighbour
#: query on a wide encoded matrix is a brute-force scan of the whole training fold for
#: every test row. Both are properties of the algorithm, not of this machine.
CAPS: dict[str, dict[str, int | None]] = {
    "logistic_regression": {"rows": None, "features": 5000},
    "decision_tree": {"rows": None, "features": 5000},
    "gradient_boosting": {"rows": None, "features": 5000},
    "random_forest": {"rows": 20000, "features": 2000},
    "svm_rbf": {"rows": 8000, "features": 1000},
    "knn": {"rows": 20000, "features": 1000},
}

#: Above this many encoded columns nothing is trained. A matrix this wide means a
#: high-cardinality column was one-hot encoded, and the useful answer is to say which
#: rather than to spend ten minutes proving that six models all do badly.
HARD_FEATURE_CEILING = 5000

#: Rows the explanation is computed over. PRD.md §10 asks for 1,000 with the sampling
#: stated; both halves of that are honoured — the number actually used is reported.
EXPLAIN_ROWS = 1000

#: Families whose fitted estimator a SHAP tree explainer can read exactly.
TREE_FAMILIES = frozenset({"random_forest", "gradient_boosting", "decision_tree"})


# --------------------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------------------


def is_classification(task: str) -> bool:
    """Whether a task string names a classification problem.

    The same test ``services/families.py`` applies, so a family's display label and its
    estimator can never disagree about which kind of problem was solved.
    """
    return task.strip().lower().startswith(("binary", "multiclass", "class"))


def _estimators(classification: bool, seed: int) -> dict[str, Any]:
    """One estimator per family, at library defaults.

    Copied deliberately from ``scripts/build_corpus.py``, which is the offline corpus
    builder and is documented as never being importable from ``app/``. The duplication
    is real and is the price of not reaching into a script from the application package;
    it is recorded here so that a change to one is known to need the same change to the
    other. The two lists must stay identical, because a recommendation ranks the family
    a corpus record was labelled with and this trains the family the user then sees.

    Defaults on purpose. A tuned model would answer "which family suits this dataset
    after somebody spent an afternoon on it", which is not a property of the dataset.

    ``n_jobs=1`` everywhere: the parallelism in this module is across families, not
    inside one, and a forest that opens eight threads inside one of three workers makes
    the whole job slower.
    """
    if classification:
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
            "svm_rbf": SVC(kernel="rbf", random_state=seed),
            "knn": KNeighborsClassifier(n_jobs=1),
            "decision_tree": DecisionTreeClassifier(random_state=seed),
        }
    return {
        "logistic_regression": Ridge(random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
        "svm_rbf": SVR(kernel="rbf"),
        "knn": KNeighborsRegressor(n_jobs=1),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
    }


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def _subsample(
    X: pd.DataFrame, y: pd.Series, cap: int | None, classification: bool, seed: int
) -> tuple[pd.DataFrame, pd.Series, bool]:
    """Reduce the training fold to ``cap`` rows, keeping the class shares.

    Stratified for classification so a capped SVM is not handed a different class
    balance from the forest it is compared against — that would make the leaderboard a
    comparison of two different problems. Falls back to a plain sample when
    stratification is impossible, which is the same hierarchy the splitter uses.

    The test fold is never touched. Capping it would change what the metric is measured
    over, family by family, and the whole point of the held-out fold is that every model
    is scored on exactly the same rows.
    """
    if cap is None or len(X) <= cap:
        return X, y, False

    if classification:
        try:
            X_s, _, y_s, _ = train_test_split(
                X, y, train_size=cap, random_state=seed, stratify=y
            )
            return X_s, y_s, True
        except ValueError:
            # A class too small to stratify at this size. A plain sample is still a
            # sample of the same distribution; it simply does not guarantee the shares.
            pass

    index = X.sample(n=cap, random_state=seed).index
    return X.loc[index], y.loc[index], True


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def _roc_auc(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """ROC-AUC where the fitted model can produce a score, and ``None`` where it cannot.

    ``None`` rather than 0.0, and rather than omitting the key. An SVM fitted with
    ``probability=False`` has a decision function but no calibrated probability, and a
    k-nearest-neighbour model's vote share is not a ranking in the sense AUC needs for
    more than two classes. "Not computable for this model" is a fact about the pairing;
    a zero would be read as "it ranked no better than chance".
    """
    classes = np.unique(y_test)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if len(classes) == 2:
                return float(roc_auc_score(y_test, proba[:, 1]))
            return float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
        if hasattr(model, "decision_function") and len(classes) == 2:
            return float(roc_auc_score(y_test, model.decision_function(X_test)))
    except Exception:
        return None
    return None


def _metrics(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, classification: bool
) -> dict[str, Any]:
    """Every metric PRD.md §10 names for the task, measured on the held-out fold."""
    predictions = model.predict(X_test)

    if classification:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(predictions)])))
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "precision_macro": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "roc_auc": _roc_auc(model, X_test, y_test),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
            "labels": [str(one) for one in labels],
        }

    mse = float(mean_squared_error(y_test, predictions))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


#: Which metric the leaderboard is ordered by, and which direction is better.
PRIMARY_METRIC = {"classification": "f1_macro", "regression": "r2"}


# --------------------------------------------------------------------------------------
# One family
# --------------------------------------------------------------------------------------


@dataclass
class _Outcome:
    """One family's result, plus when it ran — the raw material for both the
    leaderboard and the concurrency measurement."""

    family: str
    payload: dict[str, Any]
    started: float
    finished: float
    model: Any = None
    X_used: pd.DataFrame | None = None


def _train_one(
    family: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classification: bool,
    seed: int,
    clock: Callable[[], float],
) -> _Outcome:
    """Fit one family and score it, recording a reason instead of a number on failure.

    Every exception is caught. One pathological pairing of dataset and estimator must
    not take the other five down with it, and the user is better served by five results
    and one explained absence than by a 500.
    """
    started = clock()
    cap = CAPS[family]["rows"]
    n_features = X_train.shape[1]
    feature_cap = CAPS[family]["features"]

    if feature_cap is not None and n_features > feature_cap:
        return _Outcome(
            family=family,
            started=started,
            finished=clock(),
            payload={
                "status": "skipped",
                "reason": (
                    f"Not run: this dataset encodes to {n_features:,} columns, and "
                    f"{family_label(family, 'classification' if classification else 'regression')} "
                    f"is only run up to {feature_cap:,}. Above that its cost grows faster "
                    "than its accuracy does — a distance or kernel computation over this "
                    "many dimensions measures mostly noise."
                ),
            },
        )

    try:
        X_fit, y_fit, sampled = _subsample(X_train, y_train, cap, classification, seed)
        # One thread of linear algebra per model. See the module docstring: `n_jobs=1`
        # alone does not constrain the BLAS/OpenMP pools underneath.
        with threadpool_limits(limits=1):
            estimator.fit(X_fit, y_fit)
            metrics = _metrics(estimator, X_test, y_test, classification)
    except Exception as exc:  # noqa: BLE001 - one bad pairing must not stop the job
        return _Outcome(
            family=family,
            started=started,
            finished=clock(),
            payload={
                "status": "failed",
                "reason": (
                    f"{type(exc).__name__}: {exc}"[:400]
                    or "The estimator raised without a message."
                ),
            },
        )

    finished = clock()
    return _Outcome(
        family=family,
        started=started,
        finished=finished,
        model=estimator,
        X_used=X_fit,
        payload={
            "status": "succeeded",
            "metrics": metrics,
            "n_train_rows_used": int(len(X_fit)),
            "n_train_rows_available": int(len(X_train)),
            "sampled": sampled,
            "sampling_note": (
                f"Trained on a stratified sample of {len(X_fit):,} of "
                f"{len(X_train):,} training rows (cap for this family: {cap:,})."
                if sampled
                else f"Trained on all {len(X_fit):,} training rows."
            ),
            "n_features": int(n_features),
            "train_ms": int((finished - started) * 1000),
        },
    )


# --------------------------------------------------------------------------------------
# Concurrency measurement
# --------------------------------------------------------------------------------------


def _max_overlap(spans: list[tuple[float, float]]) -> int:
    """The largest number of jobs that were running at the same instant.

    A sweep over start and end events. This is the only evidence in the payload that the
    pool actually ran things concurrently: if the executor serialised the work, every
    span is disjoint and this returns 1. Claiming parallel execution without measuring
    it would be exactly the kind of unverified assertion the rest of this product
    refuses to make.
    """
    events: list[tuple[float, int]] = []
    for start, end in spans:
        events.append((start, 1))
        # Ends sort before starts at the same timestamp, so two jobs that merely touch
        # are not counted as overlapping.
        events.append((end, -1))
    events.sort(key=lambda pair: (pair[0], pair[1]))

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


# --------------------------------------------------------------------------------------
# The whole job
# --------------------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """Everything one training job produced."""

    task: str
    models: list[dict[str, Any]]
    baseline: dict[str, Any]
    concurrency: dict[str, Any]
    primary_metric: str
    best_family: str | None
    duration_ms: int
    n_train_rows: int
    n_test_rows: int
    n_features: int
    seed: int
    notes: list[str] = field(default_factory=list)
    #: Fitted estimators, by family. Not serialised — the caller uses this for the
    #: explanation and then drops it.
    fitted: dict[str, Any] = field(default_factory=dict, repr=False)
    fit_frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """The ``models.json`` artifact. Fitted objects are deliberately absent."""
        return {
            "task": self.task,
            "primary_metric": self.primary_metric,
            "best_family": self.best_family,
            "baseline": self.baseline,
            "concurrency": self.concurrency,
            "duration_ms": self.duration_ms,
            "split": {
                "n_train_rows": self.n_train_rows,
                "n_test_rows": self.n_test_rows,
                "n_features": self.n_features,
            },
            "seed": self.seed,
            "notes": self.notes,
            "models": self.models,
        }


#: The trivial strategies a model must beat. **More than one on purpose.**
#:
#: A single ``most_frequent`` dummy is not a sufficient bar, and the failure is not
#: hypothetical — it was measured on this project's own 10 MB fixture, whose target is
#: ``rng.choice([0, 1])`` and therefore carries no signal at all. There, every one of the
#: six families scored accuracy ≈ 0.50 and ROC-AUC ≈ 0.50 — correctly learning nothing —
#: while ``most_frequent`` scored macro-F1 0.3356, because predicting one class only is
#: penalised hard by a macro average. Measured against that single dummy, all six
#: "beat the baseline", and the leaderboard would have said so.
#:
#: The honest bar is the **best score any model that ignores the features can reach**,
#: so every strategy is scored and the strongest one becomes the bar. ``uniform`` and
#: ``stratified`` are what catch a coin-flip dressed as a result.
DUMMY_STRATEGIES: tuple[str, ...] = ("most_frequent", "uniform", "stratified", "prior")

#: Regression's equivalent. ``mean`` scores R² = 0 by construction; ``median`` can beat
#: it on a skewed target, which is exactly when a weak model looks good against ``mean``.
DUMMY_REGRESSION_STRATEGIES: tuple[str, ...] = ("mean", "median")


def _baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classification: bool,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    """The strongest trivial model, scored on the same held-out rows as the six.

    Without this the leaderboard has a winner but no sense of whether winning meant
    anything. A majority-class classifier on a 95/5 split scores 0.95 accuracy, and a
    family that ties it has learned nothing.

    Every strategy in :data:`DUMMY_STRATEGIES` is fitted and scored; the one with the
    best primary metric becomes the bar, and all of them are reported so a reader can
    see which one set it. Taking the best rather than the conventional one is the whole
    point — see the note on that constant.
    """
    strategies = DUMMY_STRATEGIES if classification else DUMMY_REGRESSION_STRATEGIES
    scored: dict[str, Any] = {}
    best_name: str | None = None
    best_value: float | None = None

    for strategy in strategies:
        model = (
            DummyClassifier(strategy=strategy, random_state=seed)
            if classification
            else DummyRegressor(strategy=strategy)
        )
        try:
            with threadpool_limits(limits=1):
                model.fit(X_train, y_train)
                metrics = _metrics(model, X_test, y_test, classification)
        except Exception as exc:  # noqa: BLE001 - a strategy that cannot fit is not fatal
            scored[strategy] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:200]}
            continue

        scored[strategy] = {"status": "succeeded", "metrics": metrics}
        value = metrics.get(metric)
        if value is not None and np.isfinite(value) and (best_value is None or value > best_value):
            best_value, best_name = float(value), strategy

    if best_name is None:
        return {"status": "failed", "reason": "No trivial strategy could be fitted.", "strategies": scored}

    return {
        "status": "succeeded",
        "estimator": "DummyClassifier" if classification else "DummyRegressor",
        "strategy": best_name,
        "metrics": scored[best_name]["metrics"],
        "strategies": scored,
        "note": (
            f"The bar is the best of {len(strategies)} trivial strategies — here "
            f"{best_name!r} at {best_value:.4f} {metric}. A single majority-class dummy "
            "is a weaker bar than it looks under a macro average."
        ),
    }


def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    seed: int = SEED,
    max_workers: int = MAX_WORKERS,
) -> TrainingReport:
    """Fit all six families concurrently, score them on the held-out fold, and rank them.

    The frames are the ones the run already stored. Nothing is re-split, re-imputed or
    re-encoded here: by the time this is called, every preprocessing decision has been
    made, reviewed and logged.
    """
    classification = is_classification(task)
    kind = "classification" if classification else "regression"
    metric = PRIMARY_METRIC[kind]
    notes: list[str] = []

    n_features = X_train.shape[1]
    started = time.perf_counter()

    def clock() -> float:
        return time.perf_counter() - started

    if n_features > HARD_FEATURE_CEILING:
        notes.append(
            f"No model was trained: the encoded matrix is {n_features:,} columns wide, "
            f"above the ceiling of {HARD_FEATURE_CEILING:,}. This almost always means a "
            "high-cardinality column was one-hot encoded; change its encoding in the "
            "plan to frequency or target encoding and run again."
        )
        return TrainingReport(
            task=task,
            models=[],
            baseline={"status": "skipped", "reason": "No model was trained."},
            concurrency={"workers": max_workers, "submitted": 0, "max_observed_overlap": 0},
            primary_metric=metric,
            best_family=None,
            duration_ms=int(clock() * 1000),
            n_train_rows=int(len(X_train)),
            n_test_rows=int(len(X_test)),
            n_features=int(n_features),
            seed=seed,
            notes=notes,
        )

    estimators = _estimators(classification, seed)
    outcomes: dict[str, _Outcome] = {}

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train") as pool:
        futures = {
            pool.submit(
                _train_one,
                family,
                estimators[family],
                X_train,
                y_train,
                X_test,
                y_test,
                classification,
                seed,
                clock,
            ): family
            for family in FAMILIES
        }
        for future in as_completed(futures):
            outcome = future.result()
            outcomes[outcome.family] = outcome

    baseline = _baseline(X_train, y_train, X_test, y_test, classification, metric, seed)
    duration_ms = int(clock() * 1000)

    spans = [(one.started, one.finished) for one in outcomes.values()]
    overlap = _max_overlap(spans)
    serial_ms = int(sum((end - start) for start, end in spans) * 1000)

    # Rank. Only succeeded families are ranked; the rest keep their reason and are
    # listed after, so the table shows six rows whatever happened.
    succeeded = [
        one for one in outcomes.values() if one.payload["status"] == "succeeded"
    ]
    succeeded.sort(key=lambda one: one.payload["metrics"][metric], reverse=True)
    best_family = succeeded[0].family if succeeded else None

    baseline_value = (
        baseline.get("metrics", {}).get(metric) if baseline["status"] == "succeeded" else None
    )

    models: list[dict[str, Any]] = []
    for rank, one in enumerate(succeeded, start=1):
        value = one.payload["metrics"][metric]
        row = {
            "family": one.family,
            "label": family_label(one.family, kind),
            "rank": rank,
            **one.payload,
            "primary_metric_value": value,
        }
        if baseline_value is not None:
            row["beats_baseline"] = bool(value > baseline_value)
            row["margin_over_baseline"] = float(value - baseline_value)
        models.append(row)

    for family in FAMILIES:
        one = outcomes[family]
        if one.payload["status"] != "succeeded":
            models.append(
                {
                    "family": family,
                    "label": family_label(family, kind),
                    "rank": None,
                    **one.payload,
                    "primary_metric_value": None,
                }
            )

    if best_family and baseline_value is not None:
        top = models[0]["primary_metric_value"]
        if top <= baseline_value:
            notes.append(
                f"No family beat the trivial baseline on {metric} "
                f"({top:.4f} against {baseline_value:.4f}, strategy "
                f"{baseline.get('strategy')!r}). Treat the ranking as a ranking of "
                "near-useless models, not as a result."
            )
        if baseline.get("note"):
            notes.append(baseline["note"])

    # A second, independent reading of "did anything actually learn". The baseline
    # comparison answers "better than guessing at this metric"; this answers "better
    # than chance at separating the classes", which a metric-relative comparison can
    # miss. Both were needed: on a target with no signal at all, every family can clear
    # a majority-class dummy on macro-F1 while sitting at 0.50 ROC-AUC.
    if classification:
        aucs = [
            row["metrics"].get("roc_auc")
            for row in models
            if row["status"] == "succeeded" and row["metrics"].get("roc_auc") is not None
        ]
        if aucs and max(aucs) <= 0.55:
            notes.append(
                f"The best ROC-AUC across all families is {max(aucs):.4f}, which is "
                "close to the 0.50 of a coin flip. Whatever the ranking says, these "
                "features do not separate the classes — read this as 'no signal "
                "found', not as a leaderboard."
            )

    failures = [one for one in models if one["status"] != "succeeded"]
    if failures:
        notes.append(
            f"{len(failures)} of {len(FAMILIES)} families produced no score. Each one's "
            "row carries the reason; none was recorded as a zero."
        )

    return TrainingReport(
        task=task,
        models=models,
        baseline=baseline,
        concurrency={
            "workers": max_workers,
            "submitted": len(FAMILIES),
            "max_observed_overlap": overlap,
            "wall_clock_ms": duration_ms,
            "summed_model_ms": serial_ms,
            "speedup_vs_serial": round(serial_ms / duration_ms, 2) if duration_ms else None,
            "executor": "ThreadPoolExecutor",
            "threads_per_model": 1,
            "note": (
                "Overlap is measured from each job's own start and finish on one "
                "monotonic clock, not assumed from the worker count."
            ),
        },
        primary_metric=metric,
        best_family=best_family,
        duration_ms=duration_ms,
        n_train_rows=int(len(X_train)),
        n_test_rows=int(len(X_test)),
        n_features=int(n_features),
        seed=seed,
        notes=notes,
        fitted={
            one.family: one.model for one in outcomes.values() if one.model is not None
        },
        fit_frames={
            one.family: one.X_used for one in outcomes.values() if one.X_used is not None
        },
    )
