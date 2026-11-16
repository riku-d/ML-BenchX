"""Seed the Stage 4 experiment history from OpenML.

Offline. Run by hand or on burst compute, never from a web request::

    cd backend
    MLBENCHX_DATABASE_URL="postgresql+psycopg://user@localhost:5432/mlbenchx" \\
        ../.venv/bin/python scripts/build_corpus.py --limit 4

Why this lives in ``scripts/`` and not in ``app/``
--------------------------------------------------
To label a record you have to know which algorithm family actually won, and finding that
out means training models — which is Stage 5's job. Stage 4 cannot wait for Stage 5, so a
slim training harness comes forward.

It is deliberately kept out of the application package and is never imported by it.
Stage 5 is a Celery topology over a broker with WebSocket progress reporting; this is a
``for`` loop with ``cross_val_score`` in it. If it were importable from ``app/`` somebody
would import it, and then the first draft of Stage 5 would be this file, which would be
a bad outcome for Stage 5.

What makes the labels honest
----------------------------
Each dataset is put through **this product's own Stages 1-3** — the real ingest, the
real plan proposal, the real executor — before any model sees it. A corpus labelled
from someone else's preprocessing would describe someone else's pipeline, and the
recommendation would be for a product we do not ship.

Meta-features come from ``services.metafeatures.extract``, the same function the live
recommendation path calls. That is the invariant the whole stage rests on; see
STAGE4_PLAN.md §2.

Scores are cross-validated on the **training fold only**. The held-out fold is never
touched, exactly as ``checks.sanity_baseline`` does it — these records describe how
preparable a dataset is, and spending the test fold on corpus construction would leave
nothing honest for Stage 5 to report later.

Resumability
------------
A dataset already recorded is skipped, so the job can be interrupted and restarted
without refetching or retraining. ``--refresh`` overrides that.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

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
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.config import SEED
from app.db import ExperimentRecord, session_scope
from app.runner import execute
from app.services import metafeatures as mf
from app.services.families import FAMILIES
from app.services.ingest import read_upload
from app.services.planning import propose_for_dataset

# --------------------------------------------------------------------------------------
# The seed list
# --------------------------------------------------------------------------------------

#: OpenML data ids, with the target column each one is predicted on.
#:
#: Chosen from the OpenML-CC18 benchmark suite for three properties: small enough that
#: six families cross-validate in seconds rather than hours, tabular rather than image
#: or text, and varied in the ways the meta-features measure — rows from hundreds to
#: tens of thousands, all-numeric through all-categorical, balanced through heavily
#: skewed. A corpus of twenty near-identical datasets would produce a KNN that always
#: returns the same neighbours.
#:
#: The target is named explicitly rather than left to ``detect_target``: these are known
#: benchmark tasks with a known label column, and letting the heuristic guess would
#: silently build a corpus of records predicting the wrong thing.
SEED_DATASETS: tuple[tuple[int, str, str], ...] = (
    (31, "credit-g", "class"),
    (37, "diabetes", "class"),
    (15, "breast-w", "Class"),
    (29, "credit-approval", "class"),
    (50, "tic-tac-toe", "Class"),
    (54, "vehicle", "Class"),
    (188, "eucalyptus", "Utility"),
    (307, "vowel", "Class"),
    (458, "analcatdata_authorship", "Author"),
    (469, "analcatdata_dmft", "Prevention"),
    (1049, "pc4", "c"),
    (1050, "pc3", "c"),
    (1063, "kc2", "problems"),
    (1068, "pc1", "defects"),
    (1462, "banknote-authentication", "Class"),
    (1464, "blood-transfusion-service-center", "Class"),
    (1480, "ilpd", "Class"),
    (1510, "wdbc", "Class"),
    (6332, "cylinder-bands", "band_type"),
    (23381, "dresses-sales", "Class"),
)


# --------------------------------------------------------------------------------------
# The candidate models
# --------------------------------------------------------------------------------------


def _estimators(classification: bool, seed: int) -> dict[str, Any]:
    """One estimator per family, at library defaults.

    Defaults on purpose. The question a record answers is "which family suits a dataset
    shaped like this", and a tuned model would answer "which family suits this dataset
    after somebody spent an afternoon on it" — which is not a property of the dataset
    and would not transfer to a new one.

    ``probability=False`` on the SVM and a capped ``max_iter`` on the linear model keep
    a single pathological dataset from dominating the job's runtime.
    """
    if classification:
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
            "svm_rbf": SVC(kernel="rbf", random_state=seed),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(random_state=seed),
        }
    return {
        "logistic_regression": Ridge(random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
        "svm_rbf": SVR(kernel="rbf"),
        "knn": KNeighborsRegressor(),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
    }


@dataclass
class Labelled:
    """What training produced for one dataset."""

    scores: dict[str, float]
    best_family: str
    metric: str
    folds: int
    training_ms: int
    #: The trivial model's score. A family that cannot beat it has not earned a
    #: recommendation, and a record whose winner ties the dummy is noted rather than
    #: quietly treated as a result.
    dummy_score: float


def _score_families(
    X: pd.DataFrame, y: pd.Series, classification: bool, seed: int, folds: int
) -> Labelled | None:
    """Cross-validate every family on the training fold and record all the scores.

    Every family is kept, not only the winner. A corpus of winners cannot distinguish
    "Random Forest won by a mile" from "Random Forest won by 0.001", and the second
    should never produce a confident recommendation.

    A family that raises is recorded as absent rather than as zero: scikit-learn raising
    on a dataset is a fact about that pairing, and a zero would be read as "it was tried
    and was terrible", which is a different and untrue statement.
    """
    metric = "accuracy" if classification else "r2"
    dummy = DummyClassifier(strategy="most_frequent") if classification else DummyRegressor()

    started = time.perf_counter()
    scores: dict[str, float] = {}

    with warnings.catch_warnings():
        # Convergence warnings on a hard dataset are expected and are not this job's
        # business; the score records the outcome either way.
        warnings.simplefilter("ignore")

        try:
            dummy_score = float(
                np.mean(cross_val_score(dummy, X, y, cv=folds, scoring=metric, n_jobs=1))
            )
        except Exception:
            return None

        for family, estimator in _estimators(classification, seed).items():
            try:
                result = cross_val_score(estimator, X, y, cv=folds, scoring=metric, n_jobs=1)
                value = float(np.mean(result))
            except Exception as exc:  # noqa: BLE001 - one bad pairing must not stop the job
                print(f"      {family}: skipped ({type(exc).__name__})")
                continue
            if np.isfinite(value):
                scores[family] = value

    if not scores:
        return None

    best = max(scores, key=lambda name: scores[name])
    return Labelled(
        scores=scores,
        best_family=best,
        metric=metric,
        folds=folds,
        training_ms=int((time.perf_counter() - started) * 1000),
        dummy_score=dummy_score,
    )


# --------------------------------------------------------------------------------------
# One dataset, end to end
# --------------------------------------------------------------------------------------


def _ensure_certificates() -> None:
    """Point OpenSSL at certifi's bundle when the interpreter has no trust store.

    A framework Python on macOS ships without installed root certificates, so
    ``urllib`` — which is what ``fetch_openml`` uses — fails every HTTPS request with
    ``CERTIFICATE_VERIFY_FAILED`` while ``curl`` on the same machine works fine, because
    curl uses the system keychain and Python does not.

    Set rather than worked around: the alternative people reach for is an unverified SSL
    context, which would silently accept any certificate for the rest of the process.
    An existing ``SSL_CERT_FILE`` is left alone — if the operator has pointed at a
    corporate bundle, that is the one to use.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi ships with pip and requests
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def _fetch(did: int, name: str) -> pd.DataFrame:
    """Download one OpenML dataset as a frame.

    Imported here rather than at module scope so that ``--help`` and the unit tests do
    not pay for scikit-learn's dataset machinery or touch the network.
    """
    _ensure_certificates()
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=did, as_frame=True, parser="auto")
    frame = bunch.frame
    if frame is None:
        raise ValueError(f"OpenML {did} ({name}) has no tabular frame")
    return frame


def _prepare(frame: pd.DataFrame, name: str, target: str):
    """Put a frame through this product's Stages 1-3 and return the executed run.

    The frame is serialised to CSV and read back through ``read_upload`` on purpose:
    that is the path a user's file takes, including delimiter sniffing, header tidying,
    numeric coercion and the blank-placeholder repairs. A corpus prepared by a shortcut
    around ingest would describe a pipeline this product does not run.
    """
    raw = frame.to_csv(index=False).encode()
    ingested = read_upload(raw, f"{name}.csv")
    plan = propose_for_dataset(ingested, target=target)
    result = execute(ingested.frame, plan, run_id=uuid.uuid4().hex)
    return ingested, plan, result


class Refused(Exception):
    """This dataset cannot produce an honest record.

    Distinct from a failure. A download that times out says nothing about the dataset and
    an existing record for it is still good; a dataset whose best family cannot beat the
    baseline has no winner to record, and a record saying otherwise must not survive a
    rebuild. ``--refresh`` therefore *removes* a stored record whose dataset is now
    refused, which is the whole reason this is an exception and not a ``None``.
    """


#: A winner must beat the majority-class (or mean) baseline by at least this much for
#: the record to be stored.
#:
#: Without a floor the corpus fills up with labels that are pure fold noise. Four of the
#: twenty seed datasets produced a "winner" at or within 0.006 of the dummy — ``pc3`` at
#: 0.8940 against a 0.8940 dummy, ``ilpd`` at 0.7127 against 0.7127. On those datasets no
#: family did anything at all; every model predicted the majority class and the ranking
#: was decided by which one happened to round up. Storing that as "Logistic Regression
#: suits datasets shaped like this" is a fabricated claim, and it is worse than useless
#: in the corpus because the record then casts a full vote in every ranking it is a
#: neighbour for.
#:
#: 0.02 is the floor because five-fold cross-validation on a few hundred rows moves by
#: about that much between folds, so a smaller margin is not distinguishable from noise.
#: It is a floor on *evidence*, not on usefulness: a dataset where the best family beats
#: the baseline by 0.03 is a perfectly good record.
MIN_MARGIN_OVER_DUMMY = 0.02


def build_record(
    did: int,
    name: str,
    target: str,
    *,
    seed: int,
    folds: int,
    holdout: bool,
    min_margin: float = MIN_MARGIN_OVER_DUMMY,
) -> ExperimentRecord | None:
    """Fetch, prepare, train and label one dataset. ``None`` if it cannot be used."""
    print(f"  fetching {name} (did {did})")
    frame = _fetch(did, name)

    if target not in frame.columns:
        raise Refused(
            f"no column named {target!r}; this dataset's columns are "
            f"{', '.join(map(str, frame.columns[:8]))}"
        )

    ingested, plan, result = _prepare(frame, name, target)

    # Meta-features come from the raw frame, the one the user would have uploaded —
    # see services/metafeatures.py. After Stage 3 every column is numeric and
    # `n_categorical` would be zero for every record ever stored.
    features = mf.extract(ingested.frame, plan.target, plan.task)

    classification = plan.task.strip().lower().startswith(("binary", "multiclass", "class"))
    usable_folds = folds
    if classification:
        smallest_class = int(result.y_train.value_counts().min())
        usable_folds = max(2, min(folds, smallest_class))

    print(
        f"      {features['n_rows']} x {features['n_cols']} · {plan.task} · "
        f"training on {len(result.X_train)} rows, {usable_folds} folds"
    )

    labelled = _score_families(
        result.X_train, result.y_train, classification, seed, usable_folds
    )
    if labelled is None:
        print("      skipped: no family produced a score")
        return None

    margin = labelled.scores[labelled.best_family] - labelled.dummy_score
    print(
        f"      winner: {labelled.best_family} "
        f"{labelled.scores[labelled.best_family]:.4f} {labelled.metric} "
        f"(dummy {labelled.dummy_score:.4f}, +{margin:.4f}) "
        f"in {labelled.training_ms} ms{' · HOLDOUT' if holdout else ''}"
    )

    if margin < min_margin:
        raise Refused(
            f"the winner beats the baseline by {margin:.4f}, under the "
            f"{min_margin:.4f} floor. No family did anything on this dataset, so there "
            f"is no winner to record — see MIN_MARGIN_OVER_DUMMY"
        )

    return ExperimentRecord(
        id=uuid.uuid4().hex,
        source="openml",
        source_ref=str(did),
        dataset_name=name,
        target=plan.target,
        task=plan.task,
        metafeature_version=features["metafeature_version"],
        n_rows=features["n_rows"],
        n_cols=features["n_cols"],
        n_numerical=features["n_numerical"],
        n_categorical=features["n_categorical"],
        sparsity=features["sparsity"],
        imbalance_ratio=features["imbalance_ratio"],
        mean_abs_correlation=features["mean_abs_correlation"],
        n_classes=features["n_classes"],
        meta_features=features,
        best_family=labelled.best_family,
        family_scores={
            "scores": labelled.scores,
            "dummy": labelled.dummy_score,
            # Kept so the engine can tell a decisive win from a coin toss without
            # recomputing anything.
            "margin_over_dummy": margin,
        },
        metric=labelled.metric,
        folds=labelled.folds,
        seed=seed,
        training_ms=labelled.training_ms,
        holdout=holdout,
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="only the first N datasets")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=4,
        help=(
            "reserve every Nth dataset for evaluation (NFR-018). These are never "
            "returned by the similarity search. 0 disables."
        ),
    )
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild records that already exist"
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=MIN_MARGIN_OVER_DUMMY,
        help=(
            "a winner must beat the majority-class baseline by at least this much for "
            f"the record to be stored (default {MIN_MARGIN_OVER_DUMMY}). 0 stores "
            "everything, including labels that are fold noise."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="build nothing, store nothing")
    args = parser.parse_args()

    chosen = SEED_DATASETS[: args.limit] if args.limit else SEED_DATASETS
    print(f"Stage 4 corpus: {len(chosen)} dataset(s), seed {args.seed}")

    if args.dry_run:
        for index, (did, name, target) in enumerate(chosen):
            holdout = bool(args.holdout_every) and index % args.holdout_every == args.holdout_every - 1
            print(f"  {did:>6}  {name:<36} target={target:<12}{' HOLDOUT' if holdout else ''}")
        return 0

    stored = skipped = failed = refused = 0

    for index, (did, name, target) in enumerate(chosen):
        holdout = bool(args.holdout_every) and index % args.holdout_every == args.holdout_every - 1
        try:
            with session_scope() as session:
                existing = (
                    session.query(ExperimentRecord)
                    .filter_by(source="openml", source_ref=str(did))
                    .one_or_none()
                )
                if existing is not None and not args.refresh:
                    print(f"  {name}: already recorded, skipping")
                    skipped += 1
                    continue

                try:
                    record = build_record(
                        did,
                        name,
                        target,
                        seed=args.seed,
                        folds=args.folds,
                        holdout=holdout,
                        min_margin=args.min_margin,
                    )
                except Refused as reason:
                    print(f"      REFUSED: {reason}")
                    refused += 1
                    if existing is not None:
                        print(
                            "      removing the stored record, which was built under a "
                            "rule this dataset no longer satisfies"
                        )
                        session.delete(existing)
                    continue
                if record is None:
                    failed += 1
                    continue

                if existing is not None:
                    session.delete(existing)
                    session.flush()
                session.add(record)
                stored += 1
        except Exception as exc:  # noqa: BLE001 - one dataset must not end the job
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
            failed += 1

    print(f"\nstored {stored}, skipped {skipped}, refused {refused}, failed {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
