"""Training and explaining one stored run, and persisting the results beside it.

This is the seam between the run resource — which already exists, is versioned, and has
already written ``train.parquet`` and ``test.parquet`` — and Stages 5 and 6.

Why the results are artifacts rather than a new table
-----------------------------------------------------
A leaderboard is a property of one run over one plan version. It has no independent
identity, nothing joins to it, and nothing queries across leaderboards. Stored as two
JSON artifacts of the run it belongs to, it inherits three things already built and
tested: the storage key discipline in :func:`~app.services.planning.artifact_key`, the
download route that serves any name in ``ARTIFACT_NAMES``, and the ``run.artifacts``
list the interface already renders. A table would have given the same answers and
required a migration.

The trade this makes: a leaderboard cannot be queried with SQL across runs. When
Stage 7's cross-run dashboard needs that, promote it then — the artifact is the record
either way, exactly as ``feature_schema`` was promoted onto the run row for Stage 4.

Where the data comes from
-------------------------
The two parquet artifacts, read back from storage. **Not** re-prepared. Re-running the
pipeline here would mean the models were trained on a second execution of the plan
rather than on the one the user reviewed, and the leaderboard would describe a run that
has no artifacts.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.config import SEED
from app.db import Run
from app.services import explainability, training
from app.services.families import label as family_label
from app.services.planning import ARTIFACT_NAMES, artifact_key
from app.storage import get_storage

__all__ = ["ResultsError", "train_run", "stored_results"]


class ResultsError(Exception):
    """A training request that cannot be carried out, with a sentence saying why."""


def _read_fold(run: Run, name: str) -> pd.DataFrame:
    """One of the run's two parquet folds, as the frame it was written from."""
    key = (run.artifacts or {}).get(name)
    if not key:
        raise ResultsError(
            f"This run has no {name} data stored, so nothing can be trained on it. "
            "Runs that failed do not write artifacts."
        )
    return pd.read_parquet(io.BytesIO(get_storage().get(key)))


def _split_target(frame: pd.DataFrame, target: str, fold: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target, as the executor stored them.

    The target was appended to each fold by ``planning._write_artifacts``, so splitting
    it back out here is reading the same contract rather than guessing at a convention.
    """
    if target not in frame.columns:
        raise ResultsError(
            f"The {fold} fold does not contain the target column {target!r}. "
            "The run's artifacts and its recorded target disagree, which should not "
            "be possible — re-run the plan."
        )
    return frame.drop(columns=[target]), frame[target]


def train_run(
    session: Session, run: Run, max_workers: int = training.MAX_WORKERS
) -> dict[str, Any]:
    """Train the six families on this run's data, explain the winner, and store both.

    Returns the leaderboard payload. The fitted estimators are used for the explanation
    and then dropped: persisting six models would multiply this run's storage by an
    order of magnitude for an artifact nothing in the MVP reads back.
    """
    if run.status != "succeeded":
        raise ResultsError(
            f"This run's status is {run.status!r}. Only a run that finished has data to "
            "train on." + (f" It failed: {(run.error or {}).get('message', '')}" if run.error else "")
        )

    train_frame = _read_fold(run, "train")
    test_frame = _read_fold(run, "test")
    X_train, y_train = _split_target(train_frame, run.target, "training")
    X_test, y_test = _split_target(test_frame, run.target, "held-out")

    report = training.train_all(
        X_train, y_train, X_test, y_test, run.task, seed=run.seed or SEED,
        max_workers=max_workers,
    )

    # ---- explain the winner -------------------------------------------------------
    #
    # One explanation, for the family that topped the leaderboard. Explaining all six
    # would multiply the cost for an answer about five models the user is not going to
    # use, and PRD.md §10 asks for the top model.
    if report.best_family and report.best_family in report.fitted:
        kind = "classification" if training.is_classification(run.task) else "regression"
        explanation = explainability.explain_model(
            report.fitted[report.best_family],
            report.best_family,
            family_label(report.best_family, kind),
            X_test,
            y_test,
            background=report.fit_frames.get(report.best_family, X_train),
            seed=run.seed or SEED,
        )
    else:
        explanation = {
            "status": "unavailable",
            "reason": (
                "No family produced a score, so there is no model to explain. Each "
                "family's row in the leaderboard carries its own reason."
            ),
            "is_shap": False,
            "method": None,
        }

    payload = report.to_dict()
    payload["explanation_method"] = explanation.get("method")

    # ---- persist ------------------------------------------------------------------
    #
    # Both artifacts are written before the row is touched, so `run.artifacts` never
    # names a file that is not there — the same ordering `_write_artifacts` uses.
    storage = get_storage()
    written: dict[str, str] = {}
    for name, body in (("models", payload), ("explain", explanation)):
        key = artifact_key(run.id, name)
        storage.put(key, json.dumps(body, indent=2, default=str).encode("utf-8"))
        written[name] = key

    # Reassigned rather than mutated: SQLAlchemy tracks JSON columns by identity, and an
    # in-place `dict[...] = ...` on this attribute is not seen as a change and is never
    # written back.
    run.artifacts = {**(run.artifacts or {}), **written}
    session.add(run)

    return payload


def stored_results(run: Run, name: str) -> dict[str, Any] | None:
    """A previously stored ``models`` or ``explain`` artifact, or ``None``.

    Lets the interface show a leaderboard after a reload without retraining, which on
    this hardware is the difference between an instant screen and a minute of waiting.
    """
    if name not in ARTIFACT_NAMES:
        raise ResultsError(f"There is no artifact called {name!r}.")
    key = (run.artifacts or {}).get(name)
    if not key:
        return None
    try:
        return json.loads(get_storage().get(key))
    except Exception:
        return None
