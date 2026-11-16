"""Stage 4 routes — the recommendation, and the corpus behind it.

Two reads. Neither trains anything, and neither writes.

The recommendation is computed on request rather than stored on the run. The corpus
grows — every seeding job and, later, every local run adds to it — so a ranking cached
at run time would silently describe a history that no longer exists. Recomputing is a
cosine against a few hundred rows, which costs less than the cache would.

The corpus is read whole rather than filtered in SQL. At the scale PRD.md §10 funds —
50 to 100 records — the query that avoids loading them is slower than the load, and the
engine needs the full vectors anyway to fit its scaler.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import ExperimentRecord, Run, get_session
from app.services import metafeatures as mf
from app.services import metalearning, rationale
from app.services.families import FAMILIES

router = APIRouter(tags=["meta-learning"])


def _records(session: Session) -> list[dict[str, Any]]:
    """Every stored experiment, as the plain dicts the engine consumes.

    Holdout records are loaded, not filtered out here: the engine excludes them, and one
    place that knows the rule is better than two that could disagree about it. The count
    is reported so a reader can see the corpus is larger than the part that was used.
    """
    rows = session.scalars(select(ExperimentRecord)).all()
    return [
        {
            "id": row.id,
            "dataset_name": row.dataset_name,
            "task": row.task,
            "holdout": row.holdout,
            "metafeature_version": row.metafeature_version,
            "meta_features": row.meta_features,
            "best_family": row.best_family,
            "family_scores": row.family_scores,
            "metric": row.metric,
            "n_rows": row.n_rows,
            "n_cols": row.n_cols,
        }
        for row in rows
    ]


@router.get("/api/runs/{run_id}/recommendation", summary="Which algorithms to try")
def recommend_for_run(run_id: str, session: Session = Depends(get_session)) -> Any:
    """Rank the candidate families for the dataset this run prepared.

    Reads the run's stored meta-features — not the dataset file. The run already
    recorded what its dataset looked like, so this costs no ingest and no refit, and it
    describes the data as it was when the run happened rather than as the file is now.

    A run that failed has no meta-features and gets a 409: there is nothing to compare,
    and a 404 would wrongly say the run does not exist.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")
    if run.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=(
                f"This run {run.status}, so there is nothing prepared to recommend "
                f"against. Run the plan successfully first."
            ),
        )

    features = (run.feature_schema or {}).get("meta_features")
    if not features:
        raise HTTPException(
            status_code=409,
            detail=(
                "This run was recorded before meta-features were captured, so it cannot "
                "be compared with the experiment history. Re-running the plan records "
                "them."
            ),
        )

    result = metalearning.recommend(features, _records(session))

    return {
        "run_id": run_id,
        "task": run.task,
        "declined": result.declined,
        "corpus": {
            "stored": result.corpus_size,
            "considered": result.considered,
            "metafeature_version": result.metafeature_version,
        },
        "ranking": [
            {
                "family": item.family,
                "label": item.label,
                "score": round(item.score, 4),
                "neighbours_scored": item.neighbours_scored,
                "spread": round(item.spread, 4),
            }
            for item in result.ranking
        ],
        "neighbours": [
            {
                "dataset": neighbour.dataset_name,
                "similarity": round(neighbour.similarity, 4),
                "best_family": neighbour.best_family,
                "n_rows": neighbour.n_rows,
                "n_cols": neighbour.n_cols,
                "metric": neighbour.metric,
                "scores": {k: round(v, 4) for k, v in neighbour.scores.items()},
            }
            for neighbour in result.neighbours
        ],
        "rationale": rationale.explain(result),
    }


@router.get("/api/experiments", summary="The experiment history")
def list_experiments(session: Session = Depends(get_session)) -> Any:
    """What the corpus holds, so a thin recommendation can be understood rather than
    guessed at.

    Reports the holdout count separately: those records exist and are deliberately not
    used, and a user comparing "stored" against "considered" needs the difference
    explained rather than left as a discrepancy.
    """
    rows = session.scalars(select(ExperimentRecord)).all()

    by_task: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_task[row.task] = by_task.get(row.task, 0) + 1
        by_family[row.best_family] = by_family.get(row.best_family, 0) + 1

    current = [r for r in rows if r.metafeature_version == mf.METAFEATURE_VERSION]

    return {
        "stored": len(rows),
        "usable": len([r for r in current if not r.holdout]),
        "holdout": len([r for r in rows if r.holdout]),
        "stale": len(rows) - len(current),
        "metafeature_version": mf.METAFEATURE_VERSION,
        "families": list(FAMILIES),
        "by_task": by_task,
        "winners": by_family,
        "records": [
            {
                "dataset": row.dataset_name,
                "source": row.source,
                "source_ref": row.source_ref,
                "task": row.task,
                "n_rows": row.n_rows,
                "n_cols": row.n_cols,
                "best_family": row.best_family,
                "metric": row.metric,
                "holdout": row.holdout,
            }
            for row in sorted(rows, key=lambda r: r.dataset_name)
        ],
    }
