"""Measure the recommender against datasets it has never seen.

Offline, read-only. Writes nothing to the database::

    cd backend
    MLBENCHX_DATABASE_URL="postgresql+psycopg://user@localhost:5432/mlbenchx" \\
        ../.venv/bin/python scripts/evaluate_corpus.py

This is the reading behind NFR-018 — "the recommender identifies the best-performing
algorithm family for at least 80% of held-out datasets". Every record marked
``holdout`` is queried against the corpus as if it had just been uploaded, and the
family the engine puts first is compared with the family that actually won on it.

Why it lives in ``scripts/``, like the builder
----------------------------------------------
It is an evaluation harness, not a product feature. Nothing in ``app/`` imports it, and
nothing in it may be imported by ``app/``: a KPI measured by code the application also
runs is a KPI that will eventually be tuned until it passes.

Three things this reports that a simpler harness would leave out
----------------------------------------------------------------
**The majority-class baseline.** Top-1 accuracy on its own is not evidence of anything.
If twelve of fifteen corpus records were won by Logistic Regression, a recommender that
always answered "Logistic Regression" would score well without having learned a thing.
The engine has to beat that number, and the gap is the only part of this report that
says whether the similarity search is doing work.

**Declines, counted separately.** The engine refuses to rank when the corpus is too
small, has nothing of the same task, or holds nothing similar enough. A refusal is not a
wrong answer and not a right one, and folding it into either would misreport the engine.
Accuracy is quoted over the records that produced a ranking, with the refusals stated
next to it.

**The resolution of the measurement.** With five held-out datasets each one is worth 20
percentage points, so "80%" means "four of five" and no intermediate value exists. The
report says so rather than printing a decimal that implies precision it does not have.

**Ties.** Families trained at library defaults routinely land within a thousandth of
each other. Naming a family that scored 0.9568 when the winner scored 0.9568 is not a
miss, so a near-tie is reported as its own outcome rather than being scored either way.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select

from app.db import ExperimentRecord, SessionLocal
from app.services import metalearning as ml
from app.services import rationale

#: Two families whose cross-validated scores differ by less than this are not
#: meaningfully different at library defaults, so naming either one is counted as a tie
#: rather than a hit or a miss.
TIE_TOLERANCE = 0.005


@dataclass
class Outcome:
    """One held-out dataset's result."""

    dataset: str
    truth: str
    predicted: str | None
    #: The held-out dataset's own recorded score for the family the engine named.
    predicted_score: float | None
    #: The held-out dataset's own recorded score for the family that actually won.
    best_score: float | None
    similarity: float | None
    neighbours: list[str]
    declined: str | None
    #: The sentences the user would have been shown for this dataset.
    rationale: list[str]

    @property
    def verdict(self) -> str:
        if self.declined is not None:
            return "declined"
        if self.predicted == self.truth:
            return "hit"
        if (
            self.predicted_score is not None
            and self.best_score is not None
            and self.best_score - self.predicted_score < TIE_TOLERANCE
        ):
            return "tie"
        return "miss"


def _records(session) -> list[dict]:
    """The corpus as the engine consumes it — the same shape ``metalearning_api`` builds."""
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
        for row in session.scalars(select(ExperimentRecord))
    ]


def _scores(record: dict) -> dict[str, float]:
    return dict((record.get("family_scores") or {}).get("scores", {}))


def evaluate(records: list[dict], *, k: int | None = None) -> list[Outcome]:
    """Query every holdout record against the corpus and record what came back."""
    held = [r for r in records if r.get("holdout")]
    outcomes: list[Outcome] = []

    for record in held:
        kwargs = {"k": k} if k else {}
        result = ml.recommend(record["meta_features"], records, **kwargs)

        # The validity of this entire measurement rests on the query not being able to
        # see itself. Asserted per record rather than trusted: a leak here would not
        # fail, it would quietly produce a flattering number.
        returned = [n.record_id for n in result.neighbours]
        assert record["id"] not in returned, (
            f"{record['dataset_name']} was returned as its own neighbour — "
            f"the holdout filter is not holding and this reading is worthless"
        )
        for other in held:
            assert other["id"] not in returned, (
                f"holdout record {other['dataset_name']} was returned as a neighbour "
                f"for {record['dataset_name']}"
            )

        own = _scores(record)
        predicted = result.top.family if result.top else None
        outcomes.append(
            Outcome(
                dataset=str(record["dataset_name"]),
                truth=str(record["best_family"]),
                predicted=predicted,
                predicted_score=own.get(predicted) if predicted else None,
                best_score=own.get(str(record["best_family"])),
                similarity=result.neighbours[0].similarity if result.neighbours else None,
                neighbours=[n.dataset_name for n in result.neighbours],
                declined=result.declined,
                rationale=rationale.explain(result),
            )
        )

    return outcomes


def majority_baseline(records: list[dict], outcomes: list[Outcome]) -> tuple[str, int]:
    """The family to name if you never looked at the data, and how often it is right.

    This is the number the engine has to beat. It is computed from the *usable* records
    only, because that is all a recommender with no similarity search would have.
    """
    usable = [r for r in records if not r.get("holdout")]
    counts: dict[str, int] = {}
    for record in usable:
        family = str(record["best_family"])
        counts[family] = counts.get(family, 0) + 1
    if not counts:
        return "", 0
    favourite = max(counts, key=lambda family: (counts[family], family))
    right = len([o for o in outcomes if o.truth == favourite])
    return favourite, right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=None, help="override K_NEIGHBOURS")
    parser.add_argument(
        "--rationale", action="store_true", help="print the rationale for each holdout"
    )
    args = parser.parse_args()

    with SessionLocal() as session:
        records = _records(session)

    usable = [r for r in records if not r.get("holdout")]
    held = [r for r in records if r.get("holdout")]

    print(f"corpus: {len(records)} stored · {len(usable)} usable · {len(held)} held out")
    if not held:
        print("\nNothing is marked holdout, so there is nothing to measure.")
        print("Seed with --holdout-every N to reserve an evaluation subset.")
        return 1

    outcomes = evaluate(records, k=args.k)

    print()
    for outcome in outcomes:
        mark = {"hit": "HIT ", "tie": "TIE ", "miss": "MISS", "declined": "----"}[
            outcome.verdict
        ]
        print(f"{mark} {outcome.dataset}")
        print(f"       actual best: {outcome.truth}")
        if outcome.declined:
            print(f"       declined: {outcome.declined}")
        else:
            print(
                f"       engine said: {outcome.predicted}"
                + (
                    f" (scored {outcome.predicted_score:.4f} here, "
                    f"winner {outcome.best_score:.4f})"
                    if outcome.predicted_score is not None
                    and outcome.best_score is not None
                    else ""
                )
            )
            print(
                f"       nearest: {', '.join(outcome.neighbours[:3])} "
                f"(top similarity {outcome.similarity:.3f})"
            )
        if args.rationale:
            for line in outcome.rationale:
                print(f"       | {line}")
        print()

    ranked = [o for o in outcomes if o.verdict != "declined"]
    hits = [o for o in outcomes if o.verdict == "hit"]
    ties = [o for o in outcomes if o.verdict == "tie"]
    declined = [o for o in outcomes if o.verdict == "declined"]

    print("-" * 78)
    print(f"held out            {len(outcomes)}")
    print(f"ranked              {len(ranked)}")
    print(f"declined            {len(declined)}")
    print(f"exact hits          {len(hits)}")
    print(f"within {TIE_TOLERANCE:.3f} of best {len(ties)}")

    if ranked:
        strict = len(hits) / len(ranked)
        lenient = (len(hits) + len(ties)) / len(ranked)
        print()
        print(f"top-1 (exact)       {len(hits)}/{len(ranked)} = {strict:.0%}")
        print(f"top-1 (within tie)  {len(hits) + len(ties)}/{len(ranked)} = {lenient:.0%}")

    favourite, baseline_right = majority_baseline(records, outcomes)
    if favourite:
        print()
        print(
            f"majority baseline   always answering {favourite!r} would be exactly right "
            f"on {baseline_right}/{len(outcomes)} of these"
        )
        print(
            "                    the engine has to beat this; anything at or below it "
            "means the similarity search contributed nothing"
        )

    print()
    print(
        f"resolution          {len(outcomes)} held-out datasets, so each is worth "
        f"{100 / len(outcomes):.0f} percentage points. NFR-018's 80% is "
        f"{0.8 * len(outcomes):.0f} of {len(outcomes)}; no value between the steps exists."
    )
    print(
        "                    this is an indicative reading on a small corpus, not "
        "NFR-018's formal validation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
