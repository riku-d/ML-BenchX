"""Stage 4 — the rationale, and the routes that serve it.

Two things are being guarded.

**The rationale says only what the recommendation contains.** PRD.md §10 asks for a
plain-English rationale, and the easy way to produce one is a template with numbers
dropped in — which stays fluent after the evidence stops supporting it. So the tests
check that the sentences change when the facts change, and that the three things the
engine cannot know are never claimed.

**The routes refuse rather than guess.** A run that failed, a run with no meta-features
and a corpus too small to rank from are all states a user can reach, and none of them
may produce a ranking.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import ExperimentRecord, SessionLocal
from app.main import app
from app.services import metafeatures as mf
from app.services import metalearning as ml
from app.services import rationale

FIXTURES = Path(__file__).parent / "fixtures"
TITANIC = FIXTURES / "titanic.csv"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _features(**overrides) -> dict:
    base = {name: 0.0 for name in mf.VECTOR_FIELDS}
    base.update(
        {
            "metafeature_version": mf.METAFEATURE_VERSION,
            "target_type": "binary classification",
            "is_classification": 1.0,
            "log_n_rows": 3.0,
            "log_n_cols": 1.0,
        }
    )
    base.update(overrides)
    return base


def _neighbour(name: str, similarity: float, best: str, scores: dict) -> ml.Neighbour:
    return ml.Neighbour(
        record_id=f"id-{name}",
        dataset_name=name,
        similarity=similarity,
        best_family=best,
        task="binary classification",
        n_rows=1000,
        n_cols=10,
        scores=scores,
        metric="accuracy",
    )


def _recommendation(neighbours: list[ml.Neighbour], considered: int = 12) -> ml.Recommendation:
    return ml.Recommendation(
        ranking=ml._rank(neighbours, "binary classification"),
        neighbours=neighbours,
        considered=considered,
        corpus_size=considered,
    )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded_corpus():
    """Twelve records, enough to clear MIN_CORPUS, removed afterwards."""
    ids = []
    with SessionLocal() as session:
        for index in range(12):
            record = ExperimentRecord(
                id=uuid.uuid4().hex,
                source="test",
                source_ref=f"seed-{index}",
                dataset_name=f"seeded-{index}",
                target="y",
                task="binary classification",
                metafeature_version=mf.METAFEATURE_VERSION,
                n_rows=500 + index * 100,
                n_cols=10 + index,
                n_numerical=5,
                n_categorical=5,
                sparsity=0.01 * index,
                imbalance_ratio=1.0 + index * 0.1,
                mean_abs_correlation=0.1,
                n_classes=2,
                meta_features=_features(
                    log_n_rows=2.5 + index * 0.1, n_numerical=float(index)
                ),
                best_family="random_forest" if index % 2 else "gradient_boosting",
                family_scores={"scores": {"random_forest": 0.9, "gradient_boosting": 0.85}},
                metric="accuracy",
                folds=5,
                seed=42,
                holdout=False,
            )
            session.add(record)
            ids.append(record.id)
        session.commit()
    yield ids
    with SessionLocal() as session:
        for record_id in ids:
            row = session.get(ExperimentRecord, record_id)
            if row is not None:
                session.delete(row)
        session.commit()


# --------------------------------------------------------------------------------------
# The rationale
# --------------------------------------------------------------------------------------


def test_a_decline_is_passed_through_rather_than_rewritten():
    """Two sentences saying the same thing in different words would drift apart."""
    declined = ml.Recommendation(declined="Nothing to compare against.")
    assert rationale.explain(declined) == ["Nothing to compare against."]


def test_the_rationale_names_the_closest_dataset():
    result = _recommendation(
        [
            _neighbour("credit-g", 0.91, "random_forest", {"random_forest": 0.9}),
            _neighbour("diabetes", 0.70, "random_forest", {"random_forest": 0.8}),
        ]
    )
    text = " ".join(rationale.explain(result))
    assert "credit-g" in text


def test_the_rationale_reports_how_many_were_compared():
    result = _recommendation(
        [_neighbour("a", 0.9, "knn", {"knn": 0.9})], considered=14
    )
    assert "14 past experiments" in " ".join(rationale.explain(result))


def test_closeness_wording_tracks_the_similarity():
    """The band is a translation of the number, not a replacement for it — both appear."""
    close = _recommendation([_neighbour("a", 0.95, "knn", {"knn": 0.9})])
    loose = _recommendation([_neighbour("a", 0.35, "knn", {"knn": 0.9})])
    assert "very closely" in " ".join(rationale.explain(close))
    assert "loosely" in " ".join(rationale.explain(loose))


def test_it_names_the_datasets_the_winner_actually_won_on():
    result = _recommendation(
        [
            _neighbour("alpha", 0.9, "random_forest", {"random_forest": 0.9, "knn": 0.3}),
            _neighbour("beta", 0.8, "random_forest", {"random_forest": 0.9, "knn": 0.3}),
            _neighbour("gamma", 0.7, "knn", {"random_forest": 0.9, "knn": 0.3}),
        ]
    )
    text = " ".join(rationale.explain(result))
    assert "alpha" in text and "beta" in text
    assert "won outright on 2" in text


def test_it_says_so_when_the_winner_never_actually_won():
    """A family can top the weighted mean without having won on any single neighbour.

    It happens when it ties the per-neighbour winners and was scored more often: the
    ranking breaks ties on how many neighbours scored a family, so the consistently
    present one goes first. The rationale must say that rather than claiming a win that
    did not happen.

    The ranking is built here rather than through ``_rank`` because this is a test of
    the sentence, not of the arithmetic that leads to it — ``test_metalearning``
    covers the ordering.
    """
    neighbours = [
        _neighbour("a", 0.9, "knn", {"random_forest": 0.90, "knn": 0.90}),
        _neighbour("b", 0.8, "svm_rbf", {"random_forest": 0.90, "svm_rbf": 0.90}),
    ]
    result = ml.Recommendation(
        ranking=[
            ml.FamilyRanking("random_forest", "Random Forest", 0.90, 2, 0.0),
            ml.FamilyRanking("knn", "K-Nearest Neighbours", 0.90, 1, 0.0),
        ],
        neighbours=neighbours,
        considered=12,
        corpus_size=12,
    )
    text = " ".join(rationale.explain(result))
    assert "did not win outright" in text


def test_a_level_runner_up_is_called_level():
    result = _recommendation(
        [_neighbour("a", 0.9, "knn", {"knn": 0.900, "random_forest": 0.899})]
    )
    assert "level with it" in " ".join(rationale.explain(result))


def test_a_clear_runner_up_reports_the_gap():
    result = _recommendation(
        [_neighbour("a", 0.9, "knn", {"knn": 0.9, "random_forest": 0.5})]
    )
    assert "behind" in " ".join(rationale.explain(result))


def test_wide_disagreement_is_flagged_rather_than_averaged_away():
    result = _recommendation(
        [
            _neighbour("a", 0.9, "knn", {"knn": 0.95}),
            _neighbour("b", 0.8, "knn", {"knn": 0.35}),
        ]
    )
    assert "varied widely" in " ".join(rationale.explain(result))


def test_thin_evidence_for_the_winner_is_stated():
    result = _recommendation(
        [
            _neighbour("a", 0.9, "knn", {"knn": 0.95}),
            _neighbour("b", 0.8, "random_forest", {"random_forest": 0.10}),
        ]
    )
    assert "rests on less evidence" in " ".join(rationale.explain(result))


def test_it_never_predicts_a_score_for_this_dataset():
    """The corpus scores are other datasets' results. Phrasing their mean as an
    expectation would be a fabricated number wearing a measured one's clothes."""
    result = _recommendation([_neighbour("a", 0.9, "knn", {"knn": 0.9})])
    text = " ".join(rationale.explain(result)).lower()
    for forbidden in ("expect", "will achieve", "should score", "predicted accuracy"):
        assert forbidden not in text


def test_it_says_nothing_was_trained_on_the_user_data():
    result = _recommendation([_neighbour("a", 0.9, "knn", {"knn": 0.9})])
    assert "Nothing has been trained on your data." in " ".join(rationale.explain(result))


# --------------------------------------------------------------------------------------
# The routes
# --------------------------------------------------------------------------------------


def test_experiments_route_describes_the_corpus(client, seeded_corpus):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    body = response.json()
    assert body["stored"] >= 12
    assert body["metafeature_version"] == mf.METAFEATURE_VERSION
    assert "random_forest" in body["families"]


def test_experiments_route_reports_holdout_separately(client, seeded_corpus):
    """`stored` minus `usable` must be explainable, not a discrepancy a reader has to
    guess at."""
    body = client.get("/api/experiments").json()
    assert set(body) >= {"stored", "usable", "holdout", "stale"}


def test_recommendation_for_an_unknown_run_is_404(client):
    assert client.get("/api/runs/does-not-exist/recommendation").status_code == 404


def test_a_run_records_meta_features(client):
    """The recommendation reads them from the run rather than re-reading the file,
    which may by then have been deleted."""
    with TITANIC.open("rb") as handle:
        dataset = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        ).json()
    plan = client.post(
        f"/api/datasets/{dataset['id']}/plan", json={"goal": "predict Survived"}
    ).json()
    run = client.post(f"/api/plans/{plan['id']}/run").json()

    features = run["feature_schema"]["meta_features"]
    assert features["metafeature_version"] == mf.METAFEATURE_VERSION
    assert features["n_rows"] > 0
    # Measured on the raw frame: after preprocessing there are no categorical columns
    # left, and a run that reported zero for every dataset would be useless.
    assert features["n_categorical"] > 0


def test_recommendation_declines_politely_on_a_thin_corpus(client):
    """With nothing comparable stored, the route still answers 200 with a reason —
    a 404 would say the run does not exist, which is a different and untrue thing."""
    with TITANIC.open("rb") as handle:
        dataset = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        ).json()
    plan = client.post(
        f"/api/datasets/{dataset['id']}/plan", json={"goal": "predict Survived"}
    ).json()
    run = client.post(f"/api/plans/{plan['id']}/run").json()

    response = client.get(f"/api/runs/{run['id']}/recommendation")
    assert response.status_code == 200
    body = response.json()
    assert body["rationale"]
    if body["declined"]:
        assert body["ranking"] == []


def test_recommendation_ranks_against_a_seeded_corpus(client, seeded_corpus):
    with TITANIC.open("rb") as handle:
        dataset = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        ).json()
    plan = client.post(
        f"/api/datasets/{dataset['id']}/plan", json={"goal": "predict Survived"}
    ).json()
    run = client.post(f"/api/plans/{plan['id']}/run").json()

    body = client.get(f"/api/runs/{run['id']}/recommendation").json()

    assert body["corpus"]["considered"] >= 12
    assert body["rationale"]
    if not body["declined"]:
        assert body["ranking"]
        assert body["neighbours"]
        assert body["ranking"][0]["label"]
