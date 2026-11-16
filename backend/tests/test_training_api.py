"""The Stage 5-7 routes, over HTTP, against a real uploaded file.

These go through the whole product — upload, plan, run, train — rather than calling the
services directly, because the claims being checked are claims about what a user can
reach from the interface. A leaderboard that is correct in a unit test and unreachable
over HTTP is not a feature.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = Path(__file__).resolve().parent.parent / "app" / "sample" / "campus_records.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def trained_run(client: TestClient) -> dict:
    """One dataset carried all the way to a trained run, reused by every test here.

    Module-scoped because training six families is the expensive step and nothing below
    mutates the run.
    """
    with SAMPLE.open("rb") as handle:
        upload = client.post(
            "/api/datasets", files={"file": ("campus_records.csv", handle, "text/csv")}
        )
    assert upload.status_code in (200, 201), upload.text
    dataset_id = upload.json()["id"]

    plan = client.post(
        f"/api/datasets/{dataset_id}/plan",
        json={"goal": "predict whether a student has a backlog", "target": "has_backlog"},
    )
    assert plan.status_code in (200, 201), plan.text
    plan_id = plan.json()["id"]

    run = client.post(f"/api/plans/{plan_id}/run")
    assert run.status_code in (200, 201), run.text
    run_body = run.json()
    assert run_body["status"] == "succeeded", run_body

    trained = client.post(f"/api/runs/{run_body['id']}/train")
    assert trained.status_code in (200, 201), trained.text

    return {"dataset_id": dataset_id, "run_id": run_body["id"], "board": trained.json()}


def test_training_returns_six_families_with_held_out_scores(trained_run: dict) -> None:
    board = trained_run["board"]

    assert len(board["models"]) == 6
    assert board["best_family"]
    assert board["split"]["n_test_rows"] > 0
    succeeded = [row for row in board["models"] if row["status"] == "succeeded"]
    assert succeeded, "no family scored on the bundled sample"
    for row in succeeded:
        assert isinstance(row["primary_metric_value"], float)
        assert row["metrics"]["accuracy"] >= 0.0


def test_the_leaderboard_reports_measured_concurrency(trained_run: dict) -> None:
    """Three workers, six jobs, and an overlap figure that came from a clock."""
    concurrency = trained_run["board"]["concurrency"]

    assert concurrency["submitted"] == 6
    assert concurrency["workers"] == 3
    assert concurrency["threads_per_model"] == 1
    assert 1 <= concurrency["max_observed_overlap"] <= 3


def test_no_model_row_ever_carries_a_fabricated_zero(trained_run: dict) -> None:
    """A row either succeeded and has a number, or did not and has a sentence."""
    for row in trained_run["board"]["models"]:
        if row["status"] == "succeeded":
            assert row["primary_metric_value"] is not None
            assert "reason" not in row or not row.get("reason")
        else:
            assert row["primary_metric_value"] is None
            assert row["reason"], f"{row['family']} failed without saying why"


def test_the_explanation_names_its_own_method(client: TestClient, trained_run: dict) -> None:
    response = client.get(f"/api/runs/{trained_run['run_id']}/explanation")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "succeeded"
    assert body["method"] in {"shap_tree", "shap_linear", "permutation"}
    assert isinstance(body["is_shap"], bool)
    # The one rule that must never break.
    if not body["is_shap"]:
        assert "not SHAP" in body["method_label"]
    else:
        assert "SHAP" in body["method_label"]
    assert body["features"]


def test_the_leaderboard_survives_a_reload(client: TestClient, trained_run: dict) -> None:
    """Stored as an artifact, so reopening the screen does not retrain."""
    response = client.get(f"/api/runs/{trained_run['run_id']}/models")
    assert response.status_code == 200
    assert response.json()["best_family"] == trained_run["board"]["best_family"]


def test_models_and_explanation_join_the_run_artifact_list(
    client: TestClient, trained_run: dict
) -> None:
    """Which is what puts them in the interface's download list with no new UI."""
    artifacts = client.get(f"/api/runs/{trained_run['run_id']}").json()["artifacts"]

    assert "models" in artifacts
    assert "explain" in artifacts
    for name in ("models", "explain"):
        one = client.get(f"/api/runs/{trained_run['run_id']}/artifacts/{name}")
        assert one.status_code == 200
        assert json.loads(one.content)


def test_an_untrained_run_says_so_rather_than_returning_an_empty_board(
    client: TestClient, trained_run: dict
) -> None:
    """"Nothing trained" and "six models all scored zero" are different facts."""
    plan = client.post(
        f"/api/datasets/{trained_run['dataset_id']}/plan",
        json={"goal": "", "target": "has_backlog"},
    )
    run = client.post(f"/api/plans/{plan.json()['id']}/run")
    fresh = run.json()["id"]

    response = client.get(f"/api/runs/{fresh}/models")
    assert response.status_code == 404
    # The error envelope is `{error, code, message, detail}` — AUDIT.md §8.2 O6. The
    # sentence a human reads is `message`; `detail` carries structure, or nothing.
    assert "trained" in response.json()["message"].lower()


def test_the_bundle_contains_every_artifact_and_a_readme(
    client: TestClient, trained_run: dict
) -> None:
    response = client.get(f"/api/runs/{trained_run['run_id']}/bundle.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())

    assert {
        "train.parquet",
        "test.parquet",
        "plan.json",
        "schema.json",
        "log.json",
        "pipeline.joblib",
        "pipeline.py",
        "models.json",
        "explanation.json",
        "README.txt",
        "run.json",
    } <= names

    readme = archive.read("README.txt").decode()
    assert trained_run["run_id"] in readme
    assert "held-out fold" in readme


def test_the_recommendation_is_reachable_and_states_whether_it_declined(
    client: TestClient, trained_run: dict
) -> None:
    """A refusal is a documented outcome, but it must be *reported* as a refusal.

    This test accepts either answer, because whether a corpus exists is a property of
    the environment rather than of the code. What it does not accept is a ranking
    presented alongside a refusal, or a refusal with no reason.
    """
    response = client.get(f"/api/runs/{trained_run['run_id']}/recommendation")
    assert response.status_code == 200

    body = response.json()
    if body["declined"]:
        assert not body["ranking"], "declined and yet returned a ranking"
        reason = body["declined"]
        assert (reason if isinstance(reason, str) else reason.get("reason")), (
            "declined without saying why"
        )
    else:
        assert len(body["ranking"]) == 6
        assert body["rationale"], "ranked without a plain-English rationale"
        assert body["neighbours"]
