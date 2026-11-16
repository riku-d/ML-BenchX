"""Stage 4 — the similarity search and the ranking built from it.

The engine is pure, so these tests are the engine: a list of dicts in, a recommendation
out, no database and no fixtures beyond the records themselves.

What is actually being guarded here:

* **the holdout boundary**, because NFR-018 cannot be measured on records that produced
  the answer, and a leak would never show up as a failure — only as a KPI that looked
  better than it was;
* **the three decline paths**, because a confident ranking from an empty or unlike
  corpus is the failure this stage is most likely to ship;
* **the arithmetic**, against values computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import SCALER_CLIP
from app.services import metafeatures as mf
from app.services import metalearning as ml


def _features(**overrides) -> dict:
    """A complete meta-feature dict, with named fields overridden.

    The task flags are derived from ``target_type`` rather than set independently,
    because ``extract`` derives them too — a fixture that let the two disagree would be
    testing a vector the real extractor cannot produce.
    """
    base = {name: 0.0 for name in mf.VECTOR_FIELDS}
    base.update(
        {
            "metafeature_version": mf.METAFEATURE_VERSION,
            "target_type": "binary classification",
            "log_n_rows": 3.0,
            "log_n_cols": 1.0,
            "n_numerical": 5.0,
            "n_categorical": 5.0,
        }
    )
    base.update(overrides)
    classification = str(base["target_type"]).lower().startswith(
        ("binary", "multiclass", "class")
    )
    base["is_classification"] = 1.0 if classification else 0.0
    base["is_regression"] = 0.0 if classification else 1.0
    return base


def _record(
    name: str,
    *,
    best: str = "random_forest",
    scores: dict | None = None,
    holdout: bool = False,
    task: str = "binary classification",
    version: int | None = None,
    **feature_overrides,
) -> dict:
    return {
        "id": f"id-{name}",
        "dataset_name": name,
        "task": task,
        "holdout": holdout,
        "metafeature_version": mf.METAFEATURE_VERSION if version is None else version,
        "meta_features": _features(target_type=task, **feature_overrides),
        "best_family": best,
        "family_scores": {"scores": scores or {"random_forest": 0.9, "knn": 0.6}},
        "metric": "accuracy",
        "n_rows": 1000,
        "n_cols": 10,
    }


def _corpus(n: int = 10, **kwargs) -> list[dict]:
    """A corpus large enough to clear MIN_CORPUS, varied enough to have a shape."""
    return [
        _record(f"ds{i}", log_n_rows=2.0 + i * 0.3, n_numerical=float(i), **kwargs)
        for i in range(n)
    ]


#: The three non-zero ``sparsity`` values from the first real 16-record corpus, where the
#: other nine usable records sat at exactly 0.0. That 9-of-12 split is what makes this
#: pathological rather than merely skewed: the lower quartile lands inside the block of
#: zeros and the upper quartile lands a fraction of the way to the smallest non-zero
#: value, so the IQR comes out at 0.000635 — small enough to pass a zero check and then
#: amplify that coordinate by seventy-eight.
DEGENERATE_VALUES = (0.00254, 0.03204, 0.05)


def _scaler_parts(matrix: np.ndarray):
    """``_scaler``, named at the call site so the comparability test reads clearly."""
    return ml._scaler(matrix)


def _with_degenerate_column() -> np.ndarray:
    """A 12-record matrix: one near-degenerate coordinate, two well-behaved ones."""
    degenerate = np.array([0.0] * 9 + list(DEGENERATE_VALUES))
    return np.column_stack([
        degenerate,
        np.linspace(1.0, 12.0, 12),
        np.linspace(100.0, 2000.0, 12),
    ])


# --------------------------------------------------------------------------------------
# The holdout boundary — the most important test in this file
# --------------------------------------------------------------------------------------


def test_a_holdout_record_is_never_returned_as_a_neighbour():
    """NFR-018 is measured on datasets that never seeded the engine. If a holdout record
    can be a neighbour, the KPI measures the engine against its own training data."""
    corpus = _corpus(10)
    corpus.append(_record("evaluation-only", holdout=True, log_n_rows=3.0, n_numerical=5.0))

    result = ml.recommend(_features(), corpus)

    assert "evaluation-only" not in [n.dataset_name for n in result.neighbours]


def test_a_holdout_record_that_is_an_exact_match_is_still_excluded():
    """The dangerous case: the holdout record is the single most similar thing in the
    corpus, so any ordering bug surfaces it first."""
    query = _features(log_n_rows=7.7, n_numerical=42.0)
    corpus = _corpus(10)
    corpus.append(_record("twin", holdout=True, log_n_rows=7.7, n_numerical=42.0))

    result = ml.recommend(query, corpus)

    assert all(n.dataset_name != "twin" for n in result.neighbours)


def test_holdout_records_do_not_count_towards_the_corpus_minimum():
    """Eight records, all reserved for evaluation, is an empty corpus."""
    corpus = _corpus(8, holdout=True)
    result = ml.recommend(_features(), corpus)
    assert result.declined is not None
    assert result.considered == 0


# --------------------------------------------------------------------------------------
# The decline paths
# --------------------------------------------------------------------------------------


def test_declines_on_a_corpus_that_is_too_small():
    result = ml.recommend(_features(), _corpus(3))
    assert result.ranking == []
    assert result.declined is not None
    assert "at least" in result.declined


def test_the_decline_reason_names_the_numbers():
    """A refusal that does not say how short it fell is not an explanation."""
    result = ml.recommend(_features(), _corpus(3), min_corpus=8)
    assert "3" in result.declined and "8" in result.declined


def test_declines_when_no_record_shares_the_task():
    """A classification corpus cannot inform a regression query."""
    corpus = _corpus(10, task="binary classification")
    query = _features(target_type="regression")

    result = ml.recommend(query, corpus)

    assert result.declined is not None
    assert result.considered == 0


def test_declines_when_nothing_is_similar_enough():
    result = ml.recommend(_features(), _corpus(10), min_similarity=0.999)
    assert result.ranking == []
    assert result.declined is not None
    assert "similarity" in result.declined


def test_a_similarity_decline_still_reports_the_neighbours():
    """So the user can judge the refusal rather than being asked to accept it."""
    result = ml.recommend(_features(), _corpus(10), min_similarity=0.999)
    assert result.neighbours != []


def test_records_at_a_different_metafeature_version_are_ignored():
    """A vector built by an older extractor has its coordinates in different places.
    Comparing it would not raise — it would quietly return a wrong distance."""
    corpus = _corpus(10, version=mf.METAFEATURE_VERSION + 1)
    result = ml.recommend(_features(), corpus)
    assert result.considered == 0
    assert result.declined is not None


# --------------------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------------------


def test_cosine_matches_a_hand_computed_value():
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
    got = ml._cosine(query, matrix)
    assert got == pytest.approx([1.0, 0.0, -1.0, 1 / np.sqrt(2)])


def test_cosine_gives_zero_rather_than_nan_for_a_zero_row():
    """A record sitting exactly on the corpus median has no direction. NaN would poison
    the sort; 0.0 says "uninformative", which is true."""
    got = ml._cosine(np.array([1.0, 1.0]), np.array([[0.0, 0.0]]))
    assert got[0] == 0.0
    assert np.isfinite(got).all()


def test_a_constant_coordinate_does_not_produce_infinities():
    """Zero IQR means a feature every record shares. Dividing by it would be inf."""
    corpus = _corpus(10, sparsity=0.5)
    result = ml.recommend(_features(sparsity=0.5), corpus)
    assert all(np.isfinite(n.similarity) for n in result.neighbours)


def test_a_near_degenerate_coordinate_does_not_become_the_whole_metric():
    """The defect this scaler was rebuilt for.

    Nine of twelve records leave one coordinate at zero, which is what puts its quartiles
    a sliver apart rather than at the same value — small enough to pass a zero check and
    then amplify that coordinate by seventy-eight. On the real corpus it reached a
    post-scaling spread of 24.6 against 0.84 for ``log_n_rows``.

    The assertion is on comparability rather than on a particular number: after scaling,
    no coordinate may vary by more than an order of magnitude more than another.
    """
    matrix = _with_degenerate_column()

    median, divisor = _scaler_parts(matrix)
    spread = ml._scale(matrix, median, divisor).std(axis=0)

    assert spread.max() / spread.min() < 10


def test_no_coordinate_can_contribute_more_than_another():
    """The property the divisor is chosen for, stated directly. Dividing by the observed
    range puts every fitted record within ±1 of the centre on *every* coordinate, so the
    bound is arithmetic rather than a threshold that has to hold."""
    matrix = _with_degenerate_column()

    median, divisor = ml._scaler(matrix)
    scaled = (matrix - median) / divisor  # deliberately unclipped

    assert np.abs(scaled).max() <= 1.0 + 1e-9
    assert np.abs(scaled).max() <= SCALER_CLIP + 1e-9


def test_a_coordinate_every_record_shares_gets_a_divisor_of_one():
    """A range of exactly zero cannot be divided by, and 1.0 leaves the centred value at
    0 for every record — the right contribution for a feature that does not vary."""
    matrix = np.array([[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]])

    _, divisor = ml._scaler(matrix)

    assert divisor[0] == 1.0


def test_the_clip_never_binds_on_the_corpus():
    """Following from the bound above: clipping must leave every fitted record untouched,
    or a guard meant for unprecedented queries would be quietly reshaping the corpus."""
    matrix = _with_degenerate_column()

    median, divisor = ml._scaler(matrix)

    assert np.allclose((matrix - median) / divisor, ml._scale(matrix, median, divisor))


def test_a_query_outside_the_corpus_range_is_clipped():
    """A dataset with 11 classes against a corpus holding 2 to 5 sits far outside on that
    coordinate. Unclipped it is almost the whole vector, and the dataset is judged on one
    axis."""
    matrix = np.array([[2.0, 1.0], [2.0, 2.0], [3.0, 3.0], [5.0, 4.0]])
    median, divisor = ml._scaler(matrix)

    assert ml._scale(np.array([11.0, 2.5]), median, divisor)[0] == SCALER_CLIP


def test_the_query_is_scaled_by_the_corpus_centre_not_its_own():
    """``_scale`` takes the parameters rather than deriving them, because a query scaled
    against itself would sit at the origin and be equidistant from everything."""
    matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    median, divisor = ml._scaler(matrix)

    query = np.array([2.5, 25.0])
    assert np.allclose(ml._scale(query, median, divisor), (query - median) / divisor)


@pytest.mark.parametrize(
    "query_sparsity, where",
    [(0.0, "at the corpus value"), (0.04, "inside the corpus range"), (0.9, "far outside it")],
)
def test_a_near_degenerate_coordinate_does_not_decide_the_neighbours(
    query_sparsity, where
) -> None:
    """End to end, through ``recommend``, and the test that rejected the first two
    attempts at this fix.

    Two groups of six, clearly separated on three well-behaved coordinates. The query
    belongs to the first group on every one of them. Three records in the *wrong* group
    carry the only non-zero values of a near-degenerate coordinate, so a scaler that
    amplifies that coordinate pulls the answer across to them.

    Both an IQR divisor and an IQR floored at a quarter of the range fail this for every
    non-zero ``query_sparsity``: they bound the amplification without removing it.
    """
    corpus = []
    for index in range(12):
        first_group = index < 6
        corpus.append(
            _record(
                f"ds{index}",
                best="random_forest" if first_group else "knn",
                scores=(
                    {"random_forest": 0.9, "knn": 0.6}
                    if first_group
                    else {"random_forest": 0.6, "knn": 0.9}
                ),
                # Spread within each group, so the quartiles sit inside the data rather
                # than straddling two values — which is what a real corpus looks like.
                log_n_rows=(2.0 if first_group else 7.0) + 0.3 * index,
                n_numerical=(1.0 if first_group else 30.0) + index,
                mean_abs_correlation=(0.05 if first_group else 0.5) + 0.02 * index,
                sparsity=0.0,
            )
        )
    for index, value in zip((6, 7, 8), DEGENERATE_VALUES):
        corpus[index]["meta_features"]["sparsity"] = value

    query = _features(
        log_n_rows=2.9, n_numerical=4.0, mean_abs_correlation=0.11, sparsity=query_sparsity
    )
    result = ml.recommend(query, corpus, k=3)

    assert result.declined is None
    chosen = {n.dataset_name for n in result.neighbours}
    assert chosen <= {f"ds{i}" for i in range(6)}, (
        f"with the query {where}, the degenerate coordinate pulled in the wrong "
        f"group: {sorted(chosen)}"
    )


def test_neighbours_come_back_in_descending_similarity():
    result = ml.recommend(_features(), _corpus(12))
    similarities = [n.similarity for n in result.neighbours]
    assert similarities == sorted(similarities, reverse=True)


def test_k_caps_the_neighbour_count():
    result = ml.recommend(_features(), _corpus(12), k=3)
    assert len(result.neighbours) == 3


# --------------------------------------------------------------------------------------
# The ranking
# --------------------------------------------------------------------------------------


def test_ranking_is_ordered_by_weighted_score():
    corpus = _corpus(10, scores={"random_forest": 0.95, "knn": 0.40, "decision_tree": 0.70})
    result = ml.recommend(_features(), corpus)

    assert result.top.family == "random_forest"
    assert [r.family for r in result.ranking] == ["random_forest", "decision_tree", "knn"]


def test_a_family_nobody_scored_is_absent_rather_than_zero():
    """Scoring an untried family 0.0 would read as "tried, and terrible"."""
    corpus = _corpus(10, scores={"random_forest": 0.9})
    result = ml.recommend(_features(), corpus)
    assert [r.family for r in result.ranking] == ["random_forest"]


def test_each_family_reports_how_many_neighbours_scored_it():
    corpus = _corpus(10, scores={"random_forest": 0.9, "knn": 0.5})
    result = ml.recommend(_features(), corpus, k=4)
    assert all(r.neighbours_scored == 4 for r in result.ranking)


def test_spread_records_disagreement_between_neighbours():
    """A family that won on one neighbour and lost on another is not a safe pick,
    however good its mean looks."""
    corpus = [
        _record(f"ds{i}", scores={"random_forest": 0.2 if i % 2 else 0.9}, log_n_rows=2 + i * 0.1)
        for i in range(10)
    ]
    result = ml.recommend(_features(), corpus, k=10)
    forest = next(r for r in result.ranking if r.family == "random_forest")
    assert forest.spread == pytest.approx(0.7)


def test_weighted_mean_with_equal_similarity_is_the_plain_mean():
    """Hand-checkable: two neighbours scoring 0.9 and 0.5 average 0.7 when equally
    similar."""
    neighbours = [
        ml.Neighbour("a", "a", 0.5, "x", "binary classification", 0, 0, {"knn": 0.9}, "accuracy"),
        ml.Neighbour("b", "b", 0.5, "x", "binary classification", 0, 0, {"knn": 0.5}, "accuracy"),
    ]
    ranking = ml._rank(neighbours, "binary classification")
    assert ranking[0].score == pytest.approx(0.7)


def test_a_nearer_neighbour_counts_for_more():
    near = ml.Neighbour("a", "a", 0.9, "x", "binary classification", 0, 0, {"knn": 1.0}, "accuracy")
    far = ml.Neighbour("b", "b", 0.1, "x", "binary classification", 0, 0, {"knn": 0.0}, "accuracy")
    ranking = ml._rank([near, far], "binary classification")
    assert ranking[0].score == pytest.approx(0.9)


def test_negative_similarity_never_counts_backwards():
    """A negative cosine means "unlike". It does not license treating that dataset's
    result as evidence for the opposite conclusion."""
    opposed = ml.Neighbour("a", "a", -0.8, "x", "binary classification", 0, 0, {"knn": 1.0}, "accuracy")
    neutral = ml.Neighbour("b", "b", 0.4, "x", "binary classification", 0, 0, {"knn": 0.2}, "accuracy")
    ranking = ml._rank([opposed, neutral], "binary classification")
    assert 0.0 <= ranking[0].score <= 1.0
    assert ranking[0].score == pytest.approx(0.2)


def test_families_are_labelled_for_reading():
    corpus = _corpus(10, scores={"svm_rbf": 0.9})
    result = ml.recommend(_features(), corpus)
    assert result.top.label == "SVM (RBF kernel)"


def test_regression_relabels_the_linear_family():
    """`logistic_regression` is the identifier; on a regression it is Ridge."""
    corpus = _corpus(
        10, task="regression", scores={"logistic_regression": 0.8}
    )
    query = _features(target_type="regression")
    result = ml.recommend(query, corpus)
    assert result.top.label == "Ridge Regression"


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def test_corpus_sizes_are_reported_separately():
    """`corpus_size` is everything stored; `considered` is what was eligible. A user
    looking at a thin recommendation needs to know which number is which."""
    corpus = _corpus(10) + [_record("held", holdout=True)]
    result = ml.recommend(_features(), corpus)
    assert result.corpus_size == 11
    assert result.considered == 10


def test_a_successful_recommendation_declines_nothing():
    result = ml.recommend(_features(), _corpus(10))
    assert result.declined is None
    assert result.top is not None
