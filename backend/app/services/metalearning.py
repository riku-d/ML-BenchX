"""Stage 4 — ranking algorithm families by similarity to past experiments.

Given a dataset's meta-features and a corpus of past experiments, find the datasets that
look most like it and turn what won on them into a ranking.

Pure. The caller fetches the records; nothing here touches the database, so the whole
engine is testable against a list of dicts and a recommendation is reproducible from the
inputs alone.

Why robust scaling, then cosine
-------------------------------
The raw vector is not comparable across coordinates: ``n_rows`` runs to five figures
while ``sparsity`` is bounded at 1. Unscaled, a distance metric is a measure of row
count and nothing else.

Scaling is median-centred and divided by each coordinate's observed range, so every
record lands within ±1 of the centre on every coordinate and no single meta-feature can
dominate the distance. The first version of this divided by the interquartile range —
the textbook robust choice — and that turned out to be the stage's worst defect: an IQR
can be a sliver, and dividing by a sliver makes one coordinate the entire metric. See
``_scaler`` for the measurements.

Cosine on the scaled vector then compares *shape* — the pattern of which features are
unusually high or low — rather than magnitude, which is what "a similar dataset" should
mean and what PRD.md §10 asks for. Scaling centres the data, so this is closer to a
correlation than to the cosine of two raw positive vectors, which would sit near 1.0 for
every pair and discriminate nothing.

When this declines to answer
----------------------------
Three cases, each returning a reason rather than a number:

* fewer than ``MIN_CORPUS`` usable records;
* no record of the same task — a classification corpus cannot inform a regression;
* the closest record below ``MIN_SIMILARITY``.

A ranking built on any of those would be a confident-looking extrapolation from nothing,
which is worse than silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import K_NEIGHBOURS, MIN_CORPUS, MIN_SIMILARITY, SCALER_CLIP
from app.services import metafeatures as mf
from app.services.families import FAMILIES, label as family_label


@dataclass
class Neighbour:
    """One past experiment, and how like the query it is."""

    record_id: str
    dataset_name: str
    similarity: float
    best_family: str
    task: str
    n_rows: int
    n_cols: int
    scores: dict[str, float]
    metric: str


@dataclass
class FamilyRanking:
    """One family's standing, and the evidence behind it."""

    family: str
    label: str
    #: Similarity-weighted mean of this family's score across the neighbours that have
    #: one. Comparable within a recommendation, not across datasets — it is an average
    #: of other datasets' scores, not a prediction of yours.
    score: float
    #: How many of the neighbours actually scored this family. A family scored by one
    #: neighbour out of five is a much weaker claim than one scored by all five, and the
    #: number is reported rather than folded into the score.
    neighbours_scored: int
    #: Spread of those scores. A family that won on one neighbour and lost badly on
    #: another is not a safe recommendation however good its mean is.
    spread: float


@dataclass
class Recommendation:
    """What the engine concluded, including when it concluded nothing."""

    ranking: list[FamilyRanking] = field(default_factory=list)
    neighbours: list[Neighbour] = field(default_factory=list)
    #: ``None`` when a ranking was produced. Otherwise why there is none, in words
    #: meant for the user.
    declined: str | None = None
    corpus_size: int = 0
    considered: int = 0
    metafeature_version: int = mf.METAFEATURE_VERSION

    @property
    def top(self) -> FamilyRanking | None:
        return self.ranking[0] if self.ranking else None


def _usable(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    """Records that may inform this query.

    Two filters, both of which must hold or the comparison is meaningless rather than
    merely weak:

    ``holdout`` records are excluded. NFR-018 measures the engine on datasets that never
    seeded it, and a holdout record appearing as a neighbour would make that measurement
    a measurement of nothing. This is the single most important line in the module.

    ``metafeature_version`` must match. A vector built by an older extractor has
    coordinates in different places; comparing it would not fail, it would silently
    return a wrong distance.
    """
    classification = task.strip().lower().startswith(("binary", "multiclass", "class"))
    out = []
    for record in records:
        if record.get("holdout"):
            continue
        if record.get("metafeature_version") != mf.METAFEATURE_VERSION:
            continue
        record_task = str(record.get("task", ""))
        record_is_classification = record_task.strip().lower().startswith(
            ("binary", "multiclass", "class")
        )
        if record_is_classification != classification:
            continue
        if not record.get("meta_features"):
            continue
        out.append(record)
    return out


def _scaler(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate centre and divisor for the corpus.

    Centred on the **median** and divided by the **observed range**.

    Why not the interquartile range, which is the obvious robust choice and was what
    this function used first: an IQR can be a sliver, and dividing by a sliver turns one
    coordinate into the whole metric. Measured on the first real corpus — nine of twelve
    usable records had ``sparsity`` exactly 0.0, which put the quartiles 0.000635 apart
    and gave that coordinate a post-scaling standard deviation of 24.6 against 0.84 for
    ``log_n_rows``. ``sparsity`` alone then accounted for 80% of one query's vector
    length, and ``n_classes`` for 94% of another's. The similarity search had quietly
    become a one-coordinate comparison.

    A guard against *zero* IQR was already here and was not enough: a coordinate where
    three quarters of the corpus shares a value has an IQR that is small but non-zero, so
    it passes the zero check and then amplifies. Flooring the IQR at a fraction of the
    range was tried and is not enough either — it bounds the amplification but a query
    differing on such a coordinate is still described mostly by it.

    The range has the property that is actually wanted: every record lands within ±1 of
    the median on every coordinate, so **no coordinate can contribute more than any
    other**, by construction rather than by a threshold. For meta-features in
    incomparable units — rows against a fraction missing — there is no principled
    exchange rate anyway, and normalising each to its own observed spread says "treat
    every measured property as equally important", which is the right prior when there
    is no prior.

    The cost is sensitivity to a single extreme record, which is what the IQR was there
    to avoid. Two things make that acceptable. It errs toward *under*-weighting a
    coordinate rather than letting one drown out the other fourteen, and those are not
    equally bad. And the heavy-tailed coordinates are already logged — see
    ``metafeatures.VECTOR_FIELDS`` — precisely so that one very large dataset cannot set
    the scale. At a larger corpus a trimmed range (5th to 95th percentile) would be
    strictly better; with twelve records it is the same number.

    A range of exactly zero — every record shares the value — gets a divisor of 1.0.
    Dividing by zero would produce infinities; 1.0 leaves the centred value at 0 for
    every record, which is the correct contribution for a feature that does not vary:
    none.

    Returning the two parts rather than a scaled matrix is deliberate: the query has to
    be scaled by the corpus's own centre and divisor, never by its own.
    """
    median = np.median(matrix, axis=0)
    span = matrix.max(axis=0) - matrix.min(axis=0)
    return median, np.where(span <= 0, 1.0, span)


def _scale(values: np.ndarray, median: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Centre, divide and clip. One function for the corpus and for the query.

    Scaling them in two places is how the two drift apart, and a query scaled differently
    from the records it is compared against produces a similarity that means nothing
    while still rendering as a percentage.

    The clip only ever binds on a query: the divisor already bounds every record the
    scaler was fitted on at ``SCALER_CLIP``. See that constant.
    """
    return np.clip((values - median) / divisor, -SCALER_CLIP, SCALER_CLIP)


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and each row of a matrix.

    A zero-norm row — a record identical to the corpus median on every coordinate —
    scores 0.0 rather than NaN. It is genuinely uninformative about direction, and 0.0
    says that while NaN would poison the sort.
    """
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    similarities = np.zeros(matrix.shape[0], dtype="float64")
    usable = denominator > 0
    similarities[usable] = (matrix[usable] @ query) / denominator[usable]
    return similarities


def _rank(neighbours: list[Neighbour], task: str) -> list[FamilyRanking]:
    """Turn the neighbours' recorded scores into a ranking.

    Weighted by similarity, so a near-identical dataset counts for more than a distant
    one — but every neighbour that scored the family contributes, because a single
    nearest neighbour deciding the whole ranking is a KNN with k=1 wearing a disguise.

    Weights are shifted to be non-negative. Cosine on centred data can be negative, and
    a negative weight would mean "this dataset is unlike yours, so count its result
    backwards", which is not a thing similarity licenses.
    """
    ranking: list[FamilyRanking] = []

    for family in FAMILIES:
        values: list[float] = []
        weights: list[float] = []
        for neighbour in neighbours:
            score = neighbour.scores.get(family)
            if score is None:
                continue
            values.append(float(score))
            weights.append(max(neighbour.similarity, 0.0))

        if not values:
            continue

        array = np.array(values, dtype="float64")
        weight_array = np.array(weights, dtype="float64")
        if weight_array.sum() <= 0:
            weight_array = np.ones_like(weight_array)

        ranking.append(
            FamilyRanking(
                family=family,
                label=family_label(family, task),
                score=float(np.average(array, weights=weight_array)),
                neighbours_scored=len(values),
                spread=float(array.max() - array.min()),
            )
        )

    # Ties broken by how many neighbours scored the family: between two families with
    # the same mean, the better-evidenced one goes first.
    ranking.sort(key=lambda item: (item.score, item.neighbours_scored), reverse=True)
    return ranking


def recommend(
    features: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    k: int = K_NEIGHBOURS,
    min_corpus: int = MIN_CORPUS,
    min_similarity: float = MIN_SIMILARITY,
) -> Recommendation:
    """Rank the candidate families for a dataset described by ``features``.

    ``records`` are the corpus rows as plain dicts — ``meta_features``, ``task``,
    ``holdout``, ``best_family``, ``family_scores`` and the identifying fields. The
    caller does the querying; this does the reasoning.
    """
    task = str(features.get("target_type", ""))
    corpus = _usable(records, task)

    if len(corpus) < min_corpus:
        return Recommendation(
            declined=(
                f"The experiment history holds {len(corpus)} comparable past "
                f"experiment(s) for this kind of task, and a ranking needs at least "
                f"{min_corpus}. Until then there is nothing to compare this dataset "
                f"against."
            ),
            corpus_size=len(records),
            considered=len(corpus),
        )

    query = mf.as_vector(features)
    matrix = np.vstack([mf.as_vector(record["meta_features"]) for record in corpus])

    median, divisor = _scaler(matrix)
    scaled_matrix = _scale(matrix, median, divisor)
    scaled_query = _scale(query, median, divisor)

    similarities = _cosine(scaled_query, scaled_matrix)
    order = np.argsort(similarities)[::-1][:k]

    neighbours = [
        Neighbour(
            record_id=str(corpus[index].get("id", "")),
            dataset_name=str(corpus[index].get("dataset_name", "")),
            similarity=float(similarities[index]),
            best_family=str(corpus[index].get("best_family", "")),
            task=str(corpus[index].get("task", "")),
            n_rows=int(corpus[index].get("n_rows", 0)),
            n_cols=int(corpus[index].get("n_cols", 0)),
            scores=dict((corpus[index].get("family_scores") or {}).get("scores", {})),
            metric=str(corpus[index].get("metric", "")),
        )
        for index in order
    ]

    if not neighbours or neighbours[0].similarity < min_similarity:
        closest = neighbours[0].similarity if neighbours else 0.0
        return Recommendation(
            neighbours=neighbours,
            declined=(
                f"Nothing in the experiment history resembles this dataset closely "
                f"enough to rank from — the closest of {len(corpus)} past experiments "
                f"scores {closest:.2f} similarity, below the {min_similarity:.2f} "
                f"needed. The neighbours are listed so you can judge that for yourself."
            ),
            corpus_size=len(records),
            considered=len(corpus),
        )

    return Recommendation(
        ranking=_rank(neighbours, task),
        neighbours=neighbours,
        corpus_size=len(records),
        considered=len(corpus),
    )
