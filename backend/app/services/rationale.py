"""Stage 4 — saying why, in English.

PRD.md §10 asks for "ranked algorithms with a plain-English rationale". This builds that
sentence from the recommendation it is given, and from nothing else.

The rule, which is the same rule the rest of this product follows
-----------------------------------------------------------------
**Every clause is derived from a fact in the recommendation.** Not a template with
numbers dropped into it — a template is a claim written in advance and decorated with
data afterwards, and it stays plausible when the data stops supporting it.

So: the engine names the datasets it actually used and the scores they actually
recorded. Where the evidence is thin, the sentence says it is thin rather than being
omitted, because a confident sentence over weak evidence is the failure this is meant to
prevent.

What is deliberately not said
-----------------------------
**No prediction of how well the model will do.** The scores in the corpus are other
datasets' cross-validated results. The weighted mean of them is a summary of what
happened elsewhere, not a forecast for here, and phrasing it as "expect about 0.87"
would be a fabricated number wearing a measured one's clothes.

**No causal explanation.** The engine knows that similar datasets favoured Random
Forest. It does not know *why*, and "because your data is high-dimensional and
non-linear" would be a story invented to fit the answer. The rationale reports the
similarity and the evidence; the user can see the meta-features for themselves.
"""

from __future__ import annotations

from app.services.metalearning import Recommendation


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _closeness(similarity: float) -> str:
    """Describe a cosine similarity in words a non-specialist can act on.

    Bands rather than the raw number in the prose, because "0.72" means nothing without
    knowing the metric's range and what a typical value looks like. The number itself is
    still reported alongside, so nothing is hidden by the translation.
    """
    if similarity >= 0.85:
        return "very closely"
    if similarity >= 0.6:
        return "closely"
    if similarity >= 0.4:
        return "moderately"
    return "loosely"


def explain(recommendation: Recommendation) -> list[str]:
    """The rationale, as sentences in reading order.

    A list rather than one blob: the surface that renders this will want them as
    separate lines, and joining them here would make that someone else's string
    manipulation problem.
    """
    if recommendation.declined:
        # The engine's refusal is already written for a reader, and rewriting it here
        # would be two sentences that could drift apart.
        return [recommendation.declined]

    top = recommendation.top
    neighbours = recommendation.neighbours
    if top is None or not neighbours:
        return ["No ranking was produced."]

    lines: list[str] = []

    # --- what was compared, and how well it matched ------------------------------------
    closest = neighbours[0]
    lines.append(
        f"Compared against {_count(recommendation.considered, 'past experiment')} "
        f"in the history. The closest {len(neighbours)} informed this ranking, led by "
        f"{closest.dataset_name}, which matches {_closeness(closest.similarity)} "
        f"({closest.similarity:.2f} similarity)."
    )

    # --- the recommendation, and the evidence for it -----------------------------------
    agreeing = [n for n in neighbours if n.best_family == top.family]
    if agreeing:
        names = ", ".join(n.dataset_name for n in agreeing[:3])
        more = f" and {len(agreeing) - 3} more" if len(agreeing) > 3 else ""
        was = "dataset" if len(agreeing) == 1 else "datasets"
        lines.append(
            f"{top.label} comes first: it won outright on {len(agreeing)} of those "
            f"{was} ({names}{more})."
        )
    else:
        # The weighted mean can favour a family that never actually won — consistently
        # second across five datasets beats winning one and placing last in four. That
        # is a real and defensible result, and hiding it would misrepresent the ranking.
        lines.append(
            f"{top.label} comes first on average, though it did not win outright on any "
            f"of these datasets — it scored consistently well across them rather than "
            f"winning one decisively."
        )

    # --- how much daylight there is ----------------------------------------------------
    if len(recommendation.ranking) > 1:
        runner_up = recommendation.ranking[1]
        gap = top.score - runner_up.score
        if gap < 0.01:
            lines.append(
                f"{runner_up.label} is level with it ({runner_up.score:.3f} against "
                f"{top.score:.3f} on those datasets). Treat them as joint first."
            )
        else:
            lines.append(
                f"{runner_up.label} follows, {gap:.3f} behind on those datasets' scores."
            )

    # --- where the evidence is thin ----------------------------------------------------
    if top.neighbours_scored < len(neighbours):
        lines.append(
            f"Worth knowing: only {top.neighbours_scored} of the {len(neighbours)} "
            f"neighbours recorded a score for {top.label}, so this rests on less "
            f"evidence than the others."
        )

    if top.spread >= 0.2:
        lines.append(
            f"Its results varied widely across those datasets — a spread of "
            f"{top.spread:.2f} between its best and worst — so this is a starting point "
            f"rather than a safe bet."
        )

    # --- what this is, and is not ------------------------------------------------------
    lines.append(
        f"These are {neighbours[0].metric or 'cross-validated'} scores other datasets "
        f"achieved, not a prediction for yours. Nothing has been trained on your data."
    )

    return lines
