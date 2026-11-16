"""Which strings mean "no value", and which are the answer.

The defect these tests exist for: ``MISSING_TOKENS`` held the literal word ``none``, so
every occurrence of it became a blank. OpenML's ``credit-g`` has 1,000 rows and no
missing values at all; its ``own_telephone`` column holds ``none`` 596 times and ``yes``
404 times. Blanking the negative class made the column 60% empty, which dropped it, and
the app then told the user that 11% of their cells were blank and two columns were more
than 60% empty — both false about the file they uploaded, both in the voice it uses for
measurements.

The rule now has two halves, and both halves need guarding:

* a **notation** (``?``, ``null``, an empty cell) is a gap however often it occurs —
  a column that is 90% ``?`` really is 90% empty, and a frequency test there would be
  the opposite mistake;
* a **word** (``none``, ``unknown``) is a gap only when it is a small share of what is
  written in its column, because a typed-in placeholder is incidental by nature and a
  level of a variable is not.

The share is per column and per token on purpose. Both are tested, because getting
either wrong reintroduces the defect in a form the credit-g case would not catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import MISSING_WORDS, PLACEHOLDER_WORD_MAX_SHARE
from app.services.ingest import read_upload


def _read(frame: pd.DataFrame, name: str = "f.csv"):
    """Round-trip through CSV, because that is how a user's file arrives."""
    return read_upload(frame.to_csv(index=False).encode(), name)


def _padding(n: int) -> dict[str, object]:
    """Two columns of filler, so ingestion has a target and more than one feature."""
    rng = np.random.default_rng(7)
    return {
        "amount": rng.normal(100, 10, n).round(3),
        "label": ["a", "b"] * (n // 2),
    }


# --------------------------------------------------------------------------------------
# The regression: a word that is a level of the variable
# --------------------------------------------------------------------------------------


def test_a_word_that_is_most_of_its_column_is_left_as_written():
    """``credit-g``'s ``own_telephone``, to the row. This is the defect."""
    n = 1000
    result = _read(pd.DataFrame({
        "own_telephone": ["none"] * 596 + ["yes"] * 404,
        **_padding(n),
    }))

    column = result.frame["own_telephone"]
    assert int(column.isna().sum()) == 0
    assert sorted(column.dropna().unique()) == ["none", "yes"]
    assert int((column == "none").sum()) == 596


def test_such_a_column_survives_to_the_feature_set():
    """Blanking 60% of it tripped MAX_MISSING_RATE and dropped the column outright,
    which is the part that cost the user a feature rather than just a number."""
    n = 1000
    result = _read(pd.DataFrame({
        "own_telephone": ["none"] * 596 + ["yes"] * 404,
        **_padding(n),
    }))

    assert "own_telephone" in result.frame.columns
    empty_warnings = [i.message for i in result.issues if "60% empty" in i.message]
    assert empty_warnings == []


def test_a_complete_file_is_not_reported_as_having_blanks():
    """The second half of the defect, and the worse half: having created the blanks,
    the app described them to the user as a property of their data."""
    n = 1000
    result = _read(pd.DataFrame({
        "own_telephone": ["none"] * 596 + ["yes"] * 404,
        "other_parties": ["none"] * 907 + ["co applicant"] * 93,
        **_padding(n),
    }))

    assert result.missing_cells == 0
    assert not [i.message for i in result.issues if "cells are blank" in i.message]


# --------------------------------------------------------------------------------------
# The other half: a word that really is a placeholder
# --------------------------------------------------------------------------------------


def test_an_occasional_word_is_still_read_as_a_blank():
    """The repair the product is meant to perform is not lost. 2% of a column typed as
    ``unknown`` is somebody recording that they did not know."""
    n = 1000
    result = _read(pd.DataFrame({
        "job": ["skilled"] * 980 + ["unknown"] * 20,
        **_padding(n),
    }))

    assert int(result.frame["job"].isna().sum()) == 20


def test_the_boundary_share_is_treated_as_a_placeholder():
    """At exactly the threshold the word is blanked; the cut is on *exceeding* it, so
    the two branches cannot both claim the boundary."""
    n = 1000
    at_threshold = int(n * PLACEHOLDER_WORD_MAX_SHARE)
    result = _read(pd.DataFrame({
        "job": ["unknown"] * at_threshold + ["skilled"] * (n - at_threshold),
        **_padding(n),
    }))

    assert int(result.frame["job"].isna().sum()) == at_threshold


def test_one_more_than_the_boundary_is_treated_as_a_level():
    n = 1000
    over = int(n * PLACEHOLDER_WORD_MAX_SHARE) + 1
    result = _read(pd.DataFrame({
        "job": ["unknown"] * over + ["skilled"] * (n - over),
        **_padding(n),
    }))

    assert int(result.frame["job"].isna().sum()) == 0


# --------------------------------------------------------------------------------------
# Notations are not subject to the frequency test
# --------------------------------------------------------------------------------------


def test_a_notation_is_a_blank_however_common_it_is():
    """``?`` is ARFF's defined missing marker. A column that is 70% ``?`` is 70% empty,
    and applying the word rule here would be the mirror-image mistake."""
    n = 1000
    result = _read(pd.DataFrame({
        "late_fee": ["?"] * 700 + ["12"] * 300,
        **_padding(n),
    }))

    assert int(result.frame["late_fee"].isna().sum()) == 700


def test_a_mostly_empty_column_is_still_reported_as_mostly_empty():
    """The warning the previous test's column should produce — and the one credit-g got
    wrongly. Here it is earned."""
    n = 1000
    result = _read(pd.DataFrame({
        "late_fee": ["?"] * 700 + ["12"] * 300,
        **_padding(n),
    }))

    assert any("late_fee" in i.message and "60% empty" in i.message for i in result.issues)


def test_an_empty_string_is_never_a_level():
    """Whitespace-only cells are what the writer left behind, not something typed."""
    n = 1000
    result = _read(pd.DataFrame({
        "note": ["   "] * 800 + ["ok"] * 200,
        **_padding(n),
    }))

    assert int(result.frame["note"].isna().sum()) == 800


# --------------------------------------------------------------------------------------
# The share is per column, and per token
# --------------------------------------------------------------------------------------


def test_the_share_is_measured_per_column_not_over_the_file():
    """The same word, two columns, opposite verdicts. A file-wide count would give both
    columns whichever answer the larger one dictated."""
    n = 1000
    result = _read(pd.DataFrame({
        "own_telephone": ["none"] * 596 + ["yes"] * 404,
        "guarantor": ["none"] * 10 + ["yes"] * 990,
        **_padding(n),
    }))

    assert int(result.frame["own_telephone"].isna().sum()) == 0
    assert int(result.frame["guarantor"].isna().sum()) == 10


def test_the_share_is_measured_per_token_not_over_the_words_together():
    """Three spellings of the same gap, 15% each. Each is incidental, so each is a
    blank — 45% in total. Summing them first would keep all three as levels."""
    n = 1000
    chunk = 150
    values = (
        ["unknown"] * chunk
        + ["not available"] * chunk
        + ["no data"] * chunk
        + ["skilled"] * (n - 3 * chunk)
    )
    result = _read(pd.DataFrame({"job": values, **_padding(n)}))

    assert int(result.frame["job"].isna().sum()) == 3 * chunk


def test_the_share_is_measured_against_written_values_not_rows():
    """A column that is 90% genuinely blank and 10% ``unknown``: of what somebody
    actually typed, every value was ``unknown``, so it is a level. Measuring against the
    row count would make it 10% and blank it — and the two denominators also differ
    between the app and the exported script, which does not drop all-blank rows."""
    n = 1000
    written = 100
    values: list[object] = [np.nan] * (n - written) + ["unknown"] * written
    result = _read(pd.DataFrame({"job": values, **_padding(n)}))

    assert int((result.frame["job"] == "unknown").sum()) == written


# --------------------------------------------------------------------------------------
# What the user is told
# --------------------------------------------------------------------------------------


def test_the_blanking_names_the_strings_it_actually_found():
    """The old sentence said "such as 'N/A' and '-'" whether or not either was in the
    file. A repair report that names tokens it did not see is not a report."""
    n = 1000
    result = _read(pd.DataFrame({
        "job": ["skilled"] * 980 + ["unknown"] * 20,
        **_padding(n),
    }))

    joined = " ".join(result.cleaning)
    assert "'unknown'" in joined
    assert "'N/A'" not in joined


def test_keeping_a_word_is_reported_rather_than_silent():
    """A judgement call that changes nothing visible is the easiest kind to hide, and
    this module's rule is that no repair — or declined repair — is silent."""
    n = 1000
    result = _read(pd.DataFrame({
        "own_telephone": ["none"] * 596 + ["yes"] * 404,
        **_padding(n),
    }))

    joined = " ".join(result.cleaning)
    assert "own_telephone" in joined
    assert "'none'" in joined
    assert "60%" in joined


def test_the_decision_is_recorded_per_column_for_the_export():
    n = 1000
    result = _read(pd.DataFrame({
        "own_telephone": ["none"] * 596 + ["yes"] * 404,
        "job": ["skilled"] * 980 + ["unknown"] * 20,
        **_padding(n),
    }))

    assert result.placeholders.kept == {"own_telephone": {"none": 0.596}}
    assert result.placeholders.blanked == {"job": {"unknown": 20}}
    assert result.placeholders.total == 20
    assert result.placeholders.tokens == ["unknown"]
    assert result.placeholders.for_column("job") == ["unknown"]
    assert result.placeholders.for_column("own_telephone") == []


def test_nothing_is_reported_when_there_was_nothing_to_repair():
    n = 1000
    result = _read(pd.DataFrame({
        "grade": ["a", "b"] * (n // 2),
        **_padding(n),
    }))

    joined = " ".join(result.cleaning)
    assert "placeholder" not in joined
    assert result.placeholders.total == 0
    assert result.placeholders.kept == {}


# --------------------------------------------------------------------------------------
# The vocabulary itself
# --------------------------------------------------------------------------------------


def test_every_word_in_the_set_is_subject_to_the_frequency_test():
    """A word added to ``MISSING_WORDS`` later must inherit the rule, not bypass it.
    Each is checked at a share where it has to be kept."""
    n = 200
    for word in sorted(MISSING_WORDS):
        result = _read(pd.DataFrame({
            "field": [word] * 150 + ["other"] * 50,
            **_padding(n),
        }))
        kept = result.frame["field"]
        assert int(kept.isna().sum()) == 0, f"{word!r} was blanked at a 75% share"
        assert int((kept == word).sum()) == 150, f"{word!r} did not survive"


# --------------------------------------------------------------------------------------
# The dash, which was the second instance of the same defect
# --------------------------------------------------------------------------------------


def test_a_dash_that_is_half_a_column_is_a_level_not_a_blank():
    """OpenML's ``credit-approval`` encodes its binary target as ``-`` (383) and ``+``
    (307). A bare dash reads like pure notation, and in a spreadsheet it usually is — so
    it was first put in ``MISSING_MARKERS``, exempt from the frequency test. That blanked
    the negative class on 55% of rows, those rows were dropped for having no label, and
    the constant target that survived was described by the pipeline as a multiclass task
    whose majority-class baseline scored 1.0000.
    """
    n = 690
    result = _read(pd.DataFrame({
        "class": ["-"] * 383 + ["+"] * 307,
        "A8": ["t"] * 361 + ["f"] * 329,
        **_padding(n),
    }))

    column = result.frame["class"]
    assert int(column.isna().sum()) == 0
    assert sorted(column.dropna().unique()) == ["+", "-"]
    assert int((column == "-").sum()) == 383


def test_a_dash_in_a_few_cells_is_still_a_blank():
    """The spreadsheet reading is the common one and is not lost: a dash typed into a
    handful of cells is an empty cell."""
    n = 1000
    result = _read(pd.DataFrame({
        "revenue": ["-"] * 30 + [str(i) for i in range(970)],
        **_padding(n),
    }))

    assert int(result.frame["revenue"].isna().sum()) == 30


def test_the_frequency_test_covers_every_token_that_could_be_a_level():
    """The marker set is exempt from the frequency test, so anything in it that *could*
    be somebody's category label is a latent repeat of this defect. This asserts the
    split rather than the tokens' spelling: a marker must be a notation, not a value.

    ``-`` was moved out of the marker set for exactly this reason. The test does not try
    to judge new tokens automatically — it pins the current division so that moving a
    token has to be a deliberate edit with a test change attached.
    """
    from app.config import MISSING_MARKERS

    assert "-" in MISSING_WORDS
    assert "-" not in MISSING_MARKERS
    assert MISSING_MARKERS.isdisjoint(MISSING_WORDS)
    assert sorted(MISSING_MARKERS) == [
        "",
        "#div/0!",
        "#n/a",
        "#na",
        "#value!",
        "--",
        ".",
        "?",
        "n.a.",
        "n/a",
        "nan",
        "null",
    ]
