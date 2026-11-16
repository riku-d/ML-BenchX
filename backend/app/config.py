"""Every threshold the app applies, in one place.

Nothing here is discovered from data or tuned by a search. These are editorial
decisions — the point at which we say a column is too empty to use, too crowded to
one-hot, too unique to be a measurement — and they are collected here so that the plan,
the explanations shown to the user, the exported script and the tests all quote the same
number. A threshold that appears twice will eventually disagree with itself.

Each constant carries the reasoning for its value, because the value on its own is not
reviewable: ``0.6`` means nothing, "more than 60% missing and you are modelling your own
imputation" can be argued with.

Overriding
----------
Every constant can be overridden with an environment variable of the same name prefixed
``MLBENCHX_`` — ``MLBENCHX_MAX_UPLOAD_BYTES=104857600``. Overrides are read once at
import. This exists so an operator can raise a limit without editing code; the defaults
below are what the app is designed around.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "MAX_UPLOAD_BYTES",
    "MIN_ROWS",
    "UPLOAD_CHUNK_BYTES",
    "MISSING_MARKERS",
    "MISSING_WORDS",
    "PLACEHOLDER_WORD_MAX_SHARE",
    "MISSING_TOKENS",
    "DATE_PARSE_RATE",
    "NEAR_CONSTANT_SHARE",
    "ONEHOT_MAX_CARDINALITY",
    "MAX_MISSING_RATE",
    "TEXT_UNIQUE_RATE",
    "TEXT_MIN_UNIQUE",
    "IDENTIFIER_UNIQUE_RATE",
    "NAMED_IDENTIFIER_UNIQUE_RATE",
    "LEAKAGE_CORRELATION",
    "LEAKAGE_TREE_SCORE",
    "LEAKAGE_TREE_FOLDS",
    "LEAKAGE_TREE_DEPTH",
    "HIGH_CORRELATION",
    "MIN_ROWS_FOR_CORRELATION",
    "DRIFT_P_VALUE",
    "BASELINE_CV_FOLDS",
    "MIN_ROWS_FOR_BASELINE",
    "MIN_ROWS_FOR_DRIFT",
    "OUTLIER_IQR_MULTIPLIER",
    "MISSING_INDICATOR_RATE",
    "RARE_CATEGORY_RATE",
    "HISTOGRAM_BINS",
    "TOP_VALUES",
    "SKEW_THRESHOLD",
    "IMBALANCE_RATIO",
    "MAX_TARGET_CLASSES",
    "MAX_TARGET_CARDINALITY",
    "MAX_CLASSES_LISTED",
    "TEST_SIZE",
    "SEED",
    "DEFAULT_SCALER",
    "DEFAULT_HIGH_CARD_ENCODER",
    "TARGET_ENCODER_CV",
    "HEALTH_WEIGHTS",
    "DEFAULT_PROJECT_ID",
    "DATABASE_URL",
    "STORAGE_ROOT",
    "MAX_RUNS_IN_MEMORY",
    "EXECUTION_TIMEOUT_SECONDS",
    "EXECUTION_CPU_SECONDS",
    "EXECUTION_MEMORY_BYTES",
    "EXECUTION_MAX_OUTPUT_BYTES",
    "PREVIEW_PAGE_SIZE",
    "MAX_HISTORY_ENTRIES",
    "K_NEIGHBOURS",
    "MIN_CORPUS",
    "MIN_SIMILARITY",
    "SCALER_CLIP",
    "STRIP_WIDTH",
    "MAX_COLUMNS_IN_EXPLANATION",
]


def _int(name: str, default: int) -> int:
    raw = os.environ.get(f"MLBENCHX_{name}")
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(f"MLBENCHX_{name}")
    return float(raw) if raw else default


def _str(name: str, default: str) -> str:
    return os.environ.get(f"MLBENCHX_{name}") or default


# --------------------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------------------

#: Largest upload accepted, in bytes. Big enough for any spreadsheet a person assembled
#: by hand, small enough that a mis-dragged folder cannot exhaust the machine.
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024)

#: Read size when streaming an upload. Only affects how quickly an oversized file is
#: rejected, not what is accepted.
UPLOAD_CHUNK_BYTES = _int("UPLOAD_CHUNK_BYTES", 1024 * 1024)

#: Below this many rows a train/test split cannot say anything meaningful: a 20% holdout
#: of 40 rows is 8 rows, and any score computed on 8 rows is noise.
MIN_ROWS = _int("MIN_ROWS", 50)

#: Share of a text column's non-blank values that must parse as dates before the column
#: is treated as one. The spec's rule is a rate rather than unanimity: a column where
#: 95% of values are dates and the rest are stray notes is a date column with some bad
#: rows, and leaving the whole thing as text because of them loses more than it saves.
DATE_PARSE_RATE = _float("DATE_PARSE_RATE", 0.90)

#: A column whose most common value occupies at least this share is near-constant. It
#: carries almost nothing and costs a column's width and the reader's attention.
NEAR_CONSTANT_SHARE = _float("NEAR_CONSTANT_SHARE", 0.99)

#: Notations that mean "no value" wherever they appear, however often they appear.
#: None of these is a word or a symbol anybody would choose as a label: they are the
#: marks a person or a program leaves where a value is absent — an empty cell, ARFF's
#: ``?``, SQL's ``null``, the float repr ``nan``, Excel's error cells. A column that is
#: 90% ``?`` really is 90% empty, so no frequency test applies to this set.
#: Compared case-insensitively after stripping surrounding whitespace.
#:
#: A single ``-`` is deliberately **not** here, though it reads like one of these. See
#: ``MISSING_WORDS``.
MISSING_MARKERS: frozenset[str] = frozenset({
    "", "--", ".", "?", "n/a", "n.a.", "#n/a", "#na", "nan", "null",
    "#value!", "#div/0!",
})

#: Ordinary words that *sometimes* mean "no value" and sometimes are the answer.
#:
#: Treating these as blanks unconditionally destroys real data, and does it silently.
#: The case that found this: OpenML's ``credit-g`` has 1,000 rows and not one missing
#: value. Its ``own_telephone`` column holds ``none`` 596 times and ``yes`` 404 times —
#: a balanced, informative binary feature whose negative class is spelled "none".
#: Blanking it made the column 60% empty, which dropped it; ``other_parties`` and
#: ``other_payment_plans`` went the same way. The app then reported "2 columns are more
#: than 60% empty" and "11.0% of all cells are blank" about a file with no blanks in it.
#:
#: So a word from this set is read as a gap only when it is a small share of what is
#: written in its column — see ``PLACEHOLDER_WORD_MAX_SHARE``. A typed-in placeholder is
#: incidental by nature; a level of a variable is not.
#:
#: ``-`` is in here, not in ``MISSING_MARKERS``, and that was not the first guess. A bare
#: dash looks like pure notation, and in a spreadsheet it usually is. But OpenML's
#: ``credit-approval`` — a standard benchmark — encodes its binary target as ``-`` (383)
#: and ``+`` (307). Blanking the dash emptied the target on 55% of rows, those rows were
#: then dropped for having no label, and what survived was a *constant* target that the
#: pipeline went on to describe as a multiclass task whose majority-class baseline scored
#: 1.0000. A frequency test gets both readings right: a dash holding half a column is
#: that column's negative class, and a dash in 3% of one is an empty cell.
MISSING_WORDS: frozenset[str] = frozenset({
    "-", "none", "unknown", "na", "nil", "missing", "no data", "not available",
    "not applicable", "tbd",
})

#: Share of a column's written values above which a word from ``MISSING_WORDS`` is read
#: as a category level rather than a gap.
#:
#: Measured against the non-blank values — what somebody actually typed in that column —
#: because that is what a placeholder competes with to be a level, and because it makes
#: the rule independent of how many entirely blank rows the file has.
#:
#: A fifth of a column is the line because of what sits on either side of it:
#: ``MISSING_INDICATOR_RATE`` (5%) is the point at which missingness is itself worth
#: modelling, and ``MAX_MISSING_RATE`` (60%) is the point at which a column is dropped
#: altogether. The damage this prevents is all at the top of that range, and a word
#: accounting for more than a fifth of a column is doing work in it.
#:
#: This is a rate over the whole file, like ``DATE_PARSE_RATE``, and for the same
#: reason: deciding whether a string is data or the absence of data is a parsing
#: question, not a value learned from the data and applied to rows. No target is
#: consulted and nothing is fitted.
PLACEHOLDER_WORD_MAX_SHARE = _float("PLACEHOLDER_WORD_MAX_SHARE", 0.20)

#: Every string recognised as a possible blank. The two sets above decide *how* each one
#: is applied; this is the vocabulary, and exists so that a caller needing one list — the
#: exported script, mainly — has one.
MISSING_TOKENS: frozenset[str] = MISSING_MARKERS | MISSING_WORDS


# --------------------------------------------------------------------------------------
# Column routing
# --------------------------------------------------------------------------------------

#: A categorical column with at most this many distinct values is one-hot encoded.
#: Above it, one column per value costs more width than the signal is worth, so the
#: column is encoded to a single number instead.
ONEHOT_MAX_CARDINALITY = _int("ONEHOT_MAX_CARDINALITY", 10)

#: Columns missing more than this share of their values are dropped. Past roughly this
#: point an imputed column is mostly a record of our own fill value rather than the
#: user's data, and the model learns the imputation.
MAX_MISSING_RATE = _float("MAX_MISSING_RATE", 0.60)

#: A column whose distinct values exceed this share of the rows, *and* which has more
#: than ``TEXT_MIN_UNIQUE`` of them, is free text or an identifier rather than a
#: category. Both conditions are required: 8 distinct values in 10 rows is not free text.
TEXT_UNIQUE_RATE = _float("TEXT_UNIQUE_RATE", 0.5)
TEXT_MIN_UNIQUE = _int("TEXT_MIN_UNIQUE", 50)

#: A column this close to entirely distinct identifies the row, whatever its dtype. Set
#: high on purpose: a genuine measurement like income can be 60% distinct.
IDENTIFIER_UNIQUE_RATE = _float("IDENTIFIER_UNIQUE_RATE", 0.95)

#: A weaker bar applies when the column is *named* like a key — ``student_id``,
#: ``ref_no``. The name is evidence, so less uniqueness is needed to convict.
NAMED_IDENTIFIER_UNIQUE_RATE = _float("NAMED_IDENTIFIER_UNIQUE_RATE", 0.90)


# --------------------------------------------------------------------------------------
# Profiling signals
# --------------------------------------------------------------------------------------

#: Correlation with the target above which a feature is flagged as probable leakage.
#: Deliberately just below 1.0: an exact copy of the target is rare, a rounded or
#: rescaled copy is common, and both should be caught.
LEAKAGE_CORRELATION = _float("LEAKAGE_CORRELATION", 0.99)

#: A single feature that predicts the target this well, on its own, in cross-validation,
#: is almost certainly derived from it. Catches the non-linear copies correlation
#: misses: a status string, a bucketed score, anything that splits the answer cleanly
#: without moving in step with it.
LEAKAGE_TREE_SCORE = _float("LEAKAGE_TREE_SCORE", 0.98)

#: Folds for that check. Five is enough to be stable and cheap enough to run per column.
LEAKAGE_TREE_FOLDS = _int("LEAKAGE_TREE_FOLDS", 5)
LEAKAGE_TREE_CV_FOLDS = LEAKAGE_TREE_FOLDS

#: Depth of the probe tree. Two levels can express "this column separates the classes",
#: which is the question, without being able to memorise the training rows, which would
#: make every column look like a leak.
LEAKAGE_TREE_DEPTH = _int("LEAKAGE_TREE_DEPTH", 2)

#: A feature whose train/test distributions differ at this p-value is flagged as drift.
#: Deliberately strict: at 0.05, one feature in twenty is flagged by chance, which on a
#: fifty-column dataset means two or three false alarms every run and a warning nobody
#: reads.
DRIFT_P_VALUE = _float("DRIFT_P_VALUE", 0.01)

#: Below this many rows on either side, a KS test is dominated by sampling noise.
MIN_ROWS_FOR_DRIFT = _int("MIN_ROWS_FOR_DRIFT", 20)

#: Folds for the sanity baseline. Five is the usual compromise between a stable
#: estimate and the time it costs.
BASELINE_CV_FOLDS = _int("BASELINE_CV_FOLDS", 5)

#: Below this many training rows a cross-validated score says more about the split than
#: about the data, so no baseline is reported rather than a misleading one.
MIN_ROWS_FOR_BASELINE = _int("MIN_ROWS_FOR_BASELINE", 40)

#: Correlation between two features above which they are reported as redundant.
HIGH_CORRELATION = _float("HIGH_CORRELATION", 0.90)

#: Correlation is not reported below this many overlapping non-blank rows, where a
#: coefficient is dominated by chance.
MIN_ROWS_FOR_CORRELATION = _int("MIN_ROWS_FOR_CORRELATION", 30)

#: Tukey's multiplier. A value more than this many interquartile ranges outside the
#: quartiles is reported as an outlier. 1.5 is the convention behind every box plot;
#: it is reported, never acted on.
OUTLIER_IQR_MULTIPLIER = _float("OUTLIER_IQR_MULTIPLIER", 1.5)

#: A column missing more than this share gets a companion yes/no column recording which
#: rows were blank. Below it the indicator is mostly zeros and costs more width than it
#: carries; above it, whether a value was missing is often predictive in itself.
MISSING_INDICATOR_RATE = _float("MISSING_INDICATOR_RATE", 0.05)

#: Categories occurring in less than this share of rows are grouped as "Other". A value
#: seen four times in ten thousand rows teaches the model nothing that generalises, and
#: one that appears only in the held-out fold would be unseen at prediction time.
RARE_CATEGORY_RATE = _float("RARE_CATEGORY_RATE", 0.01)

#: Bins in a numeric column's histogram. Twenty is enough to show shape — bimodality,
#: a spike at zero, a long tail — without turning a few hundred rows into noise.
HISTOGRAM_BINS = _int("HISTOGRAM_BINS", 20)

#: How many of the most frequent values to report per column.
TOP_VALUES = _int("TOP_VALUES", 5)

#: |skew| above this marks a column as one-sided enough to be worth transforming. The
#: spec's figure. Below it, a log transform usually costs more interpretability than it
#: buys in symmetry.
SKEW_THRESHOLD = _float("SKEW_THRESHOLD", 1.0)

#: Class imbalance is reported when the largest class is at least this many times the
#: smallest. At 3:1 accuracy starts to flatter a model that always guesses the majority.
IMBALANCE_RATIO = _float("IMBALANCE_RATIO", 3.0)

#: A numeric target with more distinct values than this is a regression rather than a
#: many-class classification.
MAX_TARGET_CLASSES = _int("MAX_TARGET_CLASSES", 20)

#: How many classes to enumerate in a balance report. The interface renders these as a
#: list, so an unbounded one would push everything else off the screen; the true count
#: is reported alongside, so this elides rather than hides.
MAX_CLASSES_LISTED = _int("MAX_CLASSES_LISTED", 12)

#: A non-numeric target with more distinct values than this is a label or free-text
#: field, not an outcome, and is refused with an explanation.
MAX_TARGET_CARDINALITY = _int("MAX_TARGET_CARDINALITY", 50)


# --------------------------------------------------------------------------------------
# Preprocessing defaults
# --------------------------------------------------------------------------------------

#: Share of rows held out. 20% is the default everywhere for good reason: enough to
#: measure with, cheap enough to give up.
TEST_SIZE = _float("TEST_SIZE", 0.2)

#: The only source of randomness in the app. Fixed so the same file always produces the
#: same split, and so the exported script reproduces what the user was shown.
SEED = _int("SEED", 42)

#: ``"standard"`` | ``"minmax"`` | ``"robust"``. Standard is the default because it is
#: the one every downstream model assumes.
DEFAULT_SCALER = _str("DEFAULT_SCALER", "standard")

#: ``"frequency"`` | ``"target"``. Frequency is the default because it never reads the
#: target and so cannot leak under any circumstances; target encoding is offered because
#: the spec asks for it and sklearn's implementation cross-fits internally.
DEFAULT_HIGH_CARD_ENCODER = _str("DEFAULT_HIGH_CARD_ENCODER", "frequency")

#: Inner folds used by ``TargetEncoder`` to cross-fit its training-set output. This is
#: the mechanism that stops a row's own target contributing to its own encoding.
TARGET_ENCODER_CV = _int("TARGET_ENCODER_CV", 5)


# --------------------------------------------------------------------------------------
# Health score
# --------------------------------------------------------------------------------------

#: The health score starts at 100 and loses points for each defect below. Every entry is
#: ``(points per unit of the defect, maximum deduction)``, so no single problem can sink
#: the score on its own and the total is bounded and explainable.
#:
#: The formula, in full:
#:
#:     score = 100
#:       − min(missing_cell_fraction × 100 × 1.5, 35)     blank cells
#:       − min(duplicate_row_fraction × 100 × 1.0, 15)    exact duplicate rows
#:       − min(n_empty_columns × 4, 12)                   columns with nothing in them
#:       − min(n_sparse_columns × 3, 12)                  columns over MAX_MISSING_RATE
#:       − min(n_leakage_flags × 10, 20)                  features that look like the answer
#:       − 8 if rows < SMALL_SAMPLE_ROWS                  too small to trust
#:
#: Weights are ordered by how much each defect actually costs a model. Blank cells top
#: the list because they are the most common and the most quietly damaging; a small
#: sample is a flat penalty because it is a fact about the file rather than a matter of
#: degree. The result is clamped to 0..100.
HEALTH_WEIGHTS: dict[str, tuple[float, float]] = {
    "missing_cells": (1.5, 35.0),
    "duplicate_rows": (1.0, 15.0),
    "empty_columns": (4.0, 12.0),
    "sparse_columns": (3.0, 12.0),
    "leakage_flags": (10.0, 20.0),
}

#: Flat penalty applied to files below ``SMALL_SAMPLE_ROWS`` rows.
SMALL_SAMPLE_ROWS = _int("SMALL_SAMPLE_ROWS", 200)
SMALL_SAMPLE_PENALTY = _float("SMALL_SAMPLE_PENALTY", 8.0)

#: Blank cells below this share are not worth a line in the issues list.
MISSING_MENTION_RATE = _float("MISSING_MENTION_RATE", 0.02)


# --------------------------------------------------------------------------------------
# Running user code
# --------------------------------------------------------------------------------------

#: Wall-clock ceiling for one pipeline run. Generous next to the ~2s a real pipeline
#: takes on the file sizes this accepts, so only a genuinely stuck script hits it.
EXECUTION_TIMEOUT_SECONDS = _int("EXECUTION_TIMEOUT_SECONDS", 120)

#: CPU seconds. Lower than the wall clock on purpose: a busy loop should be caught by
#: the kernel rather than waited out.
EXECUTION_CPU_SECONDS = _int("EXECUTION_CPU_SECONDS", 90)

#: Address space for the child. Enough for pandas plus a one-hot expansion of a
#: reasonable dataset, small enough that a runaway allocation dies instead of the
#: machine swapping.
EXECUTION_MEMORY_BYTES = _int("EXECUTION_MEMORY_BYTES", 2 * 1024 * 1024 * 1024)

#: Largest file the sandboxed script may write, so a loop appending to the output
#: cannot fill the disk.
EXECUTION_MAX_OUTPUT_BYTES = _int("EXECUTION_MAX_OUTPUT_BYTES", 512 * 1024 * 1024)

#: Rows sent to the browser in one dataset preview page.
PREVIEW_PAGE_SIZE = _int("PREVIEW_PAGE_SIZE", 50)

#: Pipeline states kept for the change-history panel.
MAX_HISTORY_ENTRIES = _int("MAX_HISTORY_ENTRIES", 50)


# --------------------------------------------------------------------------------------
# Storage and persistence
# --------------------------------------------------------------------------------------

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

#: Phase 1 has no accounts, but every table carries a project so that adding them later
#: is a migration rather than a redesign. Everything lands in this one.
DEFAULT_PROJECT_ID = _str("DEFAULT_PROJECT_ID", "default")

#: SQLite lives next to the code. One file, no server, trivially inspectable — the right
#: database for an app whose data is a handful of rows per upload.
DATABASE_URL = _str("DATABASE_URL", f"sqlite:///{_BACKEND_ROOT / 'mlbenchx.db'}")

#: Uploaded files and fitted pipelines are written under here by ``LocalStorage``.
STORAGE_ROOT = Path(_str("STORAGE_ROOT", str(_BACKEND_ROOT / "storage")))

#: How many analysed runs to keep hot in memory. The database is the source of truth;
#: this only avoids re-reading and re-fitting for the run the user is looking at.
MAX_RUNS_IN_MEMORY = _int("MAX_RUNS_IN_MEMORY", 20)


# --------------------------------------------------------------------------------------
# Stage 4 — meta-learning
# --------------------------------------------------------------------------------------

#: How many past experiments inform one recommendation. Five is small enough that every
#: neighbour can be named in the rationale — a ranking the user cannot audit is the
#: thing this product exists not to produce — and large enough that one unusual dataset
#: cannot decide the answer on its own.
K_NEIGHBOURS = _int("K_NEIGHBOURS", 5)

#: Below this many usable records, no recommendation is offered at all.
#:
#: Not a tuning knob. With three records a "top-ranked family" is the winner of a field
#: of three, dressed as a finding. Declining is the honest output, and the engine says
#: why rather than returning an empty list.
MIN_CORPUS = _int("MIN_CORPUS", 8)

#: Scaled coordinates are clipped to ±this before any distance is measured.
#:
#: Not an independent choice. Each coordinate is divided by its observed range across the
#: corpus, so every record the scaler was fitted on lands within ±1 of the median on
#: every coordinate — the bound is arithmetic. The clip therefore does exactly one thing:
#: it bounds a *query* whose value falls outside everything the corpus has seen, which
#: the divisor cannot help with because it is fitted on the corpus alone.
#:
#: That case is real and not rare. A dataset with 11 classes queried against a corpus
#: whose records have 2 to 5 sits far outside on that coordinate; unclipped it is
#: described almost entirely by it. Clipping says "outside anything comparable", which is
#: true, instead of letting one axis decide what the dataset resembles.
SCALER_CLIP = _float("SCALER_CLIP", 1.0)

#: Cosine similarity, after scaling, below which the corpus is treated as having
#: nothing to say about this dataset. Neighbours still get reported; the ranking does
#: not, because extrapolating from datasets that look nothing like yours is guessing.
MIN_SIMILARITY = _float("MIN_SIMILARITY", 0.30)


# --------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------

#: Values in a before/after strip. Three is enough to show a change in context and few
#: enough to read at a glance.
STRIP_WIDTH = _int("STRIP_WIDTH", 3)

#: Explanations name at most this many columns inline before switching to a summary.
#: Every value is always present in the machine-readable details block.
MAX_COLUMNS_IN_EXPLANATION = _int("MAX_COLUMNS_IN_EXPLANATION", 6)
