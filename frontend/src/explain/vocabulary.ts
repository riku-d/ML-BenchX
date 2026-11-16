/**
 * Definitions for the terms this product uses but never explains.
 *
 * An inventory of every user-facing string (see `docs/EXPLANATION_INVENTORY.md`) found
 * roughly thirty terms appearing in text the user is expected to act on, with no
 * definition anywhere in the application: no glossary, no tooltips, no first-use gloss.
 * They fall into four groups — scikit-learn identifiers that leaked into prose
 * (`onehot`, `label`, `frequency`), algorithms named without introduction (k-NN,
 * gradient descent), ordinary statistics (interquartile range, skew, p-value), and
 * terms this product coined itself (scale disparity, signal, drift).
 *
 * Two rules govern what is written here.
 *
 * **These are definitions, not replacements.** The product's own sentences are derived
 * from fitted objects after the fact, which is why they cannot drift out of sync with
 * what ran. Nothing here rewrites them. A definition sits beside a term and explains it.
 *
 * **Product vocabulary is defined, never translated away.** `docs/PRODUCT.md` lists
 * terms — glass-box, the split, the plan, compile, dataset health, impact, the dock —
 * as product language rather than jargon to be smoothed out. Those get an entry so a
 * newcomer can learn them, and keep their names.
 *
 * Each definition says what the thing is and, where it earns the line, why it matters
 * here. A definition that only restates the term is worse than none, because it costs
 * the reader a click to learn nothing.
 */

export type Definition = {
  /** What it is, in one or two sentences. */
  what: string;
  /** Why it matters for preprocessing. Omitted where `what` already carries it. */
  why?: string;
};

export const VOCABULARY: Record<string, Definition> = {
  // --- how categories become numbers ------------------------------------------------
  "one-hot": {
    what: "One new column per category, holding 1 when the row has that value and 0 when it does not.",
    why: "It lets a model use a category without implying the values have an order. Three categories become three columns, so it is only affordable when there are not many.",
  },
  ordinal: {
    what: "Each category is replaced by a whole number — first value 0, second 1, and so on.",
    why: "Compact, but it tells the model the categories have an order and even spacing. That is right for small/medium/large and wrong for red/green/blue.",
  },
  "label encoding": {
    what: "The same as ordinal encoding: each distinct value becomes a number.",
    why: "Harmless when a column has exactly two values, because any two numbers carry the same information as yes/no.",
  },
  "frequency encoding": {
    what: "Each category is replaced by how often it occurs in the training rows.",
    why: "It handles columns with many categories without creating many columns, and it never looks at the target, so it cannot leak the answer.",
  },
  "target encoding": {
    what: "Each category is replaced by the average outcome recorded for it.",
    why: "Often the most predictive option, and the most dangerous: done naively a row helps encode itself. This product cross-fits it so that cannot happen.",
  },
  "cross-fitted": {
    what: "The data is split into parts, and each part's encoding is computed only from the other parts.",
    why: "It stops a row influencing its own encoded value, which would otherwise leak the answer into the feature.",
  },

  // --- filling gaps ------------------------------------------------------------------
  median: {
    what: "The middle value when the column is sorted — half the rows are above it, half below.",
    why: "Preferred over the mean for filling gaps, because a few extreme values cannot drag it around.",
  },
  mean: {
    what: "The ordinary average: everything added up, divided by the count.",
    why: "Sensitive to extremes. One very large value pulls it away from where most of the data sits.",
  },
  imputation: {
    what: "Filling a blank cell with a value worked out from the rows that do have one.",
    why: "Most models refuse to train on blanks at all, so the choice is to fill them or to drop those rows.",
  },
  "missing indicator": {
    what: "An extra yes/no column recording which rows were blank before the gap was filled.",
    why: "Sometimes the fact that a value was missing is itself informative — an unanswered question is not the same as an average answer.",
  },

  // --- unusual values ----------------------------------------------------------------
  outlier: {
    what: "A value far outside the range the rest of the column occupies.",
    why: "An unusually large real value and a typing error look identical from here, which is why this product caps them rather than deleting the row.",
  },
  "interquartile range": {
    what: "The span covering the middle half of the data — from the 25th percentile to the 75th.",
    why: "It describes the usual spread without being moved by extremes, so it is a stable basis for deciding what counts as unusual.",
  },
  skew: {
    what: "How lopsided a column is. Zero is symmetric; a positive value means a long tail of large values on the right.",
    why: "Strongly skewed columns make scaling less effective, because most values end up squeezed into a narrow band.",
  },
  "power transform": {
    what: "A reshaping of a column — such as a log or Yeo-Johnson transform — that pulls in a long tail and makes the spread more even.",
  },
  "Yeo-Johnson": {
    what: "A power transform that makes a skewed column more symmetric, and unlike a log transform it works on zero and negative values.",
  },

  // --- scale -------------------------------------------------------------------------
  scaling: {
    what: "Putting every numeric column onto a comparable range, so that one is not numerically far larger than another.",
    why: "Any model that measures distance between rows will otherwise be dominated by whichever column happens to use the biggest numbers.",
  },
  "scale disparity": {
    what: "How much more widely the broadest column varies than the narrowest, as a ratio.",
    why: "A high ratio means one column would overwhelm the others in any distance-based model, until they are scaled.",
  },
  standardisation: {
    what: "Shifting a column so its average is zero, then dividing by its spread, so most values land between about -3 and +3.",
  },

  // --- the split and leakage ---------------------------------------------------------
  "the split": {
    what: "Dividing the rows into a training set and a held-out test set, before anything is measured or decided.",
    why: "This is the product's central guarantee. Fill values, category lists, scaler centres and the decisions about which columns survive all come from the training rows alone.",
  },
  stratified: {
    what: "A split that keeps the mix of outcomes the same on both sides.",
    why: "Without it, a random split can leave the rarer class badly represented in one half, and the test score stops meaning anything.",
  },
  leakage: {
    what: "Information reaching the model that it would not have at the moment it needs to make a real prediction.",
    why: "It makes a model look excellent in testing and fail in use. It is the single most common reason a promising result does not survive contact with reality.",
  },
  "training rows": {
    what: "The portion of the data — 80% by default — that every decision and every fitted value is computed from.",
    why: "The remaining rows are never consulted. That is what makes the test score an honest estimate.",
  },
  "cross-validation": {
    what: "Splitting the training rows into several parts, fitting on all but one and scoring on the one left out, then rotating.",
    why: "It gives a more stable estimate than a single split, and it uses only training rows, so the held-out test set stays untouched.",
  },
  fold: {
    what: "One of the parts the training rows are divided into during cross-validation.",
  },

  // --- what the run reports ----------------------------------------------------------
  drift: {
    what: "A column whose distribution differs noticeably between the training rows and the held-out rows.",
    why: "The split is random, so large differences should be rare. One that shows up is usually a sign of hidden structure — data in time order, or rows that are not independent.",
  },
  "p-value": {
    what: "The probability of seeing a difference this large purely by chance, if there were really no difference at all.",
    why: "A small value means chance is an unlikely explanation. The threshold here is set low on purpose, because many columns are tested at once and some will look unusual by luck.",
  },
  baseline: {
    what: "A deliberately simple model, used as a yardstick — here, one that always guesses the most common outcome.",
    why: "If a real model cannot beat it, the prepared features carry no usable information. It is a check on the preparation, not a result.",
  },
  lift: {
    what: "How much a simple real model beats the trivial baseline by.",
    why: "A positive lift means there is something in the features worth modelling. It is not a performance claim — it was measured on training rows only.",
  },
  signal: {
    what: "Information in the columns that actually relates to the outcome being predicted.",
  },
  "class imbalance": {
    what: "One outcome occurring far more often than another.",
    why: "A model can score well by always predicting the common one. Judge it on how it handles the rarer class, not on overall accuracy.",
  },

  // --- columns -----------------------------------------------------------------------
  "identifier-like": {
    what: "A column where almost every row has a different value — an ID, a ticket number, a name.",
    why: "It identifies the row rather than describing it, so it cannot generalise to rows the model has not seen.",
  },
  correlation: {
    what: "How closely two columns move together, from -1 to 1. Zero means knowing one tells you nothing about the other; 1 means they rise and fall in lockstep.",
    why: "Near 1 between two features means they are nearly the same column. Near 1 between a feature and the target usually means the feature is a copy of the answer.",
  },
  "redundant pair": {
    what: "Two columns that move together so closely they carry nearly the same information.",
    why: "Not harmful to most models, but it splits one signal across two features, making each look less important than it is.",
  },
  "constant column": {
    what: "A column where every row holds the same value.",
    why: "It cannot explain anything that varies, so it contributes nothing.",
  },
  cardinality: {
    what: "How many distinct values a column contains.",
    why: "It decides how a category is encoded: a few values can afford one column each, many cannot.",
  },

  // --- product vocabulary — defined, deliberately not renamed -------------------------
  "glass-box": {
    what: "The opposite of a black box: every decision is visible, explained against your own data, optional and reversible.",
  },
  "dataset health": {
    what: "A score out of 100, built from six measured dimensions each given a weight.",
    why: "It is not a grade. It is a summary of what is wrong, and every part of it can be taken apart and checked.",
  },
  compile: {
    what: "Running the pipeline on your data, in a sandbox, and measuring what came out.",
  },
  "the plan": {
    what: "The decision made for each column, with the reason it was chosen. Versioned, and never changed in place — editing it creates a new version.",
  },
};

/** Look a term up, case-insensitively. Returns undefined when it is not in the list. */
export function define(term: string): Definition | undefined {
  return VOCABULARY[term] ?? VOCABULARY[term.toLowerCase()];
}
