/**
 * Translating a step's measured facts into sentences, and finding its code.
 *
 * `StepFacts` arrives as a bag of keys whose names are honest but whose *meaning*
 * changes from step to step, and the workspace was rendering them generically. Two
 * consequences, both found by reading what the server actually sends rather than the
 * type:
 *
 * **`per_column` counts a different thing in every step.** For missing values it is
 * blank cells (`Age: 184`). For outliers it is values outside the usual range
 * (`Fare: 128`). For encoding it is the number of *distinct* values (`Sex: 2`). The
 * card listed all three as a bare number under the heading "Which columns", so `Sex 2`
 * and `Age 184` read as the same kind of quantity. They are not remotely the same
 * quantity, so each step names its own unit here.
 *
 * **`rows_affected` is not always rows that change.** For Data Type Detection it is
 * every row in the file, because every row is inspected and none is altered. For Class
 * Imbalance it is the size of the *smallest class* — 373 of 891, which is neither a
 * count of changes nor a problem. Printing "891 rows affected" or "373 rows affected"
 * under those steps would be false, so neither does.
 *
 * What is deliberately not shown
 * ------------------------------
 * `text_date_columns` is initialised to zero in `impact.step_facts` and never
 * incremented — dates stored as text are not currently counted. The key is therefore
 * always `0`, which is not a measurement of anything, so it is not displayed. Numbers
 * stored as text *are* counted, and are.
 */

import type { StepFacts, StepId, StepView } from "../api";

/** One line of evidence: what was measured, the number, and what the number means. */
export type StepFact = {
  label: string;
  value: string;
  /** The sentence under the number. Omitted where the label carries it. */
  meaning?: string;
  /** A vocabulary key, when a word in the label needs defining. */
  term?: string;
};

/** What a step's `per_column` numbers count, so the list can say so. */
export type PerColumnLegend = {
  heading: string;
  /** Singular noun for one unit, used as "184 blank cells". */
  unit: string;
  term?: string;
};

const count = (n: number) => n.toLocaleString();

const plural = (n: number, word: string) => `${count(n)} ${word}${n === 1 ? "" : "s"}`;

/**
 * One measured line for the face of the card.
 *
 * This is the glance layer, and it is written to *complement* `step.summary` rather
 * than to restate it. The summary already says what the step will do and to how many
 * things ("885 missing value(s) in 3 column(s) will be filled"); repeating those counts
 * underneath would teach the reader that the second line never repays reading. So each
 * step says here the thing its summary leaves out: which columns, what it would cost to
 * skip, how far apart the scales are.
 *
 * Returns null where the summary already carries everything measured — a dataset with
 * no duplicates needs one sentence about duplicates, not two.
 */
export function headline(step: StepView, totalRows: number): string | null {
  const f = step.facts;

  switch (step.id) {
    case "data_types": {
      const text = f.text_numeric_columns ?? 0;
      return text > 0
        ? `${plural(text, "column")} hold numbers written as text, and will be converted`
        : "No column holds numbers written as text";
    }

    case "missing_values": {
      const rows = f.rows_affected ?? 0;
      if ((f.values_affected ?? 0) === 0 || rows === 0) return null;
      return `${count(rows)} of ${count(totalRows)} rows have at least one gap — what dropping them instead would cost`;
    }

    case "duplicates": {
      const rows = f.rows_affected ?? 0;
      if (rows === 0 || f.share == null) return null;
      return `${(f.share * 100).toFixed(1)}% of all rows`;
    }

    case "outliers": {
      if ((f.values_affected ?? 0) === 0) return null;
      const names = Object.keys(f.per_column ?? {});
      if (!names.length) return null;
      const shown = names.slice(0, 3).join(", ");
      return `Found in ${shown}${names.length > 3 ? ` and ${names.length - 3} more` : ""}`;
    }

    case "encoding": {
      const widths = Object.entries(f.per_column ?? {});
      if (widths.length < 2) return null;
      // `per_column` is the number of distinct values here, which is exactly what
      // decides whether a column gets one column per value or is encoded in place.
      const sorted = [...widths].sort((a, b) => b[1] - a[1]);
      const [wide, wideN] = sorted[0];
      const [narrow, narrowN] = sorted[sorted.length - 1];
      return `${wide} has ${count(wideN)} distinct values, ${narrow} has ${count(narrowN)} — which is what decides how each is encoded`;
    }

    case "scaling": {
      const disparity = f.scale_disparity;
      if (disparity == null || disparity <= 1.5 || !f.widest_column || !f.narrowest_column) {
        return null;
      }
      return `${f.widest_column} spreads ${disparity}× wider than ${f.narrowest_column}`;
    }

    case "feature_selection": {
      const constant = f.constant_columns ?? [];
      const redundant = f.redundant_pairs ?? [];
      if (!constant.length && !redundant.length) return null;
      const parts: string[] = [];
      if (constant.length) parts.push(`${constant.slice(0, 3).join(", ")} never change`);
      if (redundant.length) {
        parts.push(
          `${redundant
            .slice(0, 2)
            .map((pair) => pair.join(" ≈ "))
            .join(", ")} carry nearly the same information`,
        );
      }
      return parts.join("; ");
    }

    case "imbalance": {
      if (!f.ratio || !f.minority) return null;
      return `The smaller class (${f.minority}) has ${count(f.rows_affected ?? 0)} of ${count(totalRows)} rows`;
    }

    default:
      return null;
  }
}

/**
 * The evidence layer: every measured number this step reported, with its meaning.
 *
 * Written per step rather than from the keys, because the same key means different
 * things in different steps — see the note at the top of this file.
 */
export function describeStep(step: StepView): StepFact[] {
  const f: StepFacts = step.facts;
  const out: StepFact[] = [];

  switch (step.id) {
    case "data_types":
      out.push({
        label: "Columns inspected",
        value: count(f.columns_affected ?? 0),
        meaning: "Every column is examined. Only the recorded type changes — no value is rewritten.",
      });
      out.push({
        label: "Numbers stored as text",
        value: count(f.text_numeric_columns ?? 0),
        meaning:
          (f.text_numeric_columns ?? 0) > 0
            ? "These convert to real numbers, and only when every non-blank value in the column converts cleanly."
            : "Nothing in this file is a number trapped in a text column.",
      });
      break;

    case "missing_values":
      out.push({
        label: "Blank cells",
        value: count(f.values_affected ?? 0),
        meaning: "Most models refuse to train on a blank, so the choice is to fill them or lose those rows.",
        term: "imputation",
      });
      out.push({
        label: "Columns with gaps",
        value: count(f.columns_affected ?? 0),
      });
      out.push({
        label: "Rows with at least one gap",
        value: count(f.rows_affected ?? 0),
        meaning: "Dropping rows instead of filling them would cost you all of these.",
      });
      if (f.columns_to_drop?.length) {
        out.push({
          label: "Too empty to fill",
          value: f.columns_to_drop.join(", "),
          meaning:
            "Past a certain emptiness a filled column is mostly invented, so these are dropped rather than imputed.",
        });
      }
      break;

    case "duplicates":
      out.push({
        label: "Repeated rows",
        value: count(f.rows_affected ?? 0),
        meaning: "An exact repeat is one observation counted twice, which quietly doubles its weight when the model fits.",
      });
      if (f.share != null) {
        out.push({ label: "Share of the dataset", value: `${(f.share * 100).toFixed(1)}%` });
      }
      break;

    case "outliers":
      out.push({
        label: "Values outside the usual range",
        value: count(f.values_affected ?? 0),
        term: "outlier",
        meaning: "An unusually large real value and a typing error look identical from here.",
      });
      out.push({ label: "Columns involved", value: count(f.columns_affected ?? 0) });
      break;

    case "encoding":
      out.push({
        label: "Text columns",
        value: count(f.columns_affected ?? 0),
        meaning: "A model does arithmetic, so every one of these has to become a number first.",
      });
      out.push({
        label: "Values to convert",
        value: count(f.values_affected ?? 0),
        meaning: "Every non-blank cell in those columns.",
      });
      if (f.columns_created != null) {
        // `columns_created` sums the widths of the columns that qualify for one-hot, so
        // it is the number of one-hot columns that exist afterwards — not the net change
        // in column count, since each of them replaces an original.
        out.push({
          label: "One-hot columns produced",
          value: count(f.columns_created),
          term: "one-hot",
          meaning:
            "Only columns with few enough distinct values get one column per value, and those replace the original. Wider columns are encoded in place and add no columns at all.",
        });
      }
      break;

    case "scaling":
      out.push({ label: "Numeric columns", value: count(f.columns_affected ?? 0) });
      out.push({
        label: "Values rescaled",
        value: count(f.values_affected ?? 0),
      });
      if (f.scale_disparity != null && f.widest_column && f.narrowest_column) {
        out.push({
          label: "Scale disparity",
          value: `${f.scale_disparity}×`,
          term: "scale disparity",
          meaning: `${f.widest_column} varies ${f.scale_disparity} times more widely than ${f.narrowest_column}. Any model that measures distance between rows would be dominated by ${f.widest_column} until both are scaled.`,
        });
      }
      break;

    case "feature_selection":
      out.push({
        label: "Columns that would be removed",
        value: count(f.columns_affected ?? 0),
      });
      if (f.constant_columns?.length) {
        out.push({
          label: "Never change",
          value: f.constant_columns.join(", "),
          term: "constant column",
          meaning: "A column that holds one value everywhere cannot explain anything that varies.",
        });
      }
      if (f.redundant_pairs?.length) {
        out.push({
          label: "Near-duplicate pairs",
          value: f.redundant_pairs.slice(0, 3).map((pair) => pair.join(" ≈ ")).join(", "),
          term: "redundant pair",
          meaning: "Two columns moving together split one signal across two features, making each look less important than it is.",
        });
      }
      break;

    case "imbalance":
      if (f.ratio) {
        out.push({
          label: "Largest class against smallest",
          value: `${f.ratio.toFixed(2)} : 1`,
          term: "class imbalance",
          meaning: `A model that always answered "${f.majority}" would still look accurate. Judge it on how it handles ${f.minority}.`,
        });
        out.push({
          label: `Rows in the smaller class (${f.minority})`,
          value: count(f.rows_affected ?? 0),
        });
      }
      break;
  }

  return out;
}

/** What this step's per-column numbers count. Null where the step reports none. */
export function perColumnLegend(id: StepId): PerColumnLegend | null {
  switch (id) {
    case "missing_values":
      return { heading: "Blank cells, by column", unit: "blank cell" };
    case "outliers":
      return { heading: "Values outside the usual range, by column", unit: "value", term: "outlier" };
    case "encoding":
      return {
        heading: "Distinct values, by column",
        unit: "distinct value",
        term: "cardinality",
      };
    default:
      return null;
  }
}

/**
 * The line in the current code where this step's section begins.
 *
 * The generator writes each step's own label as a banner comment — `# Feature Scaling`
 * — so the link is read out of the code that is actually in the editor rather than
 * from a table of line numbers that would drift the moment anyone typed. A disabled
 * step emits no section, and hand-edited code may have had the section removed; in
 * both cases this returns null and the link is simply not offered.
 */
export function findStepInCode(code: string, label: string): number | null {
  const wanted = `# ${label}`;
  const lines = code.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i].trim() === wanted) return i + 1;
  }
  return null;
}
