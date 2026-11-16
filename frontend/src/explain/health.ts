/**
 * Translating one health dimension's facts into evidence a reader can act on.
 *
 * The insights panel's docstring says the dimensions exist so that "a score a user
 * cannot take apart is a score they have to take on faith". The renderer did not keep
 * that promise. `DimensionFacts` pushed a fixed list of scalar keys and skipped
 * anything that was not a number or a flat array — which silently dropped the most
 * specific evidence the server sends:
 *
 * * **`worst_columns`** (completeness) — which columns have gaps and how badly. The
 *   summary said "across 3 column(s)" and never named one of them.
 * * **`per_column_outliers`** (validity) — `Fare: 128, SibSp: 27, Age: 15`. Not shown,
 *   so "170 values outside the usual range" was unlocatable.
 * * **`skew` and `heavily_skewed`** (distribution) — the summary said "2 heavily skewed
 *   column(s)" and the panel could not tell you which two.
 * * **`mixed_type_columns`, `inconsistent_casing`** (consistency) — both dropped, along
 *   with the examples of the offending values.
 *
 * Everything below is read out of those same facts. Nothing is computed here and
 * nothing is estimated: where a dimension reports no evidence, this returns none rather
 * than filling the space.
 */

import type { HealthDimension } from "../api";

/** One measured line. */
export type HealthFact = {
  label: string;
  value: string;
  meaning?: string;
  term?: string;
};

/** A named column with what was measured about it. */
export type ColumnFinding = { column: string; detail: string };

export type DimensionEvidence = {
  facts: HealthFact[];
  /** The heading over `columns`, when there are any. */
  columnsHeading?: string;
  columns: ColumnFinding[];
  /** Sentences that do not belong to any single number. */
  notes: string[];
};

const count = (n: number) => n.toLocaleString();
const pct = (share: number) => `${(share * 100).toFixed(1)}%`;

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function describeDimension(dimension: HealthDimension): DimensionEvidence {
  const f = dimension.facts;
  const facts: HealthFact[] = [];
  const columns: ColumnFinding[] = [];
  const notes: string[] = [];
  let columnsHeading: string | undefined;

  switch (dimension.key) {
    case "completeness": {
      const missing = asNumber(f.missing_cells);
      const total = asNumber(f.total_cells);
      if (missing != null && total != null) {
        facts.push({
          label: "Blank cells",
          value: `${count(missing)} of ${count(total)}`,
          meaning:
            missing > 0
              ? `${pct(missing / Math.max(total, 1))} of every cell in the file is empty.`
              : undefined,
        });
      }
      const withGaps = asNumber(f.columns_with_gaps);
      if (withGaps != null) facts.push({ label: "Columns with gaps", value: count(withGaps) });

      const worst = Array.isArray(f.worst_columns) ? f.worst_columns : [];
      if (worst.length) {
        columnsHeading = "Emptiest columns";
        for (const entry of worst) {
          const row = asRecord(entry);
          const rate = asNumber(row.missing_rate);
          if (typeof row.column === "string" && rate != null) {
            columns.push({ column: row.column, detail: `${pct(rate)} blank` });
          }
        }
        notes.push(
          "A column past roughly half empty is usually dropped rather than filled, because a filled column that mostly did not exist is mostly invented.",
        );
      }
      break;
    }

    case "validity": {
      const outliers = asNumber(f.outliers);
      const checked = asNumber(f.checked_columns);
      if (outliers != null) {
        facts.push({
          label: "Values outside the usual range",
          value: count(outliers),
          term: "outlier",
          meaning:
            checked != null
              ? `Found by the interquartile-range rule across ${count(checked)} numeric columns.`
              : undefined,
        });
      }
      const infinite = asNumber(f.infinite);
      if (infinite != null) {
        facts.push({
          label: "Infinite values",
          value: count(infinite),
          meaning:
            infinite > 0
              ? "An infinity is an overflow, not a measurement — there is no quantity here for a model to learn from."
              : undefined,
        });
      }

      const perColumn = asRecord(f.per_column_outliers);
      const entries = Object.entries(perColumn);
      if (entries.length) {
        columnsHeading = "Where they are";
        for (const [column, value] of entries) {
          const n = asNumber(value);
          if (n != null) columns.push({ column, detail: `${count(n)} value${n === 1 ? "" : "s"}` });
        }
        notes.push(
          "Being unusual is not being wrong. An exceptional real value and a mistyped one are indistinguishable from here, which is why the default caps them instead of deleting the row.",
        );
      }
      break;
    }

    case "consistency": {
      const mixed = asRecord(f.mixed_type_columns);
      const casing = asRecord(f.inconsistent_casing);

      facts.push({
        label: "Columns mixing numbers and text",
        value: count(Object.keys(mixed).length),
        meaning:
          "A column that is almost entirely numeric is the dangerous case: fully numeric is fine, fully textual is fine, 97% numeric is a broken export.",
      });
      facts.push({
        label: "Columns with values differing only by case or spacing",
        value: count(Object.keys(casing).length),
        meaning: '"Yes", "yes" and " Yes " are three separate categories to an encoder.',
      });

      const found: ColumnFinding[] = [];
      for (const [column, detail] of Object.entries(mixed)) {
        const row = asRecord(detail);
        const share = asNumber(row.numeric_share);
        const examples = Array.isArray(row.non_numeric_examples)
          ? row.non_numeric_examples.map(String)
          : [];
        found.push({
          column,
          detail: [
            share != null ? `${pct(share)} numeric` : null,
            examples.length ? `non-numeric: ${examples.join(", ")}` : null,
          ]
            .filter(Boolean)
            .join(" · "),
        });
      }
      for (const [column, collapsed] of Object.entries(casing)) {
        const n = asNumber(collapsed);
        found.push({
          column,
          detail: n != null ? `${count(n)} value${n === 1 ? "" : "s"} are duplicates once case and spacing are ignored` : "inconsistent casing",
        });
      }
      if (found.length) {
        columnsHeading = "Which columns";
        columns.push(...found);
      }
      break;
    }

    case "uniqueness": {
      const duplicates = asNumber(f.duplicate_rows);
      const distinct = asNumber(f.distinct_rows);
      if (duplicates != null) {
        facts.push({
          label: "Repeated rows",
          value: count(duplicates),
          meaning:
            duplicates > 0
              ? "Each repeat is one observation counted twice, which weights it double when the model fits."
              : undefined,
        });
      }
      if (distinct != null) facts.push({ label: "Distinct rows", value: count(distinct) });
      const constant = Array.isArray(f.constant_columns) ? f.constant_columns.map(String) : [];
      if (constant.length) {
        columnsHeading = "Columns that never change";
        columns.push(
          ...constant.map((column) => ({ column, detail: "one value in every row" })),
        );
        notes.push(
          "A column holding the same value everywhere cannot explain anything that varies, so it contributes nothing to a model.",
        );
      }
      break;
    }

    case "distribution": {
      const numeric = asNumber(f.numeric_columns);
      if (numeric != null) facts.push({ label: "Numeric columns", value: count(numeric) });

      const disparity = asNumber(f.scale_disparity);
      if (disparity != null) {
        facts.push({
          label: "Scale disparity",
          value: `${disparity.toFixed(1)}×`,
          term: "scale disparity",
          // The definition of the ratio belongs in the tooltip. What belongs here is
          // what *this* ratio means, which is the opposite thing at 1.0 and at 359.
          meaning:
            disparity > 1.5
              ? "Any model that measures distance between rows is dominated by the broadest column until they are scaled."
              : "These columns already vary over comparable ranges, so no one of them can dominate on size alone.",
        });
      }

      const skew = asRecord(f.skew);
      const heavy = asRecord(f.heavily_skewed);
      const entries = Object.entries(skew);
      if (entries.length) {
        columnsHeading = "How lopsided each column is";
        for (const [column, value] of entries) {
          const n = asNumber(value);
          if (n == null) continue;
          const heavily = column in heavy;
          const direction = n > 0 ? "a long tail of large values" : "a long tail of small values";
          columns.push({
            column,
            detail: `${n > 0 ? "+" : ""}${n.toFixed(2)}${heavily ? ` — heavily skewed, ${direction}` : ""}`,
          });
        }
        notes.push(
          "Zero is symmetric. Past about ±1 most of the column is squeezed into a narrow band with a few values far out, which makes plain scaling much less effective than a power transform.",
        );
      }
      break;
    }

    case "feature_quality": {
      const features = asNumber(f.feature_count);
      if (features != null) {
        facts.push({
          label: "Feature columns",
          value: count(features),
          meaning: "Every column except the target.",
        });
      }

      const identifiers = Array.isArray(f.identifier_columns)
        ? f.identifier_columns.map(String)
        : [];
      const redundant = Array.isArray(f.redundant_pairs) ? f.redundant_pairs : [];
      const leaks = Array.isArray(f.leakage_suspects) ? f.leakage_suspects : [];

      facts.push({
        label: "Identifier-like columns",
        value: count(identifiers.length),
        term: "identifier-like",
        meaning: identifiers.length
          ? `${identifiers.join(", ")} — almost every row holds a different value, so these identify the row rather than describing it.`
          : undefined,
      });

      if (redundant.length) {
        columnsHeading = "Pairs carrying nearly the same information";
        for (const entry of redundant) {
          const row = asRecord(entry);
          const pair = Array.isArray(row.columns) ? row.columns.map(String) : [];
          const correlation = asNumber(row.correlation);
          if (pair.length === 2) {
            columns.push({
              column: `${pair[0]} ≈ ${pair[1]}`,
              detail:
                correlation != null
                  ? `correlation ${correlation.toFixed(2)}`
                  : "closely correlated",
            });
          }
        }
        notes.push(
          "A correlation of 1.00 would mean the two columns are the same column twice over. Harmless to most models, but one signal split across two features makes each look less important than it is.",
        );
      }

      if (leaks.length) {
        for (const entry of leaks) {
          const row = asRecord(entry);
          const correlation = asNumber(row.correlation);
          if (typeof row.column === "string") {
            notes.push(
              `${row.column} tracks the target${correlation != null ? ` at ${correlation.toFixed(2)}` : ""}. A column that close to the answer is usually a copy of it, recorded after the fact — check whether it would exist at prediction time.`,
            );
          }
        }
      }
      break;
    }
  }

  return { facts, columnsHeading, columns, notes };
}
