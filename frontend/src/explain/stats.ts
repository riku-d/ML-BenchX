/**
 * Turning the plan's measurement keys into something a person can read.
 *
 * The plan review screen rendered its evidence panel as `Object.entries(column.stats)`
 * with underscores swapped for spaces. Across five datasets and 57 columns that is
 * exactly eight keys, every one shown raw:
 *
 *     n missing      0          top share     0.577
 *     missing rate   0          rows measured 712
 *     n unique       712        skew          0.389
 *
 * `top_share` is meaningless without being told it is the share held by the single most
 * common value. `rows_measured` is worse than meaningless — it is the number that
 * proves this product's central claim, that every decision came from the training rows
 * alone, and it was displayed as an unlabelled key beside an unexplained count.
 *
 * So each stat gets a label in words, a formatted value, and — where the number means
 * something specific for *this* column — a sentence saying what it means. The
 * interpretation is computed from the value and the column's kind, not written into a
 * lookup table, so it cannot describe a column it was not measured on.
 */

import type { ColumnPlan } from "../api";

export type StatView = {
  key: string;
  /** What to call it in words. */
  label: string;
  /** The value, formatted for reading rather than for precision. */
  value: string;
  /** What this number means for this column. Omitted when the label says it all. */
  meaning?: string;
  /** A key in the vocabulary, when the label names a term worth defining. */
  term?: string;
  /** Shown first and given emphasis — currently only the leakage evidence. */
  emphasis?: boolean;
};

const ORDER = [
  "rows_measured",
  "n_missing",
  "missing_rate",
  "n_unique",
  "top_share",
  "median",
  "mean",
  "skew",
];

function count(value: unknown): string {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : String(value);
}

function share(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n === 0) return "none";
  return n < 0.01 ? "<1%" : `${Math.round(n * 100)}%`;
}

function number(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Number.isInteger(n)) return n.toLocaleString();
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toPrecision(3);
}

function signed(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return `${n > 0 ? "+" : ""}${n.toPrecision(2)}`;
}

/**
 * Describe one column's measurements.
 *
 * `rows` is the dataset's total row count where the caller knows it. It is used only to
 * say how `rows_measured` compares — the point being that it is deliberately smaller.
 */
export function describeStats(column: ColumnPlan, rows?: number): StatView[] {
  const stats = column.stats ?? {};
  const views: StatView[] = [];

  const push = (view: StatView | null) => {
    if (view) views.push(view);
  };

  for (const key of ORDER) {
    if (!(key in stats)) continue;
    const raw = stats[key];

    switch (key) {
      case "rows_measured": {
        const measured = Number(raw);
        push({
          key,
          label: "Decided from",
          value: `${count(raw)} rows`,
          term: "training rows",
          emphasis: true,
          meaning:
            rows && Number.isFinite(measured) && measured < rows
              ? `The training rows only — ${count(rows - measured)} held-out rows took no part in this decision.`
              : "The training rows only. The held-out rows took no part in this decision.",
        });
        break;
      }

      case "n_missing": {
        const n = Number(raw);
        push({
          key,
          label: "Blank values",
          value: n === 0 ? "none" : count(raw),
          meaning: n === 0 ? "Nothing to fill." : undefined,
        });
        break;
      }

      case "missing_rate": {
        const rate = Number(raw);
        if (rate === 0) break; // "Blank values: none" already said it
        push({
          key,
          label: "Share blank",
          value: share(raw),
          meaning:
            rate >= 0.6
              ? "Too much to fill honestly — filling it would record the fill value more than your data."
              : rate >= 0.2
                ? "Enough that being blank may itself be informative."
                : undefined,
        });
        break;
      }

      case "n_unique": {
        const n = Number(raw);
        const measured = Number(stats["rows_measured"]);
        const everyRowDiffers = Number.isFinite(measured) && n >= measured;
        push({
          key,
          label: "Distinct values",
          value: count(raw),
          term: "cardinality",
          meaning: everyRowDiffers
            ? "Every row differs, so this identifies the row rather than describing it."
            : n === 1
              ? "Every row holds the same value."
              : n === 2
                ? "Two values, so a single 0/1 column carries all of it."
                : column.kind === "categorical" && n > 10
                  ? "Too many for one column each without swamping the model."
                  : undefined,
        });
        break;
      }

      case "top_share": {
        const s = Number(raw);
        push({
          key,
          label: "Most common value",
          value: `${share(raw)} of rows`,
          meaning:
            s >= 0.95
              ? "Almost every row is the same — there is little here for a model to use."
              : "The share held by whichever single value occurs most often.",
        });
        break;
      }

      case "median":
        push({
          key,
          label: "Middle value",
          value: number(raw),
          term: "median",
          meaning: "Half the rows sit above this, half below.",
        });
        break;

      case "mean":
        push({
          key,
          label: "Average",
          value: number(raw),
          term: "mean",
        });
        break;

      case "skew": {
        const s = Number(raw);
        const magnitude = Math.abs(s);
        push({
          key,
          label: "Lopsidedness",
          value: signed(raw),
          term: "skew",
          meaning:
            magnitude < 0.5
              ? "Close to symmetric."
              : magnitude < 1
                ? "Mildly tilted; most values sit near the middle."
                : s > 0
                  ? "A long tail of large values. Most rows are small, a few are very large."
                  : "A long tail of small values. Most rows are large, a few are very small.",
        });
        break;
      }
    }
  }

  // Anything the backend adds later still appears, rather than silently vanishing
  // because this file has not caught up with it.
  for (const [key, raw] of Object.entries(stats)) {
    if (ORDER.includes(key)) continue;
    push({ key, label: key.replace(/_/g, " "), value: String(raw) });
  }

  return views;
}
