/**
 * The compile, read as a sequence rather than as a log.
 *
 * `StepReport[]` is already the whole story — each entry carries the shape of the frame
 * at that moment and which columns it touched — but the panel rendered it as rows of
 * `data_types`, `feature_selection`, `missing_values` in monospace: the identifiers the
 * script uses, not the names the product uses. The reader had just toggled a switch
 * labelled "Missing Value Treatment" and was then shown `missing_values`, as if the two
 * were different things.
 *
 * So each moment gets the title the rest of the product already uses for that step. The
 * `action` sentence underneath is the script's own, passed through untouched: it was
 * written by the code as it ran, and rewriting it here is exactly the drift this product
 * exists to avoid.
 *
 * Two things this does that a log cannot
 * --------------------------------------
 * **The shape change is made visible.** Columns go 13 → 15 → 14 → 16 → 12 across a
 * compile, and every one of those jumps is a decision. Showing the delta at the moment
 * it happens turns a list into a story with a plot.
 *
 * **The split is not just another row.** It is the product's central claim, and in the
 * log it was one grey line between two others. It is marked here, because everything
 * after it learned from the training rows alone and that is the thing a reader is
 * supposed to come away knowing.
 *
 * A step id with no title still renders, under its raw name. A pipeline that grows a new
 * step should show up as an unfamiliar word, not vanish from the account of what ran.
 */

import type { StepReport } from "../api";

export type Moment = {
  /** Stable across renders; `step` repeats, so the index is part of it. */
  key: string;
  id: string;
  title: string;
  /** The script's own sentence. Never rewritten. */
  action: string;
  rows: number;
  columns: number;
  /** Change in column count against the previous moment, or null for the first. */
  columnsDelta: number | null;
  rowsDelta: number | null;
  touched: string[];
  /** The split gets its own treatment; everything else is a step. */
  kind: "step" | "split";
};

/**
 * The product's own name for each step.
 *
 * The eight pipeline steps use the labels from their switches, so the account of what
 * ran matches the controls that caused it. The other five have no switch — they are
 * things the pipeline always does — and take the names the generated script's own
 * section banners use.
 */
const TITLES: Record<string, string> = {
  load: "Reading the file",
  data_types: "Data Type Detection",
  excluded: "Columns you removed",
  target: "Rows without a target value",
  duplicates: "Duplicate Removal",
  split: "The split",
  datetime: "Calendar features",
  missing_values: "Missing Value Treatment",
  outliers: "Outlier Treatment",
  encoding: "Categorical Encoding",
  scaling: "Feature Scaling",
  feature_selection: "Feature Selection",
  imbalance: "Class Imbalance",
};

/**
 * Which moment is *the* split.
 *
 * The generator can record `split` twice — once to say a stratified split was not
 * possible, and once for the split itself. Only the second carries `detail.train`, so
 * that is the one that gets the rule drawn across the column; the other stays an
 * ordinary moment, which is right, because it is a note about the split rather than the
 * split happening.
 */
function isTheSplit(report: StepReport): boolean {
  return report.step === "split" && typeof report.detail?.train === "number";
}

export function narrate(steps: StepReport[] | undefined): Moment[] {
  if (!steps?.length) return [];

  return steps.map((report, index) => {
    const previous = index > 0 ? steps[index - 1] : null;
    return {
      key: `${report.step}-${index}`,
      id: report.step,
      title: TITLES[report.step] ?? report.step,
      action: report.action,
      rows: report.rows,
      columns: report.columns,
      columnsDelta: previous ? report.columns - previous.columns : null,
      rowsDelta: previous ? report.rows - previous.rows : null,
      touched: report.columns_affected ?? [],
      kind: isTheSplit(report) ? "split" : "step",
    };
  });
}
