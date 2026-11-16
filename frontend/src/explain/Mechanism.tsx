/**
 * Layer 3 of the explanation model: showing the transformation, not only naming it.
 *
 * The product tells the reader "One column per value" and "Turn into a single 0/1
 * column". For someone learning, the sentence is the caption — seeing one column become
 * three is the explanation. This draws that, using values out of the reader's own file
 * rather than a worked example from a textbook.
 *
 * What this will not do
 * ---------------------
 * Every number here has to be one the pipeline would really produce, which rules out
 * more than it first appears:
 *
 * * **Imputation is shown as its own step, not as a final value.** Filling `Age` writes
 *   the median, 28.3 — and then scaling runs and the value in the feature matrix is
 *   -0.084. Showing "blank → 28.3" as the output would be wrong about the result, so
 *   the fill is labelled as the fill and the later rescale is stated when the plan has
 *   a scaler on.
 * * **Frequency and target encoding get no numbers at all.** Frequency encoding fits on
 *   the training fold, so the share in the profile — computed over every row — is close
 *   to it but is not it. Target encoding is cross-fitted and cannot be derived here at
 *   all. Both are described by their rule instead.
 * * **A truncated category list says so.** The profile returns at most five values, so
 *   the output columns are only named in full when `n_unique` confirms there are no
 *   more.
 *
 * What it can state exactly, because the encoders are deterministic and sorted:
 * one-hot and ordinal/binary mappings. Both were checked against a real run on
 * `titanic.csv` rather than taken from the scikit-learn defaults — `female` really does
 * become 0 and `male` 1, and `Embarked` really does become `Embarked_C`, `_Q`, `_S` in
 * that order.
 */

import type { ColumnPlan, ColumnProfile, PlanOptions } from "../api";

/** A row of the before/after table: one input value and what it becomes. */
type Mapping = { from: string; to: string[] };

type Shape = {
  /** The output column names, in the order the pipeline produces them. */
  outputs: string[];
  /** Example rows, drawn from the reader's own data. Empty when none can be shown. */
  mappings: Mapping[];
  /** Stated under the table. Carries anything the numbers cannot. */
  note?: string;
  /** True when the category list was cut short by the profile's five-value cap. */
  truncated?: boolean;
};

/** Round for reading. The evidence panel shows 29.2; this must not show 29.24682. */
function readable(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Number.isInteger(n)) return n.toLocaleString();
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toPrecision(3);
}

function sortedCategories(profile: ColumnProfile | undefined): {
  values: string[];
  complete: boolean;
} {
  const values = (profile?.top_values ?? []).map((v) => String(v.value));
  // sklearn sorts categories when it fits, so sorted order is the output order — but
  // `top_values` arrives sorted by frequency, so it has to be re-sorted here.
  const sorted = [...values].sort((a, b) => a.localeCompare(b));
  const complete = profile ? profile.n_unique <= values.length : false;
  return { values: sorted, complete };
}

function describe(
  column: ColumnPlan,
  profile: ColumnProfile | undefined,
  options: PlanOptions,
): Shape | null {
  const name = column.column;
  const scaled = options.scaler !== "none";
  const rescale = scaled
    ? " Every numeric column is rescaled afterwards, so the value in the final table is not this number."
    : "";

  switch (column.action) {
    case "encode_binary":
    case "encode_ordinal": {
      const { values, complete } = sortedCategories(profile);
      if (values.length === 0) return null;
      return {
        outputs: [name],
        mappings: values.map((v, i) => ({ from: v, to: [String(i)] })),
        truncated: !complete,
        note:
          column.action === "encode_ordinal"
            ? "The numbers follow alphabetical order, which is arbitrary — it tells a model these categories have an order and even spacing. That is the reason to prefer one column per value where you can afford it."
            : "Two values, so one column carries all of it. Which gets 0 and which gets 1 does not matter to a model.",
      };
    }

    case "encode_onehot": {
      const { values, complete } = sortedCategories(profile);
      if (values.length === 0) return null;
      const outputs = values.map((v) => `${name}_${v}`);
      return {
        outputs,
        mappings: values.map((v) => ({
          from: v,
          to: values.map((other) => (other === v ? "1" : "0")),
        })),
        truncated: !complete,
        note: `${name} becomes ${outputs.length} column${outputs.length === 1 ? "" : "s"}, exactly one of which is 1 for any row. No ordering is implied between the categories.`,
      };
    }

    case "encode_frequency": {
      const { values } = sortedCategories(profile);
      if (values.length === 0) return null;
      return {
        outputs: [name],
        mappings: [],
        note: "Each value is replaced by how often it occurs in the training rows. One column however many categories there are, and the target is never consulted, so it cannot leak the answer. The exact numbers come from fitting on the training fold and are not shown here, because the shares in the profile are computed over every row and would be slightly different.",
      };
    }

    case "encode_target":
      return {
        outputs: [name],
        mappings: [],
        note: "Each value is replaced by the average outcome recorded for it — cross-fitted, so no row contributes to its own encoded value. The numbers are produced during fitting and cannot be worked out in advance.",
      };

    case "datetime_expand": {
      const example = profile?.top_values?.[0]?.value;
      const parts = [
        "year",
        "month",
        "month_sin",
        "month_cos",
        "day",
        "day_sin",
        "day_cos",
        "dayofweek",
        "dayofweek_sin",
        "dayofweek_cos",
        "hour",
        "hour_sin",
        "hour_cos",
      ];
      return {
        outputs: parts.map((p) => `${name}_${p}`),
        mappings: example ? [{ from: String(example), to: ["→ 13 columns"] }] : [],
        note: "A timestamp is one large number whose size means nothing. The sin/cos pairs put the repeating parts on a circle, so December sits next to January rather than eleven months away from it.",
      };
    }

    case "impute_median":
    case "impute_mean":
    case "impute_mode":
    case "impute_constant": {
      const key = column.action === "impute_mean" ? "mean" : "median";
      const fill = column.stats?.[key];
      const blanks = column.stats?.["n_missing"];
      const word =
        column.action === "impute_mean"
          ? "the average"
          : column.action === "impute_median"
            ? "the middle value"
            : column.action === "impute_mode"
              ? "the most common value"
              : "a fixed value";
      if (fill === undefined && column.action !== "impute_mode") return null;

      // The two candidates, side by side, where both were measured.
      //
      // The phase 6 comprehension test found a reader could follow what happens to Age
      // and still had no basis to judge mean against median — the panel showed both
      // numbers and the column's lopsidedness and never connected them. This is the
      // connection, and it is arithmetic on measured values: the gap between the two
      // is the lopsidedness showing up. It projects nothing.
      const median = column.stats?.["median"];
      const mean = column.stats?.["mean"];
      const comparable =
        (column.action === "impute_mean" || column.action === "impute_median") &&
        typeof median === "number" &&
        typeof mean === "number";
      // The gap is taken from the *displayed* numbers, not the raw ones. At full
      // precision 29.24682 - 28.3 is 0.947, which a reader seeing "28.3" and "29.2"
      // beside it would rightly read as a mistake. Subtracting what is on screen keeps
      // the arithmetic checkable, which is the whole point of showing it.
      const shownMedian = Number(readable(median));
      const shownMean = Number(readable(mean));
      // Float subtraction of the rounded values, then trimmed: 29.2 - 28.3 is 0.9,
      // not 0.900 and not 0.8999999999999986.
      const gap = Math.abs(shownMean - shownMedian);
      const comparison = comparable
        ? ` The middle value is ${readable(median)} and the average is ${readable(mean)}` +
          (gap === 0
            ? ", which is the same number to this many digits — the column is close to symmetric."
            : `, ${String(Number(gap.toFixed(3)))} apart. That gap is the column's lopsidedness showing: the longer one tail is, the further the average sits from the middle.`)
        : "";

      return {
        outputs: [name],
        mappings:
          fill === undefined
            ? []
            : [{ from: "(blank)", to: [readable(fill)] }],
        note:
          `${blanks ? `${Number(blanks).toLocaleString()} blank row${Number(blanks) === 1 ? "" : "s"} are filled with ` : "Blanks are filled with "}` +
          `${word}${fill === undefined ? "" : ` of the training rows`}.` +
          ` The fill is worked out from the training rows alone.${rescale}${comparison}`,
      };
    }

    case "drop":
      return {
        outputs: [],
        mappings: [],
        note: "The column is not passed to the model at all. Nothing downstream can use it.",
      };

    case "passthrough":
      return null;

    case "keep":
      return scaled
        ? {
            outputs: [name],
            mappings: [],
            note: "Used as it is, then rescaled with the other numeric columns so no one column dominates on size alone.",
          }
        : null;

    default:
      return null;
  }
}

export function Mechanism({
  column,
  profile,
  options,
}: {
  column: ColumnPlan;
  profile: ColumnProfile | undefined;
  options: PlanOptions;
}) {
  const shape = describe(column, profile, options);
  if (!shape) return null;

  const { outputs, mappings, note, truncated } = shape;

  return (
    <div>
      <p className="mb-2 text-ui font-medium text-muted">What this does to the values</p>

      {mappings.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="pb-1 pr-4 text-left font-mono text-micro font-normal text-faint">
                  {column.column}
                </th>
                {/* When the column keeps its name — imputation, binary encoding — the
                    header would read "Age   Age". An arrow says the same thing without
                    the stutter; only a change in shape is worth naming. */}
                {outputs.length === 1 && outputs[0] === column.column ? (
                  <th className="pb-1 pl-3 text-right text-micro font-normal text-faint">
                    becomes
                  </th>
                ) : outputs.length > 0 && outputs.length <= 4 ? (
                  outputs.map((out) => (
                    <th
                      key={out}
                      className="pb-1 pl-3 text-right font-mono text-micro font-normal text-accent"
                    >
                      {out}
                    </th>
                  ))
                ) : outputs.length > 4 ? (
                  <th className="pb-1 pl-3 text-right text-micro font-normal text-accent">
                    {outputs.length} columns
                  </th>
                ) : null}
              </tr>
            </thead>
            <tbody>
              {mappings.map((row) => (
                <tr key={row.from} className="border-t border-line/60">
                  <td className="py-1 pr-4 font-mono text-meta text-text">
                    {row.from === "(blank)" ? (
                      <span className="text-warn">blank</span>
                    ) : (
                      `"${row.from}"`
                    )}
                  </td>
                  {row.to.map((value, index) => (
                    <td
                      key={`${row.from}-${index}`}
                      className={`tnum py-1 pl-3 text-right font-mono text-meta ${
                        value === "0" ? "text-faint" : "text-text"
                      }`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {truncated && (
        <p className="mt-1.5 text-micro text-warn">
          Showing the most common values only — this column has more than are listed here,
          and each one gets the same treatment.
        </p>
      )}

      {outputs.length > 4 && mappings.length > 0 && (
        <p className="mt-1.5 font-mono text-micro leading-relaxed text-faint">
          {outputs.slice(0, 6).join(", ")}
          {outputs.length > 6 ? `, and ${outputs.length - 6} more` : ""}
        </p>
      )}

      {note && <p className="mt-2 text-micro leading-relaxed text-faint">{note}</p>}
    </div>
  );
}
