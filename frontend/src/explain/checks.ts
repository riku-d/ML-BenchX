/**
 * What was checked after a plan was executed, in sentences.
 *
 * `RunChecks` is the most carefully-reasoned data this product produces and, until now,
 * the only data it never showed anyone: `api.listRuns` and `api.readRun` existed and
 * nothing called either. Three independent checks are run after the pipeline finishes,
 * and all three were reaching the client and being discarded.
 *
 * Each is translated here rather than printed, because each is easy to misread in a way
 * that matters:
 *
 * * **Drift** reports `flagged`, `tested` and `skipped`, and `tested + skipped` is every
 *   feature. A reader shown "0 flagged" without "of 13 tested, 10 not testable" cannot
 *   tell a clean result from a check that barely ran.
 * * **Balance** is about the split, not about the dataset. "Balanced" here means the
 *   training and held-out rows carry the *same mix* of outcomes — which is true even for
 *   a wildly imbalanced dataset, and says nothing about whether the classes are even.
 * * **Baseline** is the one most likely to be mistaken for a result. It is cross-
 *   validated on the training rows only, so it is a check on preparation and not a
 *   measure of how a model will perform. The server says so in `note`, and that note is
 *   carried through verbatim rather than summarised away.
 */

import type { Run } from "../api";

export type CheckCard = {
  key: string;
  title: string;
  /** The headline reading. */
  verdict: string;
  tone: "good" | "warn" | "info";
  /** Supporting lines, in reading order. */
  detail: string[];
  /** The server's own caveat, passed through unchanged where it has one. */
  note?: string;
};

const share = (value: number) => `${(value * 100).toFixed(1)}%`;

export function describeChecks(run: Run): CheckCard[] {
  const cards: CheckCard[] = [];
  const { drift, balance, baseline } = run.checks;

  if (drift) {
    const { flagged, tested, skipped, threshold } = drift;
    const untestable =
      // With nothing tested there is no "more" to be had; the skipped count is the
      // whole story rather than a footnote to a result.
      tested === 0
        ? `None of the ${skipped} features could be tested: a column that is text, or constant, or too sparse gives the comparison nothing to work with.`
        : skipped > 0
          ? `${skipped} more could not be tested: a column that is text, or constant, or too sparse gives the comparison nothing to work with.`
          : "";
    cards.push({
      key: "drift",
      title: "Do the two groups look alike?",
      // A check that could not run is not a clean bill of health, so it gets the
      // neutral mark rather than the green one.
      tone: flagged > 0 ? "warn" : tested === 0 ? "info" : "good",
      verdict:
        tested === 0
          ? "No feature could be compared, so this check says nothing either way."
          : flagged === 0
            ? `None of the ${tested} features tested is distributed differently between the training and held-out rows.`
            : `${flagged} of the ${tested} features tested ${flagged === 1 ? "is" : "are"} distributed differently between the training and held-out rows.`,
      detail: [
        untestable,
        `The bar is p < ${threshold}.`,
        ...drift.features.map((feature) => `${feature.message} ${feature.suggestion}`),
      ].filter(Boolean),
      note: drift.note,
    });
  }

  if (balance) {
    const worst = balance.classes.reduce(
      (most, klass) => (Math.abs(klass.delta) > Math.abs(most.delta) ? klass : most),
      balance.classes[0],
    );
    const even = balance.classes.every((klass) => Math.abs(klass.delta) < 0.02);
    cards.push({
      key: "balance",
      title: "Did the split keep the same mix of outcomes?",
      tone: even ? "good" : "warn",
      verdict: even
        ? `Yes — the largest difference between the training and held-out shares is ${share(Math.abs(worst.delta))}.`
        : `"${worst.value}" makes up ${share(worst.train_share)} of the training rows and ${share(worst.test_share)} of the held-out rows.`,
      detail: balance.classes.map(
        (klass) =>
          `"${klass.value}": ${klass.train_count.toLocaleString()} training (${share(klass.train_share)}), ${klass.test_count.toLocaleString()} held out (${share(klass.test_share)}).`,
      ),
      note: "This is about the split, not about the dataset. Classes can be badly uneven and still be divided evenly between the two groups.",
    });
  }

  if (baseline) {
    cards.push({
      key: "baseline",
      title: "Is there anything in these features to learn from?",
      tone: baseline.severity === "warn" ? "warn" : "info",
      verdict: baseline.verdict,
      detail: [
        `Always guessing the commonest answer scores ${baseline.baseline.toFixed(3)} ${baseline.metric}.`,
        `A ${baseline.model.replace(/_/g, " ")} on the prepared features scores ${baseline.model_score.toFixed(3)}.`,
        `Measured over ${baseline.folds} folds.`,
      ],
      note: baseline.note,
    });
  }

  return cards;
}

/** One line saying which execution these checks belong to. */
export function provenance(run: Run): string {
  const when = new Date(run.created_at).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `From the plan run on ${when} — version ${run.plan_version}, ${run.n_features} features, target ${run.target}.`;
}
