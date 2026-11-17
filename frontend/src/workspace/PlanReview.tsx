/**
 * The plan review screen: see every decision, disagree with any of it, then run.
 *
 * This is the screen the product's claim rests on. "Every automated recommendation is
 * visible, explained, optional and reversible" is a statement about what happens here:
 * each column shows what will be done to it, the numbers that drove that, and a dropdown
 * offering exactly the alternatives that make sense for its kind.
 *
 * Three decisions worth stating, because the obvious alternatives are worse.
 *
 * **Every column is listed, including the ones nothing happens to.** A table of only the
 * interesting columns leaves a reader unable to tell "nothing needed doing" from "we
 * forgot about it".
 *
 * **Edits are collected, then sent once.** Each save writes a new immutable plan version
 * on the server, so sending one request per dropdown would turn five changes into five
 * versions and bury the one the user actually meant. The pending edits are held here and
 * applied together, which is also what makes "discard" a local operation.
 *
 * **The dropdown offers what the server sent.** `actions_for_kind` travels with the plan
 * rather than being duplicated in this file; a copy here would be the one that goes
 * stale, and the failure would be an action the backend rejects after the user chose it.
 */

import {
  Component,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  ApiError,
  type ColumnPlan,
  type ColumnProfile,
  type Explanation,
  type Histogram,
  type Leaderboard,
  type Plan,
  type PlanAction,
  type PlanOptions,
  type PlanWarning,
  type Recommendation,
  type Run,
} from "../api";
// Note: `Tooltip` in this file is recharts', not ours. The vocabulary tooltip arrives
// through `Term`, which wraps the shared one, so the two never have to share a name.
import { Code2 } from "lucide-react";
import { describeStats } from "../explain/stats";
import { Mechanism } from "../explain/Mechanism";
import { Term } from "../explain/Term";
import {
  Badge,
  Button,
  Dialog,
  EmptyState,
  Fact,
  PanelHeading,
  Select,
  Spinner,
} from "../components/ui";

/** Plain-English labels. The user never needs to read an action's internal name. */
const ACTION_LABELS: Record<PlanAction, string> = {
  keep: "Keep as it is",
  drop: "Remove the column",
  impute_median: "Fill gaps with the median",
  impute_mean: "Fill gaps with the mean",
  impute_mode: "Fill gaps with the most common value",
  impute_constant: "Fill gaps with a fixed value",
  encode_binary: "Turn into a single 0/1 column",
  encode_onehot: "One column per value",
  encode_ordinal: "One number per value",
  encode_target: "Average outcome per value",
  encode_frequency: "How often each value occurs",
  datetime_expand: "Split the date into parts",
  passthrough: "Leave untouched",
};

const KIND_LABELS: Record<string, string> = {
  numeric: "number",
  categorical: "category",
  boolean: "true/false",
  datetime: "date/time",
  text: "free text",
  constant: "single value",
};

const SEVERITY_TONE: Record<string, "neutral" | "warn" | "danger"> = {
  info: "neutral",
  warn: "warn",
  critical: "danger",
};

function pct(value: number): string {
  return `${(value * 100).toFixed(value < 0.01 && value > 0 ? 1 : 0)}%`;
}

/**
 * One warning, rendered by severity.
 *
 * The suggestion is shown, never hidden behind a disclosure: a warning that names a
 * problem and stops leaves the reader to work out what to do, which is the kind of
 * warning people learn to click past.
 */
function WarningCard({ warning }: { warning: PlanWarning }) {
  const tone = SEVERITY_TONE[warning.severity] ?? "neutral";
  const border =
    tone === "danger"
      ? "border-danger/40 bg-danger/5"
      : tone === "warn"
        ? "border-warn/40 bg-warn/5"
        : "border-line bg-raised";

  return (
    <div className={`rounded-md border px-3 py-2 ${border}`}>
      <div className="flex items-start gap-2">
        <Badge tone={tone}>{warning.severity}</Badge>
        <div className="min-w-0 flex-1">
          <p className="text-sm text-text">{warning.message}</p>
          {warning.suggestion ? (
            <p className="mt-1 text-xs text-muted">{warning.suggestion}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/**
 * The distribution of one column, drawn from the bins the backend computed.
 *
 * Server-side on purpose: the browser holds one page of rows and would draw a different
 * distribution on every page. A histogram of 25 rows is not a histogram of the column.
 */
function ColumnHistogram({ histogram }: { histogram: Histogram | null | undefined }) {
  const bars = useMemo(() => {
    if (!histogram?.counts?.length) return [];
    if (histogram.kind === "categorical") {
      return histogram.labels.map((label, index) => ({
        label,
        count: histogram.counts[index] ?? 0,
      }));
    }
    return histogram.counts.map((count, index) => ({
      label: histogram.edges[index]?.toPrecision(3) ?? "",
      count,
    }));
  }, [histogram]);

  // An empty or absent distribution is a real state, not a failure: a column that is
  // entirely blank, or one the profile could not summarise, has nothing to draw.
  if (!bars.length) {
    return (
      <p className="text-xs text-faint">
        No distribution to draw for this column.
      </p>
    );
  }

  return (
    <div className="h-32 w-full">
      <ResponsiveContainer width="100%" height="100%">
        {/* Colours are passed as props, not classes: recharts renders SVG and sets
            `fill` on the element directly, so a utility class is overwritten. The
            values come from the same design tokens the rest of the app uses. */}
        <BarChart data={bars} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <CartesianGrid
            strokeDasharray="2 4"
            stroke="var(--color-line)"
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9, fill: "var(--color-muted)" }}
            stroke="var(--color-line-strong)"
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 9, fill: "var(--color-muted)" }}
            stroke="var(--color-line-strong)"
            width={36}
          />
          <Tooltip
            contentStyle={{
              fontSize: 11,
              background: "var(--color-raised)",
              border: "1px solid var(--color-line-strong)",
              borderRadius: 4,
            }}
            cursor={{ fill: "var(--color-line)" }}
            labelFormatter={(label) => `from ${label}`}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]} fill="var(--color-accent)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** One row of the plan table. */
function ColumnRow({
  column,
  pending,
  actions,
  onChange,
  expanded,
  onToggle,
  profile,
  options: planOptions,
  scriptUrl,
}: {
  column: ColumnPlan;
  pending: PlanAction | undefined;
  actions: PlanAction[];
  onChange: (action: PlanAction) => void;
  expanded: boolean;
  onToggle: () => void;
  profile: ColumnProfile | undefined;
  /** Needed to say whether a scaler runs after this column's own step. */
  options: PlanOptions;
  /** The generated script from the latest run, when there has been one. */
  scriptUrl: string | null;
}) {
  const action = pending ?? column.action;
  const changed = pending !== undefined && pending !== column.action;
  const isTarget = column.alternatives.length === 0 && action === "passthrough";

  const options = useMemo(
    () =>
      // The current action is always offered even when it is not in the list for this
      // kind, so a plan can always be left exactly as it is.
      Array.from(new Set<PlanAction>([action, ...actions])).map((value) => ({
        value,
        label: ACTION_LABELS[value] ?? value,
      })),
    [action, actions],
  );

  return (
    <>
      <tr
        className={`border-t border-line align-top ${changed ? "bg-accent/5" : ""}`}
        data-testid={`column-${column.column}`}
      >
        <td className="py-2 pr-3">
          <button
            type="button"
            onClick={onToggle}
            className="text-left font-mono text-sm text-text hover:text-accent"
            aria-expanded={expanded}
          >
            {column.column}
          </button>
          {column.original_name && column.original_name !== column.column ? (
            <p className="font-mono text-[10px] text-faint">
              was “{column.original_name}”
            </p>
          ) : null}
        </td>

        <td className="py-2 pr-3 text-xs text-muted">
          {KIND_LABELS[column.kind] ?? column.kind}
        </td>

        <td className="py-2 pr-3 font-mono text-xs text-muted">
          {column.stats.n_missing ? `${pct(Number(column.stats.missing_rate))} blank` : "—"}
        </td>

        <td className="py-2 pr-3 font-mono text-xs text-muted">
          {column.stats.n_unique ?? "—"}
        </td>

        <td className="py-2 pr-3">
          {isTarget ? (
            <Badge tone="accent">the answer</Badge>
          ) : (
            <Select
              value={action}
              options={options}
              onChange={(next) => onChange(next)}
              label={`What happens to ${column.column}`}
            />
          )}
          <div className="mt-1 flex flex-wrap gap-1">
            {column.overridden && !changed ? <Badge tone="accent">your choice</Badge> : null}
            {changed ? <Badge tone="accent">unsaved</Badge> : null}
            {column.add_indicator ? <Badge>+ was-missing flag</Badge> : null}
            {column.rare_categories.length ? (
              <Badge>{column.rare_categories.length} rare → Other</Badge>
            ) : null}
          </div>
        </td>

        <td className="py-2 text-xs text-muted">
          {column.reason}
          {column.warning && column.warning.message.trim() !== column.reason.trim() ? (
            <span className="mt-1 block">
              <Badge tone={SEVERITY_TONE[column.warning.severity] ?? "neutral"}>
                {column.warning.message}
              </Badge>
            </span>
          ) : null}
          {column.warning?.suggestion &&
          column.warning.message.trim() === column.reason.trim() ? (
            <span className="mt-1 block text-[11px] text-faint">
              {column.warning.suggestion}
            </span>
          ) : null}

          {/* Overriding replaces the derived reason with "You chose …", which reads as
              an explanation and is not one. The phase 6 comprehension test caught this
              on Age: a reader could say what would happen and could not say whether it
              was a good idea. The honest answer is that nothing here measures a single
              column's choice, so this says so rather than implying it was weighed. */}
          {column.overridden ? (
            <span className="mt-1 block text-micro leading-relaxed text-faint">
              Nothing measures the effect of one column's treatment. The workspace
              measures whole steps being switched on or off; this choice is yours and is
              taken as given.
            </span>
          ) : null}
        </td>
      </tr>

      {expanded ? (
        <tr className="border-t border-line/50 bg-sunken">
          <td colSpan={6} className="px-3 py-3">
            {/* The four layers, in the order a reader needs them. The row above is
                layer 1, the glance. Here: what it was decided from, what it does to
                the values, the shape of the column, and a way through to the code. */}
            <div className="grid gap-5 lg:grid-cols-3">
              <div>
                <p className="mb-2 text-xs font-medium text-muted">
                  What this was measured from
                </p>
                {/* These were `Object.entries(column.stats)` with the underscores
                    swapped for spaces, which put "top share 0.577" and "rows measured
                    712" in front of the reader. `describeStats` gives each one a label
                    in words and, where the number means something for this particular
                    column, a sentence saying what. */}
                <dl className="space-y-2">
                  {describeStats(column, profile?.n_rows).map((stat) => (
                    <div key={stat.key}>
                      <div className="flex items-baseline justify-between gap-3">
                        <dt
                          className={`text-meta ${
                            stat.emphasis ? "font-medium text-text" : "text-muted"
                          }`}
                        >
                          {stat.term ? (
                            <Term name={stat.term}>{stat.label}</Term>
                          ) : (
                            stat.label
                          )}
                        </dt>
                        <dd
                          className={`tnum shrink-0 font-mono text-meta ${
                            stat.emphasis ? "text-accent" : "text-text"
                          }`}
                        >
                          {stat.value}
                        </dd>
                      </div>
                      {stat.meaning && (
                        <p className="mt-0.5 text-micro leading-relaxed text-faint">
                          {stat.meaning}
                        </p>
                      )}
                    </div>
                  ))}
                </dl>
              </div>
              <Mechanism column={column} profile={profile} options={planOptions} />

              <div>
                <p className="mb-2 text-ui font-medium text-muted">
                  How the values are spread
                </p>
                <ChartBoundary resetKey={column.column}>
                  <ColumnHistogram histogram={profile?.histogram} />
                </ChartBoundary>
                {scriptUrl && (
                  <a
                    href={scriptUrl}
                    className="mt-3 inline-flex items-center gap-1.5 text-micro text-muted
                      underline decoration-dotted underline-offset-2 transition-colors hover:text-accent"
                  >
                    <Code2 size={11} aria-hidden />
                    See the code that does this
                  </a>
                )}
              </div>
            </div>
          </td>
        </tr>
      ) : null}
    </>
  );
}

/** Everything the run measured about the split itself. */
/**
 * Keeps a failing chart from taking the page down with it.
 *
 * A chart is decoration around numbers that are already on screen in a table. When the
 * charting library threw during render it unmounted the entire React tree and left a
 * black page — the leaderboard, the run report and the download links all disappeared
 * because a bar chart could not draw. Losing the drawing is acceptable; losing the
 * results is not.
 *
 * A class component because that is the only thing React gives for catching a render
 * error. It deliberately does not retry: if the library cannot load, re-rendering it
 * produces the same throw.
 */
class ChartBoundary extends Component<
  {
    children: ReactNode;
    /**
     * Changing this clears a previous failure.
     *
     * Without it the boundary latched: one transient throw set `failed` and nothing
     * ever unset it, so the fallback outlived the error. Retraining the same dataset
     * re-renders this boundary rather than remounting it, which meant a chart that
     * failed once stayed "could not be drawn" for fresh, perfectly good data until the
     * page was reloaded. Keyed on the data, a new explanation gets a new attempt.
     */
    resetKey?: string | number;
    /** What to show instead of the chart. Defaults to a sentence. */
    fallback?: ReactNode;
  },
  { failed: boolean; key: string | number | undefined; reason: string }
> {
  state = { failed: false, key: this.props.resetKey, reason: "" };

  static getDerivedStateFromError(error: unknown) {
    // The message is kept and shown. A boundary that swallows the reason turns one
    // debuggable exception into an unexplained blank, and the person who can see the
    // failure is rarely the person who can read the console.
    const reason =
      error instanceof Error
        ? `${error.name}: ${error.message}`
        : String(error ?? "unknown error");
    return { failed: true, reason };
  }

  static getDerivedStateFromProps(
    props: { resetKey?: string | number },
    state: { failed: boolean; key: string | number | undefined },
  ) {
    // New data is a new attempt. Compared here rather than in an effect because a class
    // boundary must decide before it renders the fallback again.
    if (props.resetKey !== state.key) {
      return { failed: false, key: props.resetKey, reason: "" };
    }
    return null;
  }

  componentDidCatch(error: unknown, info: { componentStack?: string | null }) {
    // Kept visible in the console: the numbers are still on screen, so this would
    // otherwise fail silently and nobody would know the chart was missing.
    console.error("Chart failed to render; the figures above are unaffected.", error, info);
  }

  render() {
    if (this.state.failed) {
      return (
        <>
          {this.state.reason ? (
            <p className="mt-1 font-mono text-[10px] text-warn">
              Chart error: {this.state.reason}
            </p>
          ) : null}
          {this.props.fallback ?? (
            <p className="text-xs text-faint">
              The chart could not be drawn. The figures it would have shown are in the
              table above.
            </p>
          )}
        </>
      );
    }
    return this.props.children;
  }
}

/**
 * The same numbers the chart would have drawn, as a table.
 *
 * Shown when the chart cannot render. These are the values the explainer actually
 * produced — the identical array the chart is fed — so a failure to draw costs the
 * picture and not the result.
 */
function ImportanceTable({
  features,
  unit,
}: {
  features: { feature: string; value: number }[];
  unit: string;
}) {
  if (!features.length) {
    return <p className="text-xs text-faint">No importances were returned.</p>;
  }
  const largest = Math.max(...features.map((f) => Math.abs(f.value)), 0);
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-faint">
          <tr className="border-b border-line">
            <th className="py-1 pr-2 font-normal">#</th>
            <th className="py-1 pr-2 font-normal">Feature</th>
            <th className="py-1 pr-2 text-right font-normal">{unit}</th>
            <th className="py-1 font-normal">share of the largest</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {features.map((f, index) => (
            <tr key={f.feature} className="border-b border-line/50">
              <td className="py-1 pr-2 text-faint">{index + 1}</td>
              <td className="py-1 pr-2 font-sans">{f.feature}</td>
              <td className="py-1 pr-2 text-right">
                {Math.abs(f.value) >= 1000
                  ? f.value.toFixed(1)
                  : f.value.toPrecision(4)}
              </td>
              <td className="py-1 text-faint">
                {largest > 0 ? `${((Math.abs(f.value) / largest) * 100).toFixed(0)}%` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[10px] text-faint">
        Shown as a table because the chart could not be drawn. These are the explainer's
        own values, unchanged.
      </p>
    </div>
  );
}

/** A metric's value, formatted, or a dash where there is genuinely no number. */
function metric(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toFixed(digits);
}

/**
 * The horizontal importance chart.
 *
 * Separate from `ColumnHistogram` because the axes are transposed — feature names are
 * long and belong on the vertical axis, where they have room to be read.
 */
function ImportanceChart({
  features,
}: {
  features: { feature: string; value: number }[];
}) {
  const bars = useMemo(
    () => features.slice(0, 12).map((f) => ({ label: f.feature, value: f.value })),
    [features],
  );
  if (!bars.length) {
    return <p className="text-xs text-faint">No importances to draw.</p>;
  }
  return (
    <div className="w-full" style={{ height: Math.max(140, bars.length * 22) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={bars}
          layout="vertical"
          margin={{ top: 4, right: 12, bottom: 0, left: 8 }}
        >
          <CartesianGrid strokeDasharray="2 4" stroke="var(--color-line)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 9, fill: "var(--color-muted)" }}
            stroke="var(--color-line-strong)"
          />
          <YAxis
            type="category"
            dataKey="label"
            width={150}
            tick={{ fontSize: 9, fill: "var(--color-muted)" }}
            stroke="var(--color-line-strong)"
            // Every bar gets its name. The default thins category ticks when it judges
            // them crowded, which on this chart silently left bars unlabelled — and an
            // unlabelled importance bar says nothing at all.
            interval={0}
          />
          <Tooltip
            contentStyle={{
              fontSize: 11,
              background: "var(--color-raised)",
              border: "1px solid var(--color-line-strong)",
              borderRadius: 4,
            }}
            cursor={{ fill: "var(--color-line)" }}
          />
          {/* Animation off. The entry transition starts every bar at zero width, so any
              renderer that does not run the full 1.5s of frames — a headless capture, a
              screenshot, a tab that never gets painted — shows labelled axes with no
              bars at all, which reads as a broken chart rather than a slow one. These
              are static importances; there is nothing for the motion to convey. */}
          <Bar
            dataKey="value"
            radius={[0, 2, 2, 0]}
            fill="var(--color-accent)"
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Stages 5, 6 and 7 on one panel: train six families, rank them, explain the winner,
 * and show what the experiment history would have recommended.
 *
 * Three deliberate choices about honesty, each mirroring a backend guarantee:
 *
 * **A family that did not score shows its reason, never a number.** The row is rendered
 * from `status`, so there is no code path that can print a zero for a model that was
 * never fitted.
 *
 * **The explanation's heading comes from `is_shap`**, not from a guess about the family.
 * Permutation importance is labelled as permutation importance, in the heading and in
 * the sentence beneath it.
 *
 * **A recommender that declines is reported as incomplete.** `declined` renders as a
 * warning, not as an empty ranking that would look like "no algorithm suits your data".
 */
function ModelsPanel({
  run,
  onRunPlan,
}: {
  /** `null` until the plan has been executed. The panel is shown either way. */
  run: Run | null;
  /** Executes the current plan and resolves to the run it produced. */
  onRunPlan: () => Promise<Run | null>;
}) {
  const [board, setBoard] = useState<Leaderboard | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  /**
   * Bumped every time an explanation is fetched, and used to reset the chart's error
   * boundary. Keyed on a counter rather than on the values themselves because the
   * values are deterministic: retraining the same dataset yields an identical payload,
   * so a value-derived key would never change and a chart that failed once could never
   * recover.
   */
  const [explainEpoch, setExplainEpoch] = useState(0);
  const [advice, setAdvice] = useState<Recommendation | null>(null);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const runId = run?.status === "succeeded" ? run.id : null;

  // Results already stored are loaded rather than recomputed. Training is the expensive
  // step; a reload should not pay for it twice.
  const loadStored = useCallback(async () => {
    if (!runId) return;
    const [models, explain] = await Promise.all([
      api.readModels(runId).catch(() => null),
      api.readExplanation(runId).catch(() => null),
    ]);
    if (models) setBoard(models);
    if (explain) {
      setExplanation(explain);
      setExplainEpoch((n) => n + 1);
    }
  }, [runId]);

  useEffect(() => {
    void loadStored();
  }, [loadStored]);

  // The recommendation is independent of training — it describes the dataset, not the
  // models — so it is fetched as soon as there is a run to describe.
  useEffect(() => {
    if (!runId) return;
    let live = true;
    api
      .recommendation(runId)
      .then((value) => live && setAdvice(value))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [runId]);

  /**
   * Train, executing the plan first if that has not happened yet.
   *
   * Training needs a run, because it reads the run's stored train/test folds. Requiring
   * the reader to find and press "Run this plan" first made the whole stage invisible on
   * a freshly uploaded dataset — the button did not exist until a run did. One press now
   * does both, and says which part it is on.
   */
  const train = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      let target = run;
      if (!target || target.status !== "succeeded") {
        setStage("Preparing the data…");
        target = await onRunPlan();
      }
      if (!target || target.status !== "succeeded") {
        setError(
          "The plan could not be executed, so there is nothing to train on. The run's own error is shown above.",
        );
        return;
      }
      setStage("Fitting six models…");
      setBoard(await api.trainRun(target.id));
      setExplanation(await api.readExplanation(target.id).catch(() => null));
      setExplainEpoch((n) => n + 1);
      api
        .recommendation(target.id)
        .then(setAdvice)
        .catch(() => undefined);
    } catch (exc) {
      setError(
        exc instanceof ApiError ? exc.message : "Training failed before it could report.",
      );
    } finally {
      setBusy(false);
      setStage("");
    }
  }, [run, onRunPlan]);

  const primary = board?.primary_metric ?? "";
  const baselineValue =
    typeof board?.baseline?.metrics?.[primary] === "number"
      ? (board.baseline.metrics[primary] as number)
      : null;

  return (
    <div className="space-y-5">
      {/* ---- Stage 5: train ------------------------------------------------------ */}
      <div>
        <PanelHeading>Six candidate models</PanelHeading>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <Button onClick={train} disabled={busy}>
            {busy ? "Working…" : board ? "Train again" : "Train six models"}
          </Button>
          {busy ? (
            <span className="flex items-center gap-2 text-xs text-muted">
              <Spinner /> {stage}
              {run
                ? ` Six families on ${run.split.train_rows.toLocaleString()} training rows, three at a time.`
                : " The plan runs first, then six families train on its training rows, three at a time."}
            </span>
          ) : null}
          {error ? <span className="text-xs text-warn">{error}</span> : null}
        </div>
        {!run && !busy ? (
          <p className="mt-2 text-xs text-faint">
            This plan has not been executed yet. Training will run it first — or press
            “Run this plan” above to review the split and the checks before training.
          </p>
        ) : null}

        {board ? (
          <>
            <p className="mt-3 text-xs text-muted">
              Fitted on {board.split.n_train_rows.toLocaleString()} training rows and
              scored on {board.split.n_test_rows.toLocaleString()}{" "}
              <strong>held-out</strong> rows — the fold no transformer was fitted on.
              Ranked by {primary}.
            </p>

            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-faint">
                  <tr className="border-b border-line">
                    <th className="py-1 pr-2 font-normal">#</th>
                    <th className="py-1 pr-2 font-normal">Model</th>
                    <th className="py-1 pr-2 text-right font-normal">{primary}</th>
                    <th className="py-1 pr-2 text-right font-normal">
                      {board.task.includes("class") ? "accuracy" : "mae"}
                    </th>
                    <th className="py-1 pr-2 text-right font-normal">
                      {board.task.includes("class") ? "roc_auc" : "rmse"}
                    </th>
                    <th className="py-1 pr-2 text-right font-normal">rows used</th>
                    <th className="py-1 pr-2 text-right font-normal">ms</th>
                    <th className="py-1 font-normal">vs. baseline</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {board.models.map((row) => {
                    const classification = board.task.includes("class");
                    const second = classification ? "accuracy" : "mae";
                    const third = classification ? "roc_auc" : "rmse";
                    if (row.status !== "succeeded") {
                      return (
                        <tr key={row.family} className="border-b border-line/50 align-top">
                          <td className="py-1 pr-2 text-faint">—</td>
                          <td className="py-1 pr-2 font-sans text-muted">{row.label}</td>
                          <td className="py-1 pr-2 font-sans text-warn" colSpan={6}>
                            <Badge tone="warn">{row.status}</Badge>{" "}
                            <span className="text-xs text-muted">{row.reason}</span>
                          </td>
                        </tr>
                      );
                    }
                    const best = row.rank === 1;
                    return (
                      <tr
                        key={row.family}
                        className={`border-b border-line/50 ${best ? "bg-raised" : ""}`}
                      >
                        <td className="py-1 pr-2 text-faint">{row.rank}</td>
                        <td className="py-1 pr-2 font-sans">
                          {row.label}
                          {best ? <span className="ml-2 text-accent">★ best</span> : null}
                          {row.sampled ? (
                            <span
                              className="ml-2 text-[10px] text-warn"
                              title={row.sampling_note}
                            >
                              sampled
                            </span>
                          ) : null}
                        </td>
                        <td className="py-1 pr-2 text-right">
                          {metric(row.primary_metric_value)}
                        </td>
                        <td className="py-1 pr-2 text-right text-muted">
                          {metric(row.metrics?.[second] as number | null)}
                        </td>
                        <td className="py-1 pr-2 text-right text-muted">
                          {metric(row.metrics?.[third] as number | null)}
                        </td>
                        <td className="py-1 pr-2 text-right text-muted">
                          {row.n_train_rows_used?.toLocaleString()}
                        </td>
                        <td className="py-1 pr-2 text-right text-faint">{row.train_ms}</td>
                        <td className="py-1 font-sans">
                          {row.beats_baseline === undefined ? (
                            <span className="text-faint">—</span>
                          ) : row.beats_baseline ? (
                            <span className="text-good">
                              +{metric(row.margin_over_baseline, 3)}
                            </span>
                          ) : (
                            <span className="text-warn">
                              no better ({metric(row.margin_over_baseline, 3)})
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              <div
                title={
                  board.baseline.note ??
                  `${board.baseline.estimator ?? "Dummy"} on the same held-out rows. A model that cannot beat this has learned nothing.`
                }
              >
                <Fact
                  label={
                    board.baseline.strategy
                      ? `Trivial baseline (${board.baseline.strategy})`
                      : "Trivial baseline"
                  }
                  value={baselineValue === null ? "—" : metric(baselineValue)}
                  mono
                />
              </div>
              <div title={board.concurrency.note}>
                <Fact
                  label="Concurrency observed"
                  value={`${board.concurrency.max_observed_overlap} of ${board.concurrency.workers} workers`}
                  mono
                />
              </div>
              <div
                title={`${(board.concurrency.summed_model_ms / 1000).toFixed(2)}s of model time ran in ${(board.concurrency.wall_clock_ms / 1000).toFixed(2)}s.`}
              >
                <Fact
                  label="Wall clock"
                  value={`${(board.concurrency.wall_clock_ms / 1000).toFixed(2)}s · ${board.concurrency.speedup_vs_serial}× vs serial`}
                  mono
                />
              </div>
            </div>

            {board.notes.map((note) => (
              <p key={note} className="mt-2 text-xs text-warn">
                {note}
              </p>
            ))}
          </>
        ) : busy ? null : (
          <p className="mt-2 text-xs text-faint">
            Nothing has been trained on this run yet.
          </p>
        )}
      </div>

      {/* ---- Stage 6: explain the winner ----------------------------------------- */}
      {explanation && explanation.status === "succeeded" ? (
        <div>
          <PanelHeading>
            {explanation.is_shap ? "SHAP explanation" : "Feature importance"}
          </PanelHeading>
          <p className="mt-1 text-xs text-muted">
            <Badge tone={explanation.is_shap ? "good" : "warn"}>
              {explanation.method_label}
            </Badge>{" "}
            for {explanation.label}. {explanation.sampling_note}
          </p>
          {explanation.fallback_reason ? (
            <p className="mt-1 text-xs text-warn">{explanation.fallback_reason}</p>
          ) : null}
          <div className="mt-2">
            <ChartBoundary
              resetKey={explainEpoch}
              fallback={
                <ImportanceTable
                  features={explanation.features ?? []}
                  unit={explanation.is_shap ? "mean |SHAP|" : "importance"}
                />
              }
            >
              <ImportanceChart features={explanation.features ?? []} />
            </ChartBoundary>
          </div>
          <p className="mt-1 text-[10px] text-faint">
            Mean absolute contribution per feature, across the sampled held-out rows.
            Global importance only — no per-prediction breakdown is claimed.
          </p>
        </div>
      ) : explanation && explanation.status !== "succeeded" ? (
        <div>
          <PanelHeading>Feature importance</PanelHeading>
          <p className="mt-1 text-xs text-warn">{explanation.reason}</p>
        </div>
      ) : null}

      {/* ---- Stage 7: what the history recommends -------------------------------- */}
      {advice ? (
        <div>
          <PanelHeading>What similar datasets favoured</PanelHeading>
          {advice.declined ? (
            <div className="mt-2 rounded border border-warn/40 bg-warn/5 p-3">
              <Badge tone="warn">incomplete</Badge>
              <p className="mt-2 text-xs text-muted">
                The recommender declined to rank:{" "}
                {typeof advice.declined === "string"
                  ? advice.declined
                  : advice.declined.reason}
              </p>
              <p className="mt-1 text-xs text-faint">
                This is a refusal, not a result — {advice.corpus.considered} comparable
                experiments were available of {advice.corpus.stored} stored. Seed more of
                the history and ask again.
              </p>
            </div>
          ) : (
            <>
              <ul className="mt-2 space-y-1">
                {advice.rationale.map((line) => (
                  <li key={line} className="text-xs text-muted">
                    • {line}
                  </li>
                ))}
              </ul>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-faint">
                    <tr className="border-b border-line">
                      <th className="py-1 pr-2 font-normal">Recommended</th>
                      <th className="py-1 pr-2 text-right font-normal">score</th>
                      <th className="py-1 pr-2 text-right font-normal">spread</th>
                      <th className="py-1 font-normal">actually best here</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {advice.ranking.map((row, index) => (
                      <tr key={row.family} className="border-b border-line/50">
                        <td className="py-1 pr-2 font-sans">
                          {index + 1}. {row.label}
                        </td>
                        <td className="py-1 pr-2 text-right">{metric(row.score, 3)}</td>
                        <td className="py-1 pr-2 text-right text-faint">
                          {metric(row.spread, 3)}
                        </td>
                        <td className="py-1 font-sans">
                          {board?.best_family === row.family ? (
                            <span className="text-good">✓ it won</span>
                          ) : (
                            <span className="text-faint">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-[10px] text-faint">
                Drawn from {advice.corpus.considered} comparable past experiments of{" "}
                {advice.corpus.stored} stored. These are scores other datasets achieved,
                not a forecast for this one.
              </p>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

function RunReport({ run }: { run: Run }) {
  if (run.status === "failed") {
    return (
      <div className="rounded-md border border-danger/40 bg-danger/5 p-4">
        <PanelHeading>The run did not finish</PanelHeading>
        <p className="mt-2 text-sm text-text">{run.error?.message}</p>
        <p className="mt-1 text-xs text-muted">
          Nothing was produced, so there is nothing to download. The plan is unchanged —
          adjust a column and run it again.
        </p>
      </div>
    );
  }

  const { drift, balance, baseline } = run.checks;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Fact label="training rows" value={run.split.train_rows.toLocaleString()} />
        <Fact label="held out" value={run.split.test_rows.toLocaleString()} />
        <Fact label="features" value={String(run.n_features)} />
        <Fact
          label="stratified"
          value={run.split.stratified ? "yes" : "no"}
        />
      </div>

      {run.warnings.length ? (
        <div className="space-y-2">
          <PanelHeading>What the run found</PanelHeading>
          {run.warnings.map((warning, index) => (
            <WarningCard key={`${warning.kind}-${index}`} warning={warning} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted">
          The run raised nothing worth flagging: the folds resemble each other and the
          target is learnable from the prepared columns.
        </p>
      )}

      {baseline ? (
        <div className="rounded-md border border-line bg-raised p-3">
          <PanelHeading>Sanity baseline</PanelHeading>
          <p className="mt-1 text-sm text-text">{baseline.verdict}</p>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <Fact label="guessing" value={baseline.baseline.toFixed(3)} />
            <Fact label={baseline.model} value={baseline.model_score.toFixed(3)} />
            <Fact label={`lift (${baseline.metric})`} value={baseline.lift.toFixed(3)} />
          </div>
          <p className="mt-2 text-xs text-faint">{baseline.note}</p>
        </div>
      ) : null}

      {drift ? (
        <div className="rounded-md border border-line bg-raised p-3">
          <PanelHeading>Train/test drift</PanelHeading>
          <p className="mt-1 text-sm text-text">
            {drift.flagged} of {drift.tested} features are distributed differently across
            the two folds.
          </p>
          <p className="mt-1 text-xs text-faint">{drift.note}</p>
          {drift.features.length ? (
            <ul className="mt-2 space-y-1">
              {drift.features.slice(0, 5).map((feature) => (
                <li key={feature.column} className="font-mono text-xs text-muted">
                  {feature.column} — p={feature.p_value}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {balance?.classes?.length ? (
        <div className="rounded-md border border-line bg-raised p-3">
          <PanelHeading>Class balance in both folds</PanelHeading>
          <table className="mt-2 w-full text-xs">
            <thead className="text-faint">
              <tr>
                <th className="text-left font-normal">class</th>
                <th className="text-right font-normal">train</th>
                <th className="text-right font-normal">test</th>
                <th className="text-right font-normal">difference</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {balance.classes.map((row) => (
                <tr key={row.value}>
                  <td className="py-0.5 text-text">{row.value}</td>
                  <td className="py-0.5 text-right text-muted">{pct(row.train_share)}</td>
                  <td className="py-0.5 text-right text-muted">{pct(row.test_share)}</td>
                  <td
                    className={`py-0.5 text-right ${
                      Math.abs(row.delta) > 0.05 ? "text-warn" : "text-faint"
                    }`}
                  >
                    {row.delta > 0 ? "+" : ""}
                    {pct(row.delta)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div>
        <PanelHeading>What this run produced</PanelHeading>
        <div className="mt-2 flex flex-wrap gap-2">
          {run.artifacts.map((name) => (
            <a
              key={name}
              href={api.artifactUrl(run.id, name)}
              className="rounded border border-line bg-raised px-2 py-1 font-mono text-xs text-muted hover:border-accent hover:text-accent"
            >
              {name}
            </a>
          ))}
          {/* The archive is listed last and styled apart: it is the same bytes as the
              links beside it, collected, and a reader should not wonder whether it is
              an eighth artifact. */}
          <a
            href={api.bundleUrl(run.id)}
            className="rounded border border-accent/50 bg-raised px-2 py-1 font-mono text-xs text-accent hover:border-accent"
          >
            ⤓ all of it (.zip)
          </a>
        </div>
        {run.output_hash ? (
          <p className="mt-2 font-mono text-[10px] text-faint">
            output hash {run.output_hash.slice(0, 32)}… — the same plan on the same data
            produces this digest every time.
          </p>
        ) : null}
      </div>
    </div>
  );
}

export function PlanReview({
  datasetId,
  onExit,
}: {
  datasetId: string;
  onExit: () => void;
}) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [profiles, setProfiles] = useState<Record<string, ColumnProfile>>({});
  const [run, setRun] = useState<Run | null>(null);
  const [edits, setEdits] = useState<Record<string, PlanAction>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [busy, setBusy] = useState<"loading" | "saving" | "running" | null>("loading");
  const [error, setError] = useState<string | null>(null);
  const [confirmRun, setConfirmRun] = useState(false);

  // Only one load may be in flight. Without this, React's development double-invoke of
  // effects fires two loads at once; both find no plan, both propose one, and the user
  // arrives on version 2 of a plan they have never edited. A flag set inside the
  // effect would not help — the guard has to survive the re-invocation, which a ref
  // does and state does not.
  const loading = useRef(false);

  const load = useCallback(async () => {
    if (loading.current) return;
    loading.current = true;
    setBusy("loading");
    setError(null);
    try {
      let current: Plan;
      try {
        current = await api.readPlan(datasetId);
      } catch (exc) {
        // No plan has been proposed yet, which is the normal first visit.
        if (exc instanceof ApiError && exc.status === 404) {
          current = await api.proposePlan(datasetId, "");
        } else {
          throw exc;
        }
      }
      setPlan(current);
      setEdits({});

      // The distributions are the profile's, not the plan's. They are fetched
      // separately and joined by column name, so a profile that fails to load costs the
      // charts and nothing else — the plan is still reviewable without them.
      try {
        const profile = await api.readProfile(datasetId);
        setProfiles(
          Object.fromEntries(profile.columns.map((column) => [column.column, column])),
        );
      } catch {
        setProfiles({});
      }

      // Restore the most recent successful run of *this* plan version, so a reload
      // shows the results that already exist instead of an empty screen with a Run
      // button. Scoped to the version on purpose: a run of version 1 does not describe
      // the version 2 the reader is now looking at, and showing it under these columns
      // would attribute one plan's results to another.
      //
      // A failure here is swallowed. The plan is the screen; past runs are a
      // convenience, and losing them must not cost the review.
      try {
        const history = await api.listRuns(datasetId);
        const previous = history.runs.find(
          (one) => one.status === "succeeded" && one.plan_version === current.version,
        );
        if (previous) setRun(await api.readRun(previous.id));
      } catch {
        // No history, or it could not be read. The Run button still works.
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "The plan could not be loaded.");
    } finally {
      loading.current = false;
      setBusy(null);
    }
  }, [datasetId]);

  useEffect(() => {
    void load();
  }, [load]);

  const pendingCount = useMemo(
    () =>
      plan
        ? Object.entries(edits).filter(
            ([name, action]) => plan.columns.find((c) => c.column === name)?.action !== action,
          ).length
        : 0,
    [edits, plan],
  );

  const save = useCallback(async () => {
    if (!plan || !pendingCount) return;
    setBusy("saving");
    setError(null);
    try {
      const columns = Object.fromEntries(
        Object.entries(edits).map(([name, action]) => [name, { action }]),
      );
      setPlan(await api.editPlan(plan.id, columns));
      setEdits({});
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Those changes were refused.");
    } finally {
      setBusy(null);
    }
  }, [edits, pendingCount, plan]);

  const execute = useCallback(async (): Promise<Run | null> => {
    if (!plan) return null;
    setConfirmRun(false);
    setBusy("running");
    setError(null);
    try {
      const next = await api.runPlan(plan.id);
      setRun(next);
      return next;
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "The run could not be started.");
      return null;
    } finally {
      setBusy(null);
    }
  }, [plan]);

  if (busy === "loading") {
    return (
      <div className="flex h-screen items-center justify-center gap-2 text-muted">
        <Spinner /> Working out what to do with every column…
      </div>
    );
  }

  if (error && !plan) {
    return (
      <div className="flex h-screen items-center justify-center">
        <EmptyState
          title="The plan could not be loaded"
          description={error}
          action={<Button onClick={() => void load()}>Try again</Button>}
        />
      </div>
    );
  }

  if (!plan) return null;

  const kept = plan.columns.filter(
    (c) => c.action !== "drop" && c.action !== "passthrough",
  ).length;
  const dropped = plan.columns.filter((c) => c.action === "drop").length;

  return (
    <div className="min-h-screen bg-void text-text">
      <header className="sticky top-0 z-10 border-b border-line bg-surface/95 px-6 py-3 backdropliter">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-medium">
              Review the plan{" "}
              <span className="font-mono text-xs text-faint">v{plan.version}</span>
            </h1>
            <p className="text-xs text-muted">
              Predicting <span className="font-mono text-text">{plan.target}</span> ·{" "}
              {plan.task} · {kept} column{kept === 1 ? "" : "s"} kept, {dropped} removed
              {plan.n_overridden ? ` · ${plan.n_overridden} your choice` : ""}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" onClick={onExit}>
              Back
            </Button>
            <Button
              variant="ghost"
              onClick={() => setEdits({})}
              disabled={!pendingCount || busy !== null}
            >
              Discard changes
            </Button>
            <Button onClick={() => void save()} disabled={!pendingCount || busy !== null}>
              {busy === "saving" ? <Spinner /> : null}
              Save as v{plan.version + 1}
              {pendingCount ? ` (${pendingCount})` : ""}
            </Button>
            <Button
              variant="primary"
              onClick={() => setConfirmRun(true)}
              disabled={busy !== null}
            >
              {busy === "running" ? <Spinner /> : null}
              Run this plan
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] space-y-6 px-6 py-6">
        {error ? (
          <WarningCard warning={{ severity: "critical", message: error }} />
        ) : null}

        {pendingCount ? (
          <WarningCard
            warning={{
              severity: "info",
              message: `${pendingCount} unsaved change${pendingCount === 1 ? "" : "s"}.`,
              suggestion:
                "Saving writes a new version of the plan. The version you are looking " +
                "at stays exactly as it is, so any run that used it still means what it said.",
            }}
          />
        ) : null}

        {plan.warnings.map((warning, index) => (
          <WarningCard key={`plan-${index}`} warning={warning} />
        ))}

        <section>
          <PanelHeading aside={
            <span className="text-[11px] normal-case tracking-normal text-faint">
              Including the ones nothing happens to
            </span>
          }>
            Every column, and what happens to it
          </PanelHeading>
          <div className="mt-3 overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[900px] text-left">
              <thead className="bg-raised text-[11px] uppercase tracking-wide text-faint">
                <tr>
                  <th className="px-3 py-2 font-normal">column</th>
                  <th className="px-3 py-2 font-normal">holds</th>
                  <th className="px-3 py-2 font-normal">blank</th>
                  <th className="px-3 py-2 font-normal">distinct</th>
                  <th className="px-3 py-2 font-normal">what happens</th>
                  <th className="px-3 py-2 font-normal">why</th>
                </tr>
              </thead>
              <tbody>
                {plan.columns.map((column) => (
                  <ColumnRow
                    key={column.column}
                    column={column}
                    pending={edits[column.column]}
                    actions={plan.actions_for_kind[column.kind] ?? []}
                    onChange={(action) =>
                      setEdits((previous) => ({ ...previous, [column.column]: action }))
                    }
                    profile={profiles[column.column]}
                    options={plan.options}
                    scriptUrl={
                      // Layer 4 only exists once there is a run to read it from. No
                      // link rather than a dead one before the plan has been executed.
                      run?.status === "succeeded" && run.artifacts.includes("script")
                        ? api.artifactUrl(run.id, "script")
                        : null
                    }
                    expanded={expanded === column.column}
                    onToggle={() =>
                      setExpanded((previous) =>
                        previous === column.column ? null : column.column,
                      )
                    }
                  />
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {run ? (
          <section>
            <PanelHeading aside={
              <span className="text-[11px] normal-case tracking-normal text-faint">
                {run.duration_ms} ms · seed {run.split.seed} · {pct(run.split.test_size)} held out
              </span>
            }>
              Run of plan v{run.plan_version}
            </PanelHeading>
            <div className="mt-3">
              <RunReport run={run} />
            </div>
          </section>
        ) : null}

        {/* Stages 5-7. Rendered whether or not a run exists yet: training needs a run,
            but requiring the reader to produce one first made the entire stage
            invisible on a freshly uploaded dataset. The panel executes the plan itself
            when there is nothing to train on. */}
        <section>
          <ModelsPanel run={run} onRunPlan={execute} />
        </section>
      </main>

      <Dialog
        open={confirmRun}
        title="Run this plan?"
        onClose={() => setConfirmRun(false)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmRun(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => void execute()}>
              Run it
            </Button>
          </>
        }
      >
        <p className="text-sm text-text">
          Version {plan.version} will be executed exactly as it stands: {kept} column
          {kept === 1 ? "" : "s"} prepared, {dropped} removed.
        </p>
        {pendingCount ? (
          <p className="mt-2 text-sm text-warn">
            You have {pendingCount} unsaved change{pendingCount === 1 ? "" : "s"}. They
            will <strong>not</strong> be included — save them first if you want them to
            run.
          </p>
        ) : null}
        <p className="mt-2 text-xs text-muted">
          Nothing is re-decided while it runs. The held-out rows are set aside before
          anything is measured or fitted.
        </p>
      </Dialog>
    </div>
  );
}
