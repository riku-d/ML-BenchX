/**
 * The right panel: health, warnings, and what the last compile changed.
 *
 * The health score is the number the whole workspace is oriented around, so it is never
 * shown alone. Under it sit the six dimensions that produced it, and under those the
 * arithmetic — because a score a user cannot take apart is a score they have to take on
 * faith, and this product's entire argument is against that.
 *
 * When a compile has run, every dimension shows its before and after. That comparison
 * is the honest answer to "did this help?", including when the answer is no.
 */

import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  Info,
  Scale,
  Scissors,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import { api, type Compilation, type Health, type HealthDimension, type Run, type Workspace } from "../api";
import { Badge, Delta, Meter, ScoreRing, Spinner, Tabs } from "../components/ui";
import { describeChecks, provenance } from "../explain/checks";
import { describeDimension } from "../explain/health";
import { narrate } from "../explain/narrative";
import { Term } from "../explain/Term";

function toneFor(score: number) {
  if (score >= 90) return "good" as const;
  if (score >= 75) return "accent" as const;
  if (score >= 55) return "warn" as const;
  return "danger" as const;
}

/**
 * The evidence behind one dimension.
 *
 * Previously this pushed a fixed list of scalar keys and returned early on anything
 * that was not a number or a flat array — which threw away the only facts that said
 * *where* a problem was. `worst_columns`, `per_column_outliers`, `skew` and
 * `mixed_type_columns` are all nested, so all four were dropped: the panel could tell
 * you the dataset had 170 outliers and 2 heavily skewed columns, and could not tell you
 * which columns. `describeDimension` reads them, names the columns, and attaches the
 * meaning to the number instead of leaving "Scale disparity 359.15" to be interpreted.
 */
function DimensionFacts({ dimension }: { dimension: HealthDimension }) {
  const { facts, columnsHeading, columns, notes } = describeDimension(dimension);

  if (facts.length === 0 && columns.length === 0 && notes.length === 0) {
    return (
      <p className="mt-2 text-[11.5px] text-faint">
        This dimension reported nothing beyond its score.
      </p>
    );
  }

  return (
    <div className="mt-2.5 space-y-2.5">
      {facts.length > 0 && (
        <dl className="space-y-2">
          {facts.map((fact) => (
            <div key={fact.label}>
              <div className="flex justify-between gap-3">
                <dt className="text-micro text-faint">
                  <Term name={fact.term}>{fact.label}</Term>
                </dt>
                <dd className="tnum shrink-0 font-mono text-meta text-text">{fact.value}</dd>
              </div>
              {fact.meaning && (
                <dd className="mt-0.5 text-micro leading-relaxed text-faint">{fact.meaning}</dd>
              )}
            </div>
          ))}
        </dl>
      )}

      {columns.length > 0 && (
        <div className="border-t border-line/60 pt-2.5">
          {columnsHeading && <p className="mb-1 text-micro text-faint">{columnsHeading}</p>}
          <ul className="space-y-0.5">
            {columns.map((finding) => (
              <li key={finding.column} className="flex justify-between gap-3 text-meta">
                <span className="truncate font-mono text-muted">{finding.column}</span>
                <span className="tnum shrink-0 text-right text-faint">{finding.detail}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {notes.map((note) => (
        <p key={note} className="text-micro leading-relaxed text-faint">
          {note}
        </p>
      ))}
    </div>
  );
}

function Dimensions({
  health,
  comparison,
}: {
  health: Health;
  comparison?: { key: string; before: number; after: number; delta: number }[];
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const deltas = new Map((comparison ?? []).map((d) => [d.key, d]));

  return (
    <ul className="space-y-1">
      {health.dimensions.map((dimension) => {
        const open = expanded === dimension.key;
        const change = deltas.get(dimension.key);
        return (
          <li key={dimension.key} className="rounded-lg px-2 py-1.5 transition-colors hover:bg-raised">
            <button
              onClick={() => setExpanded(open ? null : dimension.key)}
              aria-expanded={open}
              className="w-full text-left"
            >
              <div className="mb-1.5 flex items-baseline gap-2">
                <span className="flex-1 truncate text-[12.5px] text-text">{dimension.label}</span>
                {change && change.delta !== 0 ? (
                  <Delta value={Math.round(change.delta)} />
                ) : null}
                <span className="tnum font-mono text-[11.5px] text-muted">
                  {dimension.score.toFixed(0)}
                </span>
                <ChevronDown
                  size={11}
                  className={`shrink-0 text-faint transition-transform ${open ? "rotate-180" : ""}`}
                />
              </div>
              <Meter
                value={dimension.score}
                label={dimension.label}
                showValue={false}
                tone={toneFor(dimension.score)}
              />
            </button>

            {open && (
              <div className="fade mt-2.5 border-t border-line pt-2.5">
                <p className="text-[12px] leading-relaxed text-muted">{dimension.summary}</p>
                <p className="mt-1.5 text-[11.5px] text-faint">{dimension.description}</p>
                <DimensionFacts dimension={dimension} />
                <p className="mt-2 font-mono text-[11px] text-faint">
                  {dimension.score.toFixed(1)} × {dimension.weight.toFixed(2)} ={" "}
                  {dimension.contribution.toFixed(2)} points
                </p>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/**
 * How the data was divided.
 *
 * "We split before fitting" is the product's central claim, and a claim the user cannot
 * see the evidence for is just a sentence. These numbers are read out of the report the
 * generated script wrote as it ran — not recomputed here, so they describe what actually
 * happened rather than what was configured.
 */
function SplitCard({ split }: { split: NonNullable<Compilation["result"]>["split"] }) {
  if (!split || split.train_rows == null || split.test_rows == null) return null;

  const total = split.train_rows + split.test_rows;
  const trainShare = total > 0 ? (split.train_rows / total) * 100 : 0;
  const rows: [string, string][] = [
    ["Training set", `${split.train_rows.toLocaleString()} rows`],
    ["Test set", `${split.test_rows.toLocaleString()} rows`],
    ["Split", `${Math.round(trainShare)} / ${Math.round(100 - trainShare)}`],
    ["Strategy", split.stratified ? "Stratified" : "Random"],
    ["Random seed", String(split.seed ?? "—")],
  ];

  return (
    <section className="mt-4 border-t border-line pt-3">
      <h3 className="flex items-center gap-1.5 font-display text-[11px] tracking-[0.13em] text-faint uppercase">
        <Scissors size={11} /> Train / test split
      </h3>

      <div className="mt-2.5 flex h-2 overflow-hidden rounded-full border border-line">
        <div className="bg-accent" style={{ width: `${trainShare}%` }} />
        <div
          className="bg-raised"
          style={{
            width: `${100 - trainShare}%`,
            backgroundImage:
              "repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(127,127,127,.35) 3px, rgba(127,127,127,.35) 6px)",
          }}
        />
      </div>

      <table className="mt-2.5 w-full">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="text-[11.5px]">
              <td className="py-0.5 text-muted">{label}</td>
              <td className="tnum py-0.5 text-right font-mono text-text">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="mt-2 text-[11px] leading-relaxed text-faint">
        The held-out rows were set aside before any value was measured or any decision
        made, so nothing in them influenced this pipeline.
      </p>
    </section>
  );
}

/**
 * "What happened to your data" — the compile as a story, then what was checked.
 *
 * Two sources, and they are kept apart on purpose. The sequence and the before/after
 * come from this workspace's compile. The checks come from a **plan run**, which is a
 * different execution that may have happened at a different time against a different
 * version of the plan — so they sit under their own heading with the date and version
 * they belong to, and a plain warning when the run predates the compile. Presenting the
 * two as one event would be the kind of quiet elision this product exists to refuse.
 */
function Narrative({
  compilation,
  datasetId,
}: {
  compilation: Compilation;
  datasetId: string;
}) {
  const [run, setRun] = useState<Run | null>(null);
  const [runState, setRunState] = useState<"loading" | "none" | "ready" | "error">("loading");

  useEffect(() => {
    let current = true;
    setRunState("loading");
    api
      .listRuns(datasetId)
      .then((listed) => {
        const latest = listed.runs
          .filter((entry) => entry.status === "succeeded")
          .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
        if (!latest) {
          if (current) setRunState("none");
          return;
        }
        // The listing is a summary — it carries no `checks` — so the chosen run has to
        // be read in full before anything can be said about it.
        return api.readRun(latest.id).then((full) => {
          if (!current) return;
          setRun(full);
          setRunState("ready");
        });
      })
      .catch(() => {
        if (current) setRunState("error");
      });
    return () => {
      current = false;
    };
  }, [datasetId]);

  const moments = narrate(compilation.result.steps);
  const comparison = compilation.result.comparison;
  const cards = run ? describeChecks(run) : [];

  // `comparison` is optional on a successful compile, so the shape before and after
  // falls back to the raw counts. Health has no fallback — without a comparison there
  // is no "before" to quote, and inventing one would be worse than omitting it.
  const started = comparison
    ? { rows: comparison.rows.before, columns: comparison.columns.before, score: comparison.score.before }
    : compilation.result.rows_before != null && compilation.result.columns_before != null
      ? { rows: compilation.result.rows_before, columns: compilation.result.columns_before, score: null }
      : null;
  const ended = comparison
    ? { rows: comparison.rows.after, columns: comparison.columns.after, score: comparison.score.after }
    : compilation.result.rows_after != null && compilation.result.columns_after != null
      ? { rows: compilation.result.rows_after, columns: compilation.result.columns_after, score: null }
      : null;

  return (
    <div>
      <p className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
        What happened to your data
      </p>
      <p className="mt-1 text-micro text-faint">
        Compiled from the {compilation.source} in {compilation.duration_ms} ms.
      </p>

      {/* ------------------------------------------------- where it started and ended */}
      {started && ended && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          {[
            { label: "You started with", ...started },
            { label: "You ended with", ...ended },
          ].map((side) => (
            <div key={side.label} className="card p-2.5">
              <p className="text-micro text-faint">{side.label}</p>
              <p className="tnum mt-1 font-mono text-meta text-text">
                {side.rows.toLocaleString()} × {side.columns}
              </p>
              {side.score !== null && (
                <p className="tnum mt-0.5 font-mono text-micro text-faint">
                  health {side.score}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ------------------------------------------------------------ the sequence */}
      <ol className="mt-4 space-y-0">
        {moments.map((moment) =>
          moment.kind === "split" ? (
            // The claim the whole product rests on. In the old list this was one grey
            // row between two others, which is not how you present your own thesis.
            <li key={moment.key} className="my-2 rounded-lg border border-accent/35 bg-accent/5 px-2.5 py-2">
              <p className="flex items-center gap-1.5 text-ui font-medium text-accent">
                <Scissors size={11} className="shrink-0" />
                <Term name="the split">The split</Term>
              </p>
              <p className="mt-1 text-meta leading-snug text-text">{moment.action}</p>
              <p className="mt-1 text-micro leading-relaxed text-faint">
                Everything below this line learned from the training rows alone. That is
                what makes the held-out rows an honest test.
              </p>
            </li>
          ) : (
            <li key={moment.key} className="flex gap-2.5">
              {/* The thread down the left is what makes this a sequence rather than a
                  list of cards that happen to be stacked. */}
              <div className="flex shrink-0 flex-col items-center pt-[7px]">
                <span className="h-1.5 w-1.5 rounded-full bg-hairline" />
                <span className="w-px flex-1 bg-line" />
              </div>
              <div className="min-w-0 flex-1 pb-3">
                <div className="flex items-baseline gap-2">
                  <span className="flex-1 truncate text-ui text-text">{moment.title}</span>
                  {moment.columnsDelta !== null && moment.columnsDelta !== 0 && (
                    <span
                      className={`tnum shrink-0 font-mono text-micro ${
                        moment.columnsDelta > 0 ? "text-good" : "text-warn"
                      }`}
                    >
                      {moment.columnsDelta > 0 ? "+" : ""}
                      {moment.columnsDelta} col
                    </span>
                  )}
                  {moment.rowsDelta !== null && moment.rowsDelta !== 0 && (
                    <span className="tnum shrink-0 font-mono text-micro text-warn">
                      {moment.rowsDelta > 0 ? "+" : ""}
                      {moment.rowsDelta.toLocaleString()} rows
                    </span>
                  )}
                </div>
                <p className="mt-0.5 text-meta leading-snug text-muted">{moment.action}</p>
                {moment.touched.length > 0 && (
                  <p className="mt-0.5 truncate font-mono text-micro text-faint">
                    {moment.touched.slice(0, 4).join(", ")}
                    {moment.touched.length > 4 ? `, +${moment.touched.length - 4}` : ""}
                  </p>
                )}
              </div>
            </li>
          ),
        )}
      </ol>

      {comparison && (
        <p className="border-t border-line pt-2.5 text-meta leading-relaxed text-muted">
          Dataset health moved from {comparison.score.before} to {comparison.score.after}
          {comparison.score.delta === 0
            ? ", unchanged."
            : `, ${comparison.score.delta > 0 ? "up" : "down"} ${Math.abs(comparison.score.delta)}.`}{" "}
          The Health tab takes that apart dimension by dimension.
        </p>
      )}

      {/* ----------------------------------------------------- checks from a plan run */}
      <section className="mt-5 border-t border-line pt-3">
        <h3 className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
          What was checked afterwards
        </h3>

        {runState === "loading" && (
          <p className="mt-2 flex items-center gap-2 text-meta text-muted">
            <Spinner size={11} /> looking for a plan run…
          </p>
        )}

        {runState === "none" && (
          <p className="mt-2 text-meta leading-relaxed text-faint">
            This plan has never been executed, so there is nothing independent to report.
            Running it from <span className="text-muted">Review the plan</span> produces a
            drift check, a balance check and a baseline comparison.
          </p>
        )}

        {runState === "error" && (
          <p className="mt-2 text-meta leading-relaxed text-faint">
            The run history could not be read, so these checks are not shown. Nothing
            above depends on it.
          </p>
        )}

        {runState === "ready" && run && (
          <>
            <p className="mt-1.5 text-micro leading-relaxed text-faint">{provenance(run)}</p>
            {/* A plan run and a compile are two different executions, and which came
                first changes what these checks can be taken to mean. Saying so in both
                directions costs a line; letting a reader assume they are the same event
                costs them the thing this panel is for. */}
            <p className="mt-1.5 flex gap-1.5 text-micro leading-relaxed text-warn">
              <AlertTriangle size={11} className="mt-[1px] shrink-0" />
              {run.created_at < compilation.created_at
                ? "That run is older than this compile, so these checks describe an earlier version of the pipeline — not the sequence above."
                : "A plan run is a separate execution from this compile. These checks describe that run, which may not match the sequence above."}
            </p>

            <div className="mt-2.5 space-y-2">
              {cards.map((card) => (
                <div key={card.key} className="card p-2.5">
                  <div className="flex items-start gap-1.5">
                    {card.tone === "good" ? (
                      <ShieldCheck size={12} className="mt-[2px] shrink-0 text-good" />
                    ) : card.tone === "warn" ? (
                      <AlertTriangle size={12} className="mt-[2px] shrink-0 text-warn" />
                    ) : (
                      <Info size={12} className="mt-[2px] shrink-0 text-info" />
                    )}
                    <p className="flex-1 text-ui font-medium text-text">{card.title}</p>
                  </div>
                  <p className="mt-1.5 text-meta leading-relaxed text-text">{card.verdict}</p>
                  {card.detail.length > 0 && (
                    <ul className="mt-1.5 space-y-0.5">
                      {card.detail.map((line) => (
                        <li key={line} className="text-micro leading-relaxed text-faint">
                          {line}
                        </li>
                      ))}
                    </ul>
                  )}
                  {card.note && (
                    <p className="mt-1.5 border-t border-line/60 pt-1.5 text-micro leading-relaxed text-faint">
                      {card.note}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

export function InsightsPanel({ workspace }: { workspace: Workspace }) {
  const [tab, setTab] = useState<"health" | "issues" | "changes">("health");

  const compilation = workspace.latest_compilation;
  const succeeded = compilation?.status === "success";
  const after = succeeded ? (compilation!.health_after as Health) : null;
  const shown = after ?? workspace.baseline_health;
  const comparison = compilation?.result?.comparison;

  const issues = shown.issues ?? [];
  const balance = shown.class_balance;

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-line px-3.5">
        <h2 className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
          Insights
        </h2>
        <Tabs
          label="Insights view"
          value={tab}
          onChange={setTab}
          options={[
            { value: "health", label: "Health" },
            { value: "issues", label: "Issues", count: issues.length },
            { value: "changes", label: "Changes" },
          ]}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* --------------------------------------------------------- health */}
        {tab === "health" && (
          <div className="p-3.5">
            <div className="flex items-center gap-4">
              <ScoreRing score={shown.score} size={92} />
              <div className="min-w-0">
                <p className="font-display text-[15px] font-medium text-text">{shown.rating}</p>
                <p className="mt-0.5 text-[12px] text-muted">
                  {after ? "after preprocessing" : "as uploaded"}
                </p>
                {comparison && (
                  <div className="mt-2">
                    <Delta value={comparison.score.delta} />
                    <p className="tnum mt-0.5 font-mono text-[11px] text-faint">
                      {comparison.score.before} → {comparison.score.after}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-4 border-t border-line pt-3">
              <Dimensions health={shown} comparison={comparison?.dimensions} />
            </div>

            {succeeded && <SplitCard split={compilation!.result?.split} />}

            <details className="mt-3 border-t border-line pt-3">
              <summary className="cursor-pointer list-none text-[12px] text-muted transition-colors hover:text-text">
                How this number was worked out ›
              </summary>
              <div className="fade mt-2.5">
                <p className="mb-2 text-[11.5px] leading-relaxed text-faint">
                  {shown.formula.description}
                </p>
                <table className="w-full">
                  <tbody>
                    {shown.formula.terms.map((term) => (
                      <tr key={term.label} className="text-[11.5px]">
                        <td className="py-0.5 text-muted">{term.label}</td>
                        <td className="tnum py-0.5 text-right font-mono text-faint">
                          {term.score.toFixed(1)} × {term.weight.toFixed(2)}
                        </td>
                        <td className="tnum py-0.5 pl-2 text-right font-mono text-text">
                          {term.contribution.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                    <tr className="border-t border-line text-[11.5px]">
                      <td className="py-1 text-text">Total</td>
                      <td />
                      <td className="tnum py-1 pl-2 text-right font-mono text-text">
                        {shown.formula.total.toFixed(2)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}

        {/* --------------------------------------------------------- issues */}
        {tab === "issues" && (
          <div className="space-y-2 p-3.5">
            {issues.length === 0 && !balance?.imbalanced ? (
              <div className="flex flex-col items-center py-10 text-center">
                <ShieldCheck size={20} className="text-good" />
                <p className="mt-2.5 text-[13.5px] text-text">Nothing needs your attention</p>
                <p className="mt-1 max-w-[30ch] text-[12.5px] text-muted">
                  Every dimension scored above the concern threshold.
                </p>
              </div>
            ) : (
              <>
                {issues.map((issue) => (
                  <div key={issue.dimension} className="card p-3">
                    <div className="flex items-center gap-2">
                      {issue.severity === "warning" ? (
                        <AlertTriangle size={12} className="shrink-0 text-warn" />
                      ) : (
                        <Info size={12} className="shrink-0 text-info" />
                      )}
                      <span className="flex-1 truncate text-[13px] font-medium text-text">
                        {issue.label}
                      </span>
                      <span className="tnum font-mono text-[11.5px] text-muted">
                        {issue.score.toFixed(0)}
                      </span>
                    </div>
                    <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">{issue.text}</p>
                  </div>
                ))}

                {balance?.imbalanced && (
                  <div className="card p-3">
                    <div className="flex items-center gap-2">
                      <Scale size={12} className="shrink-0 text-warn" />
                      <span className="flex-1 text-[13px] font-medium text-text">
                        Class imbalance
                      </span>
                      <Badge tone="warn">{balance.ratio}:1</Badge>
                    </div>
                    <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">
                      {balance.summary}
                    </p>
                    <div className="mt-2.5 space-y-1">
                      {balance.classes.slice(0, 5).map((klass) => (
                        <div key={klass.value} className="flex items-center gap-2">
                          <span className="w-16 truncate font-mono text-[11px] text-muted">
                            {klass.value}
                          </span>
                          <Meter
                            value={klass.share * 100}
                            label={`${klass.value} share`}
                            showValue={false}
                            tone="info"
                          />
                          <span className="tnum w-9 text-right font-mono text-[11px] text-faint">
                            {(klass.share * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* -------------------------------------------------------- changes */}
        {tab === "changes" && (
          <div className="p-3.5">
            {!compilation ? (
              <div className="flex flex-col items-center py-10 text-center">
                <Activity size={20} className="text-faint" />
                <p className="mt-2.5 text-[13.5px] text-text">Nothing compiled yet</p>
                <p className="mt-1 max-w-[32ch] text-[12.5px] text-muted">
                  Run the pipeline to see what it changed and what it cost.
                </p>
              </div>
            ) : compilation.status === "error" ? (
              <div className="card border-[color-mix(in_oklab,var(--color-danger)_35%,transparent)] p-3">
                <div className="flex items-center gap-2">
                  <ShieldAlert size={13} className="text-danger" />
                  <span className="text-[13px] font-medium text-text">
                    The last run failed
                  </span>
                </div>
                <p className="mt-1.5 font-mono text-[12px] text-danger">
                  {compilation.error?.type}: {compilation.error?.message}
                </p>
              </div>
            ) : (
              <Narrative compilation={compilation} datasetId={workspace.dataset.id} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
