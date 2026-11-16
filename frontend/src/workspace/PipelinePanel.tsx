/**
 * The left panel: every preprocessing step, as a switch that explains itself.
 *
 * This is where the product's argument lives. A step card has to answer four questions
 * without the user opening anything: what is this, is it on, what is it doing to *my*
 * data, and what would happen if I turned it off. The first three are on the face of
 * the card; the fourth is one click away and is measured, not asserted.
 *
 * Disabling never happens silently. The switch does not flip until the consequences
 * have been shown and accepted — which is the difference between a control and a trap.
 */

import { useState, type ReactNode } from "react";
import {
  AlertTriangle,
  ChevronRight,
  Code2,
  Gauge,
  Lock,
  Settings2,
  ShieldAlert,
} from "lucide-react";
import type { ImpactReport, PipelineConfig, Severity, StepId, StepView, Workspace } from "../api";
import { Badge, Button, Dialog, Segmented, Select, Spinner, Toggle, Tooltip } from "../components/ui";
import { Term } from "../explain/Term";
import { describeStep, findStepInCode, headline, perColumnLegend } from "../explain/steps";

const SEVERITY_TONE: Record<Severity, "neutral" | "info" | "warn" | "danger"> = {
  none: "neutral",
  low: "info",
  moderate: "warn",
  high: "warn",
  blocking: "danger",
};

/**
 * Severity describes the step being OFF, because that is what the server measures.
 *
 * So the badge has to be phrased differently depending on which way the switch is
 * moving: disabling, it is a warning about what you are about to cause; enabling, it is
 * a description of what you already have and are about to fix.
 */
const SEVERITY_WORD: Record<Severity, string> = {
  none: "No effect on this dataset",
  low: "Minor effect",
  moderate: "Noticeable effect",
  high: "Significant effect",
  blocking: "This will stop the pipeline working",
};

const SEVERITY_WORD_ENABLING: Record<Severity, string> = {
  none: "Nothing to fix on this dataset",
  low: "Fixes a minor problem",
  moderate: "Fixes a noticeable problem",
  high: "Fixes a significant problem",
  blocking: "The pipeline cannot run without this",
};

/** The status dot on a card. Shape and text carry the meaning; colour reinforces it. */
function StatusDot({ status }: { status: StepView["status"] }) {
  const tone =
    status === "active" ? "bg-accent" : status === "idle" ? "bg-faint" : "bg-line-strong";
  const title =
    status === "active"
      ? "Enabled and changing this dataset"
      : status === "idle"
        ? "Enabled, but nothing to do for this dataset"
        : "Disabled";
  return <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${tone}`} title={title} />;
}

// ---------------------------------------------------------------------------------
// Per-step options
// ---------------------------------------------------------------------------------

/**
 * The options for one step.
 *
 * Deliberately not generated from a schema. Each step's choices need different
 * wording, different groupings, and in two cases a warning next to one option — a
 * generic form renderer would flatten all of that into rows of identical selects.
 */
function StepOptions({
  step,
  config,
  onChange,
}: {
  step: StepView;
  config: PipelineConfig;
  onChange: (next: PipelineConfig) => void;
}) {
  const patch = (section: StepId, values: Record<string, unknown>) =>
    onChange({ ...config, [section]: { ...config[section], ...values } });

  if (step.id === "missing_values") {
    const values = config.missing_values;
    return (
      <div className="space-y-3">
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">Numeric columns</label>
          <Select
            label="Numeric fill strategy"
            value={values.numeric_strategy}
            onChange={(v) => patch("missing_values", { numeric_strategy: v })}
            options={[
              { value: "median", label: "Median — resists extreme values" },
              { value: "mean", label: "Mean — the average" },
              { value: "constant", label: "A fixed value" },
              { value: "forward_fill", label: "Carry the previous value forward" },
              { value: "backward_fill", label: "Carry the next value backward" },
              { value: "drop_rows", label: "Drop rows with blanks" },
            ]}
          />
        </div>
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">Category columns</label>
          <Select
            label="Categorical fill strategy"
            value={values.categorical_strategy}
            onChange={(v) => patch("missing_values", { categorical_strategy: v })}
            options={[
              { value: "most_frequent", label: "Most common value" },
              { value: "constant", label: "A fixed label" },
              { value: "forward_fill", label: "Carry the previous value forward" },
              { value: "backward_fill", label: "Carry the next value backward" },
              { value: "drop_rows", label: "Drop rows with blanks" },
            ]}
          />
        </div>
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">
            Drop columns emptier than{" "}
            <span className="tnum font-mono text-text">
              {Math.round(values.drop_column_threshold * 100)}%
            </span>
          </label>
          <input
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={values.drop_column_threshold}
            aria-label="Drop column threshold"
            onChange={(e) =>
              patch("missing_values", { drop_column_threshold: Number(e.target.value) })
            }
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
      </div>
    );
  }

  if (step.id === "outliers") {
    const values = config.outliers;
    return (
      <div className="space-y-3">
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">How they are found</label>
          <Segmented
            size="sm"
            label="Outlier method"
            value={values.method}
            onChange={(v) => patch("outliers", { method: v })}
            options={[
              { value: "iqr", label: "IQR", hint: "Interquartile range — the box-plot rule" },
              { value: "zscore", label: "Z-score", hint: "Standard deviations from the mean" },
            ]}
          />
        </div>
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">What happens to them</label>
          <Select
            label="Outlier action"
            value={values.action}
            onChange={(v) => patch("outliers", { action: v })}
            options={[
              { value: "cap", label: "Cap — pull back to the boundary" },
              { value: "winsorize", label: "Winsorize — pull to a percentile" },
              { value: "flag", label: "Flag — mark them, change nothing" },
              { value: "remove", label: "Remove — delete the whole row" },
            ]}
          />
          {values.action === "remove" && (
            <p className="mt-2 flex gap-1.5 text-[12px] leading-relaxed text-warn">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              This is the only option that loses data. An unusually large real value and a
              mistyped one look identical from here.
            </p>
          )}
        </div>
        {values.method === "iqr" ? (
          <div>
            <label className="mb-1.5 block text-[12px] text-muted">
              Multiplier{" "}
              <span className="tnum font-mono text-text">{values.iqr_multiplier}</span>
            </label>
            <input
              type="range" min={0.5} max={3} step={0.1}
              value={values.iqr_multiplier}
              aria-label="IQR multiplier"
              onChange={(e) => patch("outliers", { iqr_multiplier: Number(e.target.value) })}
              className="w-full accent-[var(--color-accent)]"
            />
          </div>
        ) : (
          <div>
            <label className="mb-1.5 block text-[12px] text-muted">
              Threshold{" "}
              <span className="tnum font-mono text-text">{values.zscore_threshold}σ</span>
            </label>
            <input
              type="range" min={1} max={5} step={0.5}
              value={values.zscore_threshold}
              aria-label="Z-score threshold"
              onChange={(e) => patch("outliers", { zscore_threshold: Number(e.target.value) })}
              className="w-full accent-[var(--color-accent)]"
            />
          </div>
        )}
      </div>
    );
  }

  if (step.id === "encoding") {
    const values = config.encoding;
    return (
      <div className="space-y-3">
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">Two-value columns</label>
          <Segmented
            size="sm"
            label="Binary encoding"
            value={values.binary_method}
            onChange={(v) => patch("encoding", { binary_method: v })}
            options={[
              { value: "label", label: "0 / 1" },
              { value: "onehot", label: "One-hot" },
            ]}
          />
        </div>
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">
            Up to{" "}
            <span className="tnum font-mono text-text">{values.max_onehot_cardinality}</span>{" "}
            distinct values
          </label>
          <Select
            label="Low cardinality encoding"
            value={values.low_cardinality_method}
            onChange={(v) => patch("encoding", { low_cardinality_method: v })}
            options={[
              { value: "onehot", label: "One-hot — a column per value" },
              { value: "ordinal", label: "Ordinal — one number per value" },
              { value: "label", label: "Label — one number per value" },
            ]}
          />
          <input
            type="range" min={2} max={50} step={1}
            value={values.max_onehot_cardinality}
            aria-label="One-hot cardinality threshold"
            onChange={(e) =>
              patch("encoding", { max_onehot_cardinality: Number(e.target.value) })
            }
            className="mt-2 w-full accent-[var(--color-accent)]"
          />
        </div>
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">Above that</label>
          <Select
            label="High cardinality encoding"
            value={values.high_cardinality_method}
            onChange={(v) => patch("encoding", { high_cardinality_method: v })}
            options={[
              { value: "frequency", label: "Frequency — how often each value occurs" },
              { value: "target", label: "Target — average outcome, cross-fitted" },
              { value: "ordinal", label: "Ordinal — one number per value" },
              { value: "drop", label: "Drop the column" },
            ]}
          />
          {values.high_cardinality_method === "target" && (
            <p className="mt-2 text-[12px] leading-relaxed text-muted">
              This reads the target column. scikit-learn cross-fits it internally, so the
              value a training row receives is computed from folds that exclude that row —
              but frequency encoding cannot leak at all.
            </p>
          )}
        </div>
      </div>
    );
  }

  if (step.id === "scaling") {
    return (
      <div>
        <label className="mb-1.5 block text-[12px] text-muted">Method</label>
        <Segmented
          size="sm"
          label="Scaler"
          value={config.scaling.method}
          onChange={(v) => patch("scaling", { method: v })}
          options={[
            { value: "standard", label: "Standard", hint: "Mean 0, spread 1" },
            { value: "minmax", label: "Min–max", hint: "Squeezed into 0 to 1" },
            { value: "robust", label: "Robust", hint: "Median and interquartile range" },
          ]}
        />
        <p className="mt-2.5 text-[12px] leading-relaxed text-muted">
          {
            {
              standard: "Centred on the mean and divided by the spread. What most models assume.",
              minmax: "Everything lands between 0 and 1, keeping the shape of the column. One extreme value compresses the rest.",
              robust: "Centred on the median and scaled by the interquartile range, so extreme values do not set the scale.",
            }[config.scaling.method]
          }
        </p>
      </div>
    );
  }

  if (step.id === "duplicates") {
    return (
      <div>
        <label className="mb-1.5 block text-[12px] text-muted">Which copy to keep</label>
        <Segmented
          size="sm"
          label="Duplicate keep"
          value={config.duplicates.keep}
          onChange={(v) => patch("duplicates", { keep: v })}
          options={[
            { value: "first", label: "First" },
            { value: "last", label: "Last" },
          ]}
        />
      </div>
    );
  }

  if (step.id === "feature_selection") {
    const values = config.feature_selection;
    return (
      <div className="space-y-3">
        <div>
          <label className="mb-1.5 block text-[12px] text-muted">
            Drop columns correlated above{" "}
            <span className="tnum font-mono text-text">
              {values.correlation_threshold ?? 0.95}
            </span>
          </label>
          <input
            type="range" min={0.7} max={1} step={0.01}
            value={values.correlation_threshold ?? 0.95}
            aria-label="Correlation threshold"
            onChange={(e) =>
              patch("feature_selection", { correlation_threshold: Number(e.target.value) })
            }
            className="w-full accent-[var(--color-accent)]"
          />
        </div>
        <label className="flex items-center gap-2 text-[13px] text-text">
          <input
            type="checkbox"
            checked={values.mutual_information}
            onChange={(e) =>
              patch("feature_selection", {
                mutual_information: e.target.checked,
                top_k: e.target.checked ? (values.top_k ?? 10) : null,
              })
            }
            className="accent-[var(--color-accent)]"
          />
          Rank by mutual information with the target
        </label>
        {values.mutual_information && (
          <div>
            <label className="mb-1.5 block text-[12px] text-muted">
              Keep the top <span className="tnum font-mono text-text">{values.top_k ?? 10}</span>
            </label>
            <input
              type="range" min={2} max={40} step={1}
              value={values.top_k ?? 10}
              aria-label="Top K features"
              onChange={(e) => patch("feature_selection", { top_k: Number(e.target.value) })}
              className="w-full accent-[var(--color-accent)]"
            />
          </div>
        )}
      </div>
    );
  }

  if (step.id === "imbalance") {
    return (
      <div>
        <label className="mb-1.5 block text-[12px] text-muted">Action</label>
        <Select
          label="Imbalance action"
          value={config.imbalance.action}
          onChange={(v) => patch("imbalance", { action: v })}
          options={[
            { value: "report", label: "Report only — change nothing" },
            { value: "oversample", label: "Oversample the smaller classes" },
            { value: "undersample", label: "Undersample the larger classes" },
            { value: "class_weight", label: "Report and suggest class weights" },
          ]}
        />
        {config.imbalance.action !== "report" && (
          <p className="mt-2 text-[12px] leading-relaxed text-muted">
            Applied to the training rows only. Resampling the held-out rows would mean
            testing against a distribution that does not exist.
          </p>
        )}
      </div>
    );
  }

  if (step.id === "data_types") {
    return (
      <div className="space-y-2">
        {(
          [
            ["coerce_numeric", "Recognise numbers stored as text"],
            ["parse_dates", "Recognise dates stored as text"],
          ] as const
        ).map(([key, label]) => (
          <label key={key} className="flex items-center gap-2 text-[13px] text-text">
            <input
              type="checkbox"
              checked={config.data_types[key]}
              onChange={(e) => patch("data_types", { [key]: e.target.checked })}
              className="accent-[var(--color-accent)]"
            />
            {label}
          </label>
        ))}
      </div>
    );
  }

  return <p className="text-[13px] text-faint">This step has no options.</p>;
}

// ---------------------------------------------------------------------------------
// The disable dialog
// ---------------------------------------------------------------------------------

/**
 * What turning a step off would cost, before it is turned off.
 *
 * The projected score arrives a beat after the consequences, because it requires
 * actually running the pipeline the other way. Showing the text immediately and the
 * number when it is known beats holding the whole dialog back for a second — and beats
 * inventing a number that looks precise.
 */
function ImpactDialog({
  report,
  loading,
  onConfirm,
  onCancel,
}: {
  report: ImpactReport | null;
  loading: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!report) return null;
  const enabling = report.enabling;
  const tone = SEVERITY_TONE[report.severity];

  return (
    <Dialog
      open
      onClose={onCancel}
      tone={report.severity === "blocking" ? "danger" : "warn"}
      title={
        <span className="flex items-center gap-2">
          {report.severity === "blocking" ? (
            <ShieldAlert size={16} className="text-danger" />
          ) : (
            <AlertTriangle size={16} className="text-warn" />
          )}
          {enabling ? `Enable ${report.label}?` : `Disable ${report.label}?`}
        </span>
      }
      footer={
        <>
          <Button variant="ghost" onClick={onCancel}>
            {enabling ? "Leave it off" : "Keep it enabled"}
          </Button>
          <Button variant={report.severity === "blocking" ? "danger" : "secondary"} onClick={onConfirm}>
            {enabling ? "Enable" : "Disable anyway"}
          </Button>
        </>
      }
    >
      <Badge tone={tone} className="mb-3">
        {(enabling ? SEVERITY_WORD_ENABLING : SEVERITY_WORD)[report.severity]}
      </Badge>

      {/* The consequences always describe the step being OFF — the server computes them
          with `assess_disable` whichever way the switch is moving. Read under "turning
          this on means", they stated the opposite of what would happen: enabling outlier
          treatment caps those values, and the dialog said they would be left alone. So
          the enabling case is framed as the state the user is in now, which is what
          those sentences actually describe, and which is the reason to turn it on. */}
      <p className="mb-2.5 text-[13px] text-muted">
        {enabling
          ? "This step is off. As things stand:"
          : "If you turn this off:"}
      </p>
      <ul className="space-y-2">
        {report.consequences.map((line) => (
          <li key={line} className="flex gap-2.5 text-[14px] leading-relaxed text-text">
            <span
              className={`mt-[9px] h-1 w-1 shrink-0 rounded-full ${
                report.severity === "blocking" ? "bg-danger" : "bg-warn"
              }`}
            />
            {line}
          </li>
        ))}
      </ul>

      <div className="mt-5 flex items-center gap-4 rounded-lg border border-line bg-sunken px-4 py-3.5">
        <div className="min-w-0">
          <p className="text-[11px] tracking-wide text-faint uppercase">Dataset health</p>
          {loading ? (
            <p className="mt-1 flex items-center gap-2 text-[13px] text-muted">
              <Spinner size={12} /> measuring…
            </p>
          ) : report.projected_score === null ? (
            <p className="mt-1 text-[13px] text-danger">This configuration does not run.</p>
          ) : (
            <p className="tnum mt-0.5 font-mono text-[20px] text-text">
              {report.current_score} <span className="text-faint">→</span>{" "}
              <span
                className={
                  (report.delta ?? 0) > 0
                    ? "text-good"
                    : (report.delta ?? 0) < 0
                      ? "text-warn"
                      : "text-muted"
                }
              >
                {report.projected_score}
              </span>
            </p>
          )}
        </div>
        <p className="border-l border-line pl-4 text-[12px] leading-relaxed text-faint">
          Measured by running the pipeline both ways — not estimated.
        </p>
      </div>

      <p className="mt-3 text-[12.5px] text-faint">
        You can change this back at any time. Nothing is written until you compile.
      </p>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------------
// Step card
// ---------------------------------------------------------------------------------

/** One expandable layer of a step card. */
function Layer({
  icon,
  label,
  children,
}: {
  icon: ReactNode;
  label: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 py-1 text-ui text-muted transition-colors hover:text-text"
      >
        {icon}
        {label}
        <ChevronRight
          size={11}
          className={`ml-auto shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>
      {open && <div className="fade mt-2 pb-1">{children}</div>}
    </div>
  );
}

/**
 * A step, in four layers.
 *
 * The face answers what this is, whether it is on, and what it is doing to *this*
 * dataset — the last of those measured, which it previously was not: `values_affected`
 * and friends were computed by the server on every request and then never rendered
 * anywhere. Under the face, three disclosures in the order a reader needs them: where
 * it shows up in their own data, how it works and what they can change, and the lines
 * of Python that do it.
 *
 * The code link is located by searching the editor's current contents for this step's
 * banner comment, so it cannot point at the wrong line. A step that is switched off
 * emits no section, and hand-edited code may not have one; in both cases the link is
 * absent rather than broken.
 */
function StepCard({
  step,
  index,
  config,
  code,
  totalRows,
  onToggle,
  onConfigChange,
  onJumpToLine,
}: {
  step: StepView;
  index: number;
  config: PipelineConfig;
  code: string;
  totalRows: number;
  onToggle: (next: boolean) => void;
  onConfigChange: (next: PipelineConfig) => void;
  onJumpToLine: (line: number) => void;
}) {
  const off = !step.enabled;
  const measured = headline(step, totalRows);
  const facts = describeStep(step);
  const legend = perColumnLegend(step.id);
  const perColumn = Object.entries(step.facts.per_column ?? {});
  const codeLine = findStepInCode(code, step.label);

  return (
    <li className={`card overflow-hidden transition-opacity ${off ? "opacity-60" : ""}`}>
      {/* ------------------------------------------------- layer 1: at a glance */}
      <div className="flex items-start gap-2.5 p-3">
        <span className="tnum mt-0.5 w-4 shrink-0 font-mono text-micro text-faint">
          {index + 1}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <StatusDot status={step.status} />
            <h3 className="truncate font-display text-[13.5px] font-medium text-text">
              {step.label}
            </h3>
            {step.locked && <Lock size={10} className="shrink-0 text-faint" />}
          </div>
          <p className="mt-1 text-[12.5px] leading-snug text-muted">{step.summary}</p>

          {/* The one number that makes this step about the reader's file rather than
              about preprocessing in general. Measured on every request already; it had
              simply never been shown. */}
          {measured && (
            <p className="mt-1.5 flex items-start gap-1.5 text-meta leading-snug text-faint">
              <Gauge size={11} className="mt-[2px] shrink-0" aria-hidden />
              <span>{measured}</span>
            </p>
          )}
        </div>

        {step.locked ? (
          <Tooltip content="Every other step depends on knowing what each column holds, so this one cannot be switched off.">
            <span className="mt-0.5 shrink-0 text-micro text-faint">always on</span>
          </Tooltip>
        ) : (
          <Toggle
            size="sm"
            checked={step.enabled}
            onChange={onToggle}
            label={`${step.enabled ? "Disable" : "Enable"} ${step.label}`}
          />
        )}
      </div>

      <div className="space-y-0.5 border-t border-line px-3 py-1.5">
        {/* ------------------------------------------------ layer 2: the evidence */}
        {(facts.length > 0 || perColumn.length > 0) && (
          <Layer
            icon={<Gauge size={11} className="shrink-0" />}
            label="Where this shows up in your data"
          >
            {/* Stacked, not two across: these sit inside a card whose width is set by
                the layout, and a viewport breakpoint knows nothing about that. */}
            {facts.length > 0 && (
              <dl className="space-y-2.5">
                {facts.map((fact) => (
                  <div key={fact.label}>
                    <dt className="text-micro text-faint">
                      <Term name={fact.term}>{fact.label}</Term>
                    </dt>
                    <dd className="tnum mt-0.5 font-mono text-meta text-text">{fact.value}</dd>
                    {fact.meaning && (
                      <dd className="mt-0.5 text-micro leading-relaxed text-faint">
                        {fact.meaning}
                      </dd>
                    )}
                  </div>
                ))}
              </dl>
            )}

            {/* The same key counts a different thing in every step, so the list says
                which thing. It used to read "Which columns" over bare numbers, which
                put 2 distinct values and 184 blank cells in the same column of the
                same table. */}
            {perColumn.length > 0 && legend && (
              <div className={facts.length > 0 ? "mt-3 border-t border-line/60 pt-2.5" : ""}>
                <p className="mb-1 text-micro text-faint">
                  <Term name={legend.term}>{legend.heading}</Term>
                </p>
                <ul className="space-y-0.5">
                  {perColumn.map(([column, value]) => (
                    <li key={column} className="flex justify-between gap-2 text-meta">
                      <span className="truncate font-mono text-muted">{column}</span>
                      <span className="tnum shrink-0 text-text">
                        {value.toLocaleString()}{" "}
                        <span className="text-faint">
                          {legend.unit}
                          {value === 1 ? "" : "s"}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Layer>
        )}

        {/* ----------------------------------------------- layer 3: the mechanism */}
        <Layer
          icon={<Settings2 size={11} className="shrink-0" />}
          label="How it works, and what you can change"
        >
          <div className="space-y-3">
            <p className="text-[12.5px] leading-relaxed text-muted">{step.explanation}</p>
            <StepOptions step={step} config={config} onChange={onConfigChange} />
          </div>
        </Layer>

        {/* ---------------------------------------------------- layer 4: the code */}
        {codeLine !== null && (
          <button
            onClick={() => onJumpToLine(codeLine)}
            className="flex w-full items-center gap-1.5 py-1 text-ui text-muted transition-colors hover:text-accent"
          >
            <Code2 size={11} className="shrink-0" />
            See the code that does this
            <span className="tnum ml-auto shrink-0 font-mono text-micro text-faint">
              line {codeLine}
            </span>
          </button>
        )}
      </div>
    </li>
  );
}

// ---------------------------------------------------------------------------------

export function PipelinePanel({
  workspace,
  roomy,
  onConfigChange,
  onRequestImpact,
  onJumpToLine,
  impact,
  impactLoading,
  onConfirmImpact,
  onCancelImpact,
}: {
  workspace: Workspace;
  /** True in the Explain layout, where this panel has the centre column's width. */
  roomy: boolean;
  onConfigChange: (next: PipelineConfig, summary?: string) => void;
  onRequestImpact: (step: StepId, enabled: boolean) => void;
  /** Reveal a line of the generated pipeline in the editor. */
  onJumpToLine: (line: number) => void;
  impact: ImpactReport | null;
  impactLoading: boolean;
  onConfirmImpact: () => void;
  onCancelImpact: () => void;
}) {
  const enabledCount = workspace.steps.filter((s) => s.enabled).length;

  return (
    <>
      <div className="flex h-full flex-col">
        <div className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line px-3.5">
          <h2 className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
            Pipeline
          </h2>
          <span className="tnum font-mono text-[11px] text-faint">
            {enabledCount} / {workspace.steps.length} on
          </span>
        </div>

        {workspace.code_is_authoritative && (
          <div className="mx-3 mt-3 rounded-lg border border-[color-mix(in_oklab,var(--color-warn)_35%,transparent)] bg-[color-mix(in_oklab,var(--color-warn)_8%,transparent)] px-3 py-2.5">
            <p className="flex gap-2 text-[12px] leading-relaxed text-warn">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              <span>
                You have edited the code by hand, so it is what runs. These switches
                describe the pipeline they were generated from, not what will execute.
              </span>
            </p>
          </div>
        )}

        {/* In the Explain layout this panel is the centre column, so the cards sit two
            across rather than stretching one explanation to 690px.

            `auto-rows-max` is load-bearing. A grid row defaults to `auto`, and a card
            carries `overflow-hidden`, which sets its automatic minimum size to zero —
            so once the cards needed more height than this scroller has, the rows shrank
            to fit instead of overflowing and every card was clipped mid-sentence.
            Sizing rows to their content puts the overflow back where it belongs, on the
            scroller. */}
        <ol
          className={`min-h-0 flex-1 overflow-y-auto p-3 ${
            roomy ? "grid auto-rows-max content-start gap-2 2xl:grid-cols-2" : "space-y-1.5"
          }`}
        >
          {workspace.steps.map((step, index) => (
            <StepCard
              key={step.id}
              step={step}
              index={index}
              config={workspace.config}
              code={workspace.code}
              totalRows={workspace.dataset.rows}
              onToggle={(next) => onRequestImpact(step.id, next)}
              onConfigChange={(next) => onConfigChange(next)}
              onJumpToLine={onJumpToLine}
            />
          ))}
        </ol>
      </div>

      <ImpactDialog
        report={impact}
        loading={impactLoading}
        onConfirm={onConfirmImpact}
        onCancel={onCancelImpact}
      />
    </>
  );
}
