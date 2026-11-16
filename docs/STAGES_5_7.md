# Stages 5–7 — training, explainability, recommendation

**Implemented 16 November 2026**, against the tree at `f9f8f0a`, in one three-hour
session by a single developer.

Measured on the development machine: **MacBook Pro (Mid-2015), Intel i7-4870HQ
(4 cores / 8 threads, 2.5 GHz), 16 GB RAM, macOS 12.7.6, CPython 3.12.4, x86_64.**
Every figure below was read off a run, not estimated. Where something is thin, it says
so.

This document records what was built and what it measures. It does not supersede
[PRD.md](PRD.md) §10, which sets the funded scope and the quarters; it records which
parts of Stages 5–7 now exist and on what terms.

---

## 1. What exists

| Stage | Capability | State |
|---|---|---|
| **5 — Training** | Six families trained concurrently on a stored run's data, scored on the held-out fold | **Working** |
| **6 — Explainability** | Global SHAP on the winning model, with an explicitly-labelled permutation fallback | **Working** |
| **6 — Leaderboard** | Ranked table with per-family metrics, sampling and failure reasons | **Working** |
| **7 — Recommendation** | Ranked families with a plain-English rationale, from the OpenML corpus | **Working** |
| **8 — Downloads** | Nine artifacts individually, plus a bundle with a generated README | **Working** |

**Not built, and not claimed:** Celery, Redis, WebSocket progress, per-prediction SHAP
waterfalls, beeswarm plots, cross-run comparison, model persistence between requests.

---

## 2. Architecture, and the simplifications it rests on

Four decisions were taken for the three-hour budget. Each is a real trade, and each has
a stated upgrade path.

### 2.1 Threads, not processes

`services/training.py` uses `ThreadPoolExecutor(max_workers=3)`.

Every estimator spends its time in compiled code that releases the GIL — scikit-learn's
tree builder is `nogil`, libsvm is a C extension, the histogram booster is OpenMP, the
neighbour search is BLAS — so three threads give three cores of real progress. Threads
also avoid a `spawn` start method re-importing scikit-learn three times and pickling the
matrices across a process boundary, which on this hardware is seconds per worker.

`threadpool_limits(1)` wraps each fit. `n_jobs=1` governs joblib only; without the
threadpool limit each worker would open as many BLAS/OpenMP threads as there are cores,
and 24 threads on 8 logical cores is measurably slower than 3.

**What this gives up:** crash isolation and the ability to kill a job that will not
finish. The row caps in §3 stand in for the second. **Upgrade path:** `ProcessPoolExecutor`
with the same interface, once there is time to debug the spawn/import path.

### 2.2 Concurrency is measured, never asserted

Each job records its start and finish on one monotonic clock; `_max_overlap` sweeps the
spans and reports the largest number in flight at once. If the pool serialises, the
payload reports `1`. There is no code path that can claim parallelism the executor did
not deliver, and `test_a_single_worker_pool_reports_one_not_three` pins that.

### 2.3 Results are artifacts, not a table

`models.json` and `explanation.json` are written to the run's own storage prefix and
added to `ARTIFACT_NAMES`. This inherits three things already built and tested: the key
discipline in `artifact_key`, the download route that serves any name in that dict, and
the `run.artifacts` list the interface already renders. **No migration was needed.**

**What this gives up:** a leaderboard cannot be queried with SQL across runs. When
Stage 7's cross-run dashboard needs that, promote it then — exactly as `feature_schema`
was promoted onto the run row for Stage 4.

### 2.4 Training is synchronous

`POST /api/runs/{id}/train` returns when training is done. For a single-user local
application, a status machine, a poll loop and a background task are three moving parts
that buy nothing when one user is waiting for the answer anyway.

**What this gives up:** a large file holds the request open. The caps bound it; this is
the limitation most worth revisiting first.

---

## 3. Dataset limits, and why each one exists

| Family | Row cap | Feature cap | Reason |
|---|---|---|---|
| Logistic Regression / Ridge | — | 5,000 | lbfgs is near-linear in rows |
| Decision Tree | — | 5,000 | Cheapest of the six |
| Gradient Boosting (HistGB) | — | 5,000 | Pre-binned; scales well |
| Random Forest | 20,000 | 2,000 | 100 trees single-threaded is the cost |
| **SVM (RBF)** | **8,000** | **1,000** | Kernel cost is between the square and the cube of the row count |
| K-Nearest Neighbours | 20,000 | 1,000 | k-d trees stop helping above ~20 dimensions; the query degenerates to brute force |

Above **5,000 encoded columns** nothing is trained, and the message names the likely
cause — a high-cardinality column that was one-hot encoded — rather than spending ten
minutes proving six models all do badly.

**Sampling is disclosed per family.** Every row carries `n_train_rows_used`,
`n_train_rows_available`, `sampled` and a `sampling_note`. A leaderboard where the SVM
saw 8,000 rows and the forest saw 20,000 without saying so is not a comparison, and
`test_sampling_is_disclosed_on_every_row_that_was_sampled` pins it. Classification
samples are stratified, so a capped family is not handed a different class balance from
the one it is ranked against.

**The test fold is never capped.** Every model is scored on exactly the same held-out
rows, or the ranking compares two different measurements.

---

## 4. What makes a score honest

Six properties, each enforced in code and pinned by a test.

1. **Fit on train, score on test.** The folds are the run's own stored parquet
   artifacts — the split the plan decided and the executor recreated. Nothing here
   re-splits anything.
   `test_scores_are_measured_on_the_held_out_fold_not_the_training_fold` scores one
   training fold against two different held-out folds and fails if the numbers come out
   identical, which is what would happen if someone scored on `X_train`.
2. **The strongest trivial baseline on the same fold.** Four dummy strategies for
   classification (`most_frequent`, `uniform`, `stratified`, `prior`) and two for
   regression (`mean`, `median`); the best becomes the bar, and all are reported. Every
   succeeded row carries `beats_baseline` and `margin_over_baseline`. **Taking the best
   rather than the conventional one is load-bearing** — §7b records the measurement that
   forced it, where a single `most_frequent` dummy let all six families claim to beat
   the baseline on a target made of pure noise.
3. **A family that did not run has a reason, never a zero.** A zero reads as "tried and
   terrible", which is a different and untrue statement.
4. **All six are always listed.** A table that omits the failure lets a reader believe
   only five were considered.
5. **Deterministic.** One seed, carried from the run.
   `test_the_same_data_and_seed_produce_the_same_leaderboard`.
6. **Concurrency measured, not assumed.** §2.2.

---

## 5. The labelling rule for explainability

**SHAP is only ever called SHAP when a SHAP explainer actually ran.**

Shapley values and permutation importance answer different questions. A Shapley value
decomposes one prediction into additive per-feature contributions; permutation
importance measures how much a metric degrades when a column is shuffled. They often
rank features similarly, which is exactly why mislabelling one as the other is tempting.

| Family | Explainer | `is_shap` |
|---|---|---|
| Random Forest, Decision Tree | `shap.TreeExplainer` | `true` |
| Logistic Regression | `shap.LinearExplainer` | `true` |
| Gradient Boosting (HistGB) | `permutation_importance` | **`false`** |
| SVM (RBF), KNN | `permutation_importance` | **`false`** |

Every payload carries `method`, `is_shap` and `method_label`, so the interface cannot
print a SHAP heading over permutation importances. A fallback always populates
`fallback_reason` saying why it fell back.

`HistGradientBoosting` is a boosted-tree model but its internal representation is not
one `TreeExplainer` reads, so it takes the permutation path. **`KernelExplainer` is
never used** — it is exponential in the feature count and would produce an approximation
nobody asked for, after hours.

Explanations are computed over at most **1,000 held-out rows**, with the number actually
used reported (PRD.md §10 asks for both halves of that).

**These are global importances only.** No per-prediction waterfall is produced, and the
payload does not imply one exists.

---

## 6. The recommendation corpus

Seeded with `scripts/build_corpus.py` against OpenML-CC18.

| | |
|---|---|
| Records stored | **16** |
| Held out (never returned as neighbours) | 2 |
| Refused — winner did not clear the dummy margin | 4 |
| Usable for a classification query | **10** |
| `MIN_CORPUS` | 8 |

The four refusals are the corpus builder declining to record a label it does not
believe: on `analcatdata_dmft` the best family beat the baseline by 0.0055, under the
0.0200 floor, so no winner was recorded.

**A refusal is reported as incomplete, not as a result.** When the engine declines —
too few comparable records, no record of the same task, or the closest neighbour below
`MIN_SIMILARITY` — the interface renders a warning that says the recommendation is
incomplete and names the corpus counts. It does not render an empty ranking, which
would read as "no algorithm suits your data".

---

## 7. Measured results — `campus_records.csv`

1,505 rows × 12 columns, target `has_backlog`, binary classification. 1,200 training
rows / 300 held out, stratified, 12 encoded features, seed 42.

### Leaderboard

| # | Model | f1_macro | accuracy | roc_auc | rows | ms | vs. dummy |
|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | **0.6938** | 0.7100 | 0.7633 | 1,200 | 82 | +0.327 |
| 2 | Random Forest | 0.6777 | 0.6933 | 0.7565 | 1,200 | 498 | +0.311 |
| 3 | SVM (RBF kernel) | 0.6703 | 0.7000 | 0.7390 | 1,200 | 152 | +0.303 |
| 4 | Gradient Boosting | 0.6567 | 0.6700 | 0.7060 | 1,200 | 616 | +0.290 |
| 5 | K-Nearest Neighbours | 0.6438 | 0.6600 | 0.6712 | 1,200 | 44 | +0.277 |
| 6 | Decision Tree | 0.5608 | 0.5767 | 0.5606 | 1,200 | 34 | +0.194 |

Trivial baseline: **0.3671** f1_macro. All six beat it. No family was sampled at this
size.

### Concurrency

| | |
|---|---|
| Workers | 3 |
| Jobs submitted | 6 |
| **Max observed overlap** | **3** |
| Summed model time | 2.31 s |
| Wall clock | 0.93 s |
| **Measured speed-up** | **2.5× vs. serial** |

### Explanation

`shap_linear` — `shap.LinearExplainer`, `is_shap: true`, over all 300 held-out rows.
Top features: `attendance_pct` (0.588), `prev_semester_score` (0.381),
`study_hours_week` (0.198).

### Recommendation

The engine ranked Logistic Regression first from 5 neighbours — **and Logistic
Regression went on to win the actual training run.** That is one agreement on one
dataset, not a validation of the engine; NFR-018 measures it properly on holdout
records.

---

## 7a. A pre-existing defect found while running the suite here

Running the full suite on **CPython 3.12** — this machine — surfaced a bug that
[AUDIT.md](AUDIT.md) could not have seen, because §5.1 records it running on **Python
3.14.2**.

**36 of the 41 parity tests failed.** The exported `pipeline.py` crashed on import:

```
NameError: name 'Any' is not defined
NameError: name 'ColumnPlan' is not defined
```

**Cause.** `plan_codegen.py` copies the engine's transformer classes and pipeline
builder out of the application with `inspect.getsource`, which is what makes the two
paths impossible to diverge (AUDIT.md §8.3). The copied source arrives carrying the
application's own annotations — `Any`, `ColumnPlan`, `PreprocessingPlan` — naming types
the standalone script deliberately does not contain. The generated preamble imported
none of them.

On Python 3.14 this is invisible: PEP 649 made annotations lazily evaluated by default,
so the names are never looked up. On 3.9–3.13 the annotation on a `def` is evaluated at
definition time, and the script dies before it reads a row.

**Impact.** Every downloaded `pipeline.py` was unrunnable for any user on Python 3.13 or
earlier — one of the product's headline glass-box claims, broken in the artifact rather
than in the engine, which is exactly why the engine's own tests stayed green.

**Fix.** Two lines in the generated preamble:

* `from __future__ import annotations` — makes every annotation a lazy string, so none
  of those names is ever looked up and the script needs no import of the app it came
  from. This is what does the work.
* `from typing import Any` — belt and braces, in case a copied function ever uses `Any`
  at runtime rather than in an annotation.

**Verified twice, deliberately.** 41 of 41 parity tests pass — *and* a `pipeline.py`
downloaded from a live run was executed standalone against `campus_records.csv`:

```
[prepare] removed 5 duplicate row(s)
[split] 1200 training rows, 300 held out
[fit] 8 input column(s) -> 12 feature(s)
Done. 1505 x 12  ->  1200 train + 300 test x 12 features
```

Those numbers match the run the script came from. A green parity test proves the two
code paths agree; running the downloaded file proves the feature the user actually has.

**Note for anyone with an older run:** the script is stored as an artifact at run time,
so runs executed before this fix still carry the broken copy. Re-run the plan to get a
working export.

**This was not introduced by Stages 5–7.** `git diff` shows no change to `codegen.py`,
`plan_codegen.py`, `glassbox.py`, `plan.py` or `runner.py`; the bug was latent in the
tree and was exposed by the interpreter, not by the new code.

---

## 7b. NFR-019 measured, and the defect that measuring it found

The 10 MB fixture: **35,910 rows × 50 columns → 93 encoded features**, 28,728 train /
7,182 held out.

### End-to-end latency

| Stage | Seconds |
|---|---|
| Upload + ingest | 6 |
| Profile + propose plan | 7 |
| Execute plan (preprocessing) | 8 |
| **Train six families** | **43** |
| **Total** | **64 s** |

**NFR-019 asks for under 3 minutes. Measured: 64 seconds.** The budget holds on this
hardware with the caps in §3 — and the earlier estimate in the execution plan (2.5–5
minutes for training) was too pessimistic by roughly a factor of four.

### What the caps bought

| Family | Rows used | Seconds |
|---|---|---|
| SVM (RBF) | **8,000** of 28,728 | **32.1** |
| Random Forest | 20,000 of 28,728 | 11.8 |
| Decision Tree | 28,728 | 2.7 |
| K-Nearest Neighbours | 20,000 | 2.6 |
| Gradient Boosting | 28,728 | 0.6 |
| Logistic Regression | 28,728 | 0.1 |

The SVM at its 8,000-row cap is **32 of the 50 seconds of model time**. Uncapped at
28,728 rows, quadratic scaling puts it near 7 minutes on its own, and NFR-019 fails.
The cap is what makes the budget hold.

**Observed speed-up was 1.55×, not 2.5×.** Six jobs on three workers cannot beat the
longest single job: the SVM's 32 s sets the floor for a 50 s workload, so the wall clock
is 32 s. This is Amdahl's law, not a defect in the pool, and the measured overlap was
still 3 of 3. It is recorded here because a speed-up figure quoted without the critical
path is a misleading number.

### The defect this measurement found

The 10 MB fixture's target is `rng.choice([0, 1])` — **pure noise, by construction.**
Every family scored accuracy ≈ 0.50 and ROC-AUC ≈ 0.50, correctly learning nothing.

**And all six reported `beats_baseline: true`.**

`DummyClassifier(most_frequent)` predicts one class only, which a macro average punishes
hard: it scored macro-F1 **0.3356**. A coin flip scores ≈ 0.50. So every family cleared
the bar by about 0.17 while having learned precisely nothing, and the leaderboard said
so in the interface.

**Fix, in two independent parts** — either alone would have caught this, and both are
kept because they fail for different reasons:

1. **The bar is now the best of four trivial strategies** (`most_frequent`, `uniform`,
   `stratified`, `prior`), not the conventional one. All four are scored and reported so
   a reader can see which set the bar.
2. **A near-chance ROC-AUC raises its own note.** If the best AUC across all families is
   ≤ 0.55, the leaderboard says *"these features do not separate the classes — read this
   as 'no signal found', not as a leaderboard."*

**Measured before and after, on both datasets:**

| Dataset | Bar before | Bar after | Beat the bar, before → after |
|---|---|---|---|
| `perf_10mb` (no signal) | `most_frequent` 0.3356 | **`uniform` 0.4969** | **6 of 6 → 3 of 6**, plus the no-signal note |
| `campus_records` (real signal) | `most_frequent` 0.3671 | **`stratified` 0.5494** | 6 of 6 → 6 of 6, no note |

The guard discriminates: it demotes the noise dataset and leaves the real one alone.
Pinned by `test_a_target_with_no_signal_does_not_produce_six_models_that_beat_the_baseline`.

This was a defect in Stage 5 as first written here, found by measuring rather than by
reading — the same way the seven defects in AUDIT.md §1.1 were found.

---

## 8. Test and build status

| Command | Result |
|---|---|
| `pytest tests/` (whole suite) | **623 passed, 0 failed, 9 skipped** · coverage **89.60%** (gate 80%) |
| `pytest tests/test_training.py` | **18 passed** |
| `pytest tests/test_training_api.py` | **9 passed** |
| `pytest tests/test_parity.py` | **41 passed** (36 were failing before the §7a fix) |
| `tsc -b --noEmit` | clean |
| `oxlint` | clean |
| `vite build` | 352 kB main + 392 kB lazy plan chunk |
| Full browser workflow | upload → profile → plan → override → run → train → leaderboard → SHAP → recommendation → download, all verified |

27 tests were added. None of the existing tests was weakened, skipped or deleted.

---

## 9. Known limitations

| | Limitation | Consequence |
|---|---|---|
| 1 | Training is synchronous | A large file holds the HTTP request open. Caps bound it; this is the first thing to revisit |
| 2 | Threads, not processes | No crash isolation, no hard per-model timeout |
| 3 | Fitted models are not persisted | The explanation is computed in the same call. Explaining a non-winning family later would need a retrain |
| 4 | Only the winner is explained | PRD.md §10 asks for the top model, so this is to specification — but it is a limit worth naming |
| 5 | `_estimators()` is duplicated | `scripts/build_corpus.py` and `services/training.py` carry the same six-family list. The script is documented as never importable from `app/`. **The two must be changed together** |
| 6 | Corpus is 16 records | PRD.md §10 funds 50–100. Ten usable records clears `MIN_CORPUS` but is a thin basis for a ranking, and the rationale says so when the spread is wide |
| 7 | No frontend tests | Carried forward from AUDIT.md §8.7. The new panel was verified by driving a live server |
| 8 | ~~NFR-019 not measured at 10 MB~~ | **Closed — §7b.** Measured at 64 s against a 3-minute budget |
| 9 | Speed-up is bounded by the SVM | At 10 MB the SVM is 32 s of 50 s of model time, so three workers give 1.55×, not 2.5×. More workers would not help; a cheaper kernel or a lower cap would |
| 10 | The no-signal guard is a heuristic | ROC-AUC ≤ 0.55 is a threshold, not a proof. It will not fire for a regression task, where no equivalent check is implemented |
