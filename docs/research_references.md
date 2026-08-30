# Research References

This prototype doesn't invent its core algorithms from scratch — each stage
is a simplified implementation of a published, industry-used technique.
Citations below are given at the level of detail useful for a hackathon
deck/README; verify exact venue/year details before using in a formal
academic bibliography.

## Detection — forecast-based anomaly detection on a seasonal decomposition

- **STL decomposition**: Cleveland, R. B., Cleveland, W. S., McRae, J. E., &
  Terpenning, I. (1990). *STL: A Seasonal-Trend Decomposition Procedure
  Based on Loess.* Journal of Official Statistics. The standard method for
  splitting a time series into trend, seasonal, and residual components.
- **Twitter's production anomaly detection**: Vallis, Hochenbaum, &
  Kejariwal (2014), *A Novel Technique for Long-Term Anomaly Detection in
  the Cloud* (Twitter, presented at USENIX HotCloud). Combines STL-style
  decomposition with a robust statistical test (Seasonal Hybrid ESD) on the
  residual rather than the raw series — this prototype follows the same
  "decompose, then test the residual robustly" shape.
- **Why we forecast instead of testing the in-sample residual**: fitting
  STL on data that includes the anomaly lets the trend component partially
  absorb a *sustained* shift (a real failure mode we hit and fixed — see
  `engine/anomaly.py` docstring). The fix — fit only on held-out history,
  forecast forward, compare against actual — is the same shape used by
  forecast-based systems like Facebook's Prophet.
- **Materiality = statistical AND business significance**: not from a
  single paper, but a standard practice in production monitoring/AIOps to
  avoid alert fatigue — a statistically real but tiny move isn't worth an
  investigation, and this prototype enforces both tests explicitly (see
  `kpi_contract.yaml`).

## Attribution / localization — multi-dimensional root cause analysis

- **Adtributor**: Bhagwan, R., Kumar, R., Ramjee, R., Varghese, G.,
  Mohapatra, S., Manoharan, H., & Shah, P. (2014). *Adtributor: Revenue
  Debugging in Advertising Systems.* NSDI. Introduces **Explanatory Power**
  (what fraction of the total change an element accounts for) and
  **Surprise** (a Jensen-Shannon divergence measuring how much an element's
  share of the whole shifted) as the two scoring signals for ranking
  candidate root causes — `engine/attribution.py`'s `adtributor_localize()`
  is a direct, single-dimension implementation of this idea.
- **Successors worth knowing about (not implemented here, noted for
  credibility/roadmap)**: HotSpot (Alibaba, ICDE 2018) and Squeeze
  (Alibaba, VLDB 2019) extend this to search combinations across *multiple*
  dimensions at once (e.g. region AND store AND SKU simultaneously), which
  is a combinatorially harder problem this prototype deliberately doesn't
  attempt. A recent production system, CMMD (Azure, reported ~2024), still
  cites Adtributor as a baseline it improves on — this is an active,
  real-world research area, not a solved problem.
- **The price/volume bridge, verified**: decomposing `Revenue = Price x
  Volume` into "how much of the change was volume vs. price" has an
  unavoidable interaction term (`delta_Price x delta_Volume`) that any such
  bridge has to assign somewhere. A naive bridge assigns the *entire*
  interaction term to whichever factor is paired with the "current" value
  — an arbitrary, order-dependent choice that we initially made without
  flagging it. Fixed to use the symmetric (midpoint) convention, which
  splits the interaction term evenly and gives the same split regardless
  of computation order. Both the "sums to the true total" and "order-
  independent" properties are covered by tests, not just asserted.

## Retrieval — filtered-first, then similarity

- Standard TF-IDF + cosine similarity (Salton & Buckley, 1988, classic
  information retrieval). The contribution here isn't the similarity
  metric — it's the design decision to **pre-filter by time window and
  segment before ranking by similarity**, so a textually-similar but
  wrong-region/wrong-week document can never be selected as evidence.
- **A real bug we found and fixed**: word-level TF-IDF gives a literal 0.0
  similarity between "stockout" and "out of stock" — zero shared tokens,
  despite being the same fact in different words. Fixed by expanding the
  query with the exact same cause-vocabulary `engine/confidence.py`
  already defines, rather than adding a second, separate synonym list. We
  also tested character n-gram TF-IDF as an alternative fix; it solved the
  vocabulary mismatch but inflated irrelevant documents' scores too, so we
  rejected it in favor of the vocabulary-tied approach.
- **Score presentation**: raw cosine similarity on short documents is
  small (0.03–0.15) even for a strong match. Each result also reports a
  relative score (against the best match in the already-filtered set) and
  a qualitative label, and the *scoring math* uses the relative score, not
  the raw one, since raw cosine values aren't on a comparable scale to the
  other signals being combined.

## Reconciliation — heterogeneous data integration

- Not a single-paper citation — this is standard practice in data
  integration and AIOps: different systems of record run on different
  clocks (daily sales, weekly marketing, batch/irregular inventory
  snapshots), and a defensible engine has to align them to a common
  question rather than silently pretending they're all on the same
  cadence. `engine/reconciliation.py` reports each source with its own
  grain and "as of" freshness rather than resampling away that distinction.
- **Why structured evidence outweighs text evidence**: a support ticket
  saying "we're out of stock" is a person's claim; the inventory system
  showing near-zero units on the shelf is the fact itself. `engine/
  confidence.py` weights structured reconciliation evidence (0.40) higher
  than statistical strength (0.35) and text retrieval (0.25) for this
  reason — not from a specific paper, but a reasonable and explicit
  modeling choice, stated here rather than buried in a weight constant
  with no explanation.

## Cause ranking — evidence-weighted association, and why plausibility is a gate, not a score

- Ranking combines three genuinely **independent** evidence types
  (statistical strength, structured data, text evidence) as a weighted
  sum. An earlier version added a fourth term — "plausibility" (does this
  cause's typical effect match the kind of change observed?) — as
  additional additive evidence. That double-counts: plausibility is
  *derived from* the same `dominant_driver_type` the statistical and
  structured signals already respond to, so it isn't new information, it's
  restating something already baked into the other terms. Plausibility is
  now a **multiplicative gate** applied to the sum of the three
  independent signals — a necessary sanity check (does this cause even
  make sense given what changed?), not a fourth vote in the total.

## IMPORTANT: this is not causal inference

Language elsewhere in this project (and in earlier drafts of this
document) used phrasing like "moves from correlation to causation" more
loosely than it should have. To be precise about what's actually
implemented: **there is no formal causal inference anywhere in this
pipeline.** No counterfactual modeling, no control/treatment group
comparison, no do-calculus, no causal graph learned from data — none of
the standard machinery of causal inference (e.g. Pearl's do-calculus,
difference-in-differences, instrumental variables) is present.

What the pipeline actually does — and this is a real, defensible bar, just
a different and weaker one than "causal inference" — is narrow candidate
explanations through:
1. **Temporal + segment alignment**: a cause is only considered if it's
   evidenced in the same time window and location as the anomaly.
2. **Multiple independent evidence types**, weighted rather than treated
   as equally strong (structured data outweighs a text mention).
3. **A plausibility gate** ruling out mechanistically nonsensical pairings.

The accurate description of the output is **"evidence-weighted
association, narrowed by alignment constraints"** — not "the proven
cause." This is a stronger bar than naive correlation-spotting, and a
meaningfully weaker one than causal inference. Say so plainly if asked;
claiming otherwise doesn't survive a close technical read, and a reviewer
who catches an overclaim on this point will (rightly) discount everything
else in the submission more skeptically.

## Confidence / abstention — selective prediction

- **The "reject option"**: Chow, C. K. (1970). *On Optimum Recognition
  Error and Reject Tradeoff.* IEEE Transactions on Information Theory. The
  foundational result that a classifier allowed to abstain can achieve
  higher accuracy on the cases it *does* answer, at the cost of coverage —
  formalizing exactly the tradeoff this system's confidence bands
  implement.
- **Modern framing**: recent work on LLM calibration and "conformal
  abstention" applies the same risk-coverage idea to language models
  specifically — the point being that a system saying "I don't know" is a
  designed behavior with a measurable tradeoff, not a failure state.
- **How this prototype applies it**: `utils/calibration.py` builds labeled
  synthetic scenarios, sweeps candidate thresholds, and picks the
  `confidence_bands` values in `kpi_contract.yaml` based on the resulting
  precision-coverage curve (see `docs/calibration_results.md`) — not by
  guessing a number that "feels right." Thresholds were recalibrated after
  the plausibility-gate fix, since a threshold is a property of the
  scoring function, not a constant that survives changes to it.
- **A naming collision we fixed**: `engine/anomaly.py` used to also return
  a field called "confidence" (about how much history we have), which is
  an entirely different question from "how sure are we about the cause"
  answered here. That field is now called `data_sufficiency` to avoid the
  two stages sounding like they're talking about the same thing.

## Honest scope note

Every technique above is implemented in a **simplified, single-scenario
form** appropriate for a hackathon timeline — not the full generality of
the cited papers (e.g. single-dimension localization, not multi-dimension
combinatorial search; a hand-built plausibility map, not a learned causal
graph). Say this plainly if asked — it's a stronger answer than overclaiming
and getting caught on a follow-up question.
