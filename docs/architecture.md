# Architecture

## Pipeline shape

```
Data Sources → Semantic Layer → Detection → Attribution → Reconciliation → Retrieval → Ranking → Narrative → Recommendation
   (raw)          (contract)      (stats)      (math)       (structured)     (RAG)     (rules)     (LLM)        (rules)
```

Every stage is a separate, independently testable module under `engine/`.
The rule that shapes the whole design: **the LLM is a translator, never a
calculator.** It never computes a statistic, never decides a confidence
score, never picks a root cause — those are all deterministic math,
information theory, or rules. The LLM's only job is to phrase already-
computed facts in plain language, per persona. See `docs/research_references.md`
for the published work each stage's algorithm is based on.

## Module map

| Module | Stage | LLM? | Method |
|---|---|---|---|
| `engine/anomaly.py` | Detection | No | STL decomposition fit on held-out history, forecast forward, robust (median/MAD) residual test. Materiality requires BOTH statistical AND business significance. |
| `engine/attribution.py` | Attribution | No | Adtributor-style localization (explanatory power + surprise) across stores, then a SYMMETRIC price/volume bridge decomposition on the implicated stores |
| `engine/reconciliation.py` | Reconciliation | No | Pulls STRUCTURED context from `promo.csv` (weekly) and `inventory.csv` (batch, ~3-day, irregular) at their own grain |
| `engine/evidence.py` | Retrieval | No | TF-IDF similarity over UNSTRUCTURED text, query-expanded with the same cause vocabulary `confidence.py` uses, pre-filtered by region + time window |
| `engine/confidence.py` | Ranking | No | Weighted score: statistical strength + structured evidence + text relevance, with plausibility as a multiplicative GATE, not a fourth additive term. Empirically calibrated thresholds. |
| `engine/narrator.py` | Narrative | **Yes** | Only step that calls an LLM (Gemini) — phrases the structured facts, per persona. Template fallback is also genuinely persona-specific |
| `engine/actions.py` | Recommendation | No | Lookup table: driver → lever → action → owner → expected impact → monitoring plan. Action/owner vary by persona |
| `utils/calibration.py` | Meta | No | Sweeps confidence thresholds against labeled synthetic scenarios, reports precision/coverage |

## Fixes made after a close technical review — each one, in detail

This project went through an external review that flagged five specific
issues. Rather than paper over them, here's exactly what was wrong and
what changed. If asked, this is the level of detail to answer with.

### 1. Attribution formula — verified, and fixed an ambiguity

The price/volume bridge decomposes `Revenue = Price x Volume` into a
volume-driven part and a price-driven part. Any such two-factor bridge has
an unavoidable **interaction term** (`delta_Price x delta_Volume`) that
has to be assigned somewhere. The original formula:

```
volume_contrib = delta_volume * price_BASE
price_contrib  = delta_price  * volume_NOW
```

sums exactly to the true revenue change (this is algebraically
guaranteed — verified with randomized test cases in
`tests/test_engine.py`), but it silently dumps the **entire** interaction
term into whichever factor is multiplied by the "now" value. Compute it in
the other order and you'd get a different split of the same total — an
arbitrary, order-dependent choice.

The fix: a **symmetric (midpoint) bridge**, which splits the interaction
term evenly instead of assigning it all to one side:

```
volume_contrib = delta_volume * average(price_base, price_now)
price_contrib  = delta_price  * average(volume_base, volume_now)
```

This still sums exactly to the true revenue change, and — unlike the
original — gives the *same* split regardless of computation order. Both
properties are covered by tests. The app also displays an independent
cross-check column (`actual_revenue_change`, computed directly as
`price_now×volume_now − price_base×volume_base`) right next to the
bridge's `total_change`, so the match is visible in the demo itself, not
just claimed in code comments.

### 2. Evidence relevance — a real bug, not just a display problem

Investigating the "score presentation needs work" note surfaced something
more serious than formatting: two of the three genuinely relevant
documents in the demo corpus scored **literally 0.0** relevance, because
word-level TF-IDF shares zero tokens between a query for "stockout" and a
document that only ever says "out of stock" or "restock." That's a real
retrieval failure, not a cosmetic issue.

Two fixes:
- **Query expansion**: the retrieval query is expanded with the exact same
  cause-vocabulary `engine/confidence.py` already defines
  (`KEYWORD_TO_CAUSE`), rather than inventing a second, separate synonym
  list. One vocabulary, used consistently by both retrieval and ranking.
  (We also tested switching to character n-gram TF-IDF as a fuzzier
  alternative — it fixed the vocabulary mismatch but inflated irrelevant
  documents' scores too, so we rejected it in favor of the vocabulary-tied
  fix.)
- **Relative + qualitative scoring for presentation**: raw cosine
  similarity on short documents is naturally small (0.03–0.15) even for a
  strong match. Each document now also reports `relative_relevance` (its
  score divided by the best score among the already-filtered, already-
  floor-passing candidates) and a `relevance_label` (High/Medium/Low). The
  raw score is still reported for audit purposes, but the *scoring math*
  in `engine/confidence.py` now uses `relative_relevance`, since raw
  cosine values aren't on a comparable scale to the other [0,1] signals
  being combined.

### 3. Cause ranking — fixed a real double-counting issue

The original scoring combined four things additively: statistical
strength, structured evidence, text evidence, **and** a "plausibility"
score (does this cause's typical effect match the kind of change we saw?).
The bug: plausibility is *derived from* the same `dominant_driver_type`
that the statistical and structured signals are already responding to —
it isn't independent evidence, so adding it as a fourth term inflated the
score with information that wasn't actually new.

Fix: plausibility is now a **multiplicative gate** (`IMPLAUSIBLE_PENALTY`,
applied to the weighted sum of the three genuinely independent signals),
not a fourth additive term. An implausible cause gets its score heavily
discounted rather than an plausible one getting boosted. Covered by a
regression test that checks two causes with identical statistical/
structured/retrieval inputs differ *only* by the gate's discount factor,
not by some larger, uncontrolled amount.

### 4. Causal reasoning claim — was overstated, now stated precisely

Language like "not just correlation" was doing more work than the system
actually does. To be precise: **this is not causal inference.** There's no
counterfactual modeling, no control group, no do-calculus, no causal graph
learned from data. What the pipeline actually does is narrow candidates
through temporal + segment alignment (a cause can only be considered if
it's evidenced in the same time window and location as the anomaly) and
weigh multiple, largely-independent evidence types before ranking. That's
a meaningfully stronger bar than raw correlation, but the accurate
description of the output is **"evidence-weighted association,"** not
"proven cause." This is now stated explicitly in `engine/confidence.py`'s
docstring and in the app's ranking section, rather than implied by softer
language.

### 5. Confidence — was genuinely inconsistent, now disambiguated

Two unrelated notions of "confidence" existed side by side: `engine/
anomaly.py` returned a field called `confidence` meaning "how much do we
trust this detection given how much history we have," while `engine/
confidence.py` returned a `status` meaning "how sure are we about which
cause is responsible." Same word, two different questions — a real naming
collision that made the two stages sound connected when they aren't.
`engine/anomaly.py` now returns `data_sufficiency` instead. The
thresholds themselves were also recalibrated (see
`docs/calibration_results.md`) after the plausibility-gate fix changed the
scoring function — a threshold is a property of the scoring function, not
a constant that stays valid across changes to it.

## Why reconciliation is its own stage, not folded into retrieval

Round 2 objective #2 asks the engine to "reconcile data and business
context across heterogeneous sources." Text retrieval alone doesn't do
this — a support ticket saying "we're out of stock" is a person's claim,
not a system-of-record fact. `engine/reconciliation.py` pulls the actual
inventory system's own numbers (batch, ~3-day cadence) and the marketing
system's own numbers (weekly), each reported with its own freshness/grain.
In the demo: `inventory.csv` shows stock at N1/N2 dropping to near-zero
during the anomaly window (structured confirmation, independent of and
weighted higher than the support ticket saying the same thing in words).
`promo.csv` shows marketing spend essentially flat through the same
window — a deliberate **negative control**, testing that the pipeline
correctly does *not* blame marketing for something marketing didn't cause.

## Access control — enforced, not just described

`kpi_contract.yaml` defines `can_view_grains` and `hidden_fields` per role.
`app.py` enforces both: `regional_ops_manager` sees the store-level table
with `price` dropped; `cfo` never sees store-level data at all, only a
region-level rollup. Checked directly in `tests/test_engine.py` and in
`app.py`'s `apply_access_control()` function.

## Known limitations (current prototype)

- Feedback loop (analyst confirms/corrects a cause, re-ranking improves) is
  not yet implemented
- Localization is single-dimension (stores within one region+SKU), not the
  full multi-dimension combinatorial search real systems like HotSpot/
  Squeeze perform
- The plausibility map is hand-built, not a learned causal graph
- Access control is enforced at the display layer (Streamlit), not at a
  query/database layer
- "Contradictory evidence" (two sources actively disagreeing) isn't
  separately demonstrated yet — only "insufficient evidence"
- No running counter of total model calls across a session — token/cost/
  latency are tracked per-call, not aggregated
