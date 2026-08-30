CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Installation
 * Configuration
 * How it works
 * What changed after review (read this before a demo)
 * Known issues / TODO
 * Maintainers


INTRODUCTION
------------

This is our prototype for the Accenture Innovation Challenge, Round 2,
BusinessIntelligence.ai track. The brief asks for a "KPI intelligence-to-
action engine" that can look at a business metric, tell you if the move is
real or just noise, figure out why it happened, and say what to do about
it — including when it honestly doesn't know.

Round 1 was just the idea. This is us actually building a slice of it, and
this version has been through a close technical review that flagged five
real issues (not nitpicks) — all fixed, and documented below rather than
quietly patched.

Short version of what it does: you give it a region + SKU, it checks
whether revenue moved for a real reason, figures out which store(s) are
responsible, pulls in structured context from OTHER systems (inventory,
marketing spend — different refresh cadences, on purpose), goes and finds
supporting evidence in some news/ticket/CRM text files, scores how
confident it actually is, and either gives you an answer or tells you it's
not sure and shows the top candidates instead of making something up. The
answer and the recommended action both change depending on who's asking —
a regional ops manager and a CFO get genuinely different views of the same
underlying event.

We didn't want to just wire an LLM up to a spreadsheet and call it a day —
that's the thing the brief explicitly warns against. So most of this is
plain statistics, information theory, and rules, and the LLM only shows up
in exactly one file (`engine/narrator.py`), and only to phrase things in
English, not to decide anything. Narration uses Gemini.


REQUIREMENTS
------------

 * Python 3.10+ (we built/tested this on 3.12)
 * pip
 * The packages in requirements.txt — pandas, numpy, scikit-learn,
   statsmodels, pyyaml, streamlit, google-genai, pytest
 * A Gemini API key IF you want real LLM narration. Not required to run
   the demo — there's a fallback (see Configuration below).


INSTALLATION
------------

1. Get the code onto your machine (unzip it, or clone if we've pushed it
   to a repo by the time you're reading this).

2. Install dependencies:

   pip install -r requirements.txt

3. Generate the demo data. We don't ship the CSVs in the repo — the
   generator builds them fresh every time so it's reproducible and small
   to hand around:

   python utils/data_generator.py

   You should see something like "Generated 910 sales rows, 24 promo rows,
   300 inventory rows, 5 docs." If you don't, something's wrong with the
   environment, not the logic — check your Python version first.

4. Run it:

   streamlit run app.py

   It'll print a local URL, usually http://localhost:8501. Open that.

5. (Optional, do this before you actually demo it) — run the tests to make
   sure nothing's broken on your machine:

   pytest tests/

   Should say "21 passed." If it doesn't, don't panic, read the failure —
   most likely cause is you skipped step 3.


CONFIGURATION
------------

The only real config file is kpi_contract.yaml at the project root. This
is the "semantic layer" the brief asks for — definitions, grains,
thresholds, who can see what, per-KPI ownership. Four connected KPIs
(revenue, units_sold, marketing_spend, inventory_level) across three
sources with three different refresh cadences — daily, weekly, and an
irregular ~3-day batch.

LLM narration (Gemini):

By default, if you haven't set an API key, the narrative step falls back
to a plain Python template. It still works, it's just not using a model —
and it says so on screen, it doesn't pretend. The template is genuinely
persona-specific too. To turn on real narration:

   export GEMINI_API_KEY=your-key-here
   streamlit run app.py

Optionally set GEMINI_MODEL to override the default ("gemini-2.0-flash")
if you want a different Gemini model. Do this before you record the
actual demo video, not during it.

Confidence thresholds:

The "how confident do we need to be before we answer" numbers live in
kpi_contract.yaml under confidence_bands. We didn't just pick these — run
utils/calibration.py and it'll build labeled fake scenarios matching the
real scoring function (including the plausibility-gate fix — see below),
sweep threshold values, and spit out a precision/coverage table. Results
get written to docs/calibration_results.md. If you touch the scoring
weights in engine/confidence.py, re-run this and update the yaml to match.


HOW IT WORKS
------------

Seven stages, one file each, under engine/:

anomaly.py         - is this change real, or just noise? Fits a seasonal
                     model on data BEFORE the window we're checking,
                     forecasts what "normal" should've looked like, and
                     flags it if the actual number is way off. Requires
                     the move to be big enough in dollar terms too, not
                     just statistically weird. Returns "data_sufficiency"
                     (renamed from "confidence" — see below).

attribution.py     - if it's real, where's it coming from? Breaks the
                     change down store by store, finds the smallest set of
                     stores that explain most of it, then splits the
                     change into a pricing part and a volume part using a
                     SYMMETRIC bridge formula (see below — this was fixed
                     after review).

reconciliation.py  - pulls in STRUCTURED context from the other two data
                     sources (inventory, marketing spend), each on its own
                     grain/cadence. This is what makes "stockout" a fact
                     confirmed by the inventory system, not just a guess
                     from a support ticket's wording. Marketing spend is a
                     deliberate negative control — it doesn't move during
                     the anomaly, so the pipeline has to correctly NOT
                     blame it for anything.

evidence.py        - goes looking for supporting text but only in the same
                     time window and region as the anomaly. The retrieval
                     query is expanded with the same cause-vocabulary
                     confidence.py uses, so "stockout" also catches a
                     document that only says "out of stock" (this was a
                     real bug — see below). Also reports a relative
                     relevance score and a High/Medium/Low label, since
                     raw similarity scores on short documents are small
                     even for a strong match.

confidence.py      - combines statistical strength, structured evidence,
                     and text evidence into one score. Plausibility (does
                     this cause even make sense given what changed?) is a
                     GATE that discounts implausible causes, not a fourth
                     thing added to the score — an earlier version double-
                     counted it (see below). Explicitly documented as NOT
                     causal inference — it's evidence-weighted association
                     narrowed by time/place alignment, nothing more.

narrator.py        - the only file that talks to an LLM (Gemini). Takes
                     whatever the above stages figured out and turns it
                     into a sentence a human can read, framed for whichever
                     persona asked. Falls back to a persona-specific
                     template if there's no API key.

actions.py         - a lookup table. Stockout -> expedite shipment (ops
                     manager) or -> flag as a temporary supply-side dip
                     (CFO). Same underlying driver, different action and
                     owner depending on who's asking.

app.py at the root wires all of this into a Streamlit page and actually
ENFORCES the access control from kpi_contract.yaml: a regional ops manager
sees the store-level table with pricing dropped; a CFO never sees store-
level detail at all, only a region-level rollup — with an independent
cross-check number shown so the attribution formula's correctness is
visible in the demo, not just claimed.

utils/data_generator.py builds fake sales/promo/inventory/document data
with a stockout in 2 of 3 North region stores (confirmed independently in
both the inventory numbers and a support ticket), a competitor promo in a
news file, and a marketing-spend series that deliberately doesn't move
(the negative control). Plus a brand new South-region SKU with only 5 days
of history, for the sparse-data case.

docs/research_references.md has the papers/techniques we borrowed from,
including an explicit section on what this system does NOT do (formal
causal inference) — worth reading before anyone asks "did you just make
this up" or "is this actually causal reasoning."


WHAT CHANGED AFTER REVIEW (read this before a demo)
------------

This prototype went through a close technical review. Five real issues
were flagged and fixed — not just reworded. If a judge asks "what did you
get wrong and fix," this is the honest, specific answer:

1. Attribution formula: the price/volume bridge had an unflagged ambiguity
   (it silently assigned an entire interaction term to one factor,
   depending on computation order). Fixed with a symmetric formula that's
   order-independent, verified by tests, and cross-checked live in the UI.

2. Evidence relevance: two of three genuinely relevant documents scored a
   literal 0.0 due to word-level vocabulary mismatch ("stockout" vs. "out
   of stock" share no tokens). Fixed by expanding the retrieval query with
   the same cause-vocabulary the ranking stage already uses. Also added
   relative/qualitative scoring since raw cosine scores are hard to read
   in isolation.

3. Cause ranking: plausibility was being added as a fourth "independent"
   evidence source, but it's derived from the same signal the statistical
   and structured components already use — double-counting. Fixed by
   making plausibility a multiplicative gate instead.

4. Causal reasoning claim: language like "not just correlation" implied
   more than the system does. Now stated precisely: this is evidence-
   weighted association narrowed by time/place alignment, NOT causal
   inference. No counterfactual modeling, no control group, no do-calculus.

5. Confidence terminology was genuinely inconsistent: two unrelated
   concepts ("how much history do we have" and "how sure are we about the
   cause") both used the word "confidence." Renamed the detection-stage
   field to "data_sufficiency." Thresholds were also recalibrated after
   fix #3 changed the scoring function.

Also switched the LLM provider from Anthropic to Gemini, using Google's
current `google-genai` SDK (not the older `google-generativeai` package,
which Google has deprecated).


KNOWN ISSUES / TODO
------------

Being upfront about what's NOT done:

 * No real feedback loop. The brief wants the system to learn from analyst
   corrections over time — not wired up yet.

 * Attribution only looks at one dimension (stores) at a time, not the
   full multi-dimension combinatorial search real systems perform.

 * The plausibility map is hand-built, not learned from data.

 * Access control is enforced in the Streamlit display layer, not at a
   query/database layer.

 * No genuine "contradictory evidence" scenario (two sources actively
   disagreeing) — only "insufficient evidence" is demonstrated.

 * No running total of model calls across a session — latency/tokens/cost
   are logged per call, not aggregated.

 * We have not independently verified Gemini's exact current per-token
   pricing for the cost estimate in narrator.py — it's a clearly-labeled
   placeholder rate, not a number we'd defend to the decimal.


MAINTAINERS
-----------

Team Lucids (NIT Calicut, Electrical & Electronics Engineering, 2027):
 * Aditya Sharma - Team Lead
 * Arjun K M
 * Arun A

Current module owners (update if this changes):
 * Detection + attribution + reconciliation - [assign]
 * Evidence + confidence   - [assign]
 * Narrative + actions + UI - [assign]

Questions or if something's broken and you can't figure out why: check
git blame first, then ask in the group chat before you rewrite someone
else's file.
