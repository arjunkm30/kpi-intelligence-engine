# Business Proposal

## Problem Statement

Business dashboards are excellent at showing that something changed — a KPI
graph turns red, a number drops 8 percent. But they stop there. When a
leader asks "why did this happen, and what should we do next," the burden
falls entirely on a human analyst, who must manually cross-reference sales
data, inventory records, marketing calendars, competitor news, and support
tickets to reconstruct an explanation. This investigation routinely takes
days, by which time the business has already lost its window to react.

The core problem is not a shortage of data. Most organizations already
capture rich structured metrics — revenue, pricing, inventory, CRM —
alongside a growing volume of unstructured signals such as news, emails,
reviews, and support tickets. What is missing is a translation layer that
connects a statistical anomaly to its underlying business cause, in language
a non-technical leader can trust and act on.

This is made harder because not every fluctuation is meaningful. Much of
what looks like a drop is normal week-to-week variance, seasonality, or a
one-off event, and treating every wiggle as a crisis erodes trust. Leaders
need an engine that separates signal from noise, builds an evidence-backed
explanation rather than a guess, and stays honest about uncertainty when the
true cause is genuinely unclear.

## Proposed Solution

We propose a KPI Storytelling Engine that turns a metric anomaly into a
three-part narrative: what changed, why it likely changed, and what to do
next.

First, a baseline-and-decomposition layer separates real signal from noise —
modeling each KPI's own seasonality and volatility so deviations are flagged
against what was actually expected for that segment, not a blanket
threshold. Second, a driver-attribution layer decomposes the metric
algebraically (revenue = price times volume, summed across regions or
stores) to localize exactly where and what kind of change occurred, before
any cause is proposed. Third, a retrieval layer pulls unstructured evidence
— news, CRM notes, promotions, support tickets — strictly filtered to the
same time window and segment as the localized change, then ranks candidate
causes by statistical strength, historical precedent, and business
plausibility.

An LLM narrator converts this evidence into a plain-language explanation
with a confidence score and a recommended next action drawn from a
driver-to-playbook map, so leaders get something actionable, not just
informative. Critically, when no explanation clears a confidence threshold,
the engine says so — surfacing the top competing hypotheses and a
diagnostic next step instead of a guess. This compresses a multi-day analyst
investigation into a same-day, trustworthy answer.

## Why it matters

The win isn't full automation — it's compressing a multi-day investigation
into a same-day review. The engine hands an analyst or a business leader a
narrowed, evidence-backed starting point instead of a blank slate, and is
explicitly designed to say "I don't know" rather than fabricate confidence
when the data genuinely doesn't support a clear answer.
