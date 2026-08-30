"""
Confidence / ranking — combine statistical strength (from detection),
STRUCTURED cross-source evidence (from reconciliation), and unstructured
text evidence (from retrieval) into a ranked, scored list of candidate
causes. NOT LLM: weighted scoring, no model call.

IMPORTANT HONESTY NOTE -- what this stage does NOT do:
This is NOT formal causal inference. There's no counterfactual modeling, no
control group, no do-calculus, no causal graph learned from data. What it
DOES do is narrow candidates through temporal + segment alignment (a cause
can only be considered if it's evidenced in the same time window and
location as the anomaly) and weigh multiple, largely-independent evidence
types before ranking. That's a meaningfully stronger bar than raw
correlation ("X happened and Y happened around the same time"), but the
right way to describe the output is "evidence-weighted association,"
not "proven cause." Say this plainly if asked -- overclaiming causal
inference here is exactly the kind of thing that doesn't survive a close
technical read.

FIX -- double-counting (from review): an earlier version of this module
added a "plausibility" score (does this cause's typical effect match the
kind of change we saw -- e.g. does "stockout" typically cause a volume
drop?) as a FOURTH additive term alongside statistical/structured/
retrieval. That's a mistake: plausibility is DERIVED from the same
dominant_driver_type that the statistical and structured signals are
already responding to, so adding it as independent evidence inflates the
score with information that isn't actually new. Plausibility is now a
GATE (a multiplier), not a fourth independent evidence source: an
implausible cause gets its score heavily discounted rather than boosted
for being plausible. Being "plausible" is a necessary sanity check, not
positive evidence on its own.

Framing: the abstention behavior implements "selective prediction" / the
"reject option" (Chow, 1970) -- a system that is allowed to output "I
don't know" rather than being forced to always answer. The confidence
threshold is not picked by feel; see utils/calibration.py and
docs/calibration_results.md.
"""
from engine import CONTRACT

PLAUSIBILITY_MAP = {
    "stockout": ["volume_drop"],
    "competitor_promo": ["volume_drop", "price_pressure"],
    "supply_delay": ["volume_drop"],
    "reduced_marketing": ["volume_drop"],
}

KEYWORD_TO_CAUSE = {
    "out of stock": "stockout",
    "stockout": "stockout",
    "restock": "stockout",
    "backlog": "supply_delay",
    "discount campaign": "competitor_promo",
    "competitor": "competitor_promo",
}

# Weights for the THREE independent evidence types. Plausibility is
# deliberately excluded from this list -- see module docstring, it's a
# gate applied after this weighted sum, not a fourth additive term.
WEIGHT_STATISTICAL = 0.35
WEIGHT_STRUCTURED = 0.40
WEIGHT_RETRIEVAL = 0.25

IMPLAUSIBLE_PENALTY = 0.3  # multiplier applied when a cause fails the plausibility gate

Z_SATURATION_POINT = 10.0  # matches the materiality threshold's own scale, see engine/anomaly.py


def _statistical_strength(z_score: float) -> float:
    """
    NOTE ON SHARED SIGNAL: this value is the SAME for every candidate cause
    in a given analysis -- it answers "did something real happen at all,"
    not "which cause is responsible." It's intentionally cause-agnostic;
    what differentiates candidates is the structured and retrieval
    components, and the plausibility gate below.
    """
    if z_score is None:
        return 0.0
    return min(abs(z_score) / Z_SATURATION_POINT, 1.0)


def _is_plausible(cause: str, dominant_driver_type: str) -> bool:
    plausible_effects = PLAUSIBILITY_MAP.get(cause, [])
    return f"{dominant_driver_type}_drop" in plausible_effects or \
           (dominant_driver_type == "price" and "price_pressure" in plausible_effects)


def _structured_signals(reconciliation: dict) -> dict:
    """
    Turns engine/reconciliation.py's output into a {cause: strength 0-1} map.
    This is deliberately separate from text retrieval -- a cause can be
    raised by structured data even if no document happens to mention it.
    """
    signals = {}
    inv = (reconciliation or {}).get("inventory", {})
    if inv.get("available") and inv.get("any_near_zero"):
        signals["stockout"] = 1.0  # hard data, not a text guess -> full strength

    mkt = (reconciliation or {}).get("marketing", {})
    if mkt.get("available") and mkt.get("material_shift") and mkt.get("pct_change", 0) < 0:
        signals["reduced_marketing"] = min(abs(mkt["pct_change"]) / 30.0, 1.0)

    return signals


def rank_causes(detection: dict, attribution: dict, evidence: list, reconciliation: dict = None):
    dominant = attribution["dominant_driver_type"]
    stat_strength = _statistical_strength(detection.get("z_score"))
    structured = _structured_signals(reconciliation)
    candidates = {}

    def _upsert(cause, retrieval_score, structured_score, doc_label, snippet):
        base_score = (WEIGHT_STATISTICAL * stat_strength +
                       WEIGHT_STRUCTURED * structured_score +
                       WEIGHT_RETRIEVAL * retrieval_score)
        plausible = _is_plausible(cause, dominant)
        score = base_score if plausible else base_score * IMPLAUSIBLE_PENALTY

        if cause not in candidates or score > candidates[cause]["score"]:
            candidates[cause] = {
                "cause": cause,
                "score": round(score, 3),
                "statistical_component": round(stat_strength, 3),
                "structured_component": round(structured_score, 3),
                "retrieval_component": retrieval_score,
                "plausibility_gate_passed": plausible,
                "supporting_doc": doc_label,
                "evidence_snippet": snippet,
            }

    # Text-based candidates. Uses relative_relevance (see engine/evidence.py)
    # rather than raw cosine similarity -- raw scores on this corpus are
    # small in absolute terms and not directly comparable to the other
    # [0,1]-scaled signals being combined here.
    for doc in evidence:
        text_lower = doc["text"].lower()
        retrieval_score = doc.get("relative_relevance", doc.get("relevance_score", 0.0))
        for kw, cause in KEYWORD_TO_CAUSE.items():
            if kw in text_lower:
                _upsert(cause, retrieval_score, structured.get(cause, 0.0),
                        doc["file"], doc["text"].split("Text:")[-1].strip()[:160])

    # Structured-only candidates: a cause can surface from hard data even if
    # no document happens to mention it -- this is the point of reconciliation.
    for cause, strength in structured.items():
        if cause not in candidates:
            _upsert(cause, 0.0, strength, "inventory.csv / promo.csv (structured)",
                    "Confirmed directly from structured source data, no supporting document required.")

    ranked = sorted(candidates.values(), key=lambda c: -c["score"])

    bands = CONTRACT["confidence_bands"]
    if not ranked:
        return {"status": "abstain", "reason": "no_supporting_evidence_found", "hypotheses": []}

    top_score = ranked[0]["score"]
    if top_score >= bands["high"]:
        status = "confident"
    elif top_score >= bands["medium"]:
        status = "leading_hypothesis"
    else:
        status = "abstain"

    return {"status": status, "hypotheses": ranked[:3]}
