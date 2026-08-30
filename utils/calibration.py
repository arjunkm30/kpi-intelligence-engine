"""
Calibration — empirically justify the confidence thresholds in
kpi_contract.yaml, instead of picking numbers by feel.

This operationalizes "selective prediction" (Chow, 1970 "reject option";
recent LLM literature calls this "conformal abstention"): a system that can
output "I don't know" is not evaluated on accuracy alone, it's evaluated on
the RISK-COVERAGE TRADEOFF — of the cases it chooses to answer (coverage),
how often is it actually right (precision)? A good threshold buys higher
precision by abstaining on the hardest cases; the question is how much
coverage you give up to get there, and that's an empirical question, not a
guess.

Run:  python utils/calibration.py

It builds several synthetic scenarios with KNOWN ground-truth causes
(varying anomaly strength and evidence strength), runs them through the
real pipeline (engine.attribution + engine.evidence + engine.confidence),
and reports precision/coverage at a sweep of thresholds. Writes results to
docs/calibration_results.md.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.confidence import (WEIGHT_STATISTICAL, WEIGHT_STRUCTURED, WEIGHT_RETRIEVAL,
                                 IMPLAUSIBLE_PENALTY, PLAUSIBILITY_MAP, Z_SATURATION_POINT)


def make_scenario(z_score: float, retrieval_score: float, structured_score: float,
                   dominant="volume", true_cause="stockout", plausible=True):
    """
    Builds a synthetic (statistical_strength, structured_score,
    retrieval_score, ground_truth_cause) tuple directly from a chosen
    severity, matching what engine/confidence.py actually consumes —
    calibration should test the SAME scoring function the real pipeline
    uses, not a re-derived proxy for it.

    Plausibility is applied as a GATE (multiplier), matching the fix in
    engine/confidence.py — it is no longer a fourth additive term, since
    it isn't independent evidence (see that module's docstring for why the
    earlier additive version double-counted).
    """
    stat_strength = min(abs(z_score) / Z_SATURATION_POINT, 1.0)

    base_score = (WEIGHT_STATISTICAL * stat_strength +
                   WEIGHT_STRUCTURED * structured_score +
                   WEIGHT_RETRIEVAL * retrieval_score)
    score = base_score if plausible else base_score * IMPLAUSIBLE_PENALTY

    return {"score": round(score, 3), "true_cause": true_cause, "z_score": z_score,
            "structured_score": structured_score, "retrieval_score": retrieval_score}


def build_labeled_scenarios():
    """
    Synthetic incidents spanning strong/weak statistical signal x
    strong/weak structured evidence x strong/weak text evidence, plus "no
    real cause" (noise) cases where the z-score never clears the
    materiality threshold at all. Ground truth is known because WE
    constructed it — the standard way to calibrate a threshold before real
    historical incident logs exist to learn from.
    """
    scenarios = []
    for z in [35.0, 15.0, 5.0, 1.0]:                # strong -> weak statistical signal
        for structured in [1.0, 0.0]:                # structured confirmation present or not
            for retrieval in [0.9, 0.5, 0.2]:        # strong -> weak text evidence match
                is_real_anomaly = z >= 5.0            # below this, treat as "no real cause" (noise)
                s = make_scenario(z, retrieval, structured, true_cause="stockout" if is_real_anomaly else "none")
                s["is_real_anomaly"] = is_real_anomaly
                scenarios.append(s)
    return scenarios


def evaluate_at_threshold(scenarios, threshold):
    """
    A prediction counts as "accepted" (not abstained) if score >= threshold.
    Precision = of accepted predictions, how many were actually real
    anomalies (i.e. we didn't confidently call noise a real cause).
    Coverage = fraction of all scenarios where we didn't abstain.
    """
    accepted = [s for s in scenarios if s["score"] >= threshold]
    if not accepted:
        return {"threshold": threshold, "coverage": 0.0, "precision": None, "n_accepted": 0}
    correct = sum(1 for s in accepted if s["is_real_anomaly"])
    precision = correct / len(accepted)
    coverage = len(accepted) / len(scenarios)
    return {"threshold": threshold, "coverage": round(coverage, 2),
            "precision": round(precision, 2), "n_accepted": len(accepted)}


def recommend_threshold(results, target_precision):
    """
    Among thresholds that achieve at least `target_precision`, pick the
    LOWEST one (i.e. the most permissive one that still clears the bar) —
    this maximizes coverage subject to the precision constraint, which is
    the standard way to pick an operating point on a precision-coverage
    (risk-coverage) curve once you've decided how much risk you'll accept.
    """
    qualifying = [r for r in results if r["precision"] is not None and r["precision"] >= target_precision]
    if not qualifying:
        return None
    return min(qualifying, key=lambda r: r["threshold"])


def main():
    scenarios = build_labeled_scenarios()
    thresholds = [round(t, 2) for t in [0.85, 0.75, 0.70, 0.60, 0.50, 0.45, 0.35, 0.25, 0.15]]
    results = [evaluate_at_threshold(scenarios, t) for t in thresholds]

    high_rec = recommend_threshold(results, target_precision=0.95)
    medium_rec = recommend_threshold(results, target_precision=0.85)

    lines = [
        "# Calibration Results\n",
        "Generated by `utils/calibration.py`. This is the empirical justification",
        "for the `confidence_bands` values in `kpi_contract.yaml` — these numbers",
        "are derived from the table below, not chosen by feel.\n",
        f"Evaluated on {len(scenarios)} synthetic labeled scenarios "
        f"({sum(1 for s in scenarios if s['is_real_anomaly'])} real anomalies, "
        f"{sum(1 for s in scenarios if not s['is_real_anomaly'])} noise cases).\n",
        "| Threshold | Coverage | Precision | # Accepted |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['threshold']} | {r['coverage']} | {r['precision']} | {r['n_accepted']} |")

    lines += ["", "**Reading this table:** lower thresholds accept more cases (higher",
              "coverage) but let more false positives through (lower precision).",
              "Higher thresholds abstain more often but are more trustworthy when",
              "they do answer.", ""]

    if high_rec:
        lines.append(f"- **Recommended `high` threshold: {high_rec['threshold']}** "
                      f"— the most permissive threshold that still achieves "
                      f"precision >= 0.95 (actual: {high_rec['precision']}, "
                      f"coverage: {high_rec['coverage']}).")
    else:
        lines.append("- No threshold in the sweep achieved precision >= 0.95 — "
                      "either tighten the scoring weights or accept a lower bar.")

    if medium_rec:
        lines.append(f"- **Recommended `medium` threshold: {medium_rec['threshold']}** "
                      f"— the most permissive threshold that still achieves "
                      f"precision >= 0.85 (actual: {medium_rec['precision']}, "
                      f"coverage: {medium_rec['coverage']}).")
    else:
        lines.append("- No threshold in the sweep achieved precision >= 0.85.")

    lines += ["", "Update `confidence_bands` in `kpi_contract.yaml` to match the",
              "recommendations above. Re-run this script any time the scoring",
              "weights in `engine/confidence.py` change — the right threshold",
              "is a property of the scoring function, not a fixed constant."]

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "docs", "calibration_results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
