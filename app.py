import os
import pandas as pd
import streamlit as st

from engine import CONTRACT, DATA_DIR
from engine.anomaly import detect_anomaly
from engine.attribution import attribute_change
from engine.evidence import retrieve_evidence
from engine.reconciliation import reconcile_context
from engine.confidence import rank_causes
from engine.narrator import generate_narrative
from engine.actions import recommend_action

st.set_page_config(page_title="KPI Intelligence-to-Action Engine", layout="wide")
st.title("KPI Intelligence-to-Action Engine")
st.caption("Every box below is labeled LLM or NOT-LLM — that's the point.")

sales_path = os.path.join(DATA_DIR, "sales.csv")
if not os.path.exists(sales_path):
    st.error("No data found. Run `python utils/data_generator.py` first, then reload this page.")
    st.stop()

sales = pd.read_csv(sales_path)

# --- Sidebar controls ---
st.sidebar.header("Scenario")
scenario = st.sidebar.selectbox(
    "Pick a demo case",
    ["North / SKU-A (main anomaly)", "South / SKU-C (sparse history)"],
)
persona = st.sidebar.selectbox("Persona", ["regional_ops_manager", "cfo"])
region = "North" if "North" in scenario else "South"
sku = "SKU-A" if "SKU-A" in scenario else "SKU-C"
role_rules = CONTRACT["access_control"]["roles"][persona]

st.sidebar.markdown("---")
st.sidebar.caption(f"**Role:** {persona}")
st.sidebar.caption(f"Can view grains: {', '.join(role_rules['can_view_grains'])}")
if role_rules.get("hidden_fields"):
    st.sidebar.caption(f"Hidden fields: {', '.join(role_rules['hidden_fields'])}")
st.sidebar.caption(f"Narrative focus: {role_rules['narrative_focus']}")

run = st.sidebar.button("Run analysis", type="primary")


def apply_access_control(df: pd.DataFrame, role_rules: dict) -> pd.DataFrame:
    """
    ENFORCED, not just described: drops any column named in the role's
    hidden_fields. This is deliberately dumb column-name matching -- a real
    system would do this at the query layer, but the point for the demo is
    that the same underlying data produces a visibly different table
    depending on who's asking, not just a caption saying it would.
    """
    hidden = role_rules.get("hidden_fields", [])
    cols_to_drop = [c for c in df.columns if any(h in c for h in hidden)]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


if run:
    # 1. DETECTION -----------------------------------------------------
    st.subheader("1. Detection — signal vs. noise")
    st.caption("NOT LLM — STL forecast (fit on held-out history) + robust MAD z-score. "
                "See docs/architecture.md for why this beats naive residual testing.")
    det = detect_anomaly(sales, region=region, sku=sku)
    c1, c2, c3 = st.columns(3)
    c1.metric("Material change?", "Yes" if det.get("material") else "No")
    c2.metric("% change vs baseline", f"{det.get('pct_change', 'n/a')}%")
    c3.metric("History available", f"{det.get('history_days', 'n/a')} days")
    if "statistically_significant" in det:
        st.caption(f"Materiality requires BOTH tests to pass: "
                    f"statistically significant = {det['statistically_significant']}, "
                    f"business significant (>{CONTRACT['kpis']['revenue']['materiality_min_pct_change']}% change) "
                    f"= {det['business_significant']}")
    if det.get("sparse_history"):
        st.warning("Sparse-history KPI — data sufficiency is LOW (not enough history to trust "
                    "the detection fully). This is the sparse-data demo scenario.")
    st.json(det)

    if not det.get("material") and not det.get("sparse_history"):
        st.info("No material change detected — pipeline stops here. This IS correct "
                 "behavior: not every wiggle deserves an investigation.")
        st.stop()

    # 2. ATTRIBUTION -----------------------------------------------------
    st.subheader("2. Attribution — where exactly, and what kind of change")
    st.caption("NOT LLM — Adtributor-style localization (Bhagwan et al., 2014) "
                "+ a SYMMETRIC price/volume bridge (splits the interaction term evenly "
                "instead of dumping it all into one factor — see docs/architecture.md)")
    attr = attribute_change(sales, region=region, sku=sku)
    loc = attr.get("localization", {})
    if loc:
        st.write(f"**Root-cause elements** (smallest set explaining "
                  f"\u2265{int(loc['teep_threshold']*100)}% of the change): "
                  f"{', '.join(loc['root_cause_elements'])} "
                  f"(cumulative explanatory power: {loc['cumulative_explanatory_power']})")
        with st.expander("View full Adtributor localization (explanatory power + surprise per store)"):
            st.dataframe(pd.DataFrame(loc["ranked_elements"]), use_container_width=True)

    st.write(f"**Dominant driver type:** {attr['dominant_driver_type']}")

    # --- ACCESS CONTROL, ENFORCED (not just captioned) ---
    if "store" not in role_rules["can_view_grains"] and "region" in role_rules["can_view_grains"]:
        # CFO-style role: region-level only, no store-level breakdown at all
        region_rollup = pd.DataFrame(attr["by_store"])[
            ["volume_contribution", "price_contribution", "total_change", "actual_revenue_change"]
        ].sum().to_frame().T
        st.info(f"🔒 Role '{persona}' is restricted to region-level grain — store-level breakdown is "
                 "not shown, per kpi_contract.yaml.")
        st.dataframe(apply_access_control(region_rollup, role_rules), use_container_width=True)
    else:
        store_df = apply_access_control(pd.DataFrame(attr["by_store"]), role_rules)
        if role_rules.get("hidden_fields"):
            st.caption(f"🔒 Columns hidden for role '{persona}': {', '.join(role_rules['hidden_fields'])} "
                        "(enforced above, not just described)")
        st.dataframe(store_df, use_container_width=True)
    st.caption("`actual_revenue_change` is an independent cross-check (price_now\u00d7volume_now \u2212 "
                "price_base\u00d7volume_base) shown next to `total_change` from the bridge \u2014 they match "
                "exactly, which is how you verify the decomposition formula rather than take it on faith.")

    # 3. RECONCILIATION --------------------------------------------------
    st.subheader("3. Reconciliation — structured context from OTHER sources")
    st.caption("NOT LLM — pulls promo.csv (weekly) and inventory.csv (batch, ~every 3 days) "
                "and aligns them to this KPI's window, without pretending they're on the same clock")
    reconciliation = reconcile_context(region, sku, ["N1", "N2", "N3"] if region == "North" else ["S1", "S2"])

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        mkt = reconciliation["marketing"]
        st.markdown("**Marketing spend** (`promo.csv`, weekly grain)")
        if mkt.get("available"):
            st.metric("Spend change vs. trailing avg", f"{mkt['pct_change']}%",
                       help=f"As of {mkt['as_of']}")
            if mkt["material_shift"]:
                st.warning("Material shift in spend \u2014 counts as a candidate cause.")
            else:
                st.success("No material shift \u2014 correctly NOT treated as a cause (negative control).")
        else:
            st.info("No marketing data available for this segment.")
    with rcol2:
        inv = reconciliation["inventory"]
        st.markdown("**Inventory level** (`inventory.csv`, batch ~3-day grain)")
        if inv.get("available"):
            st.dataframe(pd.DataFrame(inv["by_store"]), use_container_width=True)
            if inv["any_near_zero"]:
                st.warning("Near-zero stock confirmed directly from inventory data \u2014 "
                            "this is structured evidence, not a text guess.")
        else:
            st.info("No inventory data available for this segment.")

    # 4. RETRIEVAL -----------------------------------------------------
    st.subheader("4. Retrieval — unstructured evidence, filtered first")
    st.caption("NOT LLM — TF-IDF similarity, pre-filtered by region + time window. Query is expanded "
                "with the same cause-vocabulary engine/confidence.py uses, so 'stockout' also catches "
                "a document that only says 'out of stock' or 'restock.'")
    evidence = retrieve_evidence(region=region, query="stockout competitor promotion supply delay")
    if evidence:
        display_df = pd.DataFrame(evidence)[["file", "date", "relevance_score", "relative_relevance", "relevance_label"]]
        display_df.columns = ["file", "date", "raw cosine score", "relative to best match", "label"]
        st.dataframe(display_df, use_container_width=True)
        st.caption("Raw cosine scores on short documents are naturally small (0.03\u20130.15) even for a "
                    "strong match \u2014 'relative to best match' and the qualitative label are what's "
                    "actually meaningful, not the raw number in isolation.")
        with st.expander("View retrieved document text"):
            for e in evidence:
                st.text(f"--- {e['file']} ---\n{e['text']}")
    else:
        st.info("No evidence documents matched this region/time window.")

    # 5. RANKING -----------------------------------------------------
    st.subheader("5. Ranking — candidate causes, scored by evidence (not proven causation)")
    st.caption("NOT LLM — weighted score: statistical strength + STRUCTURED evidence "
                "(reconciliation) + text evidence (retrieval). Plausibility is a GATE that "
                "discounts implausible causes, not a fourth additive score — an earlier version "
                "double-counted it, since plausibility isn't independent evidence. "
                "⚠️ This is evidence-weighted association narrowed by time/place alignment, "
                "NOT formal causal inference — no counterfactual modeling, no control group.")
    ranked = rank_causes(det, attr, evidence, reconciliation)
    status = ranked["status"]
    hyps = ranked["hypotheses"]

    MULTI_CAUSE_MARGIN = 0.08  # if top-2 scores are this close, treat as a genuine tie
    medium_band = CONTRACT["confidence_bands"]["medium"]
    is_tied = len(hyps) >= 2 and (hyps[0]["score"] - hyps[1]["score"]) < MULTI_CAUSE_MARGIN
    has_secondary_factor = len(hyps) >= 2 and not is_tied and hyps[1]["score"] >= medium_band

    if status == "abstain":
        st.warning("⚠️ ABSTAINING — evidence doesn't clear the confidence threshold. "
                    "Showing top hypotheses instead of forcing one answer.")
    elif is_tied:
        st.info(f"🔀 MULTIPLE CONTRIBUTING CAUSES — top {min(2, len(hyps))} hypotheses score within "
                 f"{MULTI_CAUSE_MARGIN} of each other. Presenting both rather than "
                 "arbitrarily picking a single winner.")
    elif status == "leading_hypothesis":
        st.info("Leading hypothesis identified, but flagged with caveat (medium confidence).")
    else:
        st.success("Confident explanation identified.")

    if has_secondary_factor:
        st.info(f"➕ SECONDARY CONTRIBUTING FACTOR — **{hyps[1]['cause']}** independently clears the "
                 f"leading-hypothesis bar (score {hyps[1]['score']}) even though **{hyps[0]['cause']}** "
                 f"is the dominant, structurally-confirmed cause. This is the multi-factor scenario: "
                 f"two real drivers, correctly weighted rather than treated as a coin flip.")

    for h in hyps:
        structured_tag = " 🔒 structured-confirmed" if h.get("structured_component", 0) > 0.5 else ""
        gate_note = "" if h.get("plausibility_gate_passed", True) else " ⚠️ discounted (failed plausibility gate)"
        st.write(f"**{h['cause']}**{structured_tag}{gate_note} — score: {h['score']} "
                  f"(statistical: {h['statistical_component']}, structured: {h['structured_component']}, "
                  f"text: {h['retrieval_component']})  \n"
                  f"_Evidence ({h['supporting_doc']}):_ {h['evidence_snippet']}")

    top_cause = ranked["hypotheses"][0]["cause"] if ranked["hypotheses"] else "unclear"

    # 6. NARRATIVE -----------------------------------------------------
    st.subheader("6. Narrative — plain-language explanation, per persona")
    st.caption("⚡ THE ONLY LLM STEP (Gemini) — phrases the facts above, invents nothing. "
                "Persona-specific even in template-fallback mode (no API key needed to see this work).")
    facts = {
        "region": region,
        "pct_change": det.get("pct_change"),
        "dominant_driver_type": attr["dominant_driver_type"],
        "top_cause": top_cause,
        "confidence_status": status,
    }
    narrative = generate_narrative(facts, persona=persona)
    st.markdown(f"> {narrative['narrative']}")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Method", narrative["method"])
    tcol2.metric("Latency", f"{narrative['latency_seconds']}s")
    tcol3.metric("Tokens used", narrative["tokens_used"])
    tcol4.metric("Est. cost", f"${narrative['estimated_cost_usd']}")

    # 7. RECOMMENDATION -----------------------------------------------------
    st.subheader("7. Recommendation — persona-specific action")
    st.caption("NOT LLM — driver → lever → action lookup table, action/owner vary by persona")
    rec = recommend_action(top_cause, status, persona=persona)
    st.json(rec)

else:
    st.info("Pick a scenario and a persona in the sidebar, then click **Run analysis**.")
    st.markdown("""
    **What each scenario demonstrates:**
    - *North / SKU-A* — the main case: a real anomaly, multi-store attribution,
      structured confirmation from inventory data, text evidence retrieval, and
      (depending on scoring) a confidence/abstention call
    - *South / SKU-C* — sparse-history KPI, launched 5 days ago — shows the
      engine correctly flagging low confidence due to limited data

    **What switching persona demonstrates:**
    - `regional_ops_manager` sees store-level detail (minus price) and an
      operational recommendation
    - `cfo` sees a region-level rollup only (no store breakdown — enforced,
      not just described) and a financial-framing recommendation
    """)
