"""
Recommendation — driver -> controllable lever -> action -> owner ->
expected impact -> monitoring plan. NOT LLM: a lookup table. The narrator
may phrase this nicely, but the mapping itself is not model-generated.

Persona-aware: the underlying driver/lever never changes, but WHICH action
and WHOSE name goes on it does -- an ops manager gets a today-fixable
operational step, a CFO gets the financial-monitoring version of the same
underlying issue. Same facts, different decision rights.
"""

RECOMMENDATION_MAP = {
    "stockout": {
        "lever": "replenishment speed",
        "expected_impact": "Partial volume recovery within 1-2 weeks",
        "monitoring_plan": "Track daily sell-through and stock levels for 5 days",
        "by_persona": {
            "regional_ops_manager": {
                "action": "Expedite shipment to affected stores; escalate to supplier",
                "owner": "Regional Ops Manager",
            },
            "cfo": {
                "action": "Flag as a temporary, explainable revenue dip (supply-side, not demand loss) "
                           "for this reporting cycle; no pricing or margin action needed yet",
                "owner": "Regional Finance",
            },
        },
    },
    "competitor_promo": {
        "lever": "local pricing / promotion",
        "expected_impact": "Partial demand recovery, magnitude uncertain",
        "monitoring_plan": "Compare footfall vs. competitor promo end date",
        "by_persona": {
            "regional_ops_manager": {
                "action": "Evaluate a targeted counter-promotion in affected stores",
                "owner": "Regional Marketing Lead",
            },
            "cfo": {
                "action": "Model margin impact of a counter-promotion before approving spend; "
                           "treat as demand-side risk, watch next 2 reporting cycles",
                "owner": "Regional Finance",
            },
        },
    },
    "supply_delay": {
        "lever": "supplier SLA",
        "expected_impact": "Depends on supplier response time",
        "monitoring_plan": "Daily supplier delivery status check",
        "by_persona": {
            "regional_ops_manager": {
                "action": "Escalate to supplier account manager; evaluate backup supplier",
                "owner": "Supply Chain Lead",
            },
            "cfo": {
                "action": "Flag potential SLA penalty recovery with supplier; monitor for repeat pattern "
                           "before treating as a recurring cost risk",
                "owner": "Regional Finance",
            },
        },
    },
    "reduced_marketing": {
        "lever": "marketing spend",
        "expected_impact": "Depends on how much spend actually shifted",
        "monitoring_plan": "Compare next period's spend vs. this period's",
        "by_persona": {
            "regional_ops_manager": {
                "action": "Confirm with marketing whether a planned spend cut caused this",
                "owner": "Regional Marketing Lead",
            },
            "cfo": {
                "action": "Reconcile marketing budget actuals against plan for this region",
                "owner": "Regional Finance",
            },
        },
    },
}

DEFAULT_PERSONA = "regional_ops_manager"


def recommend_action(top_cause: str, confidence_status: str, persona: str = DEFAULT_PERSONA):
    rec = RECOMMENDATION_MAP.get(top_cause)
    if not rec or confidence_status == "abstain":
        abstain_by_persona = {
            "regional_ops_manager": "Pull additional data for this segment/window before acting",
            "cfo": "Hold off on any financial commentary until the analyst confirms a cause",
        }
        return {
            "action": "No confident recommendation — gather more evidence first",
            "suggested_next_step": abstain_by_persona.get(persona, abstain_by_persona[DEFAULT_PERSONA]),
            "owner": "Analyst",
        }

    persona_view = rec["by_persona"].get(persona, rec["by_persona"][DEFAULT_PERSONA])
    return {
        "driver": top_cause,
        "lever": rec["lever"],
        "action": persona_view["action"],
        "owner": persona_view["owner"],
        "expected_impact": rec["expected_impact"],
        "monitoring_plan": rec["monitoring_plan"],
        "confidence": confidence_status,
    }
