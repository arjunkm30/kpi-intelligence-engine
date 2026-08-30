"""
Reconciliation — pulls structured context from OTHER data sources (different
grain, different refresh cadence) and aligns it to the anomaly's time window
and segment. NOT LLM: pandas joins and simple aggregation.

This is what actually answers Round 2 objective #2 ("reconciles data and
business context across heterogeneous sources"), as distinct from
engine/evidence.py which handles UNSTRUCTURED text. Reconciliation handles
STRUCTURED cross-source data:

  - promo.csv      : weekly marketing spend (coarser grain than daily sales)
  - inventory.csv  : ~3-day batch snapshots (irregular grain, not even daily)

Each source is reported with its own freshness/cadence rather than silently
resampled to look like it was always daily -- the point of "reconciliation"
is being honest about what each source actually knows and when it last knew it.
"""
import os
import pandas as pd
from engine import DATA_DIR


def get_marketing_context(region: str, window_days: int = 13):
    """
    Weekly marketing spend for the region, compared: last available window
    vs. the trailing average before it. Used as a NEGATIVE-CONTROL signal in
    this dataset -- if spend didn't move, the pipeline should NOT implicate it.
    """
    path = os.path.join(DATA_DIR, "promo.csv")
    if not os.path.exists(path):
        return {"available": False, "source": "promo.csv"}

    promo = pd.read_csv(path)
    promo["week"] = pd.to_datetime(promo["week"])
    promo = promo[promo["region"] == region].sort_values("week")
    if len(promo) < 2:
        return {"available": False, "source": "promo.csv"}

    latest = promo.iloc[-1]
    trailing = promo.iloc[:-1]["marketing_spend"].mean()
    pct_change = (latest["marketing_spend"] - trailing) / trailing * 100 if trailing else 0.0

    return {
        "available": True,
        "source": "promo.csv",
        "grain": "weekly",
        "as_of": latest["week"].date().isoformat(),
        "latest_spend": round(float(latest["marketing_spend"]), 2),
        "trailing_avg_spend": round(float(trailing), 2),
        "pct_change": round(float(pct_change), 1),
        "material_shift": bool(abs(pct_change) > 15),  # threshold: is this even worth considering as a cause
    }


def get_inventory_context(region: str, sku: str, stores: list, window_days: int = 13):
    """
    Inventory snapshots are on an irregular ~3-day batch cadence -- NOT daily.
    We report the most recent snapshot per store within the window and flag
    near-zero stock explicitly, since that's structured (not text-based)
    confirmation of a stockout.
    """
    path = os.path.join(DATA_DIR, "inventory.csv")
    if not os.path.exists(path):
        return {"available": False, "source": "inventory.csv"}

    inv = pd.read_csv(path)
    inv["date"] = pd.to_datetime(inv["date"])
    cutoff = inv["date"].max() - pd.Timedelta(days=window_days)
    inv = inv[(inv["region"] == region) & (inv["sku"] == sku) & (inv["date"] >= cutoff)]
    if inv.empty:
        return {"available": False, "source": "inventory.csv"}

    by_store = []
    for store in stores:
        store_inv = inv[inv["store"] == store].sort_values("date")
        if store_inv.empty:
            continue
        latest_snapshot = store_inv.iloc[-1]
        by_store.append({
            "store": store,
            "as_of": latest_snapshot["date"].date().isoformat(),  # freshness: when this batch last ran
            "stock_level": int(latest_snapshot["stock_level"]),
            "near_zero_stock": bool(latest_snapshot["stock_level"] < 15),
        })

    return {
        "available": True,
        "source": "inventory.csv",
        "grain": "batch, ~every 3 days (irregular, not daily)",
        "by_store": by_store,
        "any_near_zero": any(s["near_zero_stock"] for s in by_store),
    }


def reconcile_context(region: str, sku: str, stores: list):
    """
    Bundles both structured sources into one context object, each still
    labeled with its own grain/freshness -- reconciliation means aligning
    them to the SAME question, not pretending they're all on the same clock.
    """
    return {
        "marketing": get_marketing_context(region),
        "inventory": get_inventory_context(region, sku, stores),
    }
