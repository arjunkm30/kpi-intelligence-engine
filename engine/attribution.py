"""
Attribution — WHERE does the change live, and WHAT KIND is it. NOT LLM:
pure arithmetic / information theory, no model call.

Two stages, matching how real multi-dimensional root-cause tools work:

  1. adtributor_localize() — ranks candidate elements (here: stores) within
     a dimension using the two concepts from Microsoft's Adtributor paper
     (Bhagwan et al., 2014, "Adtributor: Revenue Debugging in Advertising
     Systems" — the same core idea Azure's current CMMD system still cites
     as a baseline):
       - Explanatory Power (EP): what fraction of the total change this
         element accounts for
       - Surprise: how much this element's share of the total shifted,
         measured as a Jensen-Shannon divergence contribution
     Root causes = smallest set of elements (ranked by EP) whose combined
     EP clears a coverage threshold (TEEP) — not just "the biggest single
     mover," a defensible minimal explanation.

     Note: this is a simplified, single-dimension version. The full
     Adtributor paper (and successors like HotSpot/Alibaba 2018, Squeeze/
     VLDB 2019) also search combinations across MULTIPLE dimensions at
     once — out of scope for this prototype, called out here rather than
     overclaiming.

  2. attribute_change() — for the implicated element(s) from stage 1,
     decomposes revenue into price-driven vs. volume-driven components
     using a SYMMETRIC bridge (see below), answering "what kind" of
     change it is, not just "where."

FORMULA NOTE (verified, and worth being explicit about): any two-factor
revenue bridge (Revenue = Price x Volume) has an unavoidable interaction
term -- ceilta_Price * delta_Volume -- that has to be assigned somewhere.
A naive bridge that writes:
    volume_contrib = delta_volume * price_BASE
    price_contrib  = delta_price  * volume_NOW
still sums exactly to the true revenue change (this is algebraically
guaranteed), but it silently dumps the ENTIRE interaction term into
whichever factor is multiplied by the "now" value -- here, price. Swap the
order and you'd get a different split with the same total. That's an
arbitrary, order-dependent choice, not a defect in the total, but a real
ambiguity in how much of the change gets LABELED as "price" vs "volume."
We instead use the symmetric (midpoint) convention, which splits the
interaction term evenly between both factors instead of assigning it
entirely to one:
    volume_contrib = delta_volume * average(price_base, price_now)
    price_contrib  = delta_price  * average(volume_base, volume_now)
This still sums exactly to the true revenue change (verified below) and
doesn't depend on which factor you happen to compute first.
"""
import math
import pandas as pd


def adtributor_localize(baseline: dict, current: dict, teep: float = 0.7):
    """
    baseline / current: {element_name: value} for the SAME dimension
    (e.g. {"N1": 840, "N2": 900, "N3": 810} for store-level volume).
    teep: minimum cumulative explanatory power the root-cause set must
    cover (paper's "Threshold for Explanatory power," default 0.7 in the
    original Adtributor work).
    """
    total_baseline = sum(baseline.values()) or 1e-9
    total_current = sum(current.values())
    delta_total = total_current - total_baseline

    elements = []
    for key in baseline:
        b, c = baseline[key], current.get(key, 0)
        delta = c - b
        ep = delta / delta_total if delta_total != 0 else 0.0

        p = b / total_baseline if total_baseline else 0.0
        q = c / total_current if total_current else 0.0
        m = (p + q) / 2
        surprise = 0.0
        if p > 0 and m > 0:
            surprise += 0.5 * p * math.log(p / m)
        if q > 0 and m > 0:
            surprise += 0.5 * q * math.log(q / m)

        elements.append({
            "element": key,
            "baseline": round(b, 1),
            "current": round(c, 1),
            "delta": round(delta, 1),
            "explanatory_power": round(ep, 3),
            "surprise": round(surprise, 4),
        })

    elements.sort(key=lambda e: -e["explanatory_power"])

    root_cause_set, cumulative_ep = [], 0.0
    for e in elements:
        if e["explanatory_power"] <= 0:
            break  # only elements moving the SAME direction as the total change count
        root_cause_set.append(e["element"])
        cumulative_ep += e["explanatory_power"]
        if cumulative_ep >= teep:
            break

    return {
        "ranked_elements": elements,
        "root_cause_elements": root_cause_set,
        "cumulative_explanatory_power": round(cumulative_ep, 3),
        "teep_threshold": teep,
    }


def attribute_change(sales: pd.DataFrame, region: str, sku: str, teep: float = 0.7):
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["region"] == region) & (df["sku"] == sku)]
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = df.groupby(["store", "week"]).agg(price=("price", "mean"), volume=("volume", "sum")).reset_index()

    baseline_volume, current_volume = {}, {}
    per_store_detail = []
    for store, g in weekly.groupby("store"):
        g = g.sort_values("week")
        if len(g) < 2:
            continue
        now, base = g.iloc[-1], g.iloc[:-1].mean(numeric_only=True)
        baseline_volume[store] = float(base["volume"])
        current_volume[store] = float(now["volume"])

        # Symmetric (midpoint) bridge -- see module docstring. Splits the
        # price*volume interaction term evenly instead of dumping all of it
        # into one factor. Still sums exactly to the true revenue change.
        avg_price = (base["price"] + now["price"]) / 2
        avg_volume = (base["volume"] + now["volume"]) / 2
        vol_contrib = (now["volume"] - base["volume"]) * avg_price
        price_contrib = (now["price"] - base["price"]) * avg_volume

        actual_change = (now["price"] * now["volume"]) - (base["price"] * base["volume"])
        per_store_detail.append({
            "store": store,
            "volume_contribution": round(float(vol_contrib), 2),
            "price_contribution": round(float(price_contrib), 2),
            "total_change": round(float(vol_contrib + price_contrib), 2),
            "actual_revenue_change": round(float(actual_change), 2),  # cross-check: should match total_change
            "current_volume": round(float(now["volume"]), 1),
            "baseline_volume": round(float(base["volume"]), 1),
        })

    # Stage 1: Adtributor-style localization across stores (volume dimension)
    localization = adtributor_localize(baseline_volume, current_volume, teep=teep)

    # Stage 2: restrict the price/volume bridge to the implicated stores only
    implicated = set(localization["root_cause_elements"]) or {d["store"] for d in per_store_detail}
    by_store = [d for d in per_store_detail if d["store"] in implicated]
    by_store.sort(key=lambda r: r["total_change"])

    dominant = "volume" if sum(abs(r["volume_contribution"]) for r in by_store) > \
                           sum(abs(r["price_contribution"]) for r in by_store) else "price"

    return {
        "by_store": by_store,
        "dominant_driver_type": dominant,
        "localization": localization,
    }
