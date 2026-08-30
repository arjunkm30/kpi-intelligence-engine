"""
Generates fake-but-realistic data for the demo. Run from the project root:

    python utils/data_generator.py

Writes into ../data relative to this file (i.e. the project root's data/ dir):
- data/sales.csv         : daily, fine grain (region/store/sku/price/volume)
- data/promo.csv         : weekly, coarse grain (region/marketing spend)
                           -> deliberately different cadence than sales, per the brief.
                           Used as a NEGATIVE CONTROL: spend doesn't move during the
                           anomaly, so the pipeline should NOT flag marketing as a cause.
- data/inventory.csv     : batch/irregular grain (~every 3 days, region/store/sku),
                           the THIRD data source. Real structured confirmation of the
                           stockout -- this is what makes "stockout" more than a text guess.
- data/docs/*.txt        : unstructured evidence (news, ticket, CRM note, noise)

Scenario baked in:
  Region "North", last 13 days: revenue drops sharply.
  Real cause = stockout on SKU-A in stores N1 + N2 (volume driver, confirmed by
  BOTH the inventory data AND a support ticket/CRM note), amplified by a
  competitor promo mentioned in a news snippet. Marketing spend is unchanged
  during this window -- included specifically so the engine has to correctly
  NOT implicate it.
  Region "South" has SKU-C, launched 5 days ago -> sparse-history scenario.
"""
import os
import pandas as pd
import numpy as np

np.random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(DATA, "docs")
os.makedirs(DOCS, exist_ok=True)

TODAY = pd.Timestamp("2026-08-25")
DAYS = 90
dates = pd.date_range(TODAY - pd.Timedelta(days=DAYS - 1), TODAY, freq="D")

regions_stores = {
    "North": ["N1", "N2", "N3"],
    "South": ["S1", "S2"],
}
skus = ["SKU-A", "SKU-B"]  # SKU-C added only in South, only last 5 days (sparse)

rows = []
for d in dates:
    for region, stores in regions_stores.items():
        for store in stores:
            for sku in skus:
                base_price = 20 if sku == "SKU-A" else 35
                base_volume = 120 if region == "North" else 90

                seasonal = 1 + 0.05 * np.sin(d.dayofweek)
                noise = np.random.normal(0, 0.04)
                volume = base_volume * seasonal * (1 + noise)
                price = base_price * (1 + np.random.normal(0, 0.01))

                # INJECTED ANOMALY: North, SKU-A, stores N1/N2, last 13 days -> stockout.
                # Using days-ago (not week_idx) so the anomaly is confined to a clean
                # recent window and the rest of history stays a valid, uncorrupted baseline.
                days_ago = (TODAY - d).days
                if region == "North" and sku == "SKU-A" and store in ("N1", "N2") and days_ago <= 13:
                    volume *= 0.45  # sharp volume drop = stockout signature

                rows.append([d.date().isoformat(), region, store, sku, round(price, 2), max(round(volume), 0)])

sales = pd.DataFrame(rows, columns=["date", "region", "store", "sku", "price", "volume"])

sparse_dates = pd.date_range(TODAY - pd.Timedelta(days=4), TODAY, freq="D")
for d in sparse_dates:
    for store in regions_stores["South"]:
        vol = 30 * (1 + np.random.normal(0, 0.1))
        sales = pd.concat([sales, pd.DataFrame([[d.date().isoformat(), "South", store, "SKU-C", 15.0, max(round(vol), 0)]],
                                                 columns=sales.columns)], ignore_index=True)

sales.to_csv(os.path.join(DATA, "sales.csv"), index=False)

# --- promo.csv: weekly marketing spend. Deliberately FLAT through the anomaly
# window -- a negative control so the pipeline is tested on whether it
# correctly does NOT blame marketing for something marketing didn't cause. ---
weeks = pd.date_range(TODAY - pd.Timedelta(weeks=12), TODAY, freq="W")
promo_rows = []
for w in weeks:
    for region in regions_stores:
        base_spend = 4000 if region == "North" else 3000
        spend = base_spend * (1 + np.random.normal(0, 0.05))  # small noise, no real shift
        promo_rows.append([w.date().isoformat(), region, round(spend, 2)])
promo = pd.DataFrame(promo_rows, columns=["week", "region", "marketing_spend"])
promo.to_csv(os.path.join(DATA, "promo.csv"), index=False)

# --- inventory.csv: THIRD data source, batch/irregular cadence (roughly every
# 3 days, not daily) -- the point is to force the pipeline to reconcile a
# source that isn't even on a regular clock. This is the structured (not
# text-based) confirmation of the stockout: stock genuinely hits near-zero
# for SKU-A in N1/N2 during the anomaly window. ---
inventory_rows = []
snapshot_dates = [d for i, d in enumerate(dates) if i % 3 == 0]  # irregular ~3-day batch cadence
for d in snapshot_dates:
    for region, stores in regions_stores.items():
        for store in stores:
            for sku in skus:
                base_stock = 200 if sku == "SKU-A" else 150
                days_ago = (TODAY - d).days
                if region == "North" and sku == "SKU-A" and store in ("N1", "N2") and days_ago <= 13:
                    stock = max(int(base_stock * 0.03 * (1 + np.random.normal(0, 0.3))), 0)  # near-zero
                else:
                    stock = int(base_stock * (1 + np.random.normal(0, 0.1)))
                inventory_rows.append([d.date().isoformat(), region, store, sku, max(stock, 0)])
inventory = pd.DataFrame(inventory_rows, columns=["date", "region", "store", "sku", "stock_level"])
inventory.to_csv(os.path.join(DATA, "inventory.csv"), index=False)

recent = (TODAY - pd.Timedelta(days=10)).date().isoformat()
docs = {
    "ticket_001.txt": f"""Date: {recent}
Source: Support Ticket #4471
Region: North
Store: N1
Text: Customer complained SKU-A out of stock again this week. Third complaint
in 10 days. Warehouse says replenishment delayed due to supplier backlog.""",

    "news_001.txt": f"""Date: {recent}
Source: Local Business News
Region: North
Text: Competitor "ValueMart" launched an aggressive 20% discount campaign on
household staples in the North district starting this week, drawing foot
traffic away from nearby stores.""",

    "crm_001.txt": f"""Date: {recent}
Source: CRM Note
Region: North
Store: N2
Text: Store manager flagged low shelf availability for SKU-A. Requested
priority restock. No response from central inventory team yet.""",

    "noise_001.txt": """Date: 2026-05-02
Source: Internal Newsletter
Region: South
Text: The South region team celebrated their quarterly wellness day with a
yoga session and a potluck lunch. Employee satisfaction scores remain high.""",

    "noise_002.txt": """Date: 2026-06-15
Source: Press Release
Region: National
Text: The company announced a new sustainability initiative to reduce
packaging waste by 15% over the next two years across all regions.""",
}
for fname, content in docs.items():
    with open(os.path.join(DOCS, fname), "w") as f:
        f.write(content)

if __name__ == "__main__":
    print(f"Generated {len(sales)} sales rows, {len(promo)} promo rows, "
          f"{len(inventory)} inventory rows, {len(docs)} docs.")
    print(f"Data written to: {DATA}")
