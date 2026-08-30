"""
Detection — signal vs. noise. NOT LLM: statistics only.

Method: forecast-based anomaly detection, not simple residual testing.

An earlier version of this module fit STL (Cleveland et al., 1990) on the
WHOLE series (including the anomalous window) and tested the residual. That
has a known failure mode: STL's trend component (especially with
robust=True) can partially absorb a SUSTAINED level shift — like an
ongoing 2-week stockout — into "trend," leaving very little in the
residual. STL residual testing works well for short spikes; it's the wrong
tool for a sustained regime change.

The fix, matching how production forecast-based systems (e.g. Facebook's
Prophet, and Twitter's STL+ESD pipeline for shorter anomalies) actually
handle this: fit the seasonal/trend model ONLY on history the model hasn't
seen the test window in, forecast forward, and compare the forecast against
what actually happened. The model can't "learn" an anomaly it was never
shown, so a sustained shift shows up clearly as forecast error instead of
being smoothed away.

Falls back to a simple mean/std comparison when there isn't enough history
to fit a seasonal model — this fallback is what correctly triggers the
"sparse-history" demo scenario.

TERMINOLOGY NOTE (from review): this module used to return a field called
"confidence" (low/normal) meaning "how much do we trust this detection,
given how much history we have." engine/confidence.py separately returns a
"status" (confident/leading_hypothesis/abstain) meaning "how sure are we
about WHICH cause is responsible." Those are two unrelated notions that
happened to share the word "confidence" -- a real naming collision, not
just a style nitpick, since it made the two pipeline stages sound like
they were talking about the same thing when they aren't. This module now
returns "data_sufficiency" instead.
"""
import numpy as np
import pandas as pd
from engine import CONTRACT

SEASONAL_PERIOD_DAYS = 7
MIN_PERIODS_FOR_STL = 3   # need enough history LEFT OVER after holding out the test window
TEST_WINDOW_DAYS = 14    # >= the longest anomaly duration we expect, so training
                          # data never contains a partial, contaminating anomaly


def detect_anomaly(sales: pd.DataFrame, region: str, sku: str = None, window_weeks: int = 8):
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["volume"]
    if sku:
        df = df[df["sku"] == sku]
    df = df[df["region"] == region]

    daily = df.groupby("date")["revenue"].sum().sort_index()
    daily = daily.asfreq("D").interpolate()

    n_history_days = len(daily)
    min_days = CONTRACT["kpis"]["revenue"]["min_history_days_for_confidence"]
    sparse = n_history_days < min_days
    threshold = CONTRACT["kpis"]["revenue"]["materiality_zscore_threshold"]

    current_week_total = float(daily.iloc[-TEST_WINDOW_DAYS:].sum())

    # --- Not enough data to hold out a test window and still fit a seasonal
    #     model on what's left: fall back to naive mean/std, flag sparse ---
    train = daily.iloc[:-TEST_WINDOW_DAYS]
    if len(train) < SEASONAL_PERIOD_DAYS * MIN_PERIODS_FOR_STL:
        baseline_avg = float(train.tail(window_weeks * 7).mean() * 7) if len(train) else None
        pct_change = ((current_week_total - baseline_avg) / baseline_avg * 100) if baseline_avg else 0.0
        return {
            "material": bool(abs(pct_change) > 5),
            "method": "naive_baseline (insufficient history to hold out a test window)",
            "pct_change": round(pct_change, 1),
            "sparse_history": True,
            "history_days": n_history_days,
            "data_sufficiency": "low",
        }

    # --- Forecast-based detection: fit on TRAIN only, forecast the test window ---
    from statsmodels.tsa.seasonal import STL
    stl_fit = STL(train, period=SEASONAL_PERIOD_DAYS, robust=True).fit()

    # Expected value for each held-out day = last known trend level (random-walk
    # extrapolation) + that weekday's average seasonal component from training.
    last_trend = float(stl_fit.trend.iloc[-1])
    seasonal_by_weekday = (
        pd.Series(stl_fit.seasonal.values, index=train.index)
        .groupby(train.index.dayofweek).mean()
    )
    test_index = daily.index[-TEST_WINDOW_DAYS:]
    expected = pd.Series(
        [last_trend + seasonal_by_weekday.get(d.dayofweek, 0.0) for d in test_index],
        index=test_index,
    )
    actual = daily.iloc[-TEST_WINDOW_DAYS:]
    forecast_error = actual - expected

    # Robust scale of in-sample training residuals (median/MAD, not mean/std —
    # doesn't get dragged around by the anomaly itself, since it's computed
    # purely from TRAIN, which never contained the anomaly by construction).
    train_resid = stl_fit.resid
    resid_median = train_resid.median()
    mad = (train_resid - resid_median).abs().median()
    robust_std = 1.4826 * mad if mad > 0 else (train_resid.std() or 1e-9)

    z = float((forecast_error.mean() - resid_median) / robust_std)
    baseline_avg = float(expected.sum())
    pct_change = (current_week_total - baseline_avg) / baseline_avg * 100 if baseline_avg else 0.0

    min_pct = CONTRACT["kpis"]["revenue"]["materiality_min_pct_change"]
    statistically_significant = abs(z) > threshold
    business_significant = abs(pct_change) > min_pct

    return {
        "material": bool(statistically_significant and business_significant),
        "statistically_significant": bool(statistically_significant),
        "business_significant": bool(business_significant),
        "method": "STL fit on held-out history + forecast comparison (robust MAD z-score)",
        "z_score": round(z, 2),
        "pct_change": round(pct_change, 1),
        "current_week_revenue": round(current_week_total, 2),
        "baseline_avg_revenue": round(baseline_avg, 2),
        "sparse_history": sparse,
        "history_days": n_history_days,
        "data_sufficiency": "low" if sparse else "normal",
    }
