# modules/module3_cost.py  (FINAL VERSION — FRED integration + steel removed)
# PURPOSE: For each component:
#   1. Computes LANDED COST = unit price + freight + tariff + quality cost
#   2. Blends PO history with REAL commodity prices from FRED
#      (copper + aluminum only — steel series unavailable)
#   3. Fits ARIMA on whatever data is available (no minimum required)
#   4. Forecasts next-quarter price
#   5. Flags anomalies where current quote > 15% above forecast
#
# Run: python3 modules/module3_cost.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from database.db_connect import run_query


# ─────────────────────────────────────────────
# COMMODITY CONFIG — copper + aluminum only
# ─────────────────────────────────────────────
COMMODITY_FILES = {
    "copper":   "copper_price_monthly.csv",
    "aluminum": "aluminum_price_monthly.csv",
}

COMPONENT_PRICE_MAP = {
    "MAG-WIRE":   "copper",
    "ASM-XFMR":   "copper",
    "IND-":       "copper",
    "MOSFET-":    "aluminum",
    "MAG-CORE":   "copper",
}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "external"
)


# ─────────────────────────────────────────────
# LOAD COMMODITY PRICES ONCE AT MODULE LEVEL
# ─────────────────────────────────────────────
def _load_prices():
    prices = {}
    for name, fname in COMMODITY_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                # Force plain string dates → no timezone issues ever
                df["date"] = pd.to_datetime(
                    df["date"].astype(str).str[:10]
                )
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna()
                prices[name] = df
                print(f"  ✅ FRED {name}: {len(df)} months loaded")
            except Exception as e:
                print(f"  ⚠️  Could not load {name}: {e}")
                prices[name] = None
        else:
            prices[name] = None
    return prices

_COMMODITY_PRICES = _load_prices()


# ─────────────────────────────────────────────
# GET COMMODITY INDEX FOR A PART
# ─────────────────────────────────────────────
def get_price_index_for_part(part_no):
    matched = None
    for prefix, commodity in COMPONENT_PRICE_MAP.items():
        if part_no.startswith(prefix):
            matched = commodity
            break
    if matched is None:
        return None
    df = _COMMODITY_PRICES.get(matched)
    if df is None or df.empty:
        return None
    series = df.set_index("date")["value"]
    mean_val = series.mean()
    if mean_val == 0:
        return None
    return series / mean_val


def get_commodity_context(part_no):
    index = get_price_index_for_part(part_no)
    if index is None:
        return None
    recent = index.tail(6)
    return {
        "current_index": round(float(recent.iloc[-1]), 4),
        "6mo_avg_index": round(float(recent.mean()), 4),
        "trend":         "rising" if recent.iloc[-1] > recent.iloc[0] else "falling",
        "last_date":     str(recent.index[-1].date()),
    }


# ─────────────────────────────────────────────
# STEP 1: LANDED COST
# ─────────────────────────────────────────────
def compute_landed_cost(unit_price, supplier_country, quantity, reject_rate=0.0):
    freight_rates = {
        'China': 0.08, 'Taiwan': 0.07, 'South Korea': 0.07,
        'India': 0.06, 'Vietnam': 0.08, 'Germany': 0.04, 'USA': 0.02,
    }
    tariff_rates = {
        'China': 0.075, 'Taiwan': 0.00, 'South Korea': 0.00,
        'India': 0.00,  'Vietnam': 0.025, 'Germany': 0.00, 'USA': 0.00,
    }
    freight      = unit_price * freight_rates.get(supplier_country, 0.07)
    tariff       = unit_price * tariff_rates.get(supplier_country, 0.035)
    quality_cost = reject_rate * unit_price * 1.5
    landed       = unit_price + freight + tariff + quality_cost
    return {
        'unit_price':       round(unit_price, 4),
        'freight_per_unit': round(freight, 4),
        'tariff_per_unit':  round(tariff, 4),
        'quality_cost':     round(quality_cost, 4),
        'landed_cost':      round(landed, 4),
        'total_cost':       round(landed * quantity, 2)
    }


# ─────────────────────────────────────────────
# STEP 2: PRICE HISTORY — BLENDED WITH FRED
# ─────────────────────────────────────────────
def get_price_history_blended(part_no, supplier_id=None):
    if supplier_id:
        sql = """
            SELECT DATE_TRUNC('month', po_date) AS month,
                   AVG(unit_price) AS avg_price
            FROM po_history
            WHERE part_no = %s AND supplier_id = %s
            GROUP BY 1 ORDER BY 1
        """
        po_df = run_query(sql, params=(part_no, int(supplier_id)))
    else:
        sql = """
            SELECT DATE_TRUNC('month', po_date) AS month,
                   AVG(unit_price) AS avg_price
            FROM po_history
            WHERE part_no = %s
            GROUP BY 1 ORDER BY 1
        """
        po_df = run_query(sql, params=(part_no,))

    if po_df.empty:
        return po_df, False

    # KEY FIX: slice to first 10 chars (YYYY-MM-DD) then parse
    # This completely avoids any timezone issues from PostgreSQL
    po_df['month'] = pd.to_datetime(
        po_df['month'].astype(str).str[:10]
    )
    po_df = po_df.set_index('month')

    # Try FRED commodity index
    fred_index = get_price_index_for_part(part_no)

    if fred_index is not None and not fred_index.empty:
        # FRED index already has plain dates (loaded without timezone above)
        fred_reindexed = fred_index.reindex(po_df.index, method='nearest')
        po_df['blended_price'] = (
            po_df['avg_price'] * fred_reindexed.values
        ).round(6)
        po_df['blended_price'] = po_df['blended_price'].fillna(po_df['avg_price'])
        po_df = po_df.reset_index()
        return po_df, True

    po_df = po_df.reset_index()
    return po_df, False


# ─────────────────────────────────────────────
# STEP 3: ARIMA — works with ANY data length
# ─────────────────────────────────────────────
def forecast_price_arima(price_series, steps=3):
    """
    Forecasts next `steps` months using whatever data is available.
    Minimum 2 data points. Falls back to linear trend if < 4 points.
    """
    n = len(price_series)

    if n < 2:
        avg = float(np.mean(price_series)) if n == 1 else 1.0
        return [avg]*steps, [avg*0.9]*steps, [avg*1.1]*steps

    if n < 4:
        # Linear trend extrapolation — no ARIMA needed
        x = np.arange(n)
        z = np.polyfit(x, price_series, 1)
        p = np.poly1d(z)
        future_x  = np.arange(n, n + steps)
        forecast  = [max(0.001, round(p(xi), 4)) for xi in future_x]
        return forecast, [f*0.9 for f in forecast], [f*1.1 for f in forecast]

    try:
        from pmdarima import auto_arima
        model = auto_arima(
            price_series,
            start_p=0, max_p=min(3, n//3),
            start_q=0, max_q=min(3, n//3),
            d=None,
            seasonal=False,
            information_criterion='aic',
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        forecast, conf_int = model.predict(
            n_periods=steps, return_conf_int=True, alpha=0.05)
        return (
            [round(f, 4) for f in forecast],
            [round(c[0], 4) for c in conf_int],
            [round(c[1], 4) for c in conf_int]
        )
    except Exception:
        # Fallback linear
        x = np.arange(n)
        z = np.polyfit(x, price_series, 1)
        p = np.poly1d(z)
        future_x = np.arange(n, n + steps)
        forecast  = [max(0.001, round(p(xi), 4)) for xi in future_x]
        return forecast, [f*0.9 for f in forecast], [f*1.1 for f in forecast]


# ─────────────────────────────────────────────
# STEP 3b: FORECAST ACCURACY — ARIMA BACKTEST (MAPE)
# ─────────────────────────────────────────────
def backtest_arima_mape(price_series, holdout=3):
    """
    Walk-forward backtest: hold out the last `holdout` observations,
    forecast them from the earlier history only, and return the
    Mean Absolute Percentage Error (MAPE, in %).

    Returns None when there isn't enough data to both train and test.
    """
    series = [float(x) for x in price_series]
    n = len(series)
    if n < holdout + 4:          # need a usable training window + holdout
        return None

    train  = series[:-holdout]
    actual = series[-holdout:]
    forecast, _, _ = forecast_price_arima(train, steps=holdout)

    errors = [abs((a - f) / a) for a, f in zip(actual, forecast) if a != 0]
    if not errors:
        return None
    return round(float(np.mean(errors) * 100), 2)


def evaluate_arima_accuracy(holdout=3):
    """
    Backtests ARIMA on each loaded FRED commodity series.
    Returns ({commodity: mape_pct}, average_mape_pct).
    """
    results = {}
    for name, df in _COMMODITY_PRICES.items():
        if df is not None and not df.empty:
            mape = backtest_arima_mape(df["value"].astype(float).tolist(), holdout)
            if mape is not None:
                results[name] = mape
    avg = round(float(np.mean(list(results.values()))), 2) if results else None
    return results, avg


# ─────────────────────────────────────────────
# STEP 4: ANOMALY DETECTION
# ─────────────────────────────────────────────
def check_price_anomaly(current_quote, forecast_price, threshold=0.15):
    if forecast_price <= 0:
        return False, 0.0, "No forecast available"
    deviation = (current_quote - forecast_price) / forecast_price
    if deviation > threshold:
        return True,  round(deviation,4), f"⚠️ Quote {deviation:.0%} above forecast"
    elif deviation < -threshold:
        return False, round(deviation,4), f"✅ Quote {abs(deviation):.0%} below forecast"
    return False, round(deviation,4), "✅ Quote within normal range"


# ─────────────────────────────────────────────
# STEP 5: FULL COST ANALYSIS
# ─────────────────────────────────────────────
def analyze_cost_for_part(part_no, supplier_id, supplier_country,
                           quantity, component_class):
    history_df, fred_used = get_price_history_blended(part_no, supplier_id)

    if history_df.empty:
        comp = run_query(
            "SELECT unit_cost_avg FROM component_master WHERE part_no=%s",
            params=(part_no,)
        )
        base_price   = float(comp['unit_cost_avg'].iloc[0]) if not comp.empty else 1.0
        price_series = [base_price]
        current_price= base_price
    else:
        price_col    = 'blended_price' if (fred_used and 'blended_price'
                        in history_df.columns) else 'avg_price'
        price_series = history_df[price_col].astype(float).tolist()
        current_price= price_series[-1]

    rr_df = run_query(
        """SELECT AVG(CAST(reject_quantity AS FLOAT)/NULLIF(quantity,0))
           AS reject_rate FROM po_history
           WHERE part_no=%s AND supplier_id=%s""",
        params=(part_no, int(supplier_id))
    )
    reject_rate = float(rr_df['reject_rate'].iloc[0] or 0.0)

    landed = compute_landed_cost(current_price, supplier_country,
                                 quantity, reject_rate)

    forecast, lower, upper = forecast_price_arima(price_series, steps=3)
    next_q = float(np.mean(forecast))
    is_anomaly, deviation, flag_text = check_price_anomaly(current_price, next_q)

    return {
        'part_no':           part_no,
        'supplier_id':       supplier_id,
        'current_price':     round(current_price, 4),
        'forecast_q_next':   round(next_q, 4),
        'forecast_months':   forecast,
        'forecast_lower':    lower,
        'forecast_upper':    upper,
        'price_deviation':   round(deviation, 4),
        'is_price_anomaly':  is_anomaly,
        'anomaly_flag':      flag_text,
        'fred_calibrated':   fred_used,
        'commodity_context': get_commodity_context(part_no),
        **landed
    }


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  Module 3 — Cost Analysis + ARIMA + FRED Integration Test")
    print("="*65)

    available = sum(1 for v in _COMMODITY_PRICES.values() if v is not None)
    print(f"\n  FRED series loaded: {available}/2")
    for name, df in _COMMODITY_PRICES.items():
        if df is not None:
            row = df.iloc[-1]
            print(f"  {name:10s} {len(df):3d} months | "
                  f"Latest: ${row['value']:,.2f}/MT ({row['date'].strftime('%b %Y')})")
        else:
            print(f"  {name:10s} ❌ not available")

    test_cases = [
        ('MAG-WIRE-1.0MM', 7,  'India',  50,  'Commodity'),
        ('ASM-XFMR-LVCT',  5,  'India',  100, 'Custom'),
        ('IGBT-G4PC50W',   9,  'Taiwan', 200, 'Critical'),
        ('RES-0402-10K',   1,  'China',  5000,'Commodity'),
        ('IND-1210-100U',  2,  'China',  500, 'Commodity'),
    ]

    print(f"\n{'Part':<20} {'FRED?':>6} {'Curr $':>8} "
          f"{'Fcst $':>8} {'Landed $':>10} {'Flag'}")
    print("-"*70)

    for part_no, sid, country, qty, cls in test_cases:
        r = analyze_cost_for_part(part_no, sid, country, qty, cls)
        fred_tag = "✅ YES" if r['fred_calibrated'] else "   NO"
        flag     = "⚠️" if r['is_price_anomaly'] else "✅"
        print(f"  {part_no:<20} {fred_tag} "
              f"${r['current_price']:>7.4f} "
              f"${r['forecast_q_next']:>7.4f} "
              f"${r['landed_cost']:>9.4f}  {flag}")
        if r['commodity_context']:
            ctx = r['commodity_context']
            print(f"    └─ index {ctx['current_index']:.3f} "
                  f"({ctx['trend']}) as of {ctx['last_date']}")

    # Forecast accuracy backtest (MAPE) on the FRED commodity series
    print(f"\n  ARIMA forecast accuracy (walk-forward backtest, 3-month holdout):")
    mape_by_series, avg_mape = evaluate_arima_accuracy(holdout=3)
    for name, mape in mape_by_series.items():
        print(f"    {name:10s} MAPE: {mape:.2f}%")
    if avg_mape is not None:
        print(f"    {'AVERAGE':10s} MAPE: {avg_mape:.2f}%")
    else:
        print("    (not enough data to backtest)")

    print(f"\n✅ Module 3 with FRED integration complete!\n")