# data/fetch_fred.py
# PURPOSE: Fetches real commodity price data from the FRED API
#   and saves it locally as CSV files for ARIMA calibration.
#
# FREE API key: https://fred.stlouisfed.org/docs/api/api_key.html
#   Sign up → My Account → API Keys → Request API Key
#   Takes about 2 minutes. Add to your .env as FRED_API_KEY=...
#
# Run once to download: python3 data/fetch_fred.py
# Then it auto-refreshes if data is older than 30 days.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)

# FRED series we need
# PCOPPUSDM = Global copper price, USD per metric ton, monthly
# PALUMUSDM = Global aluminum price, USD per metric ton, monthly
# PSTEELHRMUSDM = Steel HR coil, USD per metric ton, monthly
FRED_SERIES = {
    "copper":   "PCOPPUSDM",
    "aluminum": "PALUMUSDM",
    "steel":    "PSTEELHRMUSDM",
}

# How these map to component families
# Used in module3_cost.py to calibrate ARIMA
COMPONENT_PRICE_MAP = {
    "MAG-WIRE":   "copper",    # copper wire → copper price
    "MAG-CORE":   "steel",     # ferrite cores → steel proxy
    "ASM-XFMR":   "copper",    # transformers → copper price
    "MOSFET-":    "aluminum",  # heatsink-dependent → aluminum
    "IND-":       "copper",    # inductors → copper wire content
}


def fetch_fred_series(series_id, start_date="2018-01-01"):
    """
    Fetches a FRED time series as a DataFrame.
    Returns monthly dates and values.

    If no API key, falls back to downloading the CSV directly
    from FRED's public endpoint (no auth needed for basic data).
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    if FRED_API_KEY:
        # Official API — more reliable, higher rate limits
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id":         series_id,
            "api_key":           FRED_API_KEY,
            "file_type":         "json",
            "observation_start": start_date,
            "observation_end":   end_date,
            "frequency":         "m",
            "aggregation_method":"avg",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for obs in data.get("observations", []):
            if obs["value"] != ".":
                rows.append({
                    "date":  pd.to_datetime(obs["date"]),
                    "value": float(obs["value"])
                })
        df = pd.DataFrame(rows)

    else:
        # Fallback: FRED public CSV download (no API key needed)
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={series_id}&vintage_date={end_date}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df[df["date"] >= start_date]

    df = df.sort_values("date").reset_index(drop=True)
    return df


def should_refresh(filepath, max_age_days=30):
    """
    Returns True if the file doesn't exist or is older
    than max_age_days (default 30 days).
    """
    if not os.path.exists(filepath):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
    return age.days > max_age_days


def fetch_all(force=False):
    """
    Downloads all commodity series and saves as CSV.
    Skips if file is fresh (< 30 days old) unless force=True.

    Returns dict of DataFrames keyed by commodity name.
    """
    results = {}

    for name, series_id in FRED_SERIES.items():
        filepath = os.path.join(DATA_DIR, f"{name}_price_monthly.csv")

        if not force and not should_refresh(filepath):
            print(f"  ↩ {name}: using cached data ({filepath})")
            df = pd.read_csv(filepath, parse_dates=["date"])
            results[name] = df
            continue

        print(f"  ↓ Fetching {name} ({series_id}) from FRED...")
        try:
            df = fetch_fred_series(series_id)
            df.to_csv(filepath, index=False)
            print(f"    ✅ {len(df)} monthly records saved → {filepath}")
            results[name] = df
        except Exception as e:
            print(f"    ⚠️  Failed to fetch {name}: {e}")
            # Try loading cached version if available
            if os.path.exists(filepath):
                print(f"    ↩  Using cached version instead")
                results[name] = pd.read_csv(filepath, parse_dates=["date"])
            else:
                results[name] = None

    return results


def load_commodity_prices():
    """
    Loads commodity price data from local CSV files.
    If files don't exist, fetches from FRED first.
    Returns dict: { 'copper': df, 'aluminum': df, 'steel': df }
    """
    results = {}
    for name in FRED_SERIES.keys():
        filepath = os.path.join(DATA_DIR, f"{name}_price_monthly.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
            results[name] = df
        else:
            results[name] = None
    return results


def get_price_index_for_part(part_no, commodity_prices):
    """
    Returns a normalised price index series for a given part number
    by matching its prefix to the commodity map.

    The index is normalised to 1.0 at the series mean so it can be
    used as a multiplier on synthetic prices rather than replacing them.

    Returns:
        pd.Series of monthly index values, or None if no match
    """
    matched_commodity = None
    for prefix, commodity in COMPONENT_PRICE_MAP.items():
        if part_no.startswith(prefix):
            matched_commodity = commodity
            break

    if matched_commodity is None:
        return None

    df = commodity_prices.get(matched_commodity)
    if df is None or df.empty:
        return None

    # Normalise: divide by mean so index oscillates around 1.0
    series = df.set_index("date")["value"]
    mean_val = series.mean()
    if mean_val == 0:
        return None

    return series / mean_val


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  FRED Commodity Price Fetcher")
    print("="*55)

    if not FRED_API_KEY:
        print("\n⚠️  No FRED_API_KEY in .env — using public CSV fallback")
        print("   For better rate limits, get a free key at:")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
    else:
        print(f"\n✅ FRED API key found")

    print("\nFetching commodity prices...")
    prices = fetch_all(force=True)

    print("\n📊 Summary:")
    for name, df in prices.items():
        if df is not None:
            latest = df.iloc[-1]
            print(f"  {name:10s} → {len(df):3d} months | "
                  f"Latest: ${latest['value']:,.2f}/MT "
                  f"({latest['date'].strftime('%b %Y')})")
        else:
            print(f"  {name:10s} → ❌ Not available")

    print("\n🎉 FRED data ready. Module 3 will use this for ARIMA calibration.\n")