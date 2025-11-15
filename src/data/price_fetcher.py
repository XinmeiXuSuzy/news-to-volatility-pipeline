# src/data/price_fetcher.py
import os
from typing import List
import yfinance as yf
import pandas as pd
from src.config import RAW_DIR

os.makedirs(RAW_DIR, exist_ok=True)

def fetch_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,   # be explicit, avoid future warnings
    )

    if df.empty:
        print(f"No price data for {ticker}")
        return df

    # ✅ Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        # For single ticker, first level is OHLCV, second is ticker
        df.columns = [c[0] for c in df.columns]

    df.reset_index(inplace=True)
    df["ticker"] = ticker
    return df


def fetch_and_save_batch(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    suffix: str = None
) -> pd.DataFrame:
    all_dfs = []
    for t in tickers:
        print(f"Fetching prices for {t}...")
        df = fetch_price_history(t, start_date, end_date, interval)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No price data fetched.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    suffix = suffix or f"{start_date}_to_{end_date}"
    out_path = os.path.join(RAW_DIR, f"prices_demo.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved combined prices to {out_path}")
    return combined
