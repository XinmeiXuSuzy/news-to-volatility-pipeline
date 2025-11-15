import pandas as pd
import numpy as np

def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    news_df columns: ['ticker','datetime','sentiment_label','sentiment_score',...]
    Returns daily aggregated sentiment features.
    """
    df = news_df.copy()
    df["date"] = df["datetime"].dt.date

    agg = df.groupby(["ticker", "date"]).agg(
        mean_sentiment_score=("sentiment_score", "mean"),
        article_count=("sentiment_score", "count"),
        frac_positive=("sentiment_label", lambda x: (x == "positive").mean() if len(x) > 0 else np.nan),
        frac_negative=("sentiment_label", lambda x: (x == "negative").mean() if len(x) > 0 else np.nan),
    ).reset_index()

    return agg

def compute_returns_and_vol(price_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    price_df columns: ['Date','Close','ticker',...]
    Adds log_return and realized_vol_{window} as target.
    """
    df = price_df.rename(columns={"Date": "datetime"}).copy()
    df["date"] = df["datetime"].dt.date

    out = []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["log_return"] = np.log(g["Close"] / g["Close"].shift(1))
        # future realized vol (target)
        g[f"realized_vol_{window}"] = (
            g["log_return"].rolling(window).std().shift(-window + 1)
        )
        g["ret_1"] = g["log_return"].shift(1)
        g["ret_5"] = g["log_return"].rolling(5).sum().shift(1)
        g["vol_5_past"] = g["log_return"].rolling(5).std().shift(1)

        out.append(g)

    df2 = pd.concat(out, ignore_index=True)
    return df2

def build_modeling_table(sent_daily: pd.DataFrame, price_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Joins sentiment and price info into a single table.
    """
    vol_df = compute_returns_and_vol(price_df, window=window)
    vol_df["date"] = vol_df["date"].astype("object")

    sent_daily["date"] = sent_daily["date"].astype("object")

    merged = pd.merge(
        vol_df,
        sent_daily,
        on=["ticker", "date"],
        how="left"
    )

    # Drop rows where target is NaN
    target_col = f"realized_vol_{window}"
    merged = merged.dropna(subset=[target_col])

    return merged
