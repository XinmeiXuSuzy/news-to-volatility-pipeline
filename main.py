import os
from src.data.news_fetcher import NewsFetcher
from src.data.price_fetcher import fetch_and_save_batch
from src.nlp.sentiment import SentimentAnalyzer
from src.features.build_dataset import aggregate_daily_sentiment, build_modeling_table
from src.models.train_vol_model import train_xgboost
from src.config import DEFAULT_TICKERS, RAW_DIR

if __name__ == "__main__":
    tickers = DEFAULT_TICKERS  # keep 1–3 tickers at first to debug
    start = "2023-01-01"
    end = "2024-10-31"

    # 1) Fetch news
    nf = NewsFetcher()
    df_news = nf.fetch_and_save_batch(
        tickers=tickers,
        from_date=start,
        to_date=end,
        suffix="demo"
    )

    # 2) Add sentiment
    sa = SentimentAnalyzer()
    df_scored = sa.add_sentiment_to_df(df_news, text_col="title")
    os.makedirs(RAW_DIR, exist_ok=True)
    df_scored.to_csv(f"{RAW_DIR}/news_sentiment_scored.csv", index=False)
    print("[INFO] Scored news saved.")

    # 3) Aggregate sentiment to daily per ticker
    sent_daily = aggregate_daily_sentiment(df_scored)
    sent_daily.to_csv(f"{RAW_DIR}/sentiment_daily.csv", index=False)
    print("[INFO] Daily sentiment saved.")

    # 4) Fetch prices
    df_price = fetch_and_save_batch(
        tickers=tickers,
        start_date=start,
        end_date=end,
        interval="1d",
        suffix="demo"
    )

    # 5) Build modeling table (joins price + daily sentiment, computes realized_vol_5)
    df_model = build_modeling_table(sent_daily, df_price, window=5)
    df_model.to_csv(f"{RAW_DIR}/modeling_table.csv", index=False)
    print("[INFO] Modeling table saved. Columns:", df_model.columns)

    # 6) Train XGBoost on the modeling table
    model = train_xgboost(df_model, window=5)

    



