import os
import finnhub
import pandas as pd
from datetime import datetime
from typing import List, Optional
from src.config import NEWS_API_KEY, RAW_DIR

os.makedirs(RAW_DIR, exist_ok=True)


class NewsFetcher:
    """
    Fetches company-specific news using Finnhub's company-news endpoint.
    Returns DataFrames with standardized columns for sentiment + ML pipeline.
    """

    def __init__(self, api_key: str = NEWS_API_KEY):
        if api_key is None:
            raise ValueError("NEWS_API_KEY not found in environment/.env")
        self.api_key = api_key

    def fetch_company_news(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        # max_articles: int = 100
    ) -> pd.DataFrame:
        """
        Fetch news for a specific ticker using finnhub.io.

        Finnhub parameters:
        - symbol: stock ticker
        - from: YYYY-MM-DD
        - to: YYYY-MM-DD

        Returns DataFrame with columns:
        ['ticker','datetime','source','title','description','content','url']
        """

        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        try:
            data = finnhub.Client(api_key=self.api_key).general_news('general', min_id=0)
        except Exception as e:
            print(f"[ERROR] Failed to fetch news for {ticker}: {e}")
            return pd.DataFrame()

        if not isinstance(data, list):
            print(f"[WARN] Unexpected response format for {ticker}: {data}")
            return pd.DataFrame()

        # Parse Finnhub news format
        rows = []
        for article in data:
            try:
                # Finnhub returns epoch seconds
                dt = datetime.fromtimestamp(article.get("datetime"))

                rows.append({
                    "ticker": ticker,
                    "datetime": dt,
                    "category": article.get("category"),
                    "source": article.get("source"),
                    "title": article.get("headline"),
                    "summary": article.get("summary"),     # full text not available on Finnhub 
                    "url": article.get("url"),
                })
            except Exception as e:
                print(f"[WARN] Error parsing article for {ticker}: {e}")

        return pd.DataFrame(rows)

    def fetch_and_save_batch(
        self,
        tickers: List[str],
        from_date: str,
        to_date: str,
        suffix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch news for multiple tickers and save combined CSV into data/raw/.
        """
        all_rows = []
        for t in tickers:
            print(f"[INFO] Fetching news for {t}...")
            df = self.fetch_company_news(t, from_date, to_date)

            if df.empty:
                print(f"[WARN] No news found for {t}.")
            else:
                all_rows.append(df)

        if not all_rows:
            print("[WARN] No news fetched for any tickers.")
            return pd.DataFrame()

        combined = pd.concat(all_rows, ignore_index=True)

        suffix = suffix or f"{from_date}_to_{to_date}"
        out_path = os.path.join(RAW_DIR, f"news_finnhub_{suffix}.csv")

        combined.to_csv(out_path, index=False)
        print(f"[SUCCESS] Saved combined news to {out_path}")

        return combined
