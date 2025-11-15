from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from typing import Optional

class SentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def score_text(self, text: str) -> Optional[dict]:
        if not text or not isinstance(text, str):
            return None
        try:
            res = self.pipe(text[:512])[0]  # truncate long text
            # res looks like: {'label': 'positive', 'score': 0.98} 
            return res
        except Exception as e:
            print(f"Error scoring text: {e}")
            return None

    def add_sentiment_to_df(self, df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        labels = []
        scores = []

        for t in df[text_col].fillna(""):
            res = self.score_text(t)
            if res is None:
                labels.append(None)
                scores.append(None)
            else:
                labels.append(res["label"])
                scores.append(res["score"])

        df = df.copy()
        df["sentiment_label"] = labels
        df["sentiment_score"] = scores
        return df
