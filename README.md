# News-to-Volatility Pipeline

A complete quantitative research system that predicts short-horizon realized volatility using company news, FinBERT sentiment, and price-based features. The project integrates data engineering, NLP, feature construction, and XGBoost modeling with a time-aware evaluation framework.

---

## Overview

The goal is to evaluate whether news sentiment enhances the prediction of short-window realized volatility, which is critical for market making, risk management, and execution.

The pipeline fetches real-time news, scores sentiment with FinBERT, aggregates daily sentiment features, computes price-based features, and trains an XGBoost model to forecast realized volatility.

---

## Data and Features

### News & Sentiment
- Source: Finnhub `company-news`
- NLP model: `ProsusAI/finbert`
- Daily aggregated sentiment features:
  - mean_sentiment_score  
  - article_count  
  - frac_positive  
  - frac_negative  

### Price-Based Features
- log_return  
- ret_1 (previous day's return)  
- ret_5 (5-day cumulative return)  
- vol_5_past (5-day realized volatility)  

### Target Variable
- `realized_vol_5`: forward 5-day realized volatility computed from log returns

---

## Modeling Approach

- Algorithm: XGBoost Regressor (`reg:squarederror`)
- Time-based train/test split:
  - 80% oldest data for training  
  - 20% most recent data for testing  
- Prevents leakage and mirrors real trading deployment

---

## Results

### Performance
- R²: **0.3197**
- MAE: **0.00810**

The model captures meaningful structure in short-term volatility despite its inherent noise.

### Feature Importance

| Feature | Importance |
|---------|------------|
| log_return | 0.3554 |
| vol_5_past | 0.2799 |
| ret_5 | 0.1939 |
| ret_1 | 0.1708 |
| sentiment features | ~0.0000 |

Short-horizon volatility is dominated by price-based persistence; sentiment provides limited signal except on major event days.

You may add:
- Predicted vs actual volatility plot  
- Feature importance bar chart

---

## Key Technologies

- Python  
- Transformers (FinBERT)  
- XGBoost  
- Pandas / NumPy  
- yfinance  
- Finnhub API  

---

## Future Extensions

- Event-day or earnings-day modeling  
- Directional movement prediction (classification)  
- SHAP interpretability  
- Larger universe of tickers  
- Intraday and high-frequency resolution  

---