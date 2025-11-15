import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

def train_xgboost(df: pd.DataFrame, window: int = 5):
    target_col = f"realized_vol_{window}"

    feature_cols = [
        "mean_sentiment_score",
        "article_count",
        "frac_positive",
        "frac_negative",
        "log_return",
        "ret_1",
        "ret_5",
        "vol_5_past",
    ]

    # --- 1) Work on a copy
    df_model = df.copy()

    # --- 2) Drop rows with missing target only
    df_model = df_model.dropna(subset=[target_col])

    # --- 3) Fill NaNs in features with neutral values
    df_model["mean_sentiment_score"] = df_model["mean_sentiment_score"].fillna(0.0)
    df_model["article_count"] = df_model["article_count"].fillna(0)
    df_model["frac_positive"] = df_model["frac_positive"].fillna(0.0)
    df_model["frac_negative"] = df_model["frac_negative"].fillna(0.0)
    df_model["log_return"] = df_model["log_return"].fillna(0.0)
    df_model["ret_1"] = df_model["ret_1"].fillna(0.0)
    df_model["ret_5"] = df_model["ret_5"].fillna(0.0)
    df_model["vol_5_past"] = df_model["vol_5_past"].fillna(0.0)

    # --- 4) Make sure required cols exist
    missing = [c for c in feature_cols + [target_col] if c not in df_model.columns]
    if missing:
        raise ValueError(f"Missing columns in df for training: {missing}")

    # --- 5) Time-based split instead of random split
    # Sort by date (string or datetime is fine)
    df_model = df_model.sort_values("date")

    split_idx = int(len(df_model) * 0.8)

    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    # --- 6) Safety check
    if len(train) < 10 or len(test) < 5:
        print(f"[WARN] Not enough samples for meaningful split (train={len(train)}, test={len(test)}).")
        return None

    # --- 7) Train XGBoost
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"{col}: {imp:.4f}")


    # --- 8) Evaluate
    y_pred = model.predict(X_test)
    print("XGBoost R^2:", r2_score(y_test, y_pred))
    print("XGBoost MAE:", mean_absolute_error(y_test, y_pred))


    return model
