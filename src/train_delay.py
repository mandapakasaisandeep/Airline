"""
Flight-delay classifier (LightGBM) + SHAP explainability.

Core task: predict P(delayed > 15 min) for a scheduled flight, and explain
*why* each prediction is what it is using SHAP values.

Dataset: vega flights-3m (date, delay, distance, origin, destination).
Runs on CPU in seconds.
"""
import os, argparse, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "flights.csv")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)

DELAY_THRESHOLD = 15  # minutes; industry-standard "on-time" cutoff


def parse_date(v):
    """date is MMDDHHMM (e.g. 01010001 -> month=1 day=1 hour=0 min=1)."""
    s = str(int(v)).zfill(8)
    month, day, hour, minute = int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])
    return month, day, hour, minute


def build_features(df):
    parsed = df["date"].apply(parse_date)
    df["month"] = parsed.apply(lambda x: x[0])
    df["day"] = parsed.apply(lambda x: x[1])
    df["dep_hour"] = parsed.apply(lambda x: x[2])
    df["dep_minute"] = parsed.apply(lambda x: x[3])
    # crude day-of-week proxy (no year, treat Jan 1 = Monday=0)
    df["dow"] = ((df["day"] - 1) % 7)
    df["origin"] = df["origin"].astype("category")
    df["destination"] = df["destination"].astype("category")
    df["route"] = (df["origin"].astype(str) + "-" + df["destination"].astype(str)).astype("category")
    return df


FEATURES = ["distance", "month", "day", "dep_hour", "dep_minute", "dow",
            "origin", "destination", "route"]
CATEGORICAL = ["origin", "destination", "route"]


def main(sample):
    print(f"Loading {DATA} ...")
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["date", "delay", "distance", "origin", "destination"])
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    print(f"Rows: {len(df):,}")

    df = build_features(df)
    df["label"] = (df["delay"] > DELAY_THRESHOLD).astype(int)
    print(f"Delayed rate (>{DELAY_THRESHOLD}m): {df['label'].mean():.1%}")

    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(Xtr, ytr, categorical_feature=CATEGORICAL,
              eval_set=[(Xte, yte)], eval_metric="auc",
              callbacks=[lgb.early_stopping(30, verbose=False)])

    prob = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, prob)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print(classification_report(yte, (prob > 0.5).astype(int), digits=3))

    # persist model + the category vocabularies the app needs for input widgets
    joblib.dump(model, os.path.join(MODELS, "delay_model.joblib"))
    meta = {
        "features": FEATURES, "categorical": CATEGORICAL,
        "threshold": DELAY_THRESHOLD, "auc": float(auc),
        "origins": sorted(df["origin"].cat.categories.tolist()),
        "destinations": sorted(df["destination"].cat.categories.tolist()),
        "routes": df["route"].cat.categories.tolist(),
    }
    joblib.dump(meta, os.path.join(MODELS, "delay_meta.joblib"))
    # save a background sample for SHAP in the app
    joblib.dump(Xtr.sample(min(500, len(Xtr)), random_state=0),
                os.path.join(MODELS, "delay_background.joblib"))
    print(f"Saved model + meta to {MODELS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120000,
                    help="rows to use (0 = all)")
    args = ap.parse_args()
    main(args.sample or None)
