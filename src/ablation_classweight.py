"""
Benchmark B — Class-weight ablation: does balanced weighting actually help?

Motivation: delays are only ~20% of flights, so a model optimizing raw accuracy
can ignore the minority class. I trained two identical LightGBM models — one with
default weighting, one with class_weight="balanced" — and compared them on the
metric that matters operationally: recall on the *delayed* class (catching real
delays). This quantifies the impact of that single design decision.

Produces honest before/after numbers for the resume / interview.
"""
import os, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             confusion_matrix)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "flights.csv")
MODELS = os.path.join(HERE, "models")
DELAY_THRESHOLD = 15


def parse_date(v):
    s = str(int(v)).zfill(8)
    return int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])


def build(df):
    p = df["date"].apply(parse_date)
    df["month"] = p.apply(lambda x: x[0]); df["day"] = p.apply(lambda x: x[1])
    df["dep_hour"] = p.apply(lambda x: x[2]); df["dep_minute"] = p.apply(lambda x: x[3])
    df["dow"] = ((df["day"] - 1) % 7)
    for c in ["origin", "destination"]:
        df[c] = df[c].astype("category")
    df["route"] = (df["origin"].astype(str) + "-" + df["destination"].astype(str)).astype("category")
    return df


FEATURES = ["distance", "month", "day", "dep_hour", "dep_minute", "dow",
            "origin", "destination", "route"]
CATS = ["origin", "destination", "route"]


def train(Xtr, ytr, Xte, yte, balanced):
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        class_weight=("balanced" if balanced else None), verbose=-1)
    m.fit(Xtr, ytr, categorical_feature=CATS)
    prob = m.predict_proba(Xte)[:, 1]
    pred = (prob > 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
    cm = confusion_matrix(yte, pred)
    return {"precision": pr, "recall": rc, "f1": f1,
            "auc": roc_auc_score(yte, prob), "cm": cm}


def main(sample=120000):
    df = pd.read_csv(DATA).dropna(subset=["date", "delay", "distance", "origin", "destination"])
    if len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    df = build(df)
    df["label"] = (df["delay"] > DELAY_THRESHOLD).astype(int)
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    base = train(Xtr, ytr, Xte, yte, balanced=False)
    bal = train(Xtr, ytr, Xte, yte, balanced=True)

    print("=" * 64)
    print("CLASS-WEIGHT ABLATION  (delayed = positive class, ~20% base rate)")
    print("=" * 64)
    print(f"{'metric':<12}{'baseline':>12}{'balanced':>12}{'change':>12}")
    for k in ["recall", "precision", "f1", "auc"]:
        d = bal[k] - base[k]
        print(f"{k:<12}{base[k]:>12.3f}{bal[k]:>12.3f}{d:>+12.3f}")
    print()
    print("Confusion matrices  [rows=actual, cols=pred]  (on-time / delayed):")
    print("  baseline :", base["cm"].tolist())
    print("  balanced :", bal["cm"].tolist())
    # missed delays = false negatives (bottom-left)
    base_fn, bal_fn = base["cm"][1][0], bal["cm"][1][0]
    print()
    print(f"Missed delays (false negatives): baseline {base_fn:,} -> balanced {bal_fn:,}")
    print(f"  => balanced weighting catches {base_fn - bal_fn:,} more real delays")
    print()
    print("TAKEAWAY: balanced weighting raises recall on delayed flights from "
          f"{base['recall']:.1%} to {bal['recall']:.1%}, trading some precision — the "
          "right call when a missed delay costs more than a false alarm.")

    joblib.dump({"baseline": {k: (v.tolist() if hasattr(v, 'tolist') else v)
                              for k, v in base.items()},
                 "balanced": {k: (v.tolist() if hasattr(v, 'tolist') else v)
                              for k, v in bal.items()}},
                os.path.join(MODELS, "classweight_ablation.joblib"))
    print("\nSaved -> models/classweight_ablation.joblib")


if __name__ == "__main__":
    main()
