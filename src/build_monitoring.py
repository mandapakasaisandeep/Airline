"""
Compute model-monitoring artifacts for the delay model:
  - Performance: confusion matrix, precision/recall/F1, ROC/PR curves, calibration
  - Inference: single + batch latency, throughput, model size
  - Drift: PSI + KS-test comparing a REFERENCE window vs a CURRENT window

Drift note: we don't have a live production stream, so we simulate the
reference/current split by partitioning the flights data into two halves
(early vs late rows) — a standard, honest way to demonstrate a drift pipeline.
Everything here is computed on real data; only the ref/current partition is a
stand-in for "training vs production".

Output: models/monitoring.joblib (loaded by the app's Monitoring tab).
"""
import os, time, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_curve, precision_recall_curve, roc_auc_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "flights.csv")
MODELS = os.path.join(HERE, "models")

DELAY_THRESHOLD = 15


def parse_date(v):
    s = str(int(v)).zfill(8)
    return int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])


def build_features(df):
    p = df["date"].apply(parse_date)
    df["month"] = p.apply(lambda x: x[0]); df["day"] = p.apply(lambda x: x[1])
    df["dep_hour"] = p.apply(lambda x: x[2]); df["dep_minute"] = p.apply(lambda x: x[3])
    df["dow"] = ((df["day"] - 1) % 7)
    df["origin"] = df["origin"].astype("category")
    df["destination"] = df["destination"].astype("category")
    df["route"] = (df["origin"].astype(str) + "-" + df["destination"].astype(str)).astype("category")
    return df


def psi(ref, cur, bins=10):
    """Population Stability Index between two 1-D distributions."""
    edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=bins)
    r = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    r = np.clip(r, 1e-4, None); c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_stat(ref, cur):
    """Kolmogorov–Smirnov statistic (max CDF gap) without scipy."""
    allv = np.sort(np.concatenate([ref, cur]))
    cdf_r = np.searchsorted(np.sort(ref), allv, side="right") / len(ref)
    cdf_c = np.searchsorted(np.sort(cur), allv, side="right") / len(cur)
    return float(np.max(np.abs(cdf_r - cdf_c)))


def main():
    model = joblib.load(os.path.join(MODELS, "delay_model.joblib"))
    meta = joblib.load(os.path.join(MODELS, "delay_meta.joblib"))
    feats, cats = meta["features"], meta["categorical"]

    df = pd.read_csv(DATA).dropna(subset=["date", "delay", "distance", "origin", "destination"])
    df = build_features(df)
    df["label"] = (df["delay"] > DELAY_THRESHOLD).astype(int)

    # ---- held-out performance ----
    X, y = df[feats], df["label"]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob > 0.5).astype(int)

    cm = confusion_matrix(yte, pred).tolist()
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
    fpr, tpr, _ = roc_curve(yte, prob)
    prec_c, rec_c, _ = precision_recall_curve(yte, prob)
    frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=10)
    perf = {
        "confusion_matrix": cm,
        "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "auc": float(roc_auc_score(yte, prob)),
        "brier": float(brier_score_loss(yte, prob)),
        "roc": {"fpr": fpr[::max(1, len(fpr)//200)].tolist(),
                "tpr": tpr[::max(1, len(tpr)//200)].tolist()},
        "pr": {"precision": prec_c[::max(1, len(prec_c)//200)].tolist(),
               "recall": rec_c[::max(1, len(rec_c)//200)].tolist()},
        "calibration": {"pred": mean_pred.tolist(), "true": frac_pos.tolist()},
        "test_n": int(len(yte)), "positive_rate": float(yte.mean()),
    }

    # ---- inference latency ----
    one = Xte.iloc[[0]]
    for _ in range(5):  # warmup
        model.predict_proba(one)
    t = time.perf_counter()
    for _ in range(200):
        model.predict_proba(one)
    single_ms = (time.perf_counter() - t) / 200 * 1000

    batch = Xte.iloc[:1000]
    t = time.perf_counter(); model.predict_proba(batch)
    batch_s = time.perf_counter() - t
    size_mb = os.path.getsize(os.path.join(MODELS, "delay_model.joblib")) / 1e6
    inference = {
        "single_ms": float(single_ms),
        "batch_1000_ms": float(batch_s * 1000),
        "throughput_per_s": float(1000 / batch_s),
        "model_size_mb": float(size_mb),
        "n_trees": int(model.n_estimators_),
        "n_features": len(feats),
    }

    # ---- drift: reference (first half) vs current (second half) ----
    half = len(df) // 2
    ref, cur = df.iloc[:half], df.iloc[half:]
    numeric = ["distance", "month", "day", "dep_hour", "dep_minute", "dow"]
    drift_rows = []
    for f in numeric:
        p = psi(ref[f].values.astype(float), cur[f].values.astype(float))
        k = ks_stat(ref[f].values.astype(float), cur[f].values.astype(float))
        level = "high" if p > 0.2 else ("moderate" if p > 0.1 else "low")
        drift_rows.append({"feature": f, "psi": round(p, 4), "ks": round(k, 4),
                           "drift": level})
    # prediction drift: score distribution ref vs cur
    ref_prob = model.predict_proba(ref[feats])[:, 1]
    cur_prob = model.predict_proba(cur[feats])[:, 1]
    pred_psi = psi(ref_prob, cur_prob)
    drift = {
        "features": drift_rows,
        "prediction_psi": round(float(pred_psi), 4),
        "ref_n": int(len(ref)), "cur_n": int(len(cur)),
        "ref_pos_rate": float(ref["label"].mean()),
        "cur_pos_rate": float(cur["label"].mean()),
        "ref_score_hist": np.histogram(ref_prob, bins=20, range=(0, 1))[0].tolist(),
        "cur_score_hist": np.histogram(cur_prob, bins=20, range=(0, 1))[0].tolist(),
    }

    out = {"perf": perf, "inference": inference, "drift": drift,
           "generated": time.strftime("%Y-%m-%d %H:%M")}
    joblib.dump(out, os.path.join(MODELS, "monitoring.joblib"))
    print("Saved monitoring.joblib")
    print(f"  AUC {perf['auc']:.3f}  P {pr:.3f}  R {rc:.3f}  F1 {f1:.3f}  Brier {perf['brier']:.3f}")
    print(f"  single {single_ms:.2f} ms  throughput {inference['throughput_per_s']:.0f}/s  size {size_mb:.1f} MB")
    print(f"  prediction PSI {pred_psi:.4f}  (ref pos {drift['ref_pos_rate']:.1%} -> cur {drift['cur_pos_rate']:.1%})")
    for d in drift_rows:
        print(f"    {d['feature']:10s} PSI {d['psi']:.3f} KS {d['ks']:.3f} -> {d['drift']}")


if __name__ == "__main__":
    main()
