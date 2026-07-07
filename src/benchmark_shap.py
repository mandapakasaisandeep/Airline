"""
Benchmark A — Explanation speed: manual perturbation attribution vs. TreeSHAP.

Motivation: to explain *why* a flight is flagged, I first tried computing feature
attributions manually — perturb each feature, re-predict, measure the change in
output. It works, but it's slow: it requires N extra model calls per feature per
prediction. I then switched to TreeSHAP, which computes exact attributions by
exploiting the tree structure. This script measures the real speedup.

Produces honest, quotable numbers for the resume / interview.
"""
import os, time, joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")


def manual_attribution(model, row, background, feats, cats, n_samples=100):
    """Naive perturbation attribution: for each feature, replace its value with
    values sampled from the background distribution, re-predict, and measure the
    average change in predicted probability. This approximates each feature's
    marginal effect — the same idea SHAP formalizes, but done by brute force."""
    base = model.predict_proba(row)[:, 1][0]
    attributions = {}
    for f in feats:
        sampled = background[f].sample(n_samples, replace=True, random_state=0).values
        perturbed = pd.concat([row] * n_samples, ignore_index=True)
        perturbed[f] = sampled
        for c in cats:
            perturbed[c] = perturbed[c].astype("category")
        preds = model.predict_proba(perturbed)[:, 1]
        # attribution = how much the prediction moves when this feature is randomized
        attributions[f] = float(base - preds.mean())
    return attributions


def main():
    model = joblib.load(os.path.join(MODELS, "delay_model.joblib"))
    meta = joblib.load(os.path.join(MODELS, "delay_meta.joblib"))
    bg = joblib.load(os.path.join(MODELS, "delay_background.joblib"))
    feats, cats = meta["features"], meta["categorical"]

    row = bg.iloc[[0]].copy()
    for c in cats:
        row[c] = row[c].astype("category")

    # ---- MANUAL perturbation attribution ----
    # warmup
    manual_attribution(model, row, bg, feats, cats, n_samples=50)
    n_runs = 20
    t = time.perf_counter()
    for _ in range(n_runs):
        manual_attr = manual_attribution(model, row, bg, feats, cats, n_samples=100)
    manual_ms = (time.perf_counter() - t) / n_runs * 1000

    # ---- TreeSHAP ----
    import shap
    explainer = shap.TreeExplainer(model)
    # warmup
    _ = explainer(row)
    t = time.perf_counter()
    for _ in range(n_runs):
        exp = explainer(row)
        shap_vals = np.asarray(exp.values)
    shap_ms = (time.perf_counter() - t) / n_runs * 1000

    speedup = manual_ms / shap_ms

    print("=" * 60)
    print("EXPLANATION SPEED BENCHMARK  (per single prediction)")
    print("=" * 60)
    print(f"Manual perturbation attribution : {manual_ms:8.1f} ms")
    print(f"TreeSHAP                        : {shap_ms:8.2f} ms")
    print(f"Speedup                         : {speedup:8.1f}x faster")
    print()
    print("Why manual is slow: it needs ~100 extra model calls per feature")
    print(f"  = {len(feats)} features x 100 samples = {len(feats)*100} predictions per explanation.")
    print("TreeSHAP computes exact attributions from the tree structure in one pass.")
    print()
    print("Sanity check — both rank the top features similarly:")
    manual_sorted = sorted(manual_attr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    shap_row = np.asarray(exp.values)[0]
    if shap_row.ndim > 1:
        shap_row = shap_row[:, -1]
    shap_pairs = sorted(zip(feats, shap_row), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"  Manual top-3 : {[f for f,_ in manual_sorted]}")
    print(f"  SHAP   top-3 : {[f for f,_ in shap_pairs]}")

    result = {"manual_ms": manual_ms, "shap_ms": shap_ms, "speedup": speedup}
    joblib.dump(result, os.path.join(MODELS, "shap_benchmark.joblib"))
    print(f"\nSaved -> models/shap_benchmark.joblib")


if __name__ == "__main__":
    main()
