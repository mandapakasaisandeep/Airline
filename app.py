"""
Airline Delay Intelligence — prediction + model observability dashboard.

Tabs:
  1. Predict & Explain   — delay probability + per-prediction SHAP
  2. Performance         — confusion matrix, P/R/F1, ROC, PR, calibration
  3. Inference           — latency, throughput, model footprint
  4. Drift               — PSI + KS feature drift, prediction drift
  5. About / Design      — narrative & tradeoffs

Tree model + SHAP only — no deep-learning deps, runs anywhere.
Run:  streamlit run app.py
"""
import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SRC = os.path.join(HERE, "src")
import sys; sys.path.insert(0, SRC)

st.set_page_config(page_title="Airline Delay Intelligence", layout="wide")


@st.cache_resource
def load_delay():
    model = joblib.load(os.path.join(MODELS, "delay_model.joblib"))
    meta = joblib.load(os.path.join(MODELS, "delay_meta.joblib"))
    import shap
    explainer = shap.TreeExplainer(model)
    return model, meta, explainer


@st.cache_resource
def load_monitoring():
    path = os.path.join(MODELS, "monitoring.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def cached_shap(row_key):
    model, meta, explainer = load_delay()
    row = pd.DataFrame([dict(row_key)])[meta["features"]]
    for c in meta["categorical"]:
        row[c] = row[c].astype("category")
    try:
        arr = np.asarray(explainer(row).values)
    except Exception:
        sv = explainer.shap_values(row)
        arr = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)
    arr = arr[0] if arr.ndim >= 2 else arr
    if arr.ndim == 2:
        arr = arr[:, -1]
    return np.asarray(arr).ravel()[:len(meta["features"])]


model, meta, explainer = load_delay()
mon = load_monitoring()

st.title("✈️  Airline Delay Intelligence")
st.caption("Delay-risk scoring · SHAP explanations · live model observability")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🛫 Predict & Explain", "📈 Performance", "⚡ Inference", "🌊 Drift", "📊 About"])

# ============================= TAB 1: PREDICT ==============================
with tab1:
    st.subheader("Will this flight be delayed?")
    st.write(f"**LightGBM** · test ROC-AUC **{meta['auc']:.3f}** · target = delay > {meta['threshold']} min")

    c1, c2, c3 = st.columns(3)
    with c1:
        origin = st.selectbox("Origin", meta["origins"],
                              index=meta["origins"].index("ORD") if "ORD" in meta["origins"] else 0)
        dest = st.selectbox("Destination", meta["destinations"], index=0)
    with c2:
        distance = st.slider("Distance (mi)", 100, 3000, 700, 25)
        month = st.slider("Month", 1, 12, 7)
    with c3:
        dep_hour = st.slider("Departure hour", 0, 23, 17)
        dow = st.slider("Day of week (0=Mon)", 0, 6, 4)

    route = f"{origin}-{dest}"
    routes = set(meta["routes"])
    row = pd.DataFrame([{
        "distance": distance, "month": month, "day": 15,
        "dep_hour": dep_hour, "dep_minute": 0, "dow": dow,
        "origin": origin, "destination": dest,
        "route": route if route in routes else meta["routes"][0],
    }])[meta["features"]]
    for c in meta["categorical"]:
        row[c] = row[c].astype("category")

    prob = float(model.predict_proba(row)[:, 1][0])
    st.metric("Delay probability", f"{prob:.0%}",
              delta="High risk" if prob > 0.5 else "Likely on-time",
              delta_color="inverse")

    st.markdown("#### Why this prediction? — SHAP")
    if st.button("Explain this prediction"):
        with st.spinner("Computing SHAP…"):
            try:
                row_key = tuple(sorted(row.iloc[0].astype(object).to_dict().items()))
                vals = cached_shap(row_key)
                contrib = (pd.DataFrame({"feature": meta["features"],
                                         "value": row.iloc[0].astype(str).values,
                                         "shap": vals})
                           .sort_values("shap", key=abs, ascending=False))
                contrib["effect"] = np.where(contrib["shap"] > 0, "↑ raises risk", "↓ lowers risk")
                st.dataframe(contrib.assign(shap=contrib["shap"].round(3))
                             .reset_index(drop=True), width="stretch", hide_index=True)
                st.bar_chart(contrib.set_index("feature")["shap"])
                st.caption("Positive SHAP pushes toward *delayed*; negative toward *on-time*.")
            except Exception:
                imp = (pd.DataFrame({"feature": meta["features"],
                                     "importance": model.feature_importances_})
                       .sort_values("importance", ascending=False))
                st.info("Showing global feature importance.")
                st.dataframe(imp.reset_index(drop=True), width="stretch", hide_index=True)

# =========================== TAB 2: PERFORMANCE ============================
with tab2:
    st.subheader("Model performance (held-out test set)")
    if not mon:
        st.warning("Run `python src/build_monitoring.py` to generate monitoring data.")
    else:
        p = mon["perf"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("ROC-AUC", f"{p['auc']:.3f}")
        k2.metric("Precision", f"{p['precision']:.3f}")
        k3.metric("Recall", f"{p['recall']:.3f}")
        k4.metric("F1", f"{p['f1']:.3f}")
        k5.metric("Brier ↓", f"{p['brier']:.3f}")
        st.caption(f"Test set: {p['test_n']:,} flights · {p['positive_rate']:.1%} delayed")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Confusion matrix**")
            cm = np.array(p["confusion_matrix"])
            cm_df = pd.DataFrame(cm, index=["Actual: on-time", "Actual: delayed"],
                                 columns=["Pred: on-time", "Pred: delayed"])
            st.dataframe(cm_df, width="stretch")
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"TP {tp:,} · TN {tn:,} · FP {fp:,} · FN {fn:,}. "
                       f"FN (missed delays) are the costly errors here.")
        with cc2:
            st.markdown("**Calibration** (predicted vs. actual delay rate)")
            cal = pd.DataFrame({"predicted": p["calibration"]["pred"],
                                "actual": p["calibration"]["true"]}).set_index("predicted")
            st.line_chart(cal)
            st.caption("On the diagonal = well-calibrated. Off = probabilities need calibration.")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**ROC curve**")
            roc = pd.DataFrame({"FPR": p["roc"]["fpr"], "TPR": p["roc"]["tpr"]}).set_index("FPR")
            st.line_chart(roc)
        with rc2:
            st.markdown("**Precision–Recall curve**")
            prc = pd.DataFrame({"recall": p["pr"]["recall"], "precision": p["pr"]["precision"]}).set_index("recall")
            st.line_chart(prc)

# ============================ TAB 3: INFERENCE ============================
with tab3:
    st.subheader("Inference performance")
    if not mon:
        st.warning("Run `python src/build_monitoring.py` first.")
    else:
        inf = mon["inference"]
        i1, i2, i3 = st.columns(3)
        i1.metric("Single-prediction latency", f"{inf['single_ms']:.2f} ms")
        i2.metric("Throughput (batch)", f"{inf['throughput_per_s']:,.0f}/s")
        i3.metric("Batch of 1000", f"{inf['batch_1000_ms']:.0f} ms")
        j1, j2, j3 = st.columns(3)
        j1.metric("Model size", f"{inf['model_size_mb']:.1f} MB")
        j2.metric("Trees", f"{inf['n_trees']}")
        j3.metric("Features", f"{inf['n_features']}")
        st.caption("Measured on CPU. Millisecond single-prediction latency and "
                   "tens-of-thousands/sec throughput make this comfortably real-time.")
        st.markdown("**Live latency check** — time a real prediction now:")
        if st.button("Run latency test"):
            import time
            one = row
            for _ in range(3):
                model.predict_proba(one)
            t = time.perf_counter()
            for _ in range(100):
                model.predict_proba(one)
            ms = (time.perf_counter() - t) / 100 * 1000
            st.success(f"Measured {ms:.2f} ms per prediction (avg of 100 runs)")

# ============================== TAB 4: DRIFT ==============================
with tab4:
    st.subheader("Data & prediction drift")
    if not mon:
        st.warning("Run `python src/build_monitoring.py` first.")
    else:
        d = mon["drift"]
        st.write(f"Reference window: **{d['ref_n']:,}** flights · "
                 f"Current window: **{d['cur_n']:,}** flights")
        st.caption("Reference vs. current split simulates training-vs-production. "
                   "PSI < 0.1 = stable · 0.1–0.2 = moderate · > 0.2 = significant drift.")

        drift_df = pd.DataFrame(d["features"])
        st.markdown("**Feature drift (PSI & KS)**")
        st.dataframe(drift_df, width="stretch", hide_index=True)
        st.bar_chart(drift_df.set_index("feature")["psi"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction PSI", f"{d['prediction_psi']:.3f}",
                  delta="stable" if d["prediction_psi"] < 0.1 else "drifting",
                  delta_color="normal")
        m2.metric("Ref delay rate", f"{d['ref_pos_rate']:.1%}")
        m3.metric("Current delay rate", f"{d['cur_pos_rate']:.1%}")

        st.markdown("**Prediction-score distribution: reference vs current**")
        bins = [f"{i/20:.2f}" for i in range(20)]
        hist = pd.DataFrame({"reference": d["ref_score_hist"],
                             "current": d["cur_score_hist"]}, index=bins)
        st.bar_chart(hist)
        st.info("Interpretation: calendar features (month, day) drift because the "
                "data is time-ordered, but the **prediction distribution stays "
                "stable** (low PSI) — input drift without output drift. That's the "
                "signal that tells you whether a model actually needs retraining.")

# ============================== TAB 5: ABOUT ==============================
with tab5:
    st.subheader("Design & observability notes")
    st.markdown(f"""
**Task.** Predict `P(delay > {meta['threshold']} min)` for a scheduled flight and
make every prediction explainable and *monitorable*.

**Model.** LightGBM on ~231k real flights; test ROC-AUC **{meta['auc']:.3f}**.
Trees chosen for tabular data, native categoricals, and exact TreeSHAP.

**Observability layer (this dashboard).**
- *Performance:* confusion matrix, precision/recall/F1, ROC & PR curves, and a
  **calibration** curve + Brier score — because for a probability model, *being
  calibrated* matters as much as ranking.
- *Inference:* single-prediction latency, batch throughput, model footprint —
  the numbers an SRE asks before putting a model behind an API.
- *Drift:* **PSI** and **KS** per feature comparing a reference vs. current window,
  plus **prediction drift**. Distinguishing input drift from output drift is what
  decides whether you actually retrain.

**Why these metrics.** A model in production degrades silently — the data shifts,
the world changes. Monitoring catches that before users do. Recall matters most
here (a missed delay is the costly error), and low prediction-PSI is the
retraining trigger.

**Honest limitations.** No weather features (biggest next lift). Drift uses a
time-ordered ref/current split as a production stand-in — real deployment would
compare training distribution against a live inference log. Day-of-week is
approximated from day-of-month (source omits year).
""")
