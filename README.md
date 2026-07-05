# ✈️ Airline Ops + Customer Intelligence

An end-to-end ML system that unifies the two levers an airline operations team
actually pulls: **operational risk** (*will this flight be late?*) and
**customer signal** (*how do flyers feel?*) — each with first-class
interpretability, in one Streamlit dashboard.

> Built as a portfolio project to demonstrate the full ML lifecycle: data →
> two model families → explainability → deployment.

---

## What it does

| Module | Task | Model | Interpretability |
|---|---|---|---|
| **Delay Risk** | Predict `P(delay > 15 min)` for a scheduled flight | LightGBM (gradient-boosted trees) | **TreeSHAP** per-prediction attributions |
| **Customer Sentiment** | Classify feedback as positive / negative | DistilBERT + **LoRA** *(prod)* or a from-scratch **attention transformer** *(portable)* | **Token attention** highlights |

Both modules ship behind a single dashboard so a prediction is never a black box —
every delay score comes with *why*, every sentiment call with *which words drove it*.

---

## Architecture

```
        data/                         src/                        models/
  ┌──────────────────┐        ┌────────────────────┐       ┌──────────────────┐
  │ flights.csv 231k │──────► │ train_delay.py     │─────► │ delay_model      │
  │ (date,delay,...) │        │  feat-eng + LGBM   │       │ + SHAP background│
  └──────────────────┘        └────────────────────┘       └──────────────────┘
  ┌──────────────────┐        ┌────────────────────┐       ┌──────────────────┐
  │ sentiment_tweets │──────► │ train_sentiment.py │─────► │ scratch .pt +vocab│
  │ .csv  100k       │        │  backend: scratch  │       │   OR              │
  │                  │        │           | hf     │       │ distilbert+LoRA   │
  └──────────────────┘        └────────────────────┘       └──────────────────┘
                                        │                            │
                                        └──────────► app.py ◄────────┘
                                              (Streamlit dashboard)
```

## Results (as trained)

- **Delay model:** ROC-AUC **0.886** on held-out data; recall 0.77 on the
  minority *delayed* class (20% base rate, handled via `class_weight="balanced"`).
- **Sentiment (scratch transformer, CPU):** acc **0.74** / F1 **0.77** on noisy
  tweets. The DistilBERT + LoRA backend typically reaches ~0.85 on the same data.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) train the delay model (seconds on CPU)
python src/train_delay.py --sample 120000

# 2a) train sentiment — portable, no downloads (CPU, a few minutes)
python src/train_sentiment.py --backend scratch --n 40000 --epochs 4

# 2b) OR the production path — DistilBERT + LoRA (needs internet once)
python src/train_sentiment.py --backend hf --n 40000 --epochs 3

# 3) launch the dashboard
streamlit run app.py
```

The app auto-detects whichever sentiment backend you trained.

### Data
- **Flights:** `flights-3m.csv` from the vega-datasets project (231k rows).
- **Sentiment:** a 100k-row binary tweet corpus. For domain fit, swap in the
  *Twitter US Airline Sentiment* dataset — the code only expects `text` + `label`.

---

## Design decisions & tradeoffs

**LightGBM over a neural net for delays.** Tabular, mixed categorical/numeric,
~230k rows — gradient-boosted trees win on accuracy *and* training cost here, and
they pair with exact, fast **TreeSHAP** so every score is explainable.

**Backend-agnostic sentiment interface.** The same dashboard runs either a
**DistilBERT + LoRA** adapter or a **from-scratch attention transformer** exposing
the identical predict/attention API. This decouples deployment from the training
environment and made it possible to develop + demo entirely offline while keeping a
production-grade path.

**Why LoRA?** Full fine-tuning updates all 66M DistilBERT params. LoRA injects
low-rank adapters into the attention `q`/`v` projections and trains a few hundred K
(<1%) — comparable accuracy, minutes on a laptop, a few-MB artifact to ship.

**Interpretability is a feature, not an afterthought.** SHAP for the tree model,
attention for the transformer. Both surface in the UI so a non-ML stakeholder can
read *why*.

## Honest limitations (and next steps)
- Sentiment corpus is generic tweets, not airline-specific → swap dataset.
- Delay data has no weather → add METAR/visibility features (biggest expected lift).
- Day-of-week is approximated from day-of-month (source omits year).
- Sentiment is binary → extend to neutral / intent / urgency for triage.

---

## Repo layout
```
airline_intel/
├── app.py                    # Streamlit dashboard (3 tabs)
├── requirements.txt
├── README.md
├── data/                     # flights.csv, sentiment_tweets.csv
├── models/                   # trained artifacts (created by the scripts)
└── src/
    ├── train_delay.py        # feature-eng + LightGBM + SHAP export
    ├── sentiment_model.py    # from-scratch attention transformer + vocab
    └── train_sentiment.py    # scratch | hf(LoRA) backends, one interface
```
