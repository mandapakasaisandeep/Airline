"""
Train the sentiment classifier.

Two backends, same interface:
  --backend scratch : portable attention transformer (CPU, no downloads). DEFAULT.
  --backend hf      : DistilBERT + LoRA (PEFT). Production path; needs internet
                      the first time to pull 'distilbert-base-uncased'.

Both save to models/ so the Streamlit app loads whichever exists.
Dataset: 100k-tweet binary sentiment corpus (Sentiment: 0=neg, 1=pos).
"""
import os, argparse, json, random, joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "sentiment_tweets.csv")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)


def load_data(n):
    df = pd.read_csv(DATA, encoding="latin-1")
    df = df.rename(columns={"SentimentText": "text", "Sentiment": "label"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    if n and len(df) > n:
        df = df.sample(n, random_state=42).reset_index(drop=True)
    return df


# ----------------------------- scratch backend -----------------------------
def train_scratch(df, epochs, max_len=40):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sentiment_model import Vocab, TransformerClassifier, collate

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    tr, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    vocab = Vocab(tr["text"].tolist())
    print(f"Vocab size: {len(vocab):,} | train {len(tr):,} test {len(te):,}")

    def make(split):
        X = [vocab.encode(t, max_len) for t in split["text"]]
        y = split["label"].tolist()
        return list(zip(X, y))

    tr_ds, te_ds = make(tr), make(te)

    def batches(ds, bs, shuffle):
        idx = list(range(len(ds)))
        if shuffle: random.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [ds[j] for j in idx[i:i+bs]]
            xb = collate([c[0] for c in chunk], max_len)
            yb = torch.tensor([c[1] for c in chunk])
            yield xb, yb

    model = TransformerClassifier(len(vocab), max_len=max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    lossf = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train(); tot = 0
        for xb, yb in batches(tr_ds, 128, True):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item()
        # eval
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for xb, yb in batches(te_ds, 256, False):
                preds += model(xb).argmax(1).tolist(); gts += yb.tolist()
        acc, f1 = accuracy_score(gts, preds), f1_score(gts, preds)
        print(f"epoch {ep+1}/{epochs}  loss {tot:.1f}  acc {acc:.3f}  f1 {f1:.3f}")

    torch.save(model.state_dict(), os.path.join(MODELS, "sentiment_scratch.pt"))
    joblib.dump(vocab, os.path.join(MODELS, "sentiment_vocab.joblib"))
    json.dump({"backend": "scratch", "max_len": max_len, "acc": acc, "f1": f1},
              open(os.path.join(MODELS, "sentiment_meta.json"), "w"))
    print(f"Saved scratch model (acc {acc:.3f}, f1 {f1:.3f}) to {MODELS}")


# ------------------------------- hf backend --------------------------------
def train_hf(df, epochs):
    """DistilBERT + LoRA. Runs on the user's machine (needs HF hub once)."""
    import torch
    from sklearn.model_selection import train_test_split
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    tr, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)

    def to_ds(s):
        d = Dataset.from_pandas(s[["text", "label"]])
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=64,
                                   padding="max_length"), batched=True)

    tr_ds, te_ds = to_ds(tr), to_ds(te)
    base = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                      lora_dropout=0.1, target_modules=["q_lin", "v_lin"])
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    def metrics(p):
        from sklearn.metrics import accuracy_score, f1_score
        pred = p.predictions.argmax(-1)
        return {"acc": accuracy_score(p.label_ids, pred),
                "f1": f1_score(p.label_ids, pred)}

    args = TrainingArguments(output_dir=os.path.join(MODELS, "hf_ckpt"),
                             per_device_train_batch_size=16, num_train_epochs=epochs,
                             eval_strategy="epoch", learning_rate=2e-4,
                             logging_steps=50, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds,
                      eval_dataset=te_ds, compute_metrics=metrics)
    trainer.train()
    model.save_pretrained(os.path.join(MODELS, "sentiment_hf"))
    tok.save_pretrained(os.path.join(MODELS, "sentiment_hf"))
    json.dump({"backend": "hf"}, open(os.path.join(MODELS, "sentiment_meta.json"), "w"))
    print(f"Saved DistilBERT+LoRA adapter to {MODELS}/sentiment_hf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scratch", "hf"], default="scratch")
    ap.add_argument("--n", type=int, default=40000, help="rows to use")
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    df = load_data(a.n)
    print(f"Loaded {len(df):,} rows | positive rate {df['label'].mean():.1%}")
    (train_hf if a.backend == "hf" else train_scratch)(df, a.epochs)
