"""Minimal sentiment-only Streamlit app to isolate the hang."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import sys, json, joblib
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
MODELS = os.path.join(HERE, "models")

st.title("Sentiment isolation test")

@st.cache_resource
def load():
    import torch
    from sentiment_model import TransformerClassifier
    meta = json.load(open(os.path.join(MODELS, "sentiment_meta.json")))
    vocab = joblib.load(os.path.join(MODELS, "sentiment_vocab.joblib"))
    model = TransformerClassifier(len(vocab), max_len=meta["max_len"])
    model.load_state_dict(torch.load(os.path.join(MODELS, "sentiment_scratch.pt"),
                                     map_location="cpu"))
    model.eval()
    return torch, model, vocab, meta

st.write("Loading model…")
torch, model, vocab, meta = load()
st.success("Model loaded ✓")

txt = st.text_area("Text", "the flight was delayed and no one helped")
if st.button("Analyze"):
    ids = torch.tensor([vocab.encode(txt, meta["max_len"])])
    with torch.no_grad():
        logits, att = model(ids, return_attn=True)
    p = torch.softmax(logits, 1)[0].tolist()
    st.metric("Sentiment", "Positive" if p[1] > p[0] else "Negative", f"{max(p):.0%}")
