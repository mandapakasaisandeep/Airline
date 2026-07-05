"""
Lightweight self-attention transformer for sentiment, trainable from scratch
on CPU. Exposes per-token attention weights so the dashboard can highlight
which words drove the prediction (interpretability for the NLP side).

This mirrors what DistilBERT gives you, but with zero external downloads so it
runs anywhere. The HF+LoRA backend (train_sentiment.py --backend hf) is the
production path; this is the portable fallback + teaching tool.
"""
import re
import torch
import torch.nn as nn

PAD, UNK = "<pad>", "<unk>"


def tokenize(text):
    return re.findall(r"[a-z']+", str(text).lower())


class Vocab:
    def __init__(self, texts=None, max_size=20000, min_freq=2):
        self.itos = [PAD, UNK]
        if texts is not None:
            from collections import Counter
            c = Counter(t for txt in texts for t in tokenize(txt))
            for w, f in c.most_common(max_size):
                if f >= min_freq:
                    self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_len=40):
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:max_len]
        if not ids:
            ids = [1]
        return ids

    def __len__(self):
        return len(self.itos)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        # mask: True where PAD -> key_padding_mask
        out, w = self.attn(x, x, x, key_padding_mask=mask, need_weights=True,
                           average_attn_weights=True)
        return self.norm(x + out), w


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, dim=96, heads=4, layers=2, max_len=40, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([SelfAttention(dim, heads) for _ in range(layers)])
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Linear(dim, n_classes)
        self.max_len = max_len

    def forward(self, ids, return_attn=False):
        mask = ids.eq(0)  # (B,L) True where pad
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.emb(ids) + self.pos(pos)
        last_w = None
        for blk in self.blocks:
            x, last_w = blk(x, mask)
        x = self.ff(x)
        # masked mean pool
        keep = (~mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        logits = self.head(pooled)
        if return_attn:
            # importance per token = attention received (col-mean), zero pads
            att = last_w.masked_fill(mask.unsqueeze(1), 0).mean(1)  # (B,L)
            return logits, att
        return logits


def collate(batch, pad_to):
    ids = [torch.tensor(b) for b in batch]
    out = torch.zeros(len(ids), pad_to, dtype=torch.long)
    for i, t in enumerate(ids):
        out[i, :len(t)] = t[:pad_to]
    return out
