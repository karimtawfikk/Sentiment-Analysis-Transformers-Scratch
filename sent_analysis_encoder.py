import re
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter

# ---------- Tokenizer & Vocab ----------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Load vocab from external file
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Ensure keys are strings, values are ints
vocab = {str(k): int(v) for k, v in vocab.items()}
inv_vocab = {v: k for k, v in vocab.items()}

def encode(text, max_len=128):
    tokens = tokenize(text)
    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    token_ids = [vocab["<CLS>"]] + token_ids[:max_len - 2] + [vocab["<SEP>"]]
    token_ids += [vocab["<PAD>"]] * (max_len - len(token_ids))
    return token_ids

# ---------- Dataset Class ----------
class IMDBDataset(Dataset):
    def __init__(self, split, dataset):
        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        token_ids = encode(text)
        return torch.tensor(token_ids), torch.tensor(label)

# ---------- Transformer Components ----------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, num_classes):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.encoder_blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        token_emb = self.token_embed(x)
        x = token_emb + self.pos_embed[:, :x.size(1), :]
        x = self.encoder_blocks(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

# ---------- Inference ----------
def predict_sentiment(text, model, vocab, max_len=128):
    model.eval()
    token_ids = encode(text, max_len)
    input_tensor = torch.tensor([token_ids]).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
    return {0: "Negative", 1: "Positive"}.get(prediction, "Unknown")
