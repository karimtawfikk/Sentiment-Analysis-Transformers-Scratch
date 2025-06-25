import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter

# ---------- Tokenizer & Vocab ----------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Initialize with minimal vocab to allow import
vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3
}

inv_vocab = {v: k for k, v in vocab.items()}

def encode(text, max_len=128):
    tokens = tokenize(text)
    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    token_ids = [vocab["<CLS>"]] + token_ids[:max_len - 2] + [vocab["<SEP>"]]
    pad_len = max_len - len(token_ids)
    token_ids += [vocab["<PAD>"]] * pad_len
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

# ---------- Transformer Encoder Block ----------
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

# ---------- Main Transformer Model ----------
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

# ---------- Inference Function ----------
def predict_sentiment(text, model, vocab, max_len=128):
    model.eval()
    tokens = tokenize(text)
    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    token_ids = [vocab["<CLS>"]] + token_ids[:max_len - 2] + [vocab["<SEP>"]]
    pad_len = max_len - len(token_ids)
    token_ids += [vocab["<PAD>"]] * pad_len

    input_tensor = torch.tensor([token_ids]).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()

    label_map = {0: "Negative", 1: "Positive"}
    return label_map[prediction]

# ---------- Only for training ----------
if __name__ == "__main__":
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]

    all_tokens = []
    for sample in train_data:
        all_tokens.extend(tokenize(sample["text"]))

    token_freq = Counter(all_tokens)
    vocab_size = 10000
    most_common = token_freq.most_common(vocab_size - 4)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<CLS>": 2,
        "<SEP>": 3
    }
    for idx, (word, _) in enumerate(most_common, start=4):
        vocab[word] = idx
    inv_vocab = {v: k for k, v in vocab.items()}

    train_loader = DataLoader(IMDBDataset("train", dataset), batch_size=32, shuffle=True)
    test_loader = DataLoader(IMDBDataset("test", dataset), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentTransformer(
        vocab_size=len(vocab),
        max_len=128,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=5,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), correct / total

    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    for epoch in range(4):
        print(f"\nEpoch {epoch + 1}/4")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), "sentiment_transformer.pth")

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
