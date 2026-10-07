"""Small PyTorch models: a 1D CNN and an optional tiny temporal transformer."""
import numpy as np
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_ch, n_classes, width=64, p_drop=0.3):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(nn.Conv1d(c_in, c_out, 7, padding=3), nn.BatchNorm1d(c_out),
                                 nn.GELU(), nn.MaxPool1d(2))
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 1), nn.BatchNorm1d(width), nn.GELU())
        self.body = nn.Sequential(block(width, width), block(width, width), block(width, width))
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(width, n_classes))

    def forward(self, x):                      # x: (B, C, T)
        h = self.body(self.stem(x))
        return self.head(h.mean(-1))


class TinyTransformer(nn.Module):
    def __init__(self, in_ch, n_classes, d=64, patch=5, layers=2, heads=4, max_tokens=256):
        super().__init__()
        self.patch = nn.Conv1d(in_ch, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.patch(x).transpose(1, 2)        # (B, tokens, d)
        h = h + self.pos[:, : h.shape[1]]
        return self.head(self.enc(h).mean(1))


def fit_predict(kind, Xtr, ytr, Xva, yva, Xte, n_classes, epochs=40, lr=1e-3,
                batch=128, patience=6, seed=0, device=None):
    """Train with class-weighted CE, early-stop on validation SUBJECTS, return test probs."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Model = CNN1D if kind == "cnn" else TinyTransformer
    model = Model(Xtr.shape[1], n_classes).to(device)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    def batches(X, y=None, shuffle=False):
        idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
        for i in range(0, len(X), batch):
            j = idx[i:i + batch]
            xb = torch.tensor(X[j], dtype=torch.float32, device=device)
            yield (xb, torch.tensor(y[j], device=device)) if y is not None else xb

    def predict(X):
        model.eval(); out = []
        with torch.no_grad():
            for xb in batches(X):
                out.append(torch.softmax(model(xb), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, n_classes))

    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(Xtr, ytr, shuffle=True):
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        p = predict(Xva)
        vloss = float(lossf(torch.tensor(np.log(p + 1e-9), device=device),
                                 torch.tensor(yva, device=device)).item()) if len(p) else 0.0
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(Xte)
