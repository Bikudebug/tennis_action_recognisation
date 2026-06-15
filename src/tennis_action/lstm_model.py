from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], x.shape[1], -1)
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)


@dataclass
class LSTMTrainResult:
    best_epoch: int
    best_val_f1: float
    history: list[dict]


def macro_f1_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray,
    device: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> tuple[LSTMClassifier, LSTMTrainResult]:
    model = LSTMClassifier(
        input_size=X_train.shape[2] * X_train.shape[3],
        hidden_size=128,
        num_layers=1,
        num_classes=len(np.unique(np.concatenate([y_train, y_val]))),
        dropout=0.2,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_state = None
    best_epoch = -1
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                val_pred.extend(pred.tolist())
                val_true.extend(yb.numpy().tolist())

        val_true = np.array(val_true)
        val_pred = np.array(val_pred)
        val_f1 = macro_f1_numpy(val_true, val_pred, num_classes=len(class_weights))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(len(y_train), 1),
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, LSTMTrainResult(best_epoch=best_epoch, best_val_f1=best_val_f1, history=history)
