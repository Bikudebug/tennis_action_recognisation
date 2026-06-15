from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_action.common import ID_TO_LABEL, LABEL_TO_ID, ensure_dir, write_json
from tennis_action.lstm_model import train_lstm
from tennis_action.preprocess import flatten_sequences, normalize_sequences


def inverse_frequency_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    return weights / weights.mean()


def downsample_neutral(X: np.ndarray, y: np.ndarray, video_ids: np.ndarray, frame_ids: np.ndarray, neutral_ratio: float):
    neutral_id = LABEL_TO_ID["neutral"]
    action_mask = y != neutral_id
    neutral_idx = np.where(~action_mask)[0]
    action_idx = np.where(action_mask)[0]
    keep_neutral = int(round(len(action_idx) * neutral_ratio))
    keep_neutral = min(keep_neutral, len(neutral_idx))
    rng = np.random.default_rng(42)
    selected_neutral = rng.choice(neutral_idx, size=keep_neutral, replace=False) if keep_neutral > 0 else np.array([], dtype=np.int64)
    keep_idx = np.sort(np.concatenate([action_idx, selected_neutral]))
    return X[keep_idx], y[keep_idx], video_ids[keep_idx], frame_ids[keep_idx]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train classical and LSTM tennis action classifiers.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neutral-ratio", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lstm-epochs", type=int, default=10)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    train_data = np.load(dataset_dir / "train_dataset.npz", allow_pickle=True)

    X_raw = train_data["X"]
    y = train_data["y"]
    video_ids = train_data["video_ids"]
    frame_ids = train_data["frame_ids"]
    X_raw, y, video_ids, frame_ids = downsample_neutral(X_raw, y, video_ids, frame_ids, args.neutral_ratio)
    X_norm = normalize_sequences(X_raw)
    X_flat = flatten_sequences(X_norm)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    linear_svm = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000, dual="auto")
    linear_svm.fit(X_scaled, y)
    joblib.dump(linear_svm, output_dir / "linear_svm.joblib")

    logistic = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    logistic.fit(X_scaled, y)
    joblib.dump(logistic, output_dir / "logistic.joblib")

    random_forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    random_forest.fit(X_flat, y)
    joblib.dump(random_forest, output_dir / "random_forest.joblib")

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    class_weights = inverse_frequency_weights(y_train, num_classes=len(LABEL_TO_ID))
    lstm_model, lstm_result = train_lstm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights,
        device=device,
        epochs=args.lstm_epochs,
    )
    torch.save(
        {
            "state_dict": lstm_model.state_dict(),
            "class_weights": class_weights.tolist(),
            "input_shape": list(X_norm.shape[1:]),
            "labels": ID_TO_LABEL,
        },
        output_dir / "lstm.pt",
    )

    training_summary = {
        "train_shape": list(X_raw.shape),
        "train_shape_after_neutral_downsample": list(X_norm.shape),
        "neutral_ratio": args.neutral_ratio,
        "device": device,
        "label_counts_after_downsample": {ID_TO_LABEL[int(idx)]: int(count) for idx, count in enumerate(np.bincount(y, minlength=len(LABEL_TO_ID)))},
        "lstm": {
            "best_epoch": lstm_result.best_epoch,
            "best_val_macro_f1": lstm_result.best_val_f1,
            "history": lstm_result.history,
        },
    }
    write_json(output_dir / "training_summary.json", training_summary)
    print(json.dumps(training_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
