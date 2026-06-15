from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_action.common import ID_TO_LABEL, LABEL_TO_ID, ensure_dir, get_video_properties, probe_video_rotation, rotate_xy, write_json
from tennis_action.dataset_utils import load_pose_csv
from tennis_action.lstm_model import LSTMClassifier
from tennis_action.preprocess import flatten_sequences, normalize_sequences, temporal_majority_filter


def build_inference_sequences(video_path: Path, pose_csv: Path, window_radius: int, conf_thresh: float):
    pose_map = load_pose_csv(pose_csv)
    props = get_video_properties(video_path)
    raw_w, raw_h = props["width"], props["height"]
    rotation = probe_video_rotation(video_path, video_path.stem)
    frame_count = props["frame_count"]
    window_size = window_radius * 2 + 1
    num_joints = next(iter(pose_map.values())).shape[0] if pose_map else 17
    X = np.full((frame_count, window_size, num_joints, 2), np.nan, dtype=np.float32)
    frame_ids = np.arange(1, frame_count + 1, dtype=np.int32)
    for row_idx, center_frame in enumerate(frame_ids):
        start = center_frame - window_radius
        for t in range(window_size):
            frame_id = start + t
            keypoints = pose_map.get(frame_id)
            if keypoints is None:
                continue
            xy = keypoints[:, :2].copy()
            conf = keypoints[:, 2]
            xy[conf < conf_thresh] = np.nan
            xy = rotate_xy(xy, rotation, raw_w, raw_h)
            X[row_idx, t] = xy
    return X, frame_ids, props


def decision_to_probs(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        decision = np.stack([-decision, decision], axis=1)
    decision = decision - np.max(decision, axis=1, keepdims=True)
    exp = np.exp(decision)
    return exp / np.sum(exp, axis=1, keepdims=True)


def infer_model_name(checkpoint_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    summary_path = checkpoint_dir / "best_model_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")).get("best_model", "random_forest")
    return "random_forest"


def load_and_predict(model_name: str, checkpoint_dir: Path, X_norm: np.ndarray):
    X_flat = flatten_sequences(X_norm)
    if model_name == "random_forest":
        model = joblib.load(checkpoint_dir / "random_forest.joblib")
        probs = model.predict_proba(X_flat)
        pred = np.argmax(probs, axis=1)
        return pred, probs
    if model_name == "logistic":
        scaler = joblib.load(checkpoint_dir / "scaler.joblib")
        model = joblib.load(checkpoint_dir / "logistic.joblib")
        X_scaled = scaler.transform(X_flat)
        probs = model.predict_proba(X_scaled)
        pred = np.argmax(probs, axis=1)
        return pred, probs
    if model_name == "linear_svm":
        scaler = joblib.load(checkpoint_dir / "scaler.joblib")
        model = joblib.load(checkpoint_dir / "linear_svm.joblib")
        X_scaled = scaler.transform(X_flat)
        decision = model.decision_function(X_scaled)
        probs = decision_to_probs(decision)
        pred = np.argmax(probs, axis=1)
        return pred, probs
    if model_name == "lstm":
        ckpt = torch.load(checkpoint_dir / "lstm.pt", map_location="cpu")
        model = LSTMClassifier(
            input_size=X_norm.shape[2] * X_norm.shape[3],
            hidden_size=128,
            num_layers=1,
            num_classes=len(LABEL_TO_ID),
            dropout=0.2,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_norm, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
            pred = np.argmax(probs, axis=1)
        return pred, probs
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_segments(video_path: Path, frame_ids: np.ndarray, smooth_pred: np.ndarray, probs: np.ndarray) -> list[dict]:
    segments = []
    start = 0
    for idx in range(1, len(frame_ids) + 1):
        boundary = idx == len(frame_ids) or smooth_pred[idx] != smooth_pred[idx - 1]
        if not boundary:
            continue
        label_id = int(smooth_pred[start])
        label_name = ID_TO_LABEL[label_id]
        if label_name != "neutral":
            seg_frames = frame_ids[start:idx]
            seg_probs = probs[start:idx, label_id]
            center_idx = int(np.argmax(seg_probs))
            contact_frame = int(seg_frames[center_idx])
            segments.append(
                {
                    "event_id": len(segments),
                    "video_label": video_path.stem,
                    "video_stem": video_path.stem,
                    "video_file": str(video_path),
                    "action_label": label_name,
                    "action_label_raw": label_name,
                    "contact_frame": contact_frame,
                    "start_frame": int(seg_frames[0]),
                    "end_frame": int(seg_frames[-1]),
                    "segment_length": int(seg_frames[-1] - seg_frames[0] + 1),
                    "score_mean": float(seg_probs.mean()),
                    "needs_review": False,
                    "review_note": "",
                }
            )
        start = idx
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Run action recognition on a raw tennis video using a pretrained checkpoint.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="auto", choices=["auto", "random_forest", "logistic", "linear_svm", "lstm"])
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--smooth-kernel", type=int, default=9)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    checkpoint_dir = Path(args.checkpoint_dir)
    model_name = infer_model_name(checkpoint_dir, args.model_type)

    X_raw, frame_ids, props = build_inference_sequences(Path(args.video), Path(args.pose_csv), args.window_radius, args.conf_thresh)
    X_norm = normalize_sequences(X_raw)
    pred, probs = load_and_predict(model_name, checkpoint_dir, X_norm)
    smooth_pred = temporal_majority_filter(pred.astype(np.int32), args.smooth_kernel)

    label_names = [ID_TO_LABEL[int(v)] for v in pred]
    smooth_names = [ID_TO_LABEL[int(v)] for v in smooth_pred]
    pred_conf = probs.max(axis=1)
    df = pd.DataFrame(
        {
            "video_stem": Path(args.video).stem,
            "frame_id": frame_ids,
            "pred_label_id": pred,
            "pred_label": label_names,
            "pred_label_smooth_id": smooth_pred,
            "pred_label_smooth": smooth_names,
            "pred_confidence": pred_conf,
        }
    )
    for idx, label in ID_TO_LABEL.items():
        df[f"prob_{label}"] = probs[:, idx]
    df.to_csv(output_dir / "frame_predictions.csv", index=False)

    segments = build_segments(Path(args.video), frame_ids, smooth_pred, probs)
    segments_df = pd.DataFrame(segments)
    segments_df.to_csv(output_dir / "predicted_action_segments.csv", index=False)
    write_json(output_dir / "predicted_action_segments.json", {"video": args.video, "segments": segments})

    summary = {
        "video": args.video,
        "pose_csv": args.pose_csv,
        "checkpoint_dir": str(checkpoint_dir),
        "model_name": model_name,
        "frame_count": int(props["frame_count"]),
        "num_predicted_segments": int(len(segments)),
        "predicted_labels": {
            str(label): int(count)
            for label, count in df["pred_label_smooth"].value_counts().items()
        },
    }
    write_json(output_dir / "prediction_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
