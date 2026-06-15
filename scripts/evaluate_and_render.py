from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_action.common import (
    DISPLAY_LABEL,
    ID_TO_LABEL,
    LABEL_COLORS_BGR,
    LABEL_TO_ID,
    draw_pose,
    ensure_dir,
    run_h264_conversion,
    write_json,
)
from tennis_action.dataset_utils import find_video_for_label, load_pose_csv
from tennis_action.lstm_model import LSTMClassifier
from tennis_action.preprocess import flatten_sequences, normalize_sequences, temporal_majority_filter



def decision_to_probs(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        decision = np.stack([-decision, decision], axis=1)
    decision = decision - np.max(decision, axis=1, keepdims=True)
    exp = np.exp(decision)
    return exp / np.sum(exp, axis=1, keepdims=True)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(LABEL_TO_ID)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[ID_TO_LABEL[i] for i in labels],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def build_segments(frame_ids: np.ndarray, label_ids: np.ndarray) -> list[dict]:
    segments = []
    if len(frame_ids) == 0:
        return segments
    start = 0
    for idx in range(1, len(frame_ids)):
        if label_ids[idx] != label_ids[idx - 1]:
            label = ID_TO_LABEL[int(label_ids[start])]
            if label != "neutral":
                segments.append(
                    {
                        "label": label,
                        "start_frame": int(frame_ids[start]),
                        "end_frame": int(frame_ids[idx - 1]),
                    }
                )
            start = idx
    label = ID_TO_LABEL[int(label_ids[start])]
    if label != "neutral":
        segments.append(
            {
                "label": label,
                "start_frame": int(frame_ids[start]),
                "end_frame": int(frame_ids[-1]),
            }
        )
    return segments


def render_video(
    video_path: Path,
    pose_csv: Path,
    frame_df: pd.DataFrame,
    metadata: dict,
    out_path: Path,
    conf_thresh: float,
) -> None:
    pose_map = load_pose_csv(pose_csv)
    cap = cv2.VideoCapture(str(video_path))
    fps = metadata["fps"]
    disp_w = int(metadata["width"])
    disp_h = int(metadata["height"])
    rotation = int(metadata["rotation"])
    target_scale = min(1.0, 1600.0 / max(disp_w * 2, 1), 900.0 / max(disp_h, 1))
    out_w = max(2, int(round(disp_w * 2 * target_scale)))
    out_h = max(2, int(round(disp_h * target_scale)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    gt_full = frame_df["gt_label"].tolist()
    pred_full = frame_df["pred_label_smooth"].tolist()
    total_frames = len(frame_df)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    timeline_w = disp_w - 50
    timeline_y_gt = disp_h - 55
    timeline_y_pred = disp_h - 25

    timeline_bg = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    cv2.putText(timeline_bg, "GT", (25, timeline_y_gt - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(timeline_bg, "Pred", (25, timeline_y_pred - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for idx, label in enumerate(gt_full):
        x = 25 + int(idx / max(total_frames - 1, 1) * timeline_w)
        cv2.line(timeline_bg, (x, timeline_y_gt), (x, timeline_y_gt + 12), LABEL_COLORS_BGR[label], 1)
    for idx, label in enumerate(pred_full):
        x = 25 + int(idx / max(total_frames - 1, 1) * timeline_w)
        cv2.line(timeline_bg, (x, timeline_y_pred), (x, timeline_y_pred + 12), LABEL_COLORS_BGR[label], 1)
    timeline_mask = np.any(timeline_bg != 0, axis=2)

    def rotate_frame(frame):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def rotate_keypoints(kp):
        if rotation == 90:
            xy_raw = kp[:, :2].copy()
            kp[:, 0] = xy_raw[:, 1]
            kp[:, 1] = raw_w - xy_raw[:, 0]
        elif rotation == 180:
            xy_raw = kp[:, :2].copy()
            kp[:, 0] = metadata["width"] - xy_raw[:, 0]
            kp[:, 1] = metadata["height"] - xy_raw[:, 1]
        elif rotation == 270:
            xy_raw = kp[:, :2].copy()
            kp[:, 0] = metadata["height"] - xy_raw[:, 1]
            kp[:, 1] = xy_raw[:, 0]
        return kp

    def center_pose_for_panel(kp: np.ndarray) -> np.ndarray:
        visible = kp[:, 2] >= conf_thresh
        if visible.sum() < 2:
            return kp.copy()
        pts = kp[visible, :2]
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        center_xy = (min_xy + max_xy) / 2.0
        span = np.maximum(max_xy - min_xy, 1.0)
        scale = 0.62 * min(disp_w / span[0], disp_h / span[1])
        target_center = np.array([disp_w * 0.58, disp_h * 0.58], dtype=np.float32)
        kp_panel = kp.copy()
        kp_panel[:, :2] = (kp_panel[:, :2] - center_xy) * scale + target_center
        return kp_panel

    frame_num = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1
        frame = rotate_frame(frame)
        skeleton_panel = np.zeros_like(frame)
        row = frame_df.iloc[frame_num - 1]
        gt_label = row["gt_label"]
        pred_label = row["pred_label_smooth"]
        color = LABEL_COLORS_BGR[pred_label]

        if frame_num in pose_map:
            kp = pose_map[frame_num].copy()
            kp = rotate_keypoints(kp)
            draw_pose(
                skeleton_panel,
                center_pose_for_panel(kp),
                (240, 240, 240),
                conf_thresh,
                line_thickness=9,
                joint_radius=8,
                joint_outline_thickness=2,
                joint_fill_color=(240, 240, 240),
            )

        header_h = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (disp_w, header_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        overlay_sk = skeleton_panel.copy()
        cv2.rectangle(overlay_sk, (0, 0), (disp_w, header_h), (16, 16, 16), -1)
        cv2.addWeighted(overlay_sk, 0.85, skeleton_panel, 0.15, 0, skeleton_panel)
        cv2.line(skeleton_panel, (0, 0), (0, disp_h), (50, 50, 50), 2)

        cv2.putText(frame, "RGB Video", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (210, 210, 210), 2, cv2.LINE_AA)
        cv2.putText(skeleton_panel, "2D Pose Motion View", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GT: {DISPLAY_LABEL[gt_label]}", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.95, LABEL_COLORS_BGR[gt_label], 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {DISPLAY_LABEL[pred_label]}", (25, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.95, LABEL_COLORS_BGR[pred_label], 2, cv2.LINE_AA)
        cv2.putText(skeleton_panel, f"GT: {DISPLAY_LABEL[gt_label]}", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.95, LABEL_COLORS_BGR[gt_label], 2, cv2.LINE_AA)
        cv2.putText(skeleton_panel, f"Pred: {DISPLAY_LABEL[pred_label]}", (25, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (235, 235, 235), 2, cv2.LINE_AA)

        frame[timeline_mask] = timeline_bg[timeline_mask]
        marker_x = 25 + int((frame_num - 1) / max(total_frames - 1, 1) * timeline_w)
        cv2.line(frame, (marker_x, timeline_y_gt - 12), (marker_x, timeline_y_pred + 18), (255, 255, 255), 2)

        combo = np.concatenate([frame, skeleton_panel], axis=1)
        if target_scale != 1.0:
            combo = cv2.resize(combo, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(combo)

    cap.release()
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained tennis action models and render test videos.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--smooth-kernel", type=int, default=9)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    models_dir = Path(args.models_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    videos_dir = Path(args.video_dir)
    pose_root = Path(args.pose_root)

    model_order = ["linear_svm", "logistic", "random_forest", "lstm"]
    if args.render_only:
        summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        best_model = summary["best_model"]
        results_df = pd.read_csv(output_dir / "frame_predictions.csv")
    else:
        test_data = np.load(dataset_dir / "test_dataset.npz", allow_pickle=True)
        X_raw = test_data["X"]
        y_true = test_data["y"]
        frame_ids = test_data["frame_ids"]
        video_ids = test_data["video_ids"]
        X_norm = normalize_sequences(X_raw)
        X_flat = flatten_sequences(X_norm)

        scaler = joblib.load(models_dir / "scaler.joblib")
        X_scaled = scaler.transform(X_flat)

        predictions = {}
        linear_svm = joblib.load(models_dir / "linear_svm.joblib")
        svm_pred = linear_svm.predict(X_scaled)
        svm_probs = decision_to_probs(linear_svm.decision_function(X_scaled))
        predictions["linear_svm"] = (svm_pred, svm_probs)

        logistic = joblib.load(models_dir / "logistic.joblib")
        predictions["logistic"] = (logistic.predict(X_scaled), logistic.predict_proba(X_scaled))

        rf = joblib.load(models_dir / "random_forest.joblib")
        predictions["random_forest"] = (rf.predict(X_flat), rf.predict_proba(X_flat))

        lstm_ckpt = torch.load(models_dir / "lstm.pt", map_location="cpu")
        lstm = LSTMClassifier(input_size=X_norm.shape[2] * X_norm.shape[3], hidden_size=128, num_layers=1, num_classes=len(LABEL_TO_ID), dropout=0.2)
        lstm.load_state_dict(lstm_ckpt["state_dict"])
        lstm.eval()
        with torch.no_grad():
            logits = lstm(torch.tensor(X_norm, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
            predictions["lstm"] = (np.argmax(probs, axis=1), probs)

        metrics = {}
        for model_name in model_order:
            y_pred = predictions[model_name][0]
            metrics[model_name] = evaluate_predictions(y_true, y_pred)

        best_model = max(model_order, key=lambda name: metrics[name]["macro_f1"])

        results_rows = []
        for idx in range(len(y_true)):
            row = {
                "video_id": str(video_ids[idx]),
                "frame_id": int(frame_ids[idx]),
                "gt_label_id": int(y_true[idx]),
                "gt_label": ID_TO_LABEL[int(y_true[idx])],
            }
            for model_name in model_order:
                pred_id = int(predictions[model_name][0][idx])
                row[f"{model_name}_pred_id"] = pred_id
                row[f"{model_name}_pred_label"] = ID_TO_LABEL[pred_id]
                row[f"{model_name}_confidence"] = float(np.max(predictions[model_name][1][idx]))
            results_rows.append(row)
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(output_dir / "frame_predictions.csv", index=False)

        write_json(
            output_dir / "metrics_summary.json",
            {
                "best_model": best_model,
                "metrics": metrics,
            },
        )

    per_video_dir = ensure_dir(output_dir / "per_video")
    for video_id, group in results_df.groupby("video_id"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        best_pred = group[f"{best_model}_pred_id"].to_numpy(dtype=np.int32)
        smooth_pred = temporal_majority_filter(best_pred, args.smooth_kernel)
        group["pred_label_raw"] = [ID_TO_LABEL[int(v)] for v in best_pred]
        group["pred_label_smooth"] = [ID_TO_LABEL[int(v)] for v in smooth_pred]

        gt_segments = build_segments(group["frame_id"].to_numpy(), group["gt_label_id"].to_numpy())
        pred_segments = build_segments(group["frame_id"].to_numpy(), smooth_pred)

        video_dir = ensure_dir(per_video_dir / video_id)
        group.to_csv(video_dir / "frame_predictions.csv", index=False)
        write_json(video_dir / "gt_segments.json", {"video_id": video_id, "segments": gt_segments})
        write_json(video_dir / "predicted_segments.json", {"video_id": video_id, "segments": pred_segments})

        pose_csv = pose_root / video_id / "pose" / f"{video_id}_tracked_pose_keypoints.csv"
        video_path = find_video_for_label(videos_dir, video_id)
        metadata_path = dataset_dir / "per_video" / f"{video_id}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))[0]
        mp4_path = video_dir / f"{video_id}_{best_model}_gt_vs_pred.mp4"
        render_video(video_path, pose_csv, group, metadata, mp4_path, args.conf_thresh)
        run_h264_conversion(mp4_path, video_dir / f"{video_id}_{best_model}_gt_vs_pred_h264.mp4")

    best_metrics = {
        "best_model": best_model,
        "test_label_distribution": Counter(results_df["gt_label"]).copy(),
    }
    if not args.render_only:
        best_metrics["frame_macro_f1"] = metrics[best_model]["macro_f1"]
        best_metrics["frame_accuracy"] = metrics[best_model]["accuracy"]
    write_json(output_dir / "best_model_summary.json", best_metrics)
    print(json.dumps(best_metrics, indent=2, default=lambda x: dict(x)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
