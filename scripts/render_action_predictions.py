from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_action.common import LABEL_COLORS_BGR, ensure_dir, run_h264_conversion
from tennis_pipeline.pose_utils import KEYPOINT_NAMES, SKELETON_EDGES_NAME


def fit_rect(src_w: int, src_h: int, dst_x: int, dst_y: int, dst_w: int, dst_h: int):
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    ox = dst_x + (dst_w - new_w) // 2
    oy = dst_y + (dst_h - new_h) // 2
    return scale, ox, oy, new_w, new_h


def transform_point(x: float, y: float, scale: float, ox: int, oy: int) -> tuple[int, int]:
    return int(round(x * scale + ox)), int(round(y * scale + oy))


def put_text(
    frame,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 2,
):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def row_keypoint(row: dict, name: str, min_conf: float = 0.25):
    conf = row.get(f"{name}_conf")
    if conf is None or pd.isna(conf) or float(conf) < min_conf:
        return None
    x = row.get(f"{name}_x")
    y = row.get(f"{name}_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y), float(conf)


def draw_pose(
    frame,
    row: dict,
    scale: float,
    ox: int,
    oy: int,
    color: tuple[int, int, int],
    min_conf: float,
    thickness: int = 3,
    radius: int = 5,
):
    for start_name, end_name in SKELETON_EDGES_NAME:
        start = row_keypoint(row, start_name, min_conf)
        end = row_keypoint(row, end_name, min_conf)
        if start is None or end is None:
            continue
        p1 = transform_point(start[0], start[1], scale, ox, oy)
        p2 = transform_point(end[0], end[1], scale, ox, oy)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    for name in KEYPOINT_NAMES:
        point = row_keypoint(row, name, min_conf)
        if point is None:
            continue
        p = transform_point(point[0], point[1], scale, ox, oy)
        cv2.circle(frame, p, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, radius + 1, color, 2, cv2.LINE_AA)


def draw_bbox(
    frame,
    row: dict,
    scale: float,
    ox: int,
    oy: int,
    color: tuple[int, int, int],
    thickness: int = 2,
):
    try:
        p1 = transform_point(float(row["bbox_x1"]), float(row["bbox_y1"]), scale, ox, oy)
        p2 = transform_point(float(row["bbox_x2"]), float(row["bbox_y2"]), scale, ox, oy)
    except KeyError:
        return
    cv2.rectangle(frame, p1, p2, color, thickness)


def safe_float(value, default=np.nan):
    if value is None or pd.isna(value):
        return default
    return float(value)


def build_predicted_segments(prediction_df: pd.DataFrame) -> list[dict]:
    segments: list[dict] = []
    if prediction_df.empty:
        return segments

    labels = prediction_df["pred_label_smooth"].astype(str).tolist()
    frame_ids = prediction_df["frame_id"].astype(int).tolist()
    confidences = prediction_df["pred_confidence"].astype(float).tolist()

    start = 0
    for idx in range(1, len(labels) + 1):
        boundary = idx == len(labels) or labels[idx] != labels[idx - 1]
        if not boundary:
            continue
        label = labels[start]
        if label != "neutral":
            seg_frames = frame_ids[start:idx]
            seg_conf = confidences[start:idx]
            segments.append(
                {
                    "label": label,
                    "start_frame": int(seg_frames[0]),
                    "end_frame": int(seg_frames[-1]),
                    "contact_frame": int(seg_frames[int(np.argmax(seg_conf))]),
                    "score_mean": float(np.mean(seg_conf)),
                }
            )
        start = idx
    return segments


def active_segment(segments: list[dict], frame_id: int) -> dict | None:
    for seg in segments:
        if int(seg["start_frame"]) <= frame_id <= int(seg["end_frame"]):
            return seg
    return None


def draw_info_panel(
    frame,
    pred_row: dict | None,
    segment: dict | None,
    frame_id: int,
    panel_x: int,
    panel_y: int,
    panel_w: int,
):
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 270), (25, 25, 25), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 270), (70, 70, 70), 2)
    put_text(frame, f"Frame {frame_id}", (panel_x + 18, panel_y + 35), 0.72)

    if pred_row is None:
        put_text(frame, "State: no prediction row found", (panel_x + 18, panel_y + 75), 0.66)
        return

    label = str(pred_row.get("pred_label_smooth", "neutral"))
    color = LABEL_COLORS_BGR.get(label, (140, 140, 140))
    conf = safe_float(pred_row.get("pred_confidence"), 0.0)

    if segment is None:
        put_text(frame, "State: no predicted stroke window", (panel_x + 18, panel_y + 75), 0.66)
        put_text(frame, f"Predicted label: {label.upper()}   confidence: {conf:.2f}", (panel_x + 18, panel_y + 112), 0.58, color)
    else:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        contact = int(segment["contact_frame"])
        progress = (frame_id - start) / max(1, end - start)
        progress = min(1.0, max(0.0, progress))
        put_text(frame, f"Predicted action: {label.upper()}", (panel_x + 18, panel_y + 75), 0.68, color)
        put_text(frame, f"Window {start}-{end}   contact/proxy frame {contact}", (panel_x + 18, panel_y + 112), 0.58)
        bar_x = panel_x + 18
        bar_y = panel_y + 134
        bar_w = panel_w - 36
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (90, 90, 90), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 12), color, -1)
        contact_x = bar_x + int(bar_w * ((contact - start) / max(1, end - start)))
        cv2.line(frame, (contact_x, bar_y - 5), (contact_x, bar_y + 18), (255, 255, 255), 2)
        put_text(frame, f"Segment mean confidence: {safe_float(segment.get('score_mean'), 0.0):.2f}", (panel_x + 18, panel_y + 165), 0.54)

    probs = []
    for label_name in ("forehand", "backhand", "serve", "neutral"):
        probs.append(f"{label_name[:2].upper()} {safe_float(pred_row.get(f'prob_{label_name}'), 0.0):.2f}")
    put_text(frame, " | ".join(probs), (panel_x + 18, panel_y + 210), 0.50, (210, 240, 255))
    put_text(frame, "Frame-wise supervised model prediction on tracked player pose", (panel_x + 18, panel_y + 248), 0.48, (180, 200, 220))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render side-by-side action-recognition video with the same explained-video layout.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rotation", type=int, default=None, help="Optional manual rotation override. Default: no extra rotation.")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--h264", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    video_path = Path(args.video)
    prediction_df = pd.read_csv(args.prediction_csv).sort_values("frame_id").reset_index(drop=True)
    pose_df = pd.read_csv(args.pose_csv).sort_values("person_conf", ascending=False).drop_duplicates("frame_id", keep="first")
    pose_by_frame = {int(row["frame_id"]): row.to_dict() for _, row in pose_df.iterrows()}
    pred_by_frame = {int(row["frame_id"]): row.to_dict() for _, row in prediction_df.iterrows()}
    segments = build_predicted_segments(prediction_df)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_mp4 = output_dir / f"{video_path.stem}_action_recognition.mp4"
    out_h264 = output_dir / f"{video_path.stem}_action_recognition_h264.mp4"
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.output_width, args.output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out_mp4}")

    panel_w = args.output_width // 2
    left_x, right_x = 0, panel_w
    scale, fit_x_left, fit_y, fit_w, fit_h = fit_rect(src_w, src_h, left_x, 0, panel_w, args.output_height)
    _, fit_x_right, _, _, _ = fit_rect(src_w, src_h, right_x, 0, panel_w, args.output_height)

    frame_num = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1

            if args.rotation is not None:
                if args.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif args.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif args.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            canvas = np.zeros((args.output_height, args.output_width, 3), dtype=np.uint8)
            canvas[:, :panel_w] = (14, 18, 22)
            canvas[:, panel_w:] = (8, 10, 14)

            resized = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
            canvas[fit_y : fit_y + fit_h, fit_x_left : fit_x_left + fit_w] = resized

            pose_row = pose_by_frame.get(frame_num)
            pred_row = pred_by_frame.get(frame_num)
            segment = active_segment(segments, frame_num)
            label = str(pred_row.get("pred_label_smooth", "neutral")) if pred_row is not None else "neutral"
            color = LABEL_COLORS_BGR.get(label, (140, 140, 140))
            conf = safe_float(pred_row.get("pred_confidence"), 0.0) if pred_row is not None else 0.0

            if pose_row is not None:
                draw_bbox(canvas, pose_row, scale, fit_x_left, fit_y, color, 3)
                draw_pose(canvas, pose_row, scale, fit_x_left, fit_y, color, args.min_conf, 4, 5)

            cv2.rectangle(canvas, (right_x, 0), (args.output_width - 1, args.output_height - 1), (45, 45, 45), 2)
            put_text(canvas, "2D Pose Motion View", (right_x + 24, 42), 0.82, (230, 230, 230))
            put_text(canvas, "Predicted action from tracked pose sequence", (right_x + 24, 76), 0.55, (180, 200, 220))

            if pose_row is not None:
                draw_bbox(canvas, pose_row, scale, fit_x_right, fit_y, (90, 90, 90), 2)
                draw_pose(canvas, pose_row, scale, fit_x_right, fit_y, (235, 235, 235), args.min_conf, 4, 5)

                if segment is not None and frame_num == int(segment["contact_frame"]):
                    wrist_candidates = ["right_wrist", "left_wrist"]
                    contact_point = None
                    for wrist_name in wrist_candidates:
                        kp = row_keypoint(pose_row, wrist_name, args.min_conf)
                        if kp is not None:
                            contact_point = transform_point(kp[0], kp[1], scale, fit_x_right, fit_y)
                            break
                    if contact_point is not None:
                        cv2.circle(canvas, contact_point, 22, (255, 255, 255), 4, cv2.LINE_AA)
                        put_text(canvas, "CONTACT / PROXY", (contact_point[0] + 20, contact_point[1]), 0.55, color)

            draw_info_panel(
                canvas,
                pred_row,
                segment,
                frame_num,
                panel_x=right_x + 24,
                panel_y=args.output_height - 292,
                panel_w=panel_w - 48,
            )

            if segment is not None:
                put_text(canvas, f"PREDICTED {label.upper()} WINDOW", (32, 54), 0.86, color, 3)
                put_text(canvas, f"Confidence {conf:.2f}", (32, 92), 0.62, (240, 240, 240), 2)
            else:
                put_text(canvas, "RGB Video + Tracked Pose", (32, 54), 0.8)
                put_text(canvas, f"Predicted: {label.upper()}   confidence {conf:.2f}", (32, 92), 0.58, color, 2)

            writer.write(canvas)
    finally:
        cap.release()
        writer.release()

    if args.h264:
        run_h264_conversion(out_mp4, out_h264)
    print(str(out_h264 if args.h264 else out_mp4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
