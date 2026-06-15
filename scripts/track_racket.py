from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_pipeline.pose_utils import (
    ensure_dir,
    get_video_properties,
    run_h264_conversion,
)


RACKET_CLASS_NAMES = {"tennis racket", "racket", "tennis_racket"}


def point(row: dict, name: str, min_conf: float):
    conf = row.get(f"{name}_conf")
    if conf is None or pd.isna(conf) or float(conf) < min_conf:
        return None
    x = row.get(f"{name}_x")
    y = row.get(f"{name}_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return np.array([float(x), float(y)], dtype=float)


def active_action(actions: pd.DataFrame | None, frame_id: int) -> dict | None:
    if actions is None or actions.empty:
        return None
    candidates = actions[
        (actions["window_start"] <= frame_id) & (actions["window_end"] >= frame_id)
    ]
    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["distance_to_contact"] = (candidates["contact_frame"] - frame_id).abs()
    return candidates.sort_values("distance_to_contact").iloc[0].to_dict()


def active_wrist_name(action: dict | None, handedness: str) -> str:
    if action is not None and action.get("active_wrist_side"):
        side = str(action["active_wrist_side"])
    elif handedness in {"left", "right"}:
        side = handedness
    else:
        side = "right"
    return f"{side}_wrist"


def bbox_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=float)


def bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def expanded_contains(player_box: np.ndarray, racket_center: np.ndarray, margin: float) -> bool:
    x1, y1, x2, y2 = player_box
    w = x2 - x1
    h = y2 - y1
    return (
        x1 - margin * w <= racket_center[0] <= x2 + margin * w
        and y1 - margin * h <= racket_center[1] <= y2 + margin * h
    )


def get_class_ids(model: YOLO, requested_names: set[str]) -> list[int]:
    names = model.names
    ids = []
    for cls_id, cls_name in names.items():
        normalized = str(cls_name).strip().lower().replace("-", " ")
        if normalized in requested_names:
            ids.append(int(cls_id))
    return ids


def farthest_bbox_corner_from_point(box: np.ndarray, reference: np.ndarray) -> np.ndarray:
    corners = np.array(
        [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
        dtype=float,
    )
    distances = np.linalg.norm(corners - reference[None, :], axis=1)
    return corners[int(np.argmax(distances))]


def choose_racket_detection(
    result,
    racket_class_ids: list[int],
    pose_row: dict,
    wrist: np.ndarray | None,
    previous_center: np.ndarray | None,
    player_margin: float,
):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    player_box = np.array(
        [
            float(pose_row["bbox_x1"]),
            float(pose_row["bbox_y1"]),
            float(pose_row["bbox_x2"]),
            float(pose_row["bbox_y2"]),
        ],
        dtype=float,
    )
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    candidates = []
    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
        if int(cls_id) not in racket_class_ids:
            continue
        center = bbox_center(box)
        if not expanded_contains(player_box, center, player_margin):
            continue
        wrist_distance = (
            float(np.linalg.norm(center - wrist)) if wrist is not None else np.nan
        )
        previous_distance = (
            float(np.linalg.norm(center - previous_center))
            if previous_center is not None
            else np.nan
        )
        candidates.append(
            {
                "det_index": int(idx),
                "class_id": int(cls_id),
                "class_name": str(result.names[int(cls_id)]),
                "conf": float(conf),
                "box": np.array(box, dtype=float),
                "center": center,
                "area": bbox_area(box),
                "wrist_distance_px": wrist_distance,
                "previous_distance_px": previous_distance,
            }
        )

    if not candidates:
        return None

    def score(candidate):
        wrist_term = (
            candidate["wrist_distance_px"]
            if not pd.isna(candidate["wrist_distance_px"])
            else 10000.0
        )
        previous_term = (
            candidate["previous_distance_px"]
            if not pd.isna(candidate["previous_distance_px"])
            else 0.0
        )
        return wrist_term + 0.35 * previous_term - 120.0 * candidate["conf"]

    return min(candidates, key=score)


def draw_racket(frame, candidate: dict, wrist: np.ndarray | None, head: np.ndarray):
    box = candidate["box"]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    color = (40, 230, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    cv2.circle(frame, (int(candidate["center"][0]), int(candidate["center"][1])), 8, color, -1)
    cv2.circle(frame, (int(head[0]), int(head[1])), 13, (0, 80, 255), -1)
    if wrist is not None:
        cv2.line(
            frame,
            (int(wrist[0]), int(wrist[1])),
            (int(head[0]), int(head[1])),
            (0, 80, 255),
            4,
            cv2.LINE_AA,
        )
    label = f"racket {candidate['conf']:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(35, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x1, max(35, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )


def add_motion_columns(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    df = df.sort_values("frame_id").reset_index(drop=True)
    if df.empty:
        return df
    consecutive = df["frame_id"].diff().fillna(1).eq(1)
    for prefix in ["racket_center", "racket_head_proxy"]:
        dx = df[f"{prefix}_x"].diff().where(consecutive)
        dy = df[f"{prefix}_y"].diff().where(consecutive)
        df[f"{prefix}_speed_px_per_frame"] = np.sqrt(dx**2 + dy**2)
        df[f"{prefix}_speed_px_per_second"] = df[f"{prefix}_speed_px_per_frame"] * fps
        if "body_scale_px" in df:
            df[f"{prefix}_speed_body_units_per_frame"] = (
                df[f"{prefix}_speed_px_per_frame"] / df["body_scale_px"]
            )
            df[f"{prefix}_accel_body_units_per_frame2"] = df[
                f"{prefix}_speed_body_units_per_frame"
            ].diff().where(consecutive)
    return df


def track_racket(
    source_video: Path,
    detector_model: str,
    pose_csv: Path,
    frame_features_csv: Path | None,
    action_features_csv: Path | None,
    output_dir: Path,
    device: str,
    imgsz: int,
    conf: float,
    handedness: str,
    player_margin: float,
    min_keypoint_conf: float,
    save_video: bool,
    make_h264: bool,
) -> dict:
    output_dir = ensure_dir(output_dir)
    props = get_video_properties(source_video)
    fps = props["fps"]
    width = props["width"]
    height = props["height"]

    pose_df = pd.read_csv(pose_csv)
    pose_by_frame = {
        int(row["frame_id"]): row.to_dict() for _, row in pose_df.iterrows()
    }
    frame_features = (
        pd.read_csv(frame_features_csv)
        if frame_features_csv and frame_features_csv.exists()
        else pd.DataFrame()
    )
    frame_features_by_id = (
        {int(row["frame_id"]): row.to_dict() for _, row in frame_features.iterrows()}
        if not frame_features.empty
        else {}
    )
    actions = (
        pd.read_csv(action_features_csv)
        if action_features_csv and action_features_csv.exists()
        else None
    )

    model = YOLO(detector_model)
    racket_class_ids = get_class_ids(model, RACKET_CLASS_NAMES)
    if not racket_class_ids:
        raise RuntimeError(
            f"Detector model does not contain a tennis racket class. Available names: {model.names}"
        )
    print(f"Using racket class IDs: {racket_class_ids}")

    csv_path = output_dir / f"{source_video.stem}_racket_track.csv"
    mp4_path = output_dir / f"{source_video.stem}_racket_track.mp4"
    h264_path = output_dir / f"{source_video.stem}_racket_track_h264.mp4"

    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not write video: {mp4_path}")

    rows = []
    previous_center = None
    stream = model.predict(
        source=str(source_video),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        classes=racket_class_ids,
    )
    try:
        for frame_id, result in enumerate(stream, start=1):
            frame = result.orig_img.copy()
            pose_row = pose_by_frame.get(frame_id)
            if pose_row is None:
                if writer is not None:
                    writer.write(frame)
                continue

            action = active_action(actions, frame_id)
            wrist_name = active_wrist_name(action, handedness)
            wrist = point(pose_row, wrist_name, min_keypoint_conf)
            candidate = choose_racket_detection(
                result,
                racket_class_ids,
                pose_row,
                wrist,
                previous_center,
                player_margin,
            )
            if candidate is None:
                if writer is not None:
                    writer.write(frame)
                continue

            previous_center = candidate["center"]
            head = (
                farthest_bbox_corner_from_point(candidate["box"], wrist)
                if wrist is not None
                else candidate["center"]
            )
            frame_feature = frame_features_by_id.get(frame_id, {})
            body_scale = frame_feature.get("body_scale_px", np.nan)
            row = {
                "frame_id": frame_id,
                "class_name": candidate["class_name"],
                "racket_conf": candidate["conf"],
                "racket_bbox_x1": candidate["box"][0],
                "racket_bbox_y1": candidate["box"][1],
                "racket_bbox_x2": candidate["box"][2],
                "racket_bbox_y2": candidate["box"][3],
                "racket_center_x": candidate["center"][0],
                "racket_center_y": candidate["center"][1],
                "racket_head_proxy_x": head[0],
                "racket_head_proxy_y": head[1],
                "active_wrist_name": wrist_name,
                "active_wrist_x": wrist[0] if wrist is not None else np.nan,
                "active_wrist_y": wrist[1] if wrist is not None else np.nan,
                "racket_center_to_wrist_px": candidate["wrist_distance_px"],
                "racket_head_to_wrist_px": float(np.linalg.norm(head - wrist))
                if wrist is not None
                else np.nan,
                "body_scale_px": body_scale,
                "action_label": action.get("action_label") if action else "",
                "contact_frame": action.get("contact_frame") if action else "",
                "window_start": action.get("window_start") if action else "",
                "window_end": action.get("window_end") if action else "",
            }
            if not pd.isna(body_scale) and float(body_scale) > 0:
                row["racket_center_to_wrist_body_units"] = (
                    row["racket_center_to_wrist_px"] / float(body_scale)
                )
                row["racket_head_to_wrist_body_units"] = (
                    row["racket_head_to_wrist_px"] / float(body_scale)
                )
            rows.append(row)

            if writer is not None:
                draw_racket(frame, candidate, wrist, head)
                writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    track_df = add_motion_columns(pd.DataFrame(rows), fps=fps)
    track_df.to_csv(csv_path, index=False)

    h264_created = False
    if save_video and make_h264:
        h264_created = run_h264_conversion(mp4_path, h264_path)

    summary = {
        "source_video": str(source_video),
        "detector_model": detector_model,
        "pose_csv": str(pose_csv),
        "csv_path": str(csv_path),
        "video_path": str(mp4_path) if save_video else None,
        "h264_video_path": str(h264_path) if h264_created else None,
        "frames_with_racket": int(len(track_df)),
        "unique_frames_with_racket": int(track_df["frame_id"].nunique())
        if not track_df.empty
        else 0,
        "fps": fps,
        "note": (
            "Racket head is a bbox-corner proxy farthest from the active wrist, not a true racket keypoint."
        ),
    }
    print(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and track tennis racket associated with the tracked player."
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument(
        "--detector-model",
        default="yolo11x.pt",
        help="YOLO object detector weights, not pose weights. Must include tennis racket class.",
    )
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--frame-features-csv")
    parser.add_argument("--action-features-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--handedness", choices=["right", "left", "unknown"], default="right")
    parser.add_argument("--player-margin", type=float, default=0.75)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.25)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--h264", action="store_true")
    args = parser.parse_args(argv)

    track_racket(
        source_video=Path(args.source_video),
        detector_model=args.detector_model,
        pose_csv=Path(args.pose_csv),
        frame_features_csv=Path(args.frame_features_csv)
        if args.frame_features_csv
        else None,
        action_features_csv=Path(args.action_features_csv)
        if args.action_features_csv
        else None,
        output_dir=Path(args.output_dir),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        handedness=args.handedness,
        player_margin=args.player_margin,
        min_keypoint_conf=args.min_keypoint_conf,
        save_video=args.save_video,
        make_h264=args.h264,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
