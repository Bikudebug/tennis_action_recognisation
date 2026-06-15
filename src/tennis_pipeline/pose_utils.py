from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

ANALYSIS_KEYPOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

SKELETON_EDGES_INDEX = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

SKELETON_EDGES_NAME = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_properties(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    properties = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return properties


def open_video_writer(path: Path, width: int, height: int, fps: float):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer


def pose_csv_fieldnames() -> list[str]:
    fields = [
        "frame_id",
        "det_person_id",
        "track_status",
        "person_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    for name in KEYPOINT_NAMES:
        fields.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    return fields


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(box_a) + box_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_distance(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return math.hypot(ax - bx, ay - by)


def select_largest_detection(detections: list[dict]) -> int:
    if not detections:
        raise ValueError("No detections available.")
    return max(range(len(detections)), key=lambda i: box_area(detections[i]["bbox"]))


def select_tracked_detection(
    detections: list[dict],
    previous_box: Iterable[float] | None,
    min_iou: float = 0.01,
) -> tuple[int, str, float]:
    if not detections:
        raise ValueError("No detections available.")
    if previous_box is None:
        idx = select_largest_detection(detections)
        return idx, "largest_initial", 0.0

    ious = [box_iou(previous_box, det["bbox"]) for det in detections]
    best_idx = int(np.argmax(ious))
    best_iou = float(ious[best_idx])
    if best_iou >= min_iou:
        return best_idx, "iou", best_iou

    distances = [center_distance(previous_box, det["bbox"]) for det in detections]
    best_idx = int(np.argmin(distances))
    return best_idx, "center_fallback", best_iou


def extract_detections(result) -> list[dict]:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    detections = []
    for idx, (box, conf, kpts) in enumerate(zip(boxes, confs, keypoints)):
        detections.append(
            {
                "det_person_id": idx,
                "person_conf": float(conf),
                "bbox": [float(v) for v in box],
                "keypoints": [[float(x), float(y), float(c)] for x, y, c in kpts],
                "area": float(box_area(box)),
            }
        )
    return detections


def draw_detection(
    frame,
    detection: dict,
    label: str,
    color: tuple[int, int, int],
    min_keypoint_conf: float = 0.25,
    thickness: int = 4,
):
    x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(45, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (0, 0, 0),
        7,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x1, max(45, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        color,
        3,
        cv2.LINE_AA,
    )

    keypoints = detection["keypoints"]
    for start, end in SKELETON_EDGES_INDEX:
        xs, ys, cs = keypoints[start]
        xe, ye, ce = keypoints[end]
        if cs >= min_keypoint_conf and ce >= min_keypoint_conf:
            cv2.line(frame, (int(xs), int(ys)), (int(xe), int(ye)), color, thickness)

    for x, y, conf in keypoints:
        if conf >= min_keypoint_conf:
            cv2.circle(frame, (int(x), int(y)), max(5, thickness + 2), color, -1)


def write_detection_row(csv_writer: csv.DictWriter, frame_id: int, detection: dict, track_status: str):
    row = {
        "frame_id": frame_id,
        "det_person_id": detection["det_person_id"],
        "track_status": track_status,
        "person_conf": detection["person_conf"],
        "bbox_x1": detection["bbox"][0],
        "bbox_y1": detection["bbox"][1],
        "bbox_x2": detection["bbox"][2],
        "bbox_y2": detection["bbox"][3],
    }
    for name, (x, y, conf) in zip(KEYPOINT_NAMES, detection["keypoints"]):
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y
        row[f"{name}_conf"] = conf
    csv_writer.writerow(row)


def frame_to_json_payload(frame_row: dict) -> dict:
    payload = {
        "frame_id": int(frame_row["frame_id"]),
        "bbox": {
            "x1": float(frame_row["bbox_x1"]),
            "y1": float(frame_row["bbox_y1"]),
            "x2": float(frame_row["bbox_x2"]),
            "y2": float(frame_row["bbox_y2"]),
        },
        "keypoints": {},
    }
    for name in KEYPOINT_NAMES:
        payload["keypoints"][name] = {
            "x": float(frame_row[f"{name}_x"]),
            "y": float(frame_row[f"{name}_y"]),
            "confidence": float(frame_row[f"{name}_conf"]),
        }
    return payload


def normalize_action_label(value) -> str:
    text = str(value).strip().lower()
    text = text.replace("forhand", "forehand")
    if "forehand" in text:
        return "forehand"
    if "backhand" in text:
        return "backhand"
    if "serve" in text:
        return "serve"
    if "smash" in text or "overhead" in text:
        return "smash"
    if "volley" in text:
        return "volley"
    if text in {"d", "ds"}:
        return "uncertain"
    return text


def normalize_video_label(value) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return f"tennis_video_{int(match.group(1))}"
    return text.lower().replace(" ", "_")


def find_video_for_label(video_dir: Path, video_label: str) -> Path | None:
    normalized = normalize_video_label(video_label)
    candidates = sorted(video_dir.glob(f"{normalized}.*"))
    if candidates:
        return candidates[0]
    match = re.search(r"(\d+)", normalized)
    if match:
        number = int(match.group(1))
        candidates = sorted(video_dir.glob(f"*{number}.*"))
        if candidates:
            return candidates[0]
    return None


def write_json(path: Path, payload: dict):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_h264_conversion(input_path: Path, output_path: Path, crf: int = 23) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True
