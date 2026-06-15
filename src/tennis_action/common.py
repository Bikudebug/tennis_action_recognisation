from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

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

KEYPOINT_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

SKELETON_EDGES = [
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

LABEL_TO_ID = {
    "forehand": 0,
    "backhand": 1,
    "serve": 2,
    "neutral": 3,
}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

DISPLAY_LABEL = {
    "forehand": "Forehand",
    "backhand": "Backhand",
    "serve": "Serve",
    "neutral": "Neutral",
}

LABEL_COLORS_BGR = {
    "forehand": (60, 190, 70),
    "backhand": (70, 90, 230),
    "serve": (230, 170, 60),
    "neutral": (140, 140, 140),
}

FALLBACK_ROTATIONS = {
    "tennis_video_1": 90,
    "tennis_video_2": 90,
    "tennis_video_4": 90,
    "tennis_video_5": 0,
    "tennis_video_6": 90,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_video_label(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    text = text.replace(".mov", "").replace(".mp4", "")
    if text.startswith("tennis_"):
        text = text[len("tennis_") :]
    if text.startswith("video") and "_" not in text:
        text = text.replace("video", "video_")
    if text.startswith("video_"):
        return f"tennis_{text}"
    return text


def normalize_action_label(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "fh": "forehand",
        "forehand": "forehand",
        "bh": "backhand",
        "backhand": "backhand",
        "serve": "serve",
        "service": "serve",
    }
    return mapping.get(text, text)


def is_empty(value) -> bool:
    if value is None:
        return True
    try:
        if math.isnan(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def is_number(value) -> bool:
    if is_empty(value):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def get_video_properties(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    props = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


def probe_video_rotation(video_path: Path, video_stem: str) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if streams:
            stream = streams[0]
            if "tags" in stream and "rotate" in stream["tags"]:
                return int(float(stream["tags"]["rotate"]))
            side_data = stream.get("side_data_list") or []
            if side_data:
                rotation = side_data[0].get("rotation")
                if rotation is not None:
                    return int(float(rotation))
    except Exception:
        pass
    return FALLBACK_ROTATIONS.get(video_stem, 0)


def rotate_xy(xy: np.ndarray, rotation: int, raw_width: int, raw_height: int) -> np.ndarray:
    out = xy.copy()
    if rotation == 90:
        out[..., 0] = xy[..., 1]
        out[..., 1] = raw_width - xy[..., 0]
    elif rotation == 180:
        out[..., 0] = raw_width - xy[..., 0]
        out[..., 1] = raw_height - xy[..., 1]
    elif rotation == 270:
        out[..., 0] = raw_height - xy[..., 1]
        out[..., 1] = xy[..., 0]
    return out


def draw_pose(
    frame,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    conf_thresh: float,
    line_thickness: int = 3,
    joint_radius: int = 5,
    joint_outline_thickness: int = 1,
    joint_fill_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    for start_name, end_name in SKELETON_EDGES:
        start_idx = KEYPOINT_INDEX[start_name]
        end_idx = KEYPOINT_INDEX[end_name]
        if keypoints[start_idx, 2] < conf_thresh or keypoints[end_idx, 2] < conf_thresh:
            continue
        pt1 = tuple(int(v) for v in keypoints[start_idx, :2])
        pt2 = tuple(int(v) for v in keypoints[end_idx, :2])
        cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)
    for idx in range(len(KEYPOINT_NAMES)):
        if keypoints[idx, 2] < conf_thresh:
            continue
        pt = tuple(int(v) for v in keypoints[idx, :2])
        cv2.circle(frame, pt, joint_radius, joint_fill_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, joint_radius + 1, color, joint_outline_thickness, cv2.LINE_AA)


def run_h264_conversion(input_path: Path, output_path: Path) -> bool:
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
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except Exception:
        return False
