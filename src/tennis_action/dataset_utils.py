from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    KEYPOINT_NAMES,
    LABEL_TO_ID,
    ensure_dir,
    get_video_properties,
    is_empty,
    is_number,
    normalize_action_label,
    normalize_video_label,
    probe_video_rotation,
    rotate_xy,
)


def parse_gt_excel(excel_path: Path, video_dir: Path) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=0, header=None)
    events_raw = []
    note_map = defaultdict(list)
    current_video = None

    for row_idx in range(1, len(raw)):
        row = raw.iloc[row_idx]
        if not is_empty(row.iloc[0]):
            current_video = str(row.iloc[0]).strip()
        if current_video is None:
            continue

        action_cell = row.iloc[1]
        if is_empty(action_cell):
            continue
        row_action_raw = str(action_cell).strip()
        row_action = normalize_action_label(row_action_raw)
        if row_action not in {"forehand", "backhand", "serve"}:
            continue

        for col_idx in range(2, len(row)):
            value = row.iloc[col_idx]
            if is_empty(value):
                continue
            key = (current_video, col_idx)
            if is_number(value):
                events_raw.append(
                    {
                        "video_label": current_video,
                        "video_stem": normalize_video_label(current_video),
                        "action_label": row_action,
                        "action_label_raw": row_action_raw,
                        "contact_frame": int(round(float(value))),
                        "source_row": row_idx + 1,
                        "source_col": col_idx + 1,
                    }
                )
            else:
                note_map[key].append(str(value).strip())

    clean_events = []
    for event_id, event in enumerate(events_raw):
        key = (event["video_label"], event["source_col"] - 1)
        notes = note_map.get(key, [])
        video_path = find_video_for_label(video_dir, event["video_stem"])
        clean_events.append(
            {
                "event_id": event_id,
                "video_label": event["video_label"],
                "video_stem": event["video_stem"],
                "video_file": str(video_path) if video_path else "",
                "action_label": event["action_label"],
                "action_label_id": LABEL_TO_ID[event["action_label"]],
                "contact_frame": event["contact_frame"],
                "needs_review": bool(notes),
                "review_note": "; ".join(notes),
                "source_row": event["source_row"],
                "source_col": event["source_col"],
            }
        )
    return pd.DataFrame(clean_events)


def find_video_for_label(video_dir: Path, video_stem: str) -> Path | None:
    candidates = [
        video_dir / f"{video_stem}.MOV",
        video_dir / f"{video_stem}.mov",
        video_dir / f"{video_stem}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_pose_csv(pose_root: Path, video_stem: str) -> Path:
    candidates = [
        pose_root / video_stem / "pose" / f"{video_stem}_tracked_pose_keypoints.csv",
        pose_root.parent / "Outputs" / "Pose_Estimation_yolo" / video_stem.replace("tennis_", "video_") / f"{video_stem}_yolo11x_pose_keypoints.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Pose CSV not found for {video_stem}. Tried: {candidates}")


def load_pose_csv(pose_csv: Path) -> dict[int, np.ndarray]:
    df = pd.read_csv(pose_csv)
    df = df.sort_values("person_conf", ascending=False).drop_duplicates("frame_id", keep="first")
    kp_cols = [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y", "conf")]
    out = {}
    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        out[frame_id] = row[kp_cols].to_numpy(dtype=np.float32).reshape(len(KEYPOINT_NAMES), 3)
    return out


def build_frame_labels(frame_count: int, events_df: pd.DataFrame, window_radius: int) -> pd.DataFrame:
    label_ids = np.full(frame_count + 1, LABEL_TO_ID["neutral"], dtype=np.int32)
    event_ids = np.full(frame_count + 1, -1, dtype=np.int32)
    contact_frames = np.full(frame_count + 1, -1, dtype=np.int32)
    distance_to_contact = np.full(frame_count + 1, 10**9, dtype=np.int32)

    for _, event in events_df.iterrows():
        start = max(1, int(event["contact_frame"]) - window_radius)
        end = min(frame_count, int(event["contact_frame"]) + window_radius)
        label_id = int(event["action_label_id"])
        event_id = int(event["event_id"])
        contact = int(event["contact_frame"])
        for frame_id in range(start, end + 1):
            distance = abs(frame_id - contact)
            if distance < distance_to_contact[frame_id]:
                distance_to_contact[frame_id] = distance
                label_ids[frame_id] = label_id
                event_ids[frame_id] = event_id
                contact_frames[frame_id] = contact

    rows = []
    for frame_id in range(1, frame_count + 1):
        label_id = int(label_ids[frame_id])
        rows.append(
            {
                "frame_id": frame_id,
                "label_id": label_id,
                "label_name": list(LABEL_TO_ID.keys())[list(LABEL_TO_ID.values()).index(label_id)],
                "event_id": int(event_ids[frame_id]),
                "contact_frame": int(contact_frames[frame_id]),
                "distance_to_contact": None
                if event_ids[frame_id] < 0
                else int(distance_to_contact[frame_id]),
            }
        )
    return pd.DataFrame(rows)


def build_sequence_dataset_for_video(
    video_path: Path,
    pose_csv: Path,
    frame_labels_df: pd.DataFrame,
    window_radius: int,
    conf_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    pose_map = load_pose_csv(pose_csv)
    props = get_video_properties(video_path)
    raw_w, raw_h = props["width"], props["height"]
    rotation = probe_video_rotation(video_path, video_path.stem)
    num_frames = len(frame_labels_df)
    window_size = window_radius * 2 + 1
    X = np.full((num_frames, window_size, len(KEYPOINT_NAMES), 2), np.nan, dtype=np.float32)
    y = frame_labels_df["label_id"].to_numpy(dtype=np.int32)
    frame_ids = frame_labels_df["frame_id"].to_numpy(dtype=np.int32)

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

    metadata = {
        "video_path": str(video_path),
        "pose_csv": str(pose_csv),
        "rotation": rotation,
        "fps": props["fps"],
        "width": raw_h if rotation in (90, 270) else raw_w,
        "height": raw_w if rotation in (90, 270) else raw_h,
        "frame_count": props["frame_count"],
        "window_radius": window_radius,
        "window_size": window_size,
    }
    return X, y, frame_ids, metadata


def save_per_video_dataset(
    output_dir: Path,
    video_stem: str,
    X: np.ndarray,
    y: np.ndarray,
    frame_ids: np.ndarray,
    frame_labels_df: pd.DataFrame,
    metadata: dict,
) -> None:
    ensure_dir(output_dir)
    np.savez_compressed(
        output_dir / f"{video_stem}_sequences.npz",
        X=X,
        y=y,
        frame_ids=frame_ids,
    )
    frame_labels_df.to_csv(output_dir / f"{video_stem}_frame_labels.csv", index=False)
    pd.DataFrame([metadata]).to_json(output_dir / f"{video_stem}_metadata.json", orient="records", indent=2)

