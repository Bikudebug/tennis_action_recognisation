from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_pipeline.pose_utils import (
    ensure_dir,
    frame_to_json_payload,
    normalize_video_label,
    write_json,
)


def filter_annotations(annotations: pd.DataFrame, video_name: str) -> pd.DataFrame:
    video_name_norm = normalize_video_label(video_name)
    filtered = annotations[
        (annotations["video_stem"].astype(str) == video_name_norm)
        | (annotations["video_file"].astype(str).apply(lambda p: Path(p).stem if p else "") == video_name)
    ].copy()
    if filtered.empty:
        filtered = annotations[
            annotations["video_label"].astype(str).apply(normalize_video_label) == video_name_norm
        ].copy()
    return filtered.sort_values("contact_frame")


def export_action_windows(
    pose_csv: Path,
    annotations_csv: Path,
    video_name: str,
    output_dir: Path,
    window_radius: int,
) -> dict:
    output_dir = ensure_dir(output_dir)
    pose_df = pd.read_csv(pose_csv)
    pose_by_frame = {int(row["frame_id"]): row.to_dict() for _, row in pose_df.iterrows()}
    annotations = filter_annotations(pd.read_csv(annotations_csv), video_name)

    counts = {}
    for _, event in annotations.iterrows():
        label = str(event["action_label"])
        contact = int(event["contact_frame"])
        start = max(1, contact - window_radius)
        end = contact + window_radius
        frames = []
        missing_frames = []
        for frame_id in range(start, end + 1):
            frame_row = pose_by_frame.get(frame_id)
            if frame_row is None:
                missing_frames.append(frame_id)
            else:
                frames.append(frame_to_json_payload(frame_row))

        label_dir = ensure_dir(output_dir / label)
        local_index = counts.get(label, 0)
        counts[label] = local_index + 1
        event_id = int(event["event_id"]) if "event_id" in event else local_index
        out_path = label_dir / f"event{event_id:03d}_{label}_s{start}e{end}.json"
        payload = {
            "event_id": event_id,
            "local_action_index": local_index,
            "video_stem": normalize_video_label(video_name),
            "label": label,
            "contact_frame": contact,
            "window": {
                "start_frame": start,
                "end_frame": end,
                "window_radius": window_radius,
                "expected_num_frames": end - start + 1,
            },
            "needs_review": bool(event.get("needs_review", False)),
            "review_note": event.get("review_note", ""),
            "missing_pose_frames": missing_frames,
            "frames": frames,
        }
        write_json(out_path, payload)

    summary = {
        "pose_csv": str(pose_csv),
        "annotations_csv": str(annotations_csv),
        "video_name": video_name,
        "output_dir": str(output_dir),
        "window_radius": window_radius,
        "num_events": int(len(annotations)),
        "counts_by_label": counts,
    }
    write_json(output_dir / "action_windows_summary.json", summary)
    print(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one JSON pose window per GT action event."
    )
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-radius", type=int, default=15)
    args = parser.parse_args(argv)
    export_action_windows(
        pose_csv=Path(args.pose_csv),
        annotations_csv=Path(args.annotations_csv),
        video_name=args.video_name,
        output_dir=Path(args.output_dir),
        window_radius=args.window_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
