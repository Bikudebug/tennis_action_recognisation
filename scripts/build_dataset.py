from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_action.common import ensure_dir, write_json
from tennis_action.dataset_utils import (
    build_frame_labels,
    build_sequence_dataset_for_video,
    discover_pose_csv,
    find_video_for_label,
    parse_gt_excel,
    save_per_video_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frame-level and sequence-level tennis action datasets.")
    parser.add_argument("--excel", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--train-videos", nargs="+", required=True)
    parser.add_argument("--test-videos", nargs="+", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    per_video_dir = ensure_dir(output_dir / "per_video")
    annotations_dir = ensure_dir(output_dir / "annotations")

    annotations_df = parse_gt_excel(Path(args.excel), Path(args.video_dir))
    annotations_df.to_csv(annotations_dir / "gt_annotations_all.csv", index=False)
    clean_df = annotations_df[~annotations_df["needs_review"]].copy()
    review_df = annotations_df[annotations_df["needs_review"]].copy()
    clean_df.to_csv(annotations_dir / "gt_annotations_clean.csv", index=False)
    review_df.to_csv(annotations_dir / "gt_annotations_review_excluded.csv", index=False)

    selected_videos = args.train_videos + args.test_videos
    bundle = {
        "train": {"X": [], "y": [], "frame_ids": [], "video_ids": []},
        "test": {"X": [], "y": [], "frame_ids": [], "video_ids": []},
    }
    summary_rows = []

    for video_stem in selected_videos:
        video_path = find_video_for_label(Path(args.video_dir), video_stem)
        if video_path is None:
            raise FileNotFoundError(f"Video not found for {video_stem}")
        pose_csv = discover_pose_csv(Path(args.pose_root), video_stem)
        video_events = clean_df[clean_df["video_stem"] == video_stem].copy().sort_values("contact_frame")
        from tennis_action.common import get_video_properties

        frame_count = get_video_properties(video_path)["frame_count"]
        frame_labels_df = build_frame_labels(frame_count, video_events, args.window_radius)
        X, y, frame_ids, metadata = build_sequence_dataset_for_video(
            video_path=video_path,
            pose_csv=pose_csv,
            frame_labels_df=frame_labels_df,
            window_radius=args.window_radius,
            conf_thresh=args.conf_thresh,
        )
        save_per_video_dataset(
            per_video_dir,
            video_stem,
            X,
            y,
            frame_ids,
            frame_labels_df,
            metadata,
        )

        split = "train" if video_stem in args.train_videos else "test"
        bundle[split]["X"].append(X)
        bundle[split]["y"].append(y)
        bundle[split]["frame_ids"].append(frame_ids)
        bundle[split]["video_ids"].append(np.array([video_stem] * len(frame_ids), dtype=object))

        label_counts = frame_labels_df["label_name"].value_counts().to_dict()
        summary_rows.append(
            {
                "video_stem": video_stem,
                "split": split,
                "frame_count": int(frame_count),
                "num_events_used": int(len(video_events)),
                "labels": json.dumps(label_counts),
                "pose_csv": str(pose_csv),
            }
        )

    for split in ("train", "test"):
        np.savez_compressed(
            output_dir / f"{split}_dataset.npz",
            X=np.concatenate(bundle[split]["X"], axis=0),
            y=np.concatenate(bundle[split]["y"], axis=0),
            frame_ids=np.concatenate(bundle[split]["frame_ids"], axis=0),
            video_ids=np.concatenate(bundle[split]["video_ids"], axis=0),
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "dataset_summary.csv", index=False)
    write_json(
        output_dir / "dataset_summary.json",
        {
            "train_videos": args.train_videos,
            "test_videos": args.test_videos,
            "window_radius": args.window_radius,
            "conf_thresh": args.conf_thresh,
            "videos": summary_rows,
        },
    )
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
