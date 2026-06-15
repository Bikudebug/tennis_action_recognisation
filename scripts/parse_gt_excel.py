from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_pipeline.pose_utils import (
    ensure_dir,
    find_video_for_label,
    normalize_action_label,
    normalize_video_label,
)


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


def parse_gt_excel(excel_path: Path, video_dir: Path, output_dir: Path) -> dict:
    output_dir = ensure_dir(output_dir)
    raw = pd.read_excel(excel_path, sheet_name=0, header=None)

    events_raw = []
    note_map = defaultdict(list)
    notes_raw = []
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

        for col_idx in range(2, len(row)):
            value = row.iloc[col_idx]
            if is_empty(value):
                continue
            key = (current_video, col_idx)
            if is_number(value):
                events_raw.append(
                    {
                        "video_label": current_video,
                        "row_action_raw": row_action_raw,
                        "action_label": row_action,
                        "contact_frame": int(round(float(value))),
                        "source_row": row_idx + 1,
                        "source_col": col_idx + 1,
                    }
                )
            else:
                note_text = str(value).strip()
                note_payload = {
                    "video_label": current_video,
                    "row_action_raw": row_action_raw,
                    "row_action": row_action,
                    "note": note_text,
                    "source_row": row_idx + 1,
                    "source_col": col_idx + 1,
                }
                note_map[key].append(note_payload)
                notes_raw.append(note_payload)

    numeric_keys = {(e["video_label"], e["source_col"] - 1) for e in events_raw}
    clean_events = []
    for event_id, event in enumerate(events_raw):
        key = (event["video_label"], event["source_col"] - 1)
        attached_notes = note_map.get(key, [])
        note_text = "; ".join(
            f"{note['row_action_raw']}:{note['note']}" for note in attached_notes
        )
        video_stem = normalize_video_label(event["video_label"])
        video_path = find_video_for_label(video_dir, event["video_label"])
        clean_events.append(
            {
                "event_id": event_id,
                "video_label": event["video_label"],
                "video_stem": video_stem,
                "video_file": str(video_path) if video_path else "",
                "action_label": event["action_label"],
                "action_label_raw": event["row_action_raw"],
                "contact_frame": event["contact_frame"],
                "window_start_15": max(1, event["contact_frame"] - 15),
                "window_end_15": event["contact_frame"] + 15,
                "needs_review": bool(attached_notes),
                "review_note": note_text,
                "source_row": event["source_row"],
                "source_col": event["source_col"],
            }
        )

    orphan_notes = [
        note
        for note in notes_raw
        if (note["video_label"], note["source_col"] - 1) not in numeric_keys
    ]

    annotations_csv = output_dir / "gt_annotations_clean.csv"
    notes_csv = output_dir / "gt_notes_needing_manual_review.csv"
    summary_json = output_dir / "gt_summary.json"

    events_df = pd.DataFrame(clean_events)
    events_df.to_csv(annotations_csv, index=False)
    pd.DataFrame(orphan_notes).to_csv(notes_csv, index=False)

    summary = {
        "excel_path": str(excel_path),
        "video_dir": str(video_dir),
        "annotations_csv": str(annotations_csv),
        "notes_csv": str(notes_csv),
        "num_events": int(len(events_df)),
        "num_notes_needing_manual_review": int(len(orphan_notes)),
        "events_by_video": events_df.groupby("video_stem").size().to_dict()
        if not events_df.empty
        else {},
        "events_by_label": events_df.groupby("action_label").size().to_dict()
        if not events_df.empty
        else {},
        "review_events_by_video": events_df[events_df["needs_review"]]
        .groupby("video_stem")
        .size()
        .to_dict()
        if not events_df.empty
        else {},
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Clean annotations: {annotations_csv}")
    print(f"Notes needing manual review: {notes_csv}")
    print(f"Summary: {summary_json}")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse the wide tennis GT Excel file into one event row per contact frame."
    )
    parser.add_argument("--excel", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    parse_gt_excel(Path(args.excel), Path(args.video_dir), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
