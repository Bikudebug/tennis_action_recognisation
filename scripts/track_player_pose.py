from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_pipeline.pose_utils import (
    draw_detection,
    ensure_dir,
    extract_detections,
    get_video_properties,
    open_video_writer,
    pose_csv_fieldnames,
    run_h264_conversion,
    select_largest_detection,
    select_tracked_detection,
    write_detection_row,
    write_json,
)


def choose_initial_detection(detections: list[dict], target_person_id: str) -> tuple[int, str]:
    if not detections:
        raise ValueError("No people detected on the selection frame.")
    if target_person_id.lower() == "largest":
        return select_largest_detection(detections), "largest_selected"
    target_id = int(target_person_id)
    for idx, detection in enumerate(detections):
        if detection["det_person_id"] == target_id:
            return idx, "manual_id_selected"
    available = [d["det_person_id"] for d in detections]
    raise ValueError(f"Selected person id {target_id} not found. Available IDs: {available}")


def track_player_pose(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    target_person_id: str,
    select_frame: int,
    device: str,
    imgsz: int,
    conf: float,
    min_iou: float,
    save_video: bool,
    make_h264: bool,
    frame_by_frame: bool,
) -> dict:
    output_dir = ensure_dir(output_dir)
    properties = get_video_properties(video_path)
    fps = properties["fps"]
    width = properties["width"]
    height = properties["height"]
    frame_count = properties["frame_count"]

    annotated_path = output_dir / f"{video_path.stem}_tracked_pose.mp4"
    h264_path = output_dir / f"{video_path.stem}_tracked_pose_h264.mp4"
    csv_path = output_dir / f"{video_path.stem}_tracked_pose_keypoints.csv"
    manifest_path = output_dir / f"{video_path.stem}_tracked_pose_manifest.json"

    model = YOLO(str(model_path))
    writer = open_video_writer(annotated_path, width, height, fps) if save_video else None

    previous_box = None
    rows_written = 0
    frames_with_no_detection = 0
    frames_before_selection = 0
    selected_detection_snapshot = None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=pose_csv_fieldnames())
        csv_writer.writeheader()

        try:
            if frame_by_frame:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {video_path}")

                def result_iterator():
                    try:
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            yield model.predict(
                                frame,
                                imgsz=imgsz,
                                conf=conf,
                                device=device,
                                verbose=False,
                            )[0]
                    finally:
                        cap.release()

                stream = result_iterator()
            else:
                stream = model.predict(
                    source=str(video_path),
                    stream=True,
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                    verbose=False,
                )

            for frame_id, result in enumerate(stream, start=1):
                frame = result.orig_img.copy()
                detections = extract_detections(result)

                if frame_id < select_frame:
                    frames_before_selection += 1
                    if writer is not None:
                        writer.write(frame)
                    continue

                if not detections:
                    frames_with_no_detection += 1
                    if writer is not None:
                        writer.write(frame)
                    continue

                if previous_box is None:
                    chosen_idx, track_status = choose_initial_detection(
                        detections, target_person_id
                    )
                    match_score = 0.0
                    selected_detection_snapshot = detections[chosen_idx]
                else:
                    chosen_idx, track_status, match_score = select_tracked_detection(
                        detections, previous_box, min_iou=min_iou
                    )

                detection = detections[chosen_idx]
                previous_box = detection["bbox"]
                label = (
                    f"tracked id {detection['det_person_id']} "
                    f"{track_status} {match_score:.2f} frame {frame_id}"
                )
                draw_detection(frame, detection, label, (40, 220, 40))
                write_detection_row(csv_writer, frame_id, detection, track_status)
                rows_written += 1
                if writer is not None:
                    writer.write(frame)
        finally:
            if writer is not None:
                writer.release()

    h264_created = False
    if save_video and make_h264:
        h264_created = run_h264_conversion(annotated_path, h264_path)

    manifest = {
        "video": str(video_path),
        "model": str(model_path),
        "target_person_id": target_person_id,
        "select_frame": select_frame,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "pose_rows": rows_written,
        "frames_before_selection": frames_before_selection,
        "frames_with_no_detection_after_selection": frames_with_no_detection,
        "csv_path": str(csv_path),
        "annotated_video_path": str(annotated_path) if save_video else None,
        "h264_video_path": str(h264_path) if h264_created else None,
        "selected_detection": selected_detection_snapshot,
    }
    write_json(manifest_path, manifest)

    print(f"Video frames: {frame_count}")
    print(f"Pose rows written: {rows_written}")
    print(f"Frames without selected-player pose after selection: {frames_with_no_detection}")
    print(f"Pose CSV: {csv_path}")
    if save_video:
        print(f"Annotated video: {annotated_path}")
    else:
        print("Annotated pose video skipped.")
    if make_h264:
        if h264_created:
            print(f"H264 annotated video: {h264_path}")
        else:
            print("H264 conversion failed or ffmpeg is unavailable.")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track one selected tennis player and save YOLO pose keypoints."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-person-id",
        default="largest",
        help="Detection ID from inspect_players.py on --select-frame, or 'largest'.",
    )
    parser.add_argument("--select-frame", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-iou", type=float, default=0.01)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--h264", action="store_true")
    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="Use OpenCV frame decoding plus per-frame YOLO prediction. Useful for MOV files that terminate in Ultralytics stream mode.",
    )
    args = parser.parse_args(argv)

    track_player_pose(
        video_path=Path(args.video),
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        target_person_id=args.target_person_id,
        select_frame=args.select_frame,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        min_iou=args.min_iou,
        save_video=not args.skip_video,
        make_h264=args.h264,
        frame_by_frame=args.frame_by_frame,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
