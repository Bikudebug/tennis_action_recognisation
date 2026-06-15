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
    write_json,
)


def read_frame(video_path: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
    return frame


def inspect_players(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    frame_id: int,
    device: str,
    imgsz: int,
    conf: float,
) -> dict:
    output_dir = ensure_dir(output_dir)
    frame = read_frame(video_path, frame_id)
    model = YOLO(str(model_path))
    result = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    detections = extract_detections(result)

    annotated = frame.copy()
    colors = [
        (40, 220, 40),
        (40, 160, 255),
        (255, 190, 40),
        (220, 60, 220),
        (60, 220, 220),
        (255, 80, 80),
    ]
    for detection in detections:
        det_id = detection["det_person_id"]
        label = (
            f"id {det_id} conf {detection['person_conf']:.2f} "
            f"area {detection['area']:.0f}"
        )
        draw_detection(annotated, detection, label, colors[det_id % len(colors)])

    stem = f"{video_path.stem}_frame{frame_id:06d}"
    image_path = output_dir / f"{stem}_player_candidates.jpg"
    csv_path = output_dir / f"{stem}_player_candidates.csv"
    json_path = output_dir / f"{stem}_player_candidates.json"
    cv2.imwrite(str(image_path), annotated)

    fieldnames = [
        "det_person_id",
        "person_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "area",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            writer.writerow(
                {
                    "det_person_id": det["det_person_id"],
                    "person_conf": det["person_conf"],
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "area": det["area"],
                }
            )

    payload = {
        "video": str(video_path),
        "frame_id": frame_id,
        "image_path": str(image_path),
        "csv_path": str(csv_path),
        "detections": detections,
    }
    write_json(json_path, payload)

    print(f"Saved player inspection image: {image_path}")
    print(f"Saved candidates CSV: {csv_path}")
    print(f"Saved candidates JSON: {json_path}")
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(
            "  id={id} conf={conf:.3f} area={area:.0f} bbox={bbox}".format(
                id=det["det_person_id"],
                conf=det["person_conf"],
                area=det["area"],
                bbox=[round(v, 1) for v in det["bbox"]],
            )
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect all people on one frame so the user can choose a player ID."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    inspect_players(
        video_path=Path(args.video),
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        frame_id=args.frame,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
