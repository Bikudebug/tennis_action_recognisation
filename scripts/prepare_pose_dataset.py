from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_action.dataset_utils import parse_gt_excel


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tracked-pose CSV files for the labeled tennis videos used in training/testing.")
    parser.add_argument("--excel", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--pose-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--videos", nargs="*", default=None, help="Optional explicit video stems such as tennis_video_1 tennis_video_2. Default: all labeled videos from the Excel file.")
    parser.add_argument("--target-person-id", default="largest")
    parser.add_argument("--select-frame", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frame-by-frame", action="store_true")
    parser.add_argument("--h264", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python = sys.executable
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.videos:
        video_stems = list(dict.fromkeys(args.videos))
    else:
        gt_df = parse_gt_excel(Path(args.excel), Path(args.video_dir))
        video_stems = sorted(gt_df["video_stem"].dropna().astype(str).unique().tolist())

    for video_stem in video_stems:
        candidates = [
            Path(args.video_dir) / f"{video_stem}.MOV",
            Path(args.video_dir) / f"{video_stem}.mov",
            Path(args.video_dir) / f"{video_stem}.mp4",
        ]
        video_path = next((path for path in candidates if path.exists()), None)
        if video_path is None:
            raise FileNotFoundError(f"Could not find video for {video_stem} in {args.video_dir}")

        output_dir = output_root / video_stem / "pose"
        command = [
            python,
            str(root / "track_player_pose.py"),
            "--video",
            str(video_path),
            "--model",
            args.pose_model,
            "--output-dir",
            str(output_dir),
            "--target-person-id",
            args.target_person_id,
            "--select-frame",
            str(args.select_frame),
            "--device",
            args.device,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
        ]
        if args.frame_by_frame:
            command.append("--frame-by-frame")
        if args.h264:
            command.append("--h264")
        run(command, root)

    print("\nFinished pose preparation for labeled videos.")
    print(f"Pose root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
