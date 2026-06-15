from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path):
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the supervised training/evaluation pipeline for action recognition.")
    parser.add_argument("--excel", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepare-pose", action="store_true", help="First generate tracked pose CSVs for the train/test videos inside --pose-root.")
    parser.add_argument("--pose-model", default=None, help="Required when --prepare-pose is used.")
    parser.add_argument("--target-person-id", default="largest")
    parser.add_argument("--select-frame", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--neutral-ratio", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--train-videos", nargs="+", default=["tennis_video_1", "tennis_video_4", "tennis_video_6"])
    parser.add_argument("--test-videos", nargs="+", default=["tennis_video_2", "tennis_video_5"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    dataset_dir = output_root / "dataset"
    models_dir = output_root / "models"
    results_dir = output_root / "results"
    python = sys.executable

    if args.prepare_pose:
        if not args.pose_model:
            raise SystemExit("--pose-model is required when --prepare-pose is used.")
        run(
            [
                python,
                str(root / "prepare_pose_dataset.py"),
                "--excel",
                args.excel,
                "--video-dir",
                args.video_dir,
                "--pose-model",
                args.pose_model,
                "--output-root",
                args.pose_root,
                "--videos",
                *(args.train_videos + args.test_videos),
                "--target-person-id",
                args.target_person_id,
                "--select-frame",
                str(args.select_frame),
                "--device",
                args.device.replace("cuda:", ""),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.pose_conf),
            ],
            root,
        )

    run([python, str(root / "build_dataset.py"), "--excel", args.excel, "--video-dir", args.video_dir, "--pose-root", args.pose_root, "--output-dir", str(dataset_dir), "--window-radius", str(args.window_radius), "--conf-thresh", str(args.conf_thresh), "--train-videos", *args.train_videos, "--test-videos", *args.test_videos], root)
    run([python, str(root / "train_models.py"), "--dataset-dir", str(dataset_dir), "--output-dir", str(models_dir), "--neutral-ratio", str(args.neutral_ratio), "--device", args.device, "--lstm-epochs", str(args.lstm_epochs)], root)
    run([python, str(root / "evaluate_and_render.py"), "--dataset-dir", str(dataset_dir), "--models-dir", str(models_dir), "--video-dir", args.video_dir, "--pose-root", args.pose_root, "--output-dir", str(results_dir), "--conf-thresh", str(args.conf_thresh)], root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
