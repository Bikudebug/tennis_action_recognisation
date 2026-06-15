from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path):
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete merged tennis pipeline on a raw video using a pretrained action model.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-person-id", default="largest")
    parser.add_argument("--select-frame", type=int, default=1)
    parser.add_argument("--pose-device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--frame-by-frame", action="store_true")
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--smooth-kernel", type=int, default=9)
    parser.add_argument("--handedness", choices=["right", "left", "unknown"], default="right")
    parser.add_argument("--detector-model", help="Optional racket detector model path")
    parser.add_argument("--rotation", type=int, default=None, help="Optional manual override for action-video display rotation. Default: no extra rotation.")
    parser.add_argument("--h264", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python = sys.executable
    video_path = Path(args.video)
    run_dir = Path(args.output_root) / video_path.stem
    inspection_dir = run_dir / "inspection"
    pose_dir = run_dir / "pose"
    pred_dir = run_dir / "action_predictions"
    action_video_dir = run_dir / "action_video"
    motion_dir = run_dir / "motion_analysis"
    racket_dir = run_dir / "racket_tracking"
    explained_video_dir = run_dir / "explained_video"

    run([python, str(root / "inspect_players.py"), "--video", str(video_path), "--model", args.pose_model, "--output-dir", str(inspection_dir), "--frame", str(args.select_frame), "--device", args.pose_device, "--imgsz", str(args.imgsz), "--conf", str(args.pose_conf)], root)

    track_cmd = [python, str(root / "track_player_pose.py"), "--video", str(video_path), "--model", args.pose_model, "--output-dir", str(pose_dir), "--target-person-id", args.target_person_id, "--select-frame", str(args.select_frame), "--device", args.pose_device, "--imgsz", str(args.imgsz), "--conf", str(args.pose_conf)]
    if args.frame_by_frame:
        track_cmd.append("--frame-by-frame")
    if args.h264:
        track_cmd.append("--h264")
    run(track_cmd, root)

    pose_csv = pose_dir / f"{video_path.stem}_tracked_pose_keypoints.csv"
    run([python, str(root / "predict_actions_on_video.py"), "--video", str(video_path), "--pose-csv", str(pose_csv), "--checkpoint-dir", args.checkpoint_dir, "--output-dir", str(pred_dir), "--window-radius", str(args.window_radius), "--conf-thresh", str(args.conf_thresh), "--smooth-kernel", str(args.smooth_kernel)], root)

    render_cmd = [python, str(root / "render_action_predictions.py"), "--video", str(video_path), "--pose-csv", str(pose_csv), "--prediction-csv", str(pred_dir / "frame_predictions.csv"), "--output-dir", str(action_video_dir)]
    if args.rotation is not None:
        render_cmd.extend(["--rotation", str(args.rotation)])
    if args.h264:
        render_cmd.append("--h264")
    run(render_cmd, root)

    run([python, str(root / "analyze_motion_features.py"), "--pose-csv", str(pose_csv), "--annotations-csv", str(pred_dir / "predicted_action_segments.csv"), "--video-name", video_path.stem, "--source-video", str(video_path), "--output-dir", str(motion_dir), "--handedness", args.handedness, "--window-radius", str(args.window_radius)], root)

    racket_csv = None
    if args.detector_model:
        racket_cmd = [python, str(root / "track_racket.py"), "--source-video", str(video_path), "--detector-model", args.detector_model, "--pose-csv", str(pose_csv), "--frame-features-csv", str(motion_dir / "frame_motion_features.csv"), "--action-features-csv", str(motion_dir / "action_window_features.csv"), "--output-dir", str(racket_dir), "--device", args.pose_device, "--imgsz", str(args.imgsz), "--conf", "0.10", "--handedness", args.handedness, "--save-video"]
        if args.h264:
            racket_cmd.append("--h264")
        run(racket_cmd, root)
        racket_csv = racket_dir / f"{video_path.stem}_racket_track.csv"

    motion_video_cmd = [python, str(root / "make_explained_motion_video.py"), "--source-video", str(video_path), "--pose-csv", str(pose_csv), "--action-features-csv", str(motion_dir / "action_window_features.csv"), "--frame-features-csv", str(motion_dir / "frame_motion_features.csv"), "--output-dir", str(explained_video_dir), "--output-name", f"{video_path.stem}_explained_motion.mp4"]
    if racket_csv is not None:
        motion_video_cmd.extend(["--racket-csv", str(racket_csv)])
    if args.h264:
        motion_video_cmd.append("--h264")
    run(motion_video_cmd, root)

    print("\nComplete pipeline finished.")
    print(f"Run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
