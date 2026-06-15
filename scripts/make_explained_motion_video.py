from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_pipeline.pose_utils import (
    KEYPOINT_NAMES,
    SKELETON_EDGES_NAME,
    ensure_dir,
    get_video_properties,
    run_h264_conversion,
)


ACTION_COLORS = {
    "forehand": (70, 220, 80),
    "backhand": (255, 170, 55),
    "serve": (50, 170, 255),
    "smash": (70, 70, 240),
    "volley": (200, 100, 255),
}


def fit_rect(src_w: int, src_h: int, dst_x: int, dst_y: int, dst_w: int, dst_h: int):
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    ox = dst_x + (dst_w - new_w) // 2
    oy = dst_y + (dst_h - new_h) // 2
    return scale, ox, oy, new_w, new_h


def transform_point(x: float, y: float, scale: float, ox: int, oy: int) -> tuple[int, int]:
    return int(round(x * scale + ox)), int(round(y * scale + oy))


def put_text(
    frame,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 2,
):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def keypoint(row: dict, name: str, min_conf: float = 0.25):
    conf = row.get(f"{name}_conf")
    if conf is None or pd.isna(conf) or float(conf) < min_conf:
        return None
    x = row.get(f"{name}_x")
    y = row.get(f"{name}_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y), float(conf)


def draw_pose(
    frame,
    row: dict,
    scale: float,
    ox: int,
    oy: int,
    color: tuple[int, int, int],
    min_conf: float,
    thickness: int = 3,
):
    for start_name, end_name in SKELETON_EDGES_NAME:
        start = keypoint(row, start_name, min_conf)
        end = keypoint(row, end_name, min_conf)
        if start is None or end is None:
            continue
        p1 = transform_point(start[0], start[1], scale, ox, oy)
        p2 = transform_point(end[0], end[1], scale, ox, oy)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    for name in KEYPOINT_NAMES:
        point = keypoint(row, name, min_conf)
        if point is None:
            continue
        p = transform_point(point[0], point[1], scale, ox, oy)
        cv2.circle(frame, p, max(3, thickness + 1), color, -1, cv2.LINE_AA)


def draw_bbox(
    frame,
    row: dict,
    scale: float,
    ox: int,
    oy: int,
    color: tuple[int, int, int],
    thickness: int = 2,
):
    try:
        p1 = transform_point(float(row["bbox_x1"]), float(row["bbox_y1"]), scale, ox, oy)
        p2 = transform_point(float(row["bbox_x2"]), float(row["bbox_y2"]), scale, ox, oy)
    except KeyError:
        return
    cv2.rectangle(frame, p1, p2, color, thickness)


def active_action(actions: pd.DataFrame, frame_id: int) -> dict | None:
    candidates = actions[
        (actions["window_start"] <= frame_id) & (actions["window_end"] >= frame_id)
    ]
    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["distance_to_contact"] = (candidates["contact_frame"] - frame_id).abs()
    row = candidates.sort_values("distance_to_contact").iloc[0]
    return row.to_dict()


def safe_float(value, default=np.nan):
    if value is None or pd.isna(value):
        return default
    return float(value)


def draw_wrist_trail(
    frame,
    pose_by_frame: dict[int, dict],
    frame_id: int,
    wrist_name: str,
    scale: float,
    ox: int,
    oy: int,
    color: tuple[int, int, int],
    trail_frames: int,
    min_conf: float,
):
    points = []
    for fid in range(max(1, frame_id - trail_frames), frame_id + 1):
        row = pose_by_frame.get(fid)
        if row is None:
            continue
        point = keypoint(row, wrist_name, min_conf)
        if point is None:
            continue
        points.append(transform_point(point[0], point[1], scale, ox, oy))

    if len(points) < 2:
        return
    for i in range(1, len(points)):
        alpha = i / max(1, len(points) - 1)
        blended = tuple(int(c * (0.35 + 0.65 * alpha)) for c in color)
        cv2.line(frame, points[i - 1], points[i], blended, 4, cv2.LINE_AA)
    cv2.circle(frame, points[-1], 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, points[-1], 6, color, -1, cv2.LINE_AA)


def draw_racket_overlay(
    frame,
    racket_row: dict,
    scale: float,
    ox: int,
    oy: int,
    color: tuple[int, int, int] = (40, 230, 255),
):
    try:
        x1 = float(racket_row["racket_bbox_x1"])
        y1 = float(racket_row["racket_bbox_y1"])
        x2 = float(racket_row["racket_bbox_x2"])
        y2 = float(racket_row["racket_bbox_y2"])
        center = (
            float(racket_row["racket_center_x"]),
            float(racket_row["racket_center_y"]),
        )
        head = (
            float(racket_row["racket_head_proxy_x"]),
            float(racket_row["racket_head_proxy_y"]),
        )
    except (KeyError, TypeError, ValueError):
        return

    p1 = transform_point(x1, y1, scale, ox, oy)
    p2 = transform_point(x2, y2, scale, ox, oy)
    center_p = transform_point(center[0], center[1], scale, ox, oy)
    head_p = transform_point(head[0], head[1], scale, ox, oy)
    cv2.rectangle(frame, p1, p2, color, 3)
    cv2.circle(frame, center_p, 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, head_p, 10, (0, 80, 255), -1, cv2.LINE_AA)
    cv2.line(frame, center_p, head_p, (0, 80, 255), 3, cv2.LINE_AA)


def draw_racket_trail(
    frame,
    racket_by_frame: dict[int, dict],
    frame_id: int,
    scale: float,
    ox: int,
    oy: int,
    trail_frames: int,
):
    points = []
    for fid in range(max(1, frame_id - trail_frames), frame_id + 1):
        row = racket_by_frame.get(fid)
        if row is None:
            continue
        try:
            points.append(
                transform_point(
                    float(row["racket_head_proxy_x"]),
                    float(row["racket_head_proxy_y"]),
                    scale,
                    ox,
                    oy,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 80, 255), 4, cv2.LINE_AA)


def draw_info_panel(
    frame,
    action: dict | None,
    frame_id: int,
    frame_feature: dict | None,
    racket_row: dict | None,
    panel_x: int,
    panel_y: int,
    panel_w: int,
):
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 270), (25, 25, 25), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 270), (70, 70, 70), 2)
    put_text(frame, f"Frame {frame_id}", (panel_x + 18, panel_y + 35), 0.72)

    if action is None:
        put_text(frame, "State: no annotated stroke window", (panel_x + 18, panel_y + 75), 0.66)
        put_text(frame, "Watching pose, bbox, and wrist motion trail", (panel_x + 18, panel_y + 112), 0.58)
        return

    label = str(action["action_label"])
    color = ACTION_COLORS.get(label, (220, 220, 220))
    contact = int(action["contact_frame"])
    start = int(action["window_start"])
    end = int(action["window_end"])
    active_side = str(action.get("active_wrist_side", "right"))
    wrist_name = f"{active_side}_wrist"
    progress = (frame_id - start) / max(1, end - start)
    progress = min(1.0, max(0.0, progress))

    put_text(
        frame,
        f"Action: {label.upper()}   active wrist: {active_side}",
        (panel_x + 18, panel_y + 75),
        0.68,
        color,
    )
    put_text(
        frame,
        f"Window {start}-{end}   contact/proxy frame {contact}",
        (panel_x + 18, panel_y + 112),
        0.58,
    )

    bar_x = panel_x + 18
    bar_y = panel_y + 134
    bar_w = panel_w - 36
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (90, 90, 90), -1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * progress), bar_y + 12),
        color,
        -1,
    )
    contact_x = bar_x + int(bar_w * ((contact - start) / max(1, end - start)))
    cv2.line(frame, (contact_x, bar_y - 5), (contact_x, bar_y + 18), (255, 255, 255), 2)

    path = safe_float(action.get("active_wrist_path_length_norm"))
    speed = safe_float(action.get("active_wrist_max_speed_norm_per_frame"))
    follow = safe_float(action.get("active_wrist_follow_through_length_norm"))
    torso = safe_float(action.get("torso_rotation_proxy_range_deg"))
    put_text(
        frame,
        f"Motion proxies: wrist path {path:.2f} body units | max speed {speed:.2f}/frame",
        (panel_x + 18, panel_y + 165),
        0.52,
    )
    put_text(
        frame,
        f"Follow-through {follow:.2f} body units | torso rotation range {torso:.1f} deg",
        (panel_x + 18, panel_y + 193),
        0.52,
    )

    if frame_feature is not None and wrist_name + "_speed_norm_per_frame" in frame_feature:
        inst_speed = safe_float(frame_feature.get(wrist_name + "_speed_norm_per_frame"))
        put_text(
            frame,
            f"Current wrist speed proxy: {inst_speed:.2f} body units/frame",
            (panel_x + 18, panel_y + 222),
            0.52,
            (210, 240, 255),
        )
    if racket_row is not None:
        center_speed = safe_float(racket_row.get("racket_center_speed_body_units_per_frame"))
        head_speed = safe_float(racket_row.get("racket_head_proxy_speed_body_units_per_frame"))
        conf = safe_float(racket_row.get("racket_conf"))
        put_text(
            frame,
            f"Racket: conf {conf:.2f} | center speed {center_speed:.2f} | head-proxy speed {head_speed:.2f}",
            (panel_x + 18, panel_y + 250),
            0.50,
            (80, 230, 255),
        )


def make_video(
    source_video: Path,
    pose_csv: Path,
    action_features_csv: Path,
    frame_features_csv: Path,
    racket_csv: Path | None,
    output_dir: Path,
    output_name: str,
    output_width: int,
    output_height: int,
    trail_frames: int,
    min_conf: float,
    start_frame: int | None,
    end_frame: int | None,
    make_h264: bool,
) -> dict:
    output_dir = ensure_dir(output_dir)
    props = get_video_properties(source_video)
    fps = props["fps"]
    src_w = props["width"]
    src_h = props["height"]

    pose_df = pd.read_csv(pose_csv)
    pose_by_frame = {
        int(row["frame_id"]): row.to_dict() for _, row in pose_df.iterrows()
    }
    actions = pd.read_csv(action_features_csv)
    frame_features = pd.read_csv(frame_features_csv)
    frame_features_by_id = {
        int(row["frame_id"]): row.to_dict() for _, row in frame_features.iterrows()
    }
    racket_by_frame = {}
    if racket_csv and racket_csv.exists():
        racket_df = pd.read_csv(racket_csv)
        racket_by_frame = {
            int(row["frame_id"]): row.to_dict() for _, row in racket_df.iterrows()
        }

    mp4_path = output_dir / output_name
    if mp4_path.suffix.lower() != ".mp4":
        mp4_path = mp4_path.with_suffix(".mp4")
    h264_path = mp4_path.with_name(mp4_path.stem + "_h264.mp4")

    panel_w = output_width // 2
    left_x, right_x = 0, panel_w
    scale, fit_x_left, fit_y, fit_w, fit_h = fit_rect(src_w, src_h, left_x, 0, panel_w, output_height)
    _, fit_x_right, _, _, _ = fit_rect(src_w, src_h, right_x, 0, panel_w, output_height)

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not write video: {mp4_path}")

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video}")

    written = 0
    try:
        frame_id = 0
        while True:
            ok, rgb = cap.read()
            if not ok:
                break
            frame_id += 1
            if start_frame and frame_id < start_frame:
                continue
            if end_frame and frame_id > end_frame:
                break

            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            canvas[:, :panel_w] = (14, 18, 22)
            canvas[:, panel_w:] = (8, 10, 14)

            resized = cv2.resize(rgb, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
            canvas[fit_y : fit_y + fit_h, fit_x_left : fit_x_left + fit_w] = resized

            row = pose_by_frame.get(frame_id)
            racket_row = racket_by_frame.get(frame_id)
            action = active_action(actions, frame_id)
            label = str(action["action_label"]) if action else "none"
            color = ACTION_COLORS.get(label, (90, 210, 240))
            active_side = str(action.get("active_wrist_side", "right")) if action else "right"
            active_wrist = f"{active_side}_wrist"

            if row is not None:
                draw_bbox(canvas, row, scale, fit_x_left, fit_y, color, 3)
                draw_pose(canvas, row, scale, fit_x_left, fit_y, color, min_conf, 3)
            if racket_row is not None:
                draw_racket_overlay(canvas, racket_row, scale, fit_x_left, fit_y)

            cv2.rectangle(
                canvas,
                (right_x, 0),
                (output_width - 1, output_height - 1),
                (45, 45, 45),
                2,
            )
            put_text(canvas, "2D Pose Motion View", (right_x + 24, 42), 0.82, (230, 230, 230))
            put_text(
                canvas,
                "Pelvis/body-normalized motion proxies",
                (right_x + 24, 76),
                0.55,
                (180, 200, 220),
            )
            if row is not None:
                draw_bbox(canvas, row, scale, fit_x_right, fit_y, (80, 80, 80), 2)
                draw_pose(canvas, row, scale, fit_x_right, fit_y, (230, 230, 230), min_conf, 3)
                draw_wrist_trail(
                    canvas,
                    pose_by_frame,
                    frame_id,
                    active_wrist,
                    scale,
                    fit_x_right,
                    fit_y,
                    color,
                    trail_frames,
                    min_conf,
                )
                draw_racket_trail(
                    canvas,
                    racket_by_frame,
                    frame_id,
                    scale,
                    fit_x_right,
                    fit_y,
                    trail_frames,
                )
                if racket_row is not None:
                    draw_racket_overlay(canvas, racket_row, scale, fit_x_right, fit_y)

                if action is not None and frame_id == int(action["contact_frame"]):
                    point = keypoint(row, active_wrist, min_conf)
                    if point is not None:
                        contact_p = transform_point(point[0], point[1], scale, fit_x_right, fit_y)
                        cv2.circle(canvas, contact_p, 22, (255, 255, 255), 4, cv2.LINE_AA)
                        put_text(canvas, "CONTACT / PROXY", (contact_p[0] + 20, contact_p[1]), 0.55, color)

            frame_feature = frame_features_by_id.get(frame_id)
            draw_info_panel(
                canvas,
                action,
                frame_id,
                frame_feature,
                racket_row,
                panel_x=right_x + 24,
                panel_y=output_height - 292,
                panel_w=panel_w - 48,
            )

            if action is not None:
                put_text(
                    canvas,
                    f"{label.upper()} WINDOW",
                    (32, 54),
                    0.9,
                    color,
                    3,
                )
                put_text(
                    canvas,
                    f"contact/proxy: {int(action['contact_frame'])}",
                    (32, 92),
                    0.62,
                    (240, 240, 240),
                    2,
                )
            else:
                put_text(canvas, "RGB Video + Tracked Pose", (32, 54), 0.8)

            writer.write(canvas)
            written += 1
    finally:
        cap.release()
        writer.release()

    h264_created = False
    if make_h264:
        h264_created = run_h264_conversion(mp4_path, h264_path)

    summary = {
        "source_video": str(source_video),
        "pose_csv": str(pose_csv),
        "action_features_csv": str(action_features_csv),
        "frame_features_csv": str(frame_features_csv),
        "racket_csv": str(racket_csv) if racket_csv else None,
        "output_video": str(mp4_path),
        "h264_video": str(h264_path) if h264_created else None,
        "frames_written": written,
        "fps": fps,
        "output_width": output_width,
        "output_height": output_height,
    }
    print(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side RGB + skeleton motion explanation video."
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--action-features-csv", required=True)
    parser.add_argument("--frame-features-csv", required=True)
    parser.add_argument("--racket-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="explained_motion_video.mp4")
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--trail-frames", type=int, default=30)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--h264", action="store_true")
    args = parser.parse_args(argv)

    make_video(
        source_video=Path(args.source_video),
        pose_csv=Path(args.pose_csv),
        action_features_csv=Path(args.action_features_csv),
        frame_features_csv=Path(args.frame_features_csv),
        racket_csv=Path(args.racket_csv) if args.racket_csv else None,
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
        output_width=args.output_width,
        output_height=args.output_height,
        trail_frames=args.trail_frames,
        min_conf=args.min_conf,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        make_h264=args.h264,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
