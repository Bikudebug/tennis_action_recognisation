from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_pipeline.pose_utils import (
    ANALYSIS_KEYPOINTS,
    ensure_dir,
    get_video_properties,
    normalize_video_label,
)


def valid_point(row, name: str, min_conf: float) -> np.ndarray | None:
    conf = row.get(f"{name}_conf", np.nan)
    if pd.isna(conf) or float(conf) < min_conf:
        return None
    x = row.get(f"{name}_x", np.nan)
    y = row.get(f"{name}_y", np.nan)
    if pd.isna(x) or pd.isna(y):
        return None
    return np.array([float(x), float(y)], dtype=float)


def midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def line_angle_deg(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    delta = b - a
    return float(math.degrees(math.atan2(delta[1], delta[0])))


def wrap_angle_deg(angle: float) -> float:
    if pd.isna(angle):
        return float("nan")
    return float((angle + 180.0) % 360.0 - 180.0)


def angle_diff_deg(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return wrap_angle_deg(a - b)


def joint_angle_deg(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    if a is None or b is None or c is None:
        return float("nan")
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm <= 1e-6:
        return float("nan")
    cos_value = float(np.dot(ba, bc) / norm)
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(math.degrees(math.acos(cos_value)))


def robust_body_reference(row, min_conf: float) -> tuple[np.ndarray, float, dict]:
    left_hip = valid_point(row, "left_hip", min_conf)
    right_hip = valid_point(row, "right_hip", min_conf)
    left_shoulder = valid_point(row, "left_shoulder", min_conf)
    right_shoulder = valid_point(row, "right_shoulder", min_conf)

    pelvis = midpoint(left_hip, right_hip)
    if pelvis is None:
        x1, y1, x2, y2 = [
            float(row.get("bbox_x1", 0.0)),
            float(row.get("bbox_y1", 0.0)),
            float(row.get("bbox_x2", 0.0)),
            float(row.get("bbox_y2", 0.0)),
        ]
        pelvis = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    bbox_h = max(1.0, float(row.get("bbox_y2", 0.0)) - float(row.get("bbox_y1", 0.0)))
    candidate_scales = [
        distance(left_shoulder, right_shoulder),
        distance(left_hip, right_hip),
        distance(shoulder_mid, hip_mid),
        bbox_h * 0.25,
    ]
    valid_scales = [scale for scale in candidate_scales if not pd.isna(scale) and scale > 5.0]
    scale = float(np.median(valid_scales)) if valid_scales else bbox_h * 0.25
    scale = max(scale, 1.0)
    meta = {
        "pelvis_source": "hips" if hip_mid is not None else "bbox_center",
        "body_scale_px": scale,
        "shoulder_width_px": candidate_scales[0],
        "hip_width_px": candidate_scales[1],
        "torso_length_px": candidate_scales[2],
    }
    return pelvis, scale, meta


def compute_frame_features(pose_df: pd.DataFrame, fps: float, min_conf: float) -> pd.DataFrame:
    rows = []
    for _, row in pose_df.iterrows():
        pelvis, scale, meta = robust_body_reference(row, min_conf)
        left_shoulder = valid_point(row, "left_shoulder", min_conf)
        right_shoulder = valid_point(row, "right_shoulder", min_conf)
        left_hip = valid_point(row, "left_hip", min_conf)
        right_hip = valid_point(row, "right_hip", min_conf)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)

        feature = {
            "frame_id": int(row["frame_id"]),
            "time_seconds": (int(row["frame_id"]) - 1) / fps if fps > 0 else np.nan,
            "person_conf": row.get("person_conf", np.nan),
            "bbox_center_x": (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2.0,
            "bbox_center_y": (float(row["bbox_y1"]) + float(row["bbox_y2"])) / 2.0,
            "bbox_width_px": float(row["bbox_x2"]) - float(row["bbox_x1"]),
            "bbox_height_px": float(row["bbox_y2"]) - float(row["bbox_y1"]),
            "pelvis_x": pelvis[0],
            "pelvis_y": pelvis[1],
            "body_scale_px": scale,
            "shoulder_line_angle_deg": line_angle_deg(left_shoulder, right_shoulder),
            "hip_line_angle_deg": line_angle_deg(left_hip, right_hip),
            "torso_lean_angle_deg": line_angle_deg(hip_mid, shoulder_mid),
            "left_elbow_angle_deg": joint_angle_deg(
                left_shoulder,
                valid_point(row, "left_elbow", min_conf),
                valid_point(row, "left_wrist", min_conf),
            ),
            "right_elbow_angle_deg": joint_angle_deg(
                right_shoulder,
                valid_point(row, "right_elbow", min_conf),
                valid_point(row, "right_wrist", min_conf),
            ),
            **meta,
        }
        feature["torso_rotation_proxy_deg"] = angle_diff_deg(
            feature["shoulder_line_angle_deg"], feature["hip_line_angle_deg"]
        )

        for name in ANALYSIS_KEYPOINTS:
            point = valid_point(row, name, min_conf)
            if point is None:
                feature[f"{name}_x"] = np.nan
                feature[f"{name}_y"] = np.nan
                feature[f"{name}_norm_x"] = np.nan
                feature[f"{name}_norm_y"] = np.nan
                feature[f"{name}_conf"] = row.get(f"{name}_conf", np.nan)
            else:
                feature[f"{name}_x"] = point[0]
                feature[f"{name}_y"] = point[1]
                feature[f"{name}_norm_x"] = (point[0] - pelvis[0]) / scale
                feature[f"{name}_norm_y"] = (point[1] - pelvis[1]) / scale
                feature[f"{name}_conf"] = row.get(f"{name}_conf", np.nan)
        rows.append(feature)

    features = pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)
    features["is_consecutive"] = features["frame_id"].diff().fillna(1).eq(1)
    for side in ["left", "right"]:
        x = features[f"{side}_wrist_x"]
        y = features[f"{side}_wrist_y"]
        nx = features[f"{side}_wrist_norm_x"]
        ny = features[f"{side}_wrist_norm_y"]
        consecutive = features["is_consecutive"]
        dx = x.diff().where(consecutive)
        dy = y.diff().where(consecutive)
        dnx = nx.diff().where(consecutive)
        dny = ny.diff().where(consecutive)
        features[f"{side}_wrist_speed_px_per_frame"] = np.sqrt(dx**2 + dy**2)
        features[f"{side}_wrist_speed_px_per_second"] = (
            features[f"{side}_wrist_speed_px_per_frame"] * fps
        )
        features[f"{side}_wrist_speed_norm_per_frame"] = np.sqrt(dnx**2 + dny**2)
        features[f"{side}_wrist_accel_norm_per_frame2"] = features[
            f"{side}_wrist_speed_norm_per_frame"
        ].diff().where(consecutive)
    return features


def trajectory_stats(window: pd.DataFrame, side: str, start_frame: int | None = None) -> dict:
    if start_frame is not None:
        window = window[window["frame_id"] >= start_frame]
    x_col = f"{side}_wrist_norm_x"
    y_col = f"{side}_wrist_norm_y"
    points = window[["frame_id", x_col, y_col]].dropna().sort_values("frame_id")
    if len(points) < 2:
        return {
            "path_length_norm": np.nan,
            "straight_line_displacement_norm": np.nan,
            "curvature_deg_total": np.nan,
            "curvature_deg_per_segment": np.nan,
        }

    coords = points[[x_col, y_col]].to_numpy(dtype=float)
    frame_ids = points["frame_id"].to_numpy(dtype=int)
    path_length = 0.0
    vectors = []
    for i in range(1, len(coords)):
        if frame_ids[i] - frame_ids[i - 1] > 2:
            continue
        vec = coords[i] - coords[i - 1]
        length = float(np.linalg.norm(vec))
        if length > 1e-6:
            path_length += length
            vectors.append(vec)
    displacement = float(np.linalg.norm(coords[-1] - coords[0]))

    angle_changes = []
    for i in range(1, len(vectors)):
        angle_a = math.degrees(math.atan2(vectors[i - 1][1], vectors[i - 1][0]))
        angle_b = math.degrees(math.atan2(vectors[i][1], vectors[i][0]))
        angle_changes.append(abs(wrap_angle_deg(angle_b - angle_a)))
    curvature_total = float(np.sum(angle_changes)) if angle_changes else 0.0
    curvature_per_segment = curvature_total / len(angle_changes) if angle_changes else 0.0
    return {
        "path_length_norm": path_length,
        "straight_line_displacement_norm": displacement,
        "curvature_deg_total": curvature_total,
        "curvature_deg_per_segment": curvature_per_segment,
    }


def range_value(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(series.max() - series.min())


def nearest_row(window: pd.DataFrame, contact_frame: int) -> pd.Series | None:
    if window.empty:
        return None
    idx = (window["frame_id"] - contact_frame).abs().idxmin()
    return window.loc[idx]


def compute_action_features(
    frame_features: pd.DataFrame,
    annotations: pd.DataFrame,
    video_name: str,
    window_radius: int,
    handedness: str,
) -> pd.DataFrame:
    if annotations.empty:
        return pd.DataFrame()

    video_name_norm = normalize_video_label(video_name)
    filtered = annotations[
        (annotations["video_stem"].astype(str) == video_name_norm)
        | (annotations["video_file"].astype(str).apply(lambda p: Path(p).stem if p else "") == video_name)
    ].copy()
    if filtered.empty:
        filtered = annotations[annotations["video_label"].astype(str).apply(normalize_video_label) == video_name_norm].copy()
    if filtered.empty:
        return pd.DataFrame()

    rows = []
    for _, event in filtered.sort_values("contact_frame").iterrows():
        contact = int(event["contact_frame"])
        start = max(1, contact - window_radius)
        end = contact + window_radius
        window = frame_features[
            (frame_features["frame_id"] >= start) & (frame_features["frame_id"] <= end)
        ].copy()
        expected_frames = end - start + 1

        if handedness in {"right", "left"}:
            active_side = handedness
        else:
            left_path = trajectory_stats(window, "left")["path_length_norm"]
            right_path = trajectory_stats(window, "right")["path_length_norm"]
            active_side = "right" if pd.isna(left_path) or right_path >= left_path else "left"

        all_stats = trajectory_stats(window, active_side)
        follow_stats = trajectory_stats(window, active_side, start_frame=contact)
        contact_row = nearest_row(window, contact)

        row = {
            "event_id": int(event["event_id"]) if "event_id" in event else len(rows),
            "video_stem": video_name_norm,
            "action_label": event["action_label"],
            "action_label_raw": event.get("action_label_raw", event["action_label"]),
            "contact_frame": contact,
            "window_start": start,
            "window_end": end,
            "expected_window_frames": expected_frames,
            "pose_frames_available": int(len(window)),
            "missing_pose_frames": int(expected_frames - len(window)),
            "needs_review": bool(event.get("needs_review", False)),
            "review_note": event.get("review_note", ""),
            "handedness_assumption": handedness,
            "active_wrist_side": active_side,
            "active_wrist_path_length_norm": all_stats["path_length_norm"],
            "active_wrist_straight_displacement_norm": all_stats[
                "straight_line_displacement_norm"
            ],
            "active_wrist_curvature_deg_total": all_stats["curvature_deg_total"],
            "active_wrist_curvature_deg_per_segment": all_stats[
                "curvature_deg_per_segment"
            ],
            "active_wrist_follow_through_length_norm": follow_stats["path_length_norm"],
            "active_wrist_max_speed_norm_per_frame": float(
                window[f"{active_side}_wrist_speed_norm_per_frame"].max()
            )
            if not window.empty
            else np.nan,
            "active_wrist_mean_speed_norm_per_frame": float(
                window[f"{active_side}_wrist_speed_norm_per_frame"].mean()
            )
            if not window.empty
            else np.nan,
            "active_wrist_max_speed_px_per_second": float(
                window[f"{active_side}_wrist_speed_px_per_second"].max()
            )
            if not window.empty
            else np.nan,
            "active_wrist_max_accel_norm_per_frame2": float(
                window[f"{active_side}_wrist_accel_norm_per_frame2"].abs().max()
            )
            if not window.empty
            else np.nan,
            "active_wrist_x_range_norm": range_value(
                window[f"{active_side}_wrist_norm_x"]
            ),
            "active_wrist_y_range_norm": range_value(
                window[f"{active_side}_wrist_norm_y"]
            ),
            "torso_rotation_proxy_range_deg": range_value(
                window["torso_rotation_proxy_deg"]
            ),
            "shoulder_line_angle_range_deg": range_value(
                window["shoulder_line_angle_deg"]
            ),
            "hip_line_angle_range_deg": range_value(window["hip_line_angle_deg"]),
            "torso_lean_angle_range_deg": range_value(window["torso_lean_angle_deg"]),
            "bbox_center_x_range_px": range_value(window["bbox_center_x"]),
            "bbox_center_y_range_px": range_value(window["bbox_center_y"]),
        }
        if contact_row is not None:
            row["nearest_pose_frame_to_contact"] = int(contact_row["frame_id"])
            row["contact_frame_pose_offset"] = int(contact_row["frame_id"] - contact)
            row["contact_active_wrist_norm_x"] = float(
                contact_row[f"{active_side}_wrist_norm_x"]
            )
            row["contact_active_wrist_norm_y"] = float(
                contact_row[f"{active_side}_wrist_norm_y"]
            )
            row["contact_torso_rotation_proxy_deg"] = float(
                contact_row["torso_rotation_proxy_deg"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(
    frame_features: pd.DataFrame,
    action_features: pd.DataFrame,
    output_dir: Path,
    max_plots: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = ensure_dir(output_dir / "plots")
    if not action_features.empty:
        colors = {
            "forehand": "#2ca02c",
            "backhand": "#1f77b4",
            "serve": "#ff7f0e",
            "smash": "#d62728",
            "volley": "#9467bd",
        }
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(frame_features["frame_id"], np.zeros(len(frame_features)), alpha=0.15)
        for _, event in action_features.iterrows():
            label = str(event["action_label"])
            color = colors.get(label, "#444444")
            ax.axvspan(event["window_start"], event["window_end"], color=color, alpha=0.15)
            ax.scatter(event["contact_frame"], 0, color=color, s=45, label=label)
            if event["needs_review"]:
                ax.scatter(event["contact_frame"], 0.08, color="red", marker="x", s=60)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")
        ax.set_yticks([])
        ax.set_xlabel("Frame")
        ax.set_title("Annotated action timeline")
        fig.tight_layout()
        fig.savefig(plots_dir / "action_timeline.png", dpi=160)
        plt.close(fig)

        numeric = action_features.dropna(subset=["active_wrist_path_length_norm"])
        if not numeric.empty:
            labels = sorted(numeric["action_label"].astype(str).unique())
            values = [
                numeric[numeric["action_label"].astype(str) == label][
                    "active_wrist_path_length_norm"
                ].to_numpy()
                for label in labels
            ]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(values, tick_labels=labels)
            ax.set_ylabel("Body-normalized wrist path length")
            ax.set_title("Wrist swing path by action label")
            fig.tight_layout()
            fig.savefig(plots_dir / "wrist_path_by_label.png", dpi=160)
            plt.close(fig)

    for _, event in action_features.head(max_plots).iterrows():
        window = frame_features[
            (frame_features["frame_id"] >= event["window_start"])
            & (frame_features["frame_id"] <= event["window_end"])
        ]
        if window.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for side, color in [("right", "#1f77b4"), ("left", "#ff7f0e")]:
            x_col = f"{side}_wrist_norm_x"
            y_col = f"{side}_wrist_norm_y"
            valid = window[[x_col, y_col, "frame_id"]].dropna()
            if valid.empty:
                continue
            ax.plot(valid[x_col], valid[y_col], "-o", ms=3, color=color, label=f"{side} wrist")
        contact = nearest_row(window, int(event["contact_frame"]))
        if contact is not None:
            side = event["active_wrist_side"]
            ax.scatter(
                contact[f"{side}_wrist_norm_x"],
                contact[f"{side}_wrist_norm_y"],
                color="red",
                s=70,
                marker="x",
                label="contact/proxy",
            )
        ax.invert_yaxis()
        ax.set_xlabel("Normalized x from pelvis")
        ax.set_ylabel("Normalized y from pelvis")
        ax.set_title(f"{event['action_label']} event {event['event_id']} wrist path")
        ax.legend()
        fig.tight_layout()
        name = (
            f"event{int(event['event_id']):03d}_"
            f"{event['action_label']}_f{int(event['contact_frame'])}_wrist_path.png"
        )
        fig.savefig(plots_dir / name, dpi=160)
        plt.close(fig)


def analyze_motion_features(
    pose_csv: Path,
    output_dir: Path,
    annotations_csv: Path | None,
    video_name: str,
    source_video: Path | None,
    window_radius: int,
    handedness: str,
    min_keypoint_conf: float,
    max_plots: int,
) -> dict:
    output_dir = ensure_dir(output_dir)
    pose_df = pd.read_csv(pose_csv)
    if "frame_id" not in pose_df.columns:
        raise ValueError(f"Pose CSV missing frame_id column: {pose_csv}")
    pose_df = pose_df.sort_values("frame_id").reset_index(drop=True)

    if source_video:
        fps = get_video_properties(source_video)["fps"]
    else:
        fps = 30.0

    frame_features = compute_frame_features(pose_df, fps=fps, min_conf=min_keypoint_conf)
    frame_features_csv = output_dir / "frame_motion_features.csv"
    frame_features.to_csv(frame_features_csv, index=False)

    annotations = pd.DataFrame()
    action_features = pd.DataFrame()
    if annotations_csv and annotations_csv.exists():
        annotations = pd.read_csv(annotations_csv)
        action_features = compute_action_features(
            frame_features,
            annotations,
            video_name=video_name,
            window_radius=window_radius,
            handedness=handedness,
        )
    action_features_csv = output_dir / "action_window_features.csv"
    action_features.to_csv(action_features_csv, index=False)
    action_features_json = output_dir / "action_window_features.json"
    with action_features_json.open("w", encoding="utf-8") as f:
        json.dump(action_features.to_dict(orient="records"), f, indent=2)

    plot_outputs(frame_features, action_features, output_dir, max_plots=max_plots)

    summary = {
        "pose_csv": str(pose_csv),
        "annotations_csv": str(annotations_csv) if annotations_csv else None,
        "video_name": video_name,
        "source_video": str(source_video) if source_video else None,
        "fps": fps,
        "num_pose_frames": int(len(pose_df)),
        "frame_features_csv": str(frame_features_csv),
        "num_action_windows": int(len(action_features)),
        "action_features_csv": str(action_features_csv),
        "action_features_json": str(action_features_json),
        "handedness_assumption": handedness,
        "valid_measurement_note": (
            "Speeds and distances are image-plane/body-normalized proxies, not real-world m/s or meters."
        ),
    }
    summary_json = output_dir / "motion_analysis_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute body-normalized tennis pose and wrist-motion features."
    )
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotations-csv")
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--source-video")
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument(
        "--handedness",
        choices=["right", "left", "unknown"],
        default="right",
        help="Racket hand assumption. Use unknown to pick the wrist with larger motion per window.",
    )
    parser.add_argument("--min-keypoint-conf", type=float, default=0.25)
    parser.add_argument("--max-plots", type=int, default=40)
    args = parser.parse_args(argv)

    analyze_motion_features(
        pose_csv=Path(args.pose_csv),
        output_dir=Path(args.output_dir),
        annotations_csv=Path(args.annotations_csv) if args.annotations_csv else None,
        video_name=args.video_name,
        source_video=Path(args.source_video) if args.source_video else None,
        window_radius=args.window_radius,
        handedness=args.handedness,
        min_keypoint_conf=args.min_keypoint_conf,
        max_plots=args.max_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
