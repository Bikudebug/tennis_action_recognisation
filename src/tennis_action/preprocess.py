from __future__ import annotations

import numpy as np

from .common import KEYPOINT_INDEX


def normalize_sequences(X_raw: np.ndarray) -> np.ndarray:
    X = X_raw.copy().astype(np.float32)
    left_hip = KEYPOINT_INDEX["left_hip"]
    right_hip = KEYPOINT_INDEX["right_hip"]
    left_shoulder = KEYPOINT_INDEX["left_shoulder"]
    right_shoulder = KEYPOINT_INDEX["right_shoulder"]
    n_samples, seq_len, n_joints, _ = X.shape

    for i in range(n_samples):
        for t in range(seq_len):
            lh = X[i, t, left_hip].copy()
            rh = X[i, t, right_hip].copy()
            if np.all(np.isnan(lh)) and np.all(np.isnan(rh)):
                continue
            if np.all(np.isnan(lh)):
                root = rh
            elif np.all(np.isnan(rh)):
                root = lh
            else:
                root = (lh + rh) / 2.0
            X[i, t] -= root

        shoulder_dists = []
        for t in range(seq_len):
            ls = X[i, t, left_shoulder]
            rs = X[i, t, right_shoulder]
            if not np.any(np.isnan(ls)) and not np.any(np.isnan(rs)):
                shoulder_dists.append(float(np.linalg.norm(ls - rs)))
        scale = float(np.mean(shoulder_dists)) if shoulder_dists else 1.0
        if scale < 1e-6:
            scale = 1.0
        X[i] /= scale

        for joint_idx in range(n_joints):
            for axis_idx in range(2):
                series = X[i, :, joint_idx, axis_idx]
                valid = ~np.isnan(series)
                if valid.sum() == 0:
                    series[:] = 0.0
                elif valid.sum() < seq_len:
                    idx = np.arange(seq_len)
                    series[~valid] = np.interp(idx[~valid], idx[valid], series[valid])
                X[i, :, joint_idx, axis_idx] = series
    return X


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def temporal_majority_filter(labels: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or len(labels) == 0:
        return labels.copy()
    radius = kernel_size // 2
    out = labels.copy()
    for idx in range(len(labels)):
        start = max(0, idx - radius)
        end = min(len(labels), idx + radius + 1)
        window = labels[start:end]
        values, counts = np.unique(window, return_counts=True)
        out[idx] = values[np.argmax(counts)]
    return out

