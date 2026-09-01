"""
Stage 2: 3D Reconstruction + HTML + MP4
- 讀取雙目配對點與每幀相對位姿
- 對位姿做平滑與平移向量正規化
- 三角化出每幀 3D 點
- 同時輸出互動式 3D HTML、3D 動畫 MP4、RT 文字檔
- 同時輸出 pts3d、pts2d_obs_cam0、pts2d_obs_cam2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_tool
from scipy.spatial.transform import Slerp


def triangulate_point(P1, P2, x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
        return np.full(3, np.nan)
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-8:
        return np.full(3, np.nan)
    return X[:3] / X[3]


COCO17_BODY_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def load_k_matrix(path):
    return np.loadtxt(path)


def _normalize_baseline_np(t: np.ndarray, baseline: float = 1.0) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    norm = float(np.linalg.norm(t))
    if norm < 1e-8:
        return t.astype(np.float32)
    return (t * (float(baseline) / norm)).astype(np.float32)


def _smooth_rt_ema_np(R: np.ndarray, t: np.ndarray, state, alpha: float):
    # 0429_test.py style: Slerp rotation and EMA translation.
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    if state is None:
        R0 = R.astype(np.float32)
        t0 = t.astype(np.float32)
        return R0, t0, (R0, t0)

    R_prev, t_prev = state
    alpha = float(np.clip(alpha, 0.0, 1.0))
    try:
        rots = R_tool.from_matrix([np.asarray(R_prev, dtype=np.float64), R])
        R_ema = Slerp([0, 1], rots)([alpha])[0].as_matrix()
    except ValueError:
        R_ema = R

    t_ema = (1.0 - alpha) * np.asarray(t_prev, dtype=np.float64).reshape(3, 1) + alpha * t
    R_ema = R_ema.astype(np.float32)
    t_ema = t_ema.astype(np.float32)
    return R_ema, t_ema, (R_ema, t_ema)


def load_matches(path, threshold=0.5, min_pairs=8, device="cpu"):
    matches = []
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[0])
                pairs = json.loads(parts[1])
            except Exception:
                continue
            pts1, pts2, ids = [], [], []
            for jid, p in enumerate(pairs):
                if p[2] >= threshold and p[3] >= threshold:
                    pts1.append(p[0])
                    pts2.append(p[1])
                    ids.append(jid)
            if len(pts1) >= min_pairs:
                matches.append(
                    {
                        "ts": ts,
                        "pts1": torch.tensor(pts1, dtype=torch.float32, device=device),
                        "pts2": torch.tensor(pts2, dtype=torch.float32, device=device),
                        "ids": torch.tensor(ids, dtype=torch.int64, device=device),
                    }
                )
    return matches


def load_stable_trajectory(path, window_size=5, smooth_alpha=0.1, baseline=1.0):
    # 讀取 rt quaternion 並做平滑。
    raw_data = []
    if not Path(path).exists():
        raise SystemExit(f"RT quaternion file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 8:
                raw_data.append(vals)

    if not raw_data:
        raise SystemExit(f"No valid RT rows loaded from: {path}")
    raw_data = np.array(raw_data, dtype=np.float64)
    ts = raw_data[:, 0]
    q_vals = raw_data[:, 4:8]

    rt_dict = {}
    ema_state = None
    for i in range(len(ts)):
        q = q_vals[i] / (np.linalg.norm(q_vals[i]) + 1e-8)
        R = R_tool.from_quat(q).as_matrix()
        t_raw = raw_data[i, 1:4].reshape(3, 1)
        R_sm, t_sm, ema_state = _smooth_rt_ema_np(R, t_raw, ema_state, float(smooth_alpha))
        t_norm = _normalize_baseline_np(t_sm, baseline=float(baseline))
        rt_dict[ts[i]] = (R_sm.astype(np.float32), t_norm.reshape(3, 1).astype(np.float32))
    return rt_dict


def _reprojection_errors(X_np: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray):
    if X_np is None or X_np.shape[0] == 0:
        return None
    X_h = np.hstack([X_np.astype(np.float64), np.ones((X_np.shape[0], 1), dtype=np.float64)])
    proj1 = (P1 @ X_h.T).T
    proj2 = (P2 @ X_h.T).T
    z1 = proj1[:, 2:3]
    z2 = proj2[:, 2:3]
    valid = (np.abs(z1[:, 0]) > 1e-8) & (np.abs(z2[:, 0]) > 1e-8)
    p1_hat = np.full_like(pts1, np.nan, dtype=np.float64)
    p2_hat = np.full_like(pts2, np.nan, dtype=np.float64)
    p1_hat[valid] = proj1[valid, :2] / z1[valid]
    p2_hat[valid] = proj2[valid, :2] / z2[valid]
    err1 = np.linalg.norm(p1_hat - pts1, axis=1)
    err2 = np.linalg.norm(p2_hat - pts2, axis=1)
    err_max = np.maximum(err1, err2)
    return {
        "valid": valid,
        "err1": err1,
        "err2": err2,
        "err_max": err_max,
    }


def _summarize_reprojection_errors(X_np: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray):
    errs = _reprojection_errors(X_np, pts1, pts2, P1, P2)
    if errs is None:
        return {"count": 0, "max_mean": np.nan, "max_median": np.nan, "max_max": np.nan}
    valid_vals = errs["err_max"][np.isfinite(errs["err_max"])]
    if valid_vals.size == 0:
        return {"count": 0, "max_mean": np.nan, "max_median": np.nan, "max_max": np.nan}
    return {
        "count": int(valid_vals.size),
        "max_mean": float(np.mean(valid_vals)),
        "max_median": float(np.median(valid_vals)),
        "max_max": float(np.max(valid_vals)),
    }


def _adaptive_reproj_filter(
    X_np: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    ids_np: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    max_reproj: float,
    min_inliers: int,
    min_keep: int,
    relax_times: int,
    relax_mul: float,
):
    if max_reproj <= 0:
        return X_np, pts1, pts2, ids_np, float(max_reproj), None

    thr = float(max_reproj)
    best = None
    best_thr = thr
    best_stats = None

    for _ in range(max(1, int(relax_times))):
        errs = _reprojection_errors(X_np, pts1, pts2, P1, P2)
        if errs is None:
            break
        mask = np.isfinite(errs["err_max"]) & (errs["err_max"] <= thr)
        n = int(mask.sum())
        if n >= max(int(min_inliers), 1):
            Xf = X_np[mask]
            p1f = pts1[mask]
            p2f = pts2[mask]
            idf = ids_np[mask] if ids_np is not None and ids_np.shape[0] == X_np.shape[0] else ids_np
            st = _summarize_reprojection_errors(Xf, p1f, p2f, P1, P2)
            if n >= int(min_keep):
                return Xf, p1f, p2f, idf, thr, st
            if best is None or n > best[0].shape[0]:
                best = (Xf, p1f, p2f, idf)
                best_thr = thr
                best_stats = st
        thr *= float(relax_mul)

    if best is None:
        return None, None, None, None, thr, None
    return best[0], best[1], best[2], best[3], best_thr, best_stats


def _diagnose_pose_failure(pts1: np.ndarray, pts2: np.ndarray, X_np: np.ndarray, R_np: np.ndarray, t_np: np.ndarray):
    if pts1.shape[0] != pts2.shape[0]:
        return f"len_mismatch pts1={pts1.shape[0]} pts2={pts2.shape[0]}"
    if pts1.shape[0] < 2:
        return f"too_few_pairs={pts1.shape[0]}"
    if X_np is None or X_np.shape[0] == 0:
        return "empty_triangulation"
    finite = np.isfinite(X_np).all(axis=1)
    Xf = X_np[finite]
    if Xf.shape[0] == 0:
        return f"all_nan triangulated={X_np.shape[0]}"
    z1 = Xf[:, 2]
    z2 = (R_np @ Xf.T + t_np.reshape(3, 1))[2, :]
    valid = (z1 > 0) & (z2 > 0)
    return (
        f"triangulated_valid={int(valid.sum())}/{Xf.shape[0]} "
        f"behind_cam1={int((z1 <= 0).sum())} behind_cam2={int((z2 <= 0).sum())} "
        f"t_norm={float(np.linalg.norm(t_np)):.6f}"
    )


def _camera2_center_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (-R.T @ t).astype(np.float32)


def _triangulate_points_0429_style(
    K1_np: np.ndarray,
    K2_np: np.ndarray,
    R_np: np.ndarray,
    t_np: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    ids_np: np.ndarray,
):
    P1 = K1_np @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2_np @ np.hstack([R_np, t_np.reshape(3, 1)])

    X_out, p1_out, p2_out, ids_out = [], [], [], []
    for p1, p2, jid in zip(pts1, pts2, ids_np):
        Xj = triangulate_point(P1, P2, p1, p2)
        if np.any(np.isnan(Xj)):
            continue
        z1 = float(Xj[2])
        z2 = float((R_np @ Xj + t_np.reshape(3,))[2])
        if z1 <= 0 or z2 <= 0:
            continue
        X_out.append(Xj)
        p1_out.append(p1)
        p2_out.append(p2)
        ids_out.append(int(jid))

    if not X_out:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            P1.astype(np.float64),
            P2.astype(np.float64),
        )

    return (
        np.asarray(X_out, dtype=np.float32),
        np.asarray(p1_out, dtype=np.float32),
        np.asarray(p2_out, dtype=np.float32),
        np.asarray(ids_out, dtype=np.int64),
        P1.astype(np.float64),
        P2.astype(np.float64),
    )


def _build_skeleton17_from_points(X_np: np.ndarray, ids_np: np.ndarray) -> np.ndarray:
    sk = np.full((17, 3), np.nan, dtype=np.float32)
    if ids_np is None or ids_np.shape[0] != X_np.shape[0]:
        return sk
    for i, jid in enumerate(ids_np):
        if 0 <= int(jid) < 17:
            sk[int(jid)] = X_np[i]
    return sk


def _midpoint(sk: np.ndarray, a: int, b: int):
    if np.isfinite(sk[a]).all() and np.isfinite(sk[b]).all():
        return 0.5 * (sk[a] + sk[b])
    return None


def _dist_if_valid(sk: np.ndarray, a: int, b: int):
    if np.isfinite(sk[a]).all() and np.isfinite(sk[b]).all():
        return float(np.linalg.norm(sk[a] - sk[b]))
    return None


def _skeleton_body_height(sk: np.ndarray) -> float | None:
    mid_shoulder = _midpoint(sk, 5, 6)
    mid_hip = _midpoint(sk, 11, 12)
    parts = []
    if mid_shoulder is not None:
        head_candidates = [sk[j] for j in (0, 1, 2, 3, 4) if np.isfinite(sk[j]).all()]
        if head_candidates:
            head = np.mean(np.stack(head_candidates, axis=0), axis=0)
            parts.append(float(np.linalg.norm(head - mid_shoulder)))
    if mid_shoulder is not None and mid_hip is not None:
        parts.append(float(np.linalg.norm(mid_shoulder - mid_hip)))
    leg_lengths = []
    left_upper = _dist_if_valid(sk, 11, 13)
    left_lower = _dist_if_valid(sk, 13, 15)
    if left_upper is not None and left_lower is not None:
        leg_lengths.append(left_upper + left_lower)
    right_upper = _dist_if_valid(sk, 12, 14)
    right_lower = _dist_if_valid(sk, 14, 16)
    if right_upper is not None and right_lower is not None:
        leg_lengths.append(right_upper + right_lower)
    if leg_lengths:
        parts.append(float(np.median(leg_lengths)))
    if len(parts) < 2:
        return None
    h = float(np.sum(parts))
    return h if np.isfinite(h) and h > 1e-8 else None


def _skeleton_bbox_y_height(sk: np.ndarray) -> float | None:
    valid = sk[np.isfinite(sk).all(axis=1)]
    if valid.shape[0] < 4:
        return None
    h = float(np.max(valid[:, 1]) - np.min(valid[:, 1]))
    return h if np.isfinite(h) and h > 1e-8 else None


def _estimate_frame_scales(
    skeleton_frames: list[np.ndarray],
    real_height: float,
    metric: str,
    min_joints: int,
    mode: str,
    smooth_alpha: float,
    clip_min: float,
    clip_max: float,
):
    if real_height <= 0:
        return np.ones((len(skeleton_frames),), dtype=np.float32), 1.0

    raw = []
    for sk in skeleton_frames:
        valid_joints = int(np.isfinite(sk).all(axis=1).sum())
        if valid_joints < int(min_joints):
            raw.append(np.nan)
            continue
        h = _skeleton_bbox_y_height(sk) if metric == "bbox_y" else _skeleton_body_height(sk)
        if h is None:
            raw.append(np.nan)
        else:
            raw.append(float(real_height) / float(h))

    raw_np = np.asarray(raw, dtype=np.float64)
    finite_raw = raw_np[np.isfinite(raw_np)]
    global_scale = float(np.median(finite_raw)) if finite_raw.size else 1.0
    raw_np[~np.isfinite(raw_np)] = global_scale
    raw_np = np.clip(raw_np, float(clip_min), float(clip_max))

    if mode == "global":
        return np.full((len(skeleton_frames),), global_scale, dtype=np.float32), global_scale

    alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
    sm = np.empty_like(raw_np, dtype=np.float64)
    prev = float(raw_np[0]) if raw_np.size else global_scale
    for i, v in enumerate(raw_np):
        prev = (1.0 - alpha) * prev + alpha * float(v)
        sm[i] = prev
    return sm.astype(np.float32), global_scale


def _collect_skeleton_heights(skeleton_frames: list[np.ndarray], metric: str, min_joints: int) -> list[float]:
    heights = []
    for sk in skeleton_frames:
        if sk is None:
            continue
        valid_joints = int(np.isfinite(sk).all(axis=1).sum())
        if valid_joints < int(min_joints):
            continue
        h = _skeleton_bbox_y_height(sk) if metric == "bbox_y" else _skeleton_body_height(sk)
        if h is not None and np.isfinite(h) and h > 1e-8:
            heights.append(float(h))
    return heights


def _estimate_global_scale(
    skeleton_frames: list[np.ndarray],
    real_height: float,
    metric: str,
    min_joints: int,
    max_ref_frames: int = 0,
):
    if real_height <= 0:
        return 1.0, None, 0
    refs = skeleton_frames[: int(max_ref_frames)] if max_ref_frames and max_ref_frames > 0 else skeleton_frames
    heights = _collect_skeleton_heights(refs, metric=metric, min_joints=min_joints)
    if not heights:
        return 1.0, None, 0
    h3d = float(np.median(np.asarray(heights, dtype=np.float64)))
    return float(real_height) / h3d, h3d, len(heights)


def _estimate_height_scale_for_frames(
    skeleton_frames: list[np.ndarray],
    real_height: float,
    metric: str,
    min_joints: int,
    height_stat: str = "median",
):
    if real_height <= 0:
        return None, None, 0
    heights = _collect_skeleton_heights(skeleton_frames, metric=metric, min_joints=min_joints)
    if not heights:
        return None, None, 0
    heights_np = np.asarray(heights, dtype=np.float64)
    h3d = float(np.max(heights_np)) if height_stat == "max" else float(np.median(heights_np))
    return float(real_height) / h3d, h3d, len(heights)


def _compute_block_scales(
    skeleton_frames: list[np.ndarray],
    frame_block_starts: list[int],
    real_height: float,
    metric: str,
    min_joints: int,
    fallback_height_scale: float,
    mode: str,
    height_stat: str,
    smooth_alpha: float,
    max_change: float,
    clip_min: float,
    clip_max: float,
):
    unique_blocks = []
    for block_start in frame_block_starts:
        if not unique_blocks or unique_blocks[-1] != block_start:
            unique_blocks.append(block_start)

    block_scales = {}
    raw_scales = {}
    block_h3d = {}
    block_refs = {}
    prev_scale = None
    alpha = float(np.clip(smooth_alpha, 0.0, 1.0))

    for block_start in unique_blocks:
        block_skeletons = [
            sk for sk, bs in zip(skeleton_frames, frame_block_starts)
            if bs == block_start
        ]
        raw_scale, h3d, n_refs = _estimate_height_scale_for_frames(
            block_skeletons,
            real_height=real_height,
            metric=metric,
            min_joints=min_joints,
            height_stat=height_stat,
        )
        if raw_scale is None:
            raw_scale = prev_scale if prev_scale is not None else fallback_height_scale

        raw_scale = float(np.clip(raw_scale, clip_min, clip_max))
        raw_scales[block_start] = raw_scale
        block_h3d[block_start] = np.nan if h3d is None else float(h3d)
        block_refs[block_start] = int(n_refs)

        if mode == "block":
            scale = raw_scale
        elif prev_scale is None:
            scale = raw_scale
        else:
            if max_change > 0:
                lo = prev_scale * max(0.0, 1.0 - float(max_change))
                hi = prev_scale * (1.0 + float(max_change))
                raw_scale = float(np.clip(raw_scale, lo, hi))
            scale = (1.0 - alpha) * prev_scale + alpha * raw_scale

        block_scales[block_start] = float(scale)
        prev_scale = float(scale)

    return block_scales, raw_scales, block_h3d, block_refs


def _cv_to_world(arr: np.ndarray) -> np.ndarray:
    # 將 OpenCV 座標系轉成視覺化世界座標系。
    a = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
    if a.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    Xw = a[:, 0]
    Yw = a[:, 2]
    Zw = -a[:, 1]
    return np.stack([Xw, Yw, Zw], axis=1)


def _finite_xyz(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return pts
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def _sample_pts(pts: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0 or pts.shape[0] <= max_n:
        return pts
    idx = rng.choice(pts.shape[0], max_n, replace=False)
    return pts[idx]


def _make_cube_ranges(min_xyz: np.ndarray, max_xyz: np.ndarray, pad: float):
    center = (min_xyz + max_xyz) / 2.0
    span = float(np.max(max_xyz - min_xyz))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    span *= 1.0 + 2.0 * pad
    half = span / 2.0
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
        (float(center[2] - half), float(center[2] + half)),
    )


def render_3d_video(
    frames,
    cam2_centers,
    out_path,
    fps: int = 10,
    size=(960, 540),
    zoom_level: float = 2.5,
    skeleton_frames=None,
    skeleton_edges=None,
) -> None:
    # 將 3D 點、相機位置與骨架渲染成 mp4。
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:
        raise SystemExit("Required libraries (cv2, matplotlib) are missing.")

    if not frames:
        raise SystemExit("No frames to render.")

    frames_w = [_cv_to_world(f[np.isfinite(f).all(axis=1)]) if len(f) else np.zeros((0, 3)) for f in frames]
    cam2_w = [_cv_to_world(c) if len(c) else np.zeros((0, 3)) for c in cam2_centers]
    sk_w = None
    if skeleton_frames is not None:
        sk_w = [_cv_to_world(s) if s is not None else None for s in skeleton_frames]

    base_limit = 3.0 / zoom_level
    v_min, v_max = -base_limit, base_limit

    width, height = size
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(v_min, v_max)
    ax.set_zlim(v_min, v_max)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=-70)

    pts_scatter = ax.scatter([], [], [], c="g", s=10, label="Points")
    ax.scatter([0], [0], [0], c="r", marker="^", s=60, label="Cam1")
    cam2_scatter = ax.scatter([], [], [], c="b", marker="^", s=60, label="Cam2")

    line_objs = []
    if sk_w and skeleton_edges:
        line_objs = [ax.plot([], [], [], "-", c="orange", lw=2, alpha=0.8)[0] for _ in skeleton_edges]

    ax.legend(loc="upper right", fontsize="small")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    total = len(frames_w)
    for idx in range(total):
        pts = frames_w[idx]
        if pts.size:
            pts_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        if idx < len(cam2_w) and cam2_w[idx].size:
            c2 = cam2_w[idx]
            cam2_scatter._offsets3d = (c2[:1, 0], c2[:1, 1], c2[:1, 2])

        if sk_w and idx < len(sk_w) and sk_w[idx] is not None:
            this_sk = sk_w[idx]
            for line, (i, j) in zip(line_objs, skeleton_edges):
                if i < len(this_sk) and j < len(this_sk):
                    p1, p2 = this_sk[i], this_sk[j]
                    if np.isfinite(p1).all() and np.isfinite(p2).all():
                        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                        line.set_3d_properties([p1[2], p2[2]])
                    else:
                        line.set_data([], [])
                        line.set_3d_properties([])

        ax.set_title(f"Frame {idx + 1}/{total} (Zoom: {zoom_level}x)")
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        writer.write(img)

    writer.release()
    plt.close(fig)


def write_interactive_html(
    frames,
    cam2_centers,
    out_path,
    sample: int = 3000,
    step: int = 1,
    start: int = 0,
    end: int = -1,
    pad: float = 0.05,
    seed: int = 0,
):
    # 將重建結果輸出成 Plotly 互動式 HTML。
    import plotly.graph_objects as go

    n_total = len(frames)
    if n_total == 0:
        raise SystemExit("No reconstructed frames to export.")

    start = max(0, int(start))
    end = n_total - 1 if int(end) < 0 else min(n_total - 1, int(end))
    step = max(1, int(step))
    sel = list(range(start, end + 1, step))
    if not sel:
        raise SystemExit("No frames selected. Check --start/--end/--step.")

    rng = np.random.default_rng(seed)
    pts_list = []
    c2_list = []

    min_xyz = np.array([np.inf, np.inf, np.inf], np.float32)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], np.float32)

    for i in sel:
        pts = _finite_xyz(frames[i])
        pts = _sample_pts(pts, sample, rng)
        pts_list.append(pts)

        if i < len(cam2_centers) and cam2_centers[i] is not None:
            c2 = _finite_xyz(cam2_centers[i])[:1]
        else:
            c2 = np.zeros((0, 3), np.float32)
        c2_list.append(c2)

        if pts.shape[0]:
            min_xyz = np.minimum(min_xyz, pts.min(axis=0))
            max_xyz = np.maximum(max_xyz, pts.max(axis=0))
        if c2.shape[0]:
            min_xyz = np.minimum(min_xyz, c2.min(axis=0))
            max_xyz = np.maximum(max_xyz, c2.max(axis=0))

    min_xyz = np.minimum(min_xyz, np.zeros(3, np.float32))
    max_xyz = np.maximum(max_xyz, np.zeros(3, np.float32))
    xr, yr, zr = _make_cube_ranges(min_xyz, max_xyz, float(pad))

    def make_frame(k: int):
        pts = pts_list[k]
        c2 = c2_list[k]
        idx = sel[k]
        return go.Frame(
            name=str(idx),
            data=[
                go.Scatter3d(
                    x=pts[:, 0] if pts.size else [],
                    y=pts[:, 1] if pts.size else [],
                    z=pts[:, 2] if pts.size else [],
                    mode="markers",
                    marker=dict(size=2),
                    name="Points",
                ),
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[0],
                    mode="markers",
                    marker=dict(size=6, symbol="diamond"),
                    name="Cam1",
                ),
                go.Scatter3d(
                    x=c2[:, 0] if c2.size else [],
                    y=c2[:, 1] if c2.size else [],
                    z=c2[:, 2] if c2.size else [],
                    mode="markers",
                    marker=dict(size=6, symbol="diamond"),
                    name="Cam2",
                ),
            ],
        )

    frames_plotly = [make_frame(k) for k in range(len(sel))]
    fig = go.Figure(
        data=frames_plotly[0].data,
        frames=frames_plotly,
        layout=go.Layout(
            title="Interactive 3D Reconstruction",
            uirevision="lock",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(range=[xr[0], xr[1]], autorange=True, showticklabels=False, title="X"),
                yaxis=dict(range=[yr[0], yr[1]], autorange=True, showticklabels=False, title="Y"),
                zaxis=dict(range=[zr[0], zr[1]], autorange=True, showticklabels=False, title="Z"),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=True,
            updatemenus=[
                dict(
                    type="buttons",
                    x=0.05,
                    y=0.02,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    x=0.05,
                    y=0.0,
                    len=0.9,
                    currentvalue=dict(prefix="Frame: "),
                    pad=dict(t=30),
                    steps=[
                        dict(
                            method="animate",
                            args=[[str(sel[k])], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                            label=str(sel[k]),
                        )
                        for k in range(len(sel))
                    ],
                )
            ],
        ),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    print(f"[SAVE] Interactive HTML saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", required=True)
    parser.add_argument("--rt_quat", required=True)
    parser.add_argument("--k1", required=True)
    parser.add_argument("--k2", required=True)
    parser.add_argument("--smooth_alpha", type=float, default=0.1, help="0429-style RT EMA smoothing alpha; 1 disables smoothing.")
    parser.add_argument("--baseline", type=float, default=1.0, help="Normalize translation length after RT smoothing.")
    parser.add_argument("--window", type=int, default=5, help="平滑視窗大小")
    parser.add_argument("--out", default="output/final_stable_result.html")
    parser.add_argument("--out_mp4", default=None, help="Optional output path for rendered 3D mp4.")
    parser.add_argument(
        "--out_rt",
        default=None,
        help="Optional output path for per-frame RT (timestamp tx ty tz qx qy qz qw).",
    )
    parser.add_argument("--fps", type=int, default=30, help="mp4 輸出 fps")
    parser.add_argument("--zoom", type=float, default=2.5, help="mp4 視角縮放倍率")
    parser.add_argument("--sample", type=int, default=3000, help="HTML 中每幀最多顯示多少點")
    parser.add_argument("--step", type=int, default=1, help="每隔幾幀取一幀輸出到 HTML")
    parser.add_argument("--start", type=int, default=0, help="起始 frame index")
    parser.add_argument("--end", type=int, default=-1, help="結束 frame index，-1 代表最後一幀")
    parser.add_argument("--pad", type=float, default=0.05, help="3D 邊界額外留白比例")
    parser.add_argument("--seed", type=int, default=0, help="隨機抽樣點的 seed")
    parser.add_argument("--out_pts3d", default=None, help="Optional output path for pts3d numpy array.")
    parser.add_argument("--out_pts2d_cam0", default=None, help="Optional output path for cam0 2D observations.")
    parser.add_argument("--out_pts2d_cam2", default=None, help="Optional output path for cam2 2D observations.")
    parser.add_argument("--max_reproj", type=float, default=80.0, help="Max reprojection error in pixels; 0 disables.")
    parser.add_argument("--min_inliers", type=int, default=2, help="Minimum inliers to consider filtered result valid.")
    parser.add_argument("--min_keep", type=int, default=4, help="After reprojection filter, minimum points to keep.")
    parser.add_argument("--adaptive_reproj", action="store_true", help="Auto relax reprojection threshold when too few points.")
    parser.add_argument("--relax_times", type=int, default=3, help="Adaptive reprojection relax rounds.")
    parser.add_argument("--relax_mul", type=float, default=1.5, help="Adaptive reprojection threshold multiplier.")
    parser.add_argument("--strict_reproj", action="store_true", help="Skip frame if filtered max reprojection still exceeds threshold.")
    parser.add_argument("--debug_reproj", action="store_true", help="Print per-frame reprojection summary.")
    parser.add_argument("--debug_skip", action="store_true", help="Print reasons for skipped frames.")
    parser.add_argument("--scale", type=float, default=1.0, help="Manual global multiplier applied after reconstruction.")
    parser.add_argument("--real_height", type=float, default=0.0, help="Known subject height (meters); 0 disables scale fitting.")
    parser.add_argument("--scale_mode", choices=["off", "global", "frame_smooth", "block", "block_smooth"], default="off")
    parser.add_argument("--scale_metric", choices=["body_path", "bbox_y"], default="body_path")
    parser.add_argument("--scale_ref_min_joints", type=int, default=6)
    parser.add_argument("--scale_ref_max_frames", type=int, default=0, help="0 means use all valid skeleton frames for global scale.")
    parser.add_argument("--scale_block", type=int, default=5, help="Frames per scale block for block/block_smooth.")
    parser.add_argument("--scale_smooth_alpha", type=float, default=0.3)
    parser.add_argument("--block_scale_stat", choices=["max", "median"], default="max")
    parser.add_argument("--block_scale_smooth_alpha", type=float, default=0.3)
    parser.add_argument("--block_scale_max_change", type=float, default=0.15)
    parser.add_argument("--scale_clip_min", type=float, default=0.25)
    parser.add_argument("--scale_clip_max", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K1 = torch.tensor(load_k_matrix(args.k1), dtype=torch.float32, device=device)
    K2 = torch.tensor(load_k_matrix(args.k2), dtype=torch.float32, device=device)
    K1_np = K1.detach().cpu().numpy().astype(np.float64)
    K2_np = K2.detach().cpu().numpy().astype(np.float64)

    matches = load_matches(args.match, device=device)
    rt_dict = load_stable_trajectory(
        args.rt_quat,
        window_size=args.window,
        smooth_alpha=args.smooth_alpha,
        baseline=args.baseline,
    )
    if not matches:
        raise SystemExit("No valid matches loaded.")

    frames = []
    cam2_centers = []
    skeleton_frames = []
    frame_block_starts = []
    used_rt_rows = []
    pts3d_all = []
    pts2d_cam0_all = []
    pts2d_cam2_all = []
    reproj_stats = []
    skipped_empty = 0
    skipped_reproj = 0

    print(f"[INFO] ?? 3D ?? (Window: {args.window})...")

    for match_idx, match in enumerate(matches):
        ts = match["ts"]
        if ts not in rt_dict:
            continue

        R_np, t_np = rt_dict[ts]
        qx, qy, qz, qw = R_tool.from_matrix(R_np).as_quat()

        p1_obs = match["pts1"].cpu().numpy().astype(np.float32)
        p2_obs = match["pts2"].cpu().numpy().astype(np.float32)
        ids_np = _to_np(match["ids"]).astype(np.int64)
        X_np, p1_obs, p2_obs, ids_np, P1_np, P2_np = _triangulate_points_0429_style(
            K1_np,
            K2_np,
            R_np,
            t_np,
            p1_obs,
            p2_obs,
            ids_np,
        )

        if X_np.shape[0] == 0:
            skipped_empty += 1
            if args.debug_skip:
                print(f"[DBG_SKIP] ts={ts:.6f} empty_after_depth_check")
            continue

        before_stat = _summarize_reprojection_errors(X_np, p1_obs, p2_obs, P1_np, P2_np)

        if args.adaptive_reproj:
            Xf, p1f, p2f, idsf, used_thr, after_stat = _adaptive_reproj_filter(
                X_np,
                p1_obs,
                p2_obs,
                ids_np,
                P1_np,
                P2_np,
                max_reproj=args.max_reproj,
                min_inliers=args.min_inliers,
                min_keep=args.min_keep,
                relax_times=args.relax_times,
                relax_mul=args.relax_mul,
            )
        else:
            Xf, p1f, p2f, idsf, used_thr = X_np, p1_obs, p2_obs, ids_np, float(args.max_reproj)
            if args.max_reproj > 0:
                errs = _reprojection_errors(Xf, p1f, p2f, P1_np, P2_np)
                mask = np.isfinite(errs["err_max"]) & (errs["err_max"] <= float(args.max_reproj))
                if int(mask.sum()) >= int(args.min_inliers):
                    Xf = Xf[mask]
                    p1f = p1f[mask]
                    p2f = p2f[mask]
                    idsf = idsf[mask] if idsf is not None and idsf.shape[0] == mask.shape[0] else idsf
            after_stat = _summarize_reprojection_errors(Xf, p1f, p2f, P1_np, P2_np)

        if Xf is None or Xf.shape[0] < int(args.min_keep):
            skipped_reproj += 1
            if args.debug_skip:
                reason = _diagnose_pose_failure(p1_obs, p2_obs, X_np, R_np, t_np)
                print(
                    f"[DBG_SKIP] ts={ts:.6f} too_few_after_filter "
                    f"before={before_stat['count']} after={0 if Xf is None else Xf.shape[0]} "
                    f"thr={used_thr:.3f} {reason}"
                )
            continue

        if args.strict_reproj and args.max_reproj > 0 and np.isfinite(after_stat["max_max"]) and after_stat["max_max"] > float(args.max_reproj):
            skipped_reproj += 1
            if args.debug_skip:
                print(
                    f"[DBG_SKIP] ts={ts:.6f} strict_reproj "
                    f"maxerr={after_stat['max_max']:.3f} thr={float(args.max_reproj):.3f}"
                )
            continue

        if args.debug_reproj:
            print(
                f"[DBG_REPROJ] ts={ts:.6f} "
                f"before(n={before_stat['count']}, max_mean={before_stat['max_mean']:.3f}, max_max={before_stat['max_max']:.3f}) "
                f"after(n={after_stat['count']}, max_mean={after_stat['max_mean']:.3f}, max_max={after_stat['max_max']:.3f}) "
                f"thr={used_thr:.3f}"
            )
        reproj_stats.append((before_stat, after_stat, float(used_thr)))

        X_np = Xf.astype(np.float32)
        p1_obs = p1f.astype(np.float32)
        p2_obs = p2f.astype(np.float32)
        ids_np = idsf.astype(np.int64) if idsf is not None else np.full((X_np.shape[0],), -1, dtype=np.int64)

        tx, ty, tz = t_np.flatten()
        used_rt_rows.append((ts, tx, ty, tz, qx, qy, qz, qw))

        frames.append(X_np)
        pts3d_all.append(X_np.copy())
        pts2d_cam0_all.append(np.asarray(p1_obs, dtype=np.float32))
        pts2d_cam2_all.append(np.asarray(p2_obs, dtype=np.float32))

        c2 = _camera2_center_from_rt(R_np, t_np)
        cam2_centers.append(c2.reshape(1, 3))

        sk = _build_skeleton17_from_points(X_np, ids_np)
        skeleton_frames.append(sk)
        block_size = max(1, int(args.scale_block))
        frame_block_starts.append((int(match_idx) // block_size) * block_size)

    if not frames:
        raise SystemExit("No reconstructed frames available after RT matching.")

    frame_scales = np.ones((len(frames),), dtype=np.float32)
    global_scale = 1.0
    if args.scale_mode != "off" and args.real_height > 0:
        if args.scale_mode == "frame_smooth":
            frame_scales, global_scale = _estimate_frame_scales(
                skeleton_frames=skeleton_frames,
                real_height=float(args.real_height),
                metric=args.scale_metric,
                min_joints=int(args.scale_ref_min_joints),
                mode="frame_smooth",
                smooth_alpha=float(args.scale_smooth_alpha),
                clip_min=float(args.scale_clip_min),
                clip_max=float(args.scale_clip_max),
            )
            frame_scales = frame_scales.astype(np.float32) * float(args.scale)
        else:
            height_scale, h3d_median, n_scale_refs = _estimate_global_scale(
                skeleton_frames,
                real_height=float(args.real_height),
                metric=args.scale_metric,
                min_joints=int(args.scale_ref_min_joints),
                max_ref_frames=int(args.scale_ref_max_frames),
            )
            height_scale = float(height_scale)
            if args.scale_mode == "global":
                frame_scales = np.full((len(frames),), float(args.scale) * height_scale, dtype=np.float32)
            else:
                block_scales, raw_scale_by_block, h3d_by_block, refs_by_block = _compute_block_scales(
                    skeleton_frames,
                    frame_block_starts,
                    real_height=float(args.real_height),
                    metric=args.scale_metric,
                    min_joints=int(args.scale_ref_min_joints),
                    fallback_height_scale=height_scale,
                    mode=args.scale_mode,
                    height_stat=args.block_scale_stat,
                    smooth_alpha=float(args.block_scale_smooth_alpha),
                    max_change=float(args.block_scale_max_change),
                    clip_min=float(args.scale_clip_min),
                    clip_max=float(args.scale_clip_max),
                )
                frame_scales = np.asarray(
                    [float(args.scale) * block_scales.get(bs, height_scale) for bs in frame_block_starts],
                    dtype=np.float32,
                )
            global_scale = float(np.median(frame_scales)) if frame_scales.size else 1.0
            print(
                f"[INFO] scale enabled mode={args.scale_mode} global_scale={global_scale:.6f} "
                f"range=[{float(np.min(frame_scales)):.6f}, {float(np.max(frame_scales)):.6f}]"
            )
    elif abs(float(args.scale) - 1.0) > 1e-8:
        frame_scales = np.full((len(frames),), float(args.scale), dtype=np.float32)
        global_scale = float(args.scale)
        print(f"[INFO] manual global scale={global_scale:.6f}")

    if any(abs(float(s) - 1.0) > 1e-8 for s in frame_scales):
        for i, s in enumerate(frame_scales):
            sf = float(s)
            frames[i] *= sf
            pts3d_all[i] *= sf
            cam2_centers[i] *= sf
            if skeleton_frames[i] is not None:
                valid = np.isfinite(skeleton_frames[i]).all(axis=1)
                skeleton_frames[i][valid] *= sf
            ts, tx, ty, tz, qx, qy, qz, qw = used_rt_rows[i]
            used_rt_rows[i] = (ts, tx * sf, ty * sf, tz * sf, qx, qy, qz, qw)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_rt_path = Path(args.out_rt) if args.out_rt else out_path.with_name(f"{out_path.stem}_rt.txt")
    out_rt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rt_path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in used_rt_rows:
            f.write(" ".join(f"{float(v):.8f}" for v in row) + "\n")
    print(f"[SAVE] Per-frame RT saved: {out_rt_path}")

    pts3d_arr = np.vstack(pts3d_all).astype(np.float32)
    pts2d_cam0_arr = np.vstack(pts2d_cam0_all).astype(np.float32)
    pts2d_cam2_arr = np.vstack(pts2d_cam2_all).astype(np.float32)

    out_pts3d_path = Path(args.out_pts3d) if args.out_pts3d else out_path.with_name(f"{out_path.stem}_pts3d.npy")
    out_pts2d_cam0_path = (
        Path(args.out_pts2d_cam0) if args.out_pts2d_cam0 else out_path.with_name(f"{out_path.stem}_pts2d_cam0.npy")
    )
    out_pts2d_cam2_path = (
        Path(args.out_pts2d_cam2) if args.out_pts2d_cam2 else out_path.with_name(f"{out_path.stem}_pts2d_cam2.npy")
    )

    out_pts3d_path.parent.mkdir(parents=True, exist_ok=True)
    out_pts2d_cam0_path.parent.mkdir(parents=True, exist_ok=True)
    out_pts2d_cam2_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_pts3d_path, pts3d_arr)
    np.save(out_pts2d_cam0_path, pts2d_cam0_arr)
    np.save(out_pts2d_cam2_path, pts2d_cam2_arr)
    print(f"[SAVE] pts3d saved: {out_pts3d_path} shape={pts3d_arr.shape}")
    print(f"[SAVE] pts2d_cam0 saved: {out_pts2d_cam0_path} shape={pts2d_cam0_arr.shape}")
    print(f"[SAVE] pts2d_cam2 saved: {out_pts2d_cam2_path} shape={pts2d_cam2_arr.shape}")
    if reproj_stats:
        after_mean = np.array([r[1]["max_mean"] for r in reproj_stats if np.isfinite(r[1]["max_mean"])], dtype=np.float64)
        after_max = np.array([r[1]["max_max"] for r in reproj_stats if np.isfinite(r[1]["max_max"])], dtype=np.float64)
        if after_mean.size:
            print(
                f"[INFO] reprojection after-filter mean={float(np.mean(after_mean)):.3f} "
                f"median={float(np.median(after_mean)):.3f} max={float(np.max(after_max)):.3f}"
            )
    if skipped_empty or skipped_reproj:
        print(f"[INFO] skipped empty={skipped_empty} skipped reproj={skipped_reproj}")

    write_interactive_html(
        frames=frames,
        cam2_centers=cam2_centers,
        out_path=out_path,
        sample=args.sample,
        step=args.step,
        start=args.start,
        end=args.end,
        pad=args.pad,
        seed=args.seed,
    )

    out_mp4_path = Path(args.out_mp4) if args.out_mp4 else out_path.with_suffix(".mp4")
    render_3d_video(
        frames=frames,
        cam2_centers=cam2_centers,
        out_path=out_mp4_path,
        fps=args.fps,
        zoom_level=args.zoom,
        skeleton_frames=skeleton_frames,
        skeleton_edges=COCO17_BODY_EDGES,
    )
    print(f"[SAVE] 3D mp4 saved: {out_mp4_path}")


if __name__ == "__main__":
    main()
