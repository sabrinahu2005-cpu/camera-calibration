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


def load_stable_trajectory(path, window_size=5):
    # 讀取 rt quaternion 並做平滑。
    raw_data = []
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 8:
                raw_data.append(vals)

    if not raw_data:
        return {}
    raw_data = np.array(raw_data)
    ts = raw_data[:, 0]
    t_vals = raw_data[:, 1:4]
    q_vals = raw_data[:, 4:8]

    smoothed_t = np.copy(t_vals)
    if len(ts) > window_size:
        for i in range(3):
            smoothed_t[:, i] = np.convolve(t_vals[:, i], np.ones(window_size) / window_size, mode="same")

    smoothed_q = np.copy(q_vals)
    if len(ts) > window_size:
        for i in range(4):
            smoothed_q[:, i] = np.convolve(q_vals[:, i], np.ones(window_size) / window_size, mode="same")

    rt_dict = {}
    for i in range(len(ts)):
        q = smoothed_q[i] / (np.linalg.norm(smoothed_q[i]) + 1e-8)
        R = R_tool.from_quat(q).as_matrix()
        t_raw = smoothed_t[i]
        t_norm = t_raw / (np.linalg.norm(t_raw) + 1e-8)
        rt_dict[ts[i]] = (R.astype(np.float32), t_norm.reshape(3, 1).astype(np.float32))
    return rt_dict


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K1 = torch.tensor(load_k_matrix(args.k1), dtype=torch.float32, device=device)
    K2 = torch.tensor(load_k_matrix(args.k2), dtype=torch.float32, device=device)

    matches = load_matches(args.match, device=device)
    rt_dict = load_stable_trajectory(args.rt_quat, window_size=args.window)
    if not matches:
        raise SystemExit("No valid matches loaded.")

    frames = []
    cam2_centers = []
    skeleton_frames = []
    used_rt_rows = []
    pts3d_all = []
    pts2d_cam0_all = []
    pts2d_cam2_all = []

    print(f"[INFO] 開始 3D 重建 (Window: {args.window})...")

    for match in matches:
        ts = match["ts"]
        if ts not in rt_dict:
            continue

        R_np, t_np = rt_dict[ts]
        R = torch.tensor(R_np, device=device)
        t = torch.tensor(t_np, device=device)
        qx, qy, qz, qw = R_tool.from_matrix(R_np).as_quat()
        tx, ty, tz = t_np.flatten()
        used_rt_rows.append((ts, tx, ty, tz, qx, qy, qz, qw))

        P1 = K1 @ torch.eye(3, 4, device=device)
        ext2 = torch.cat([R, t], dim=1)
        P2 = K2 @ ext2

        p1n = match["pts1"].T.cpu().numpy()
        p2n = match["pts2"].T.cpu().numpy()
        X4 = cv2.triangulatePoints(P1.cpu().numpy(), P2.cpu().numpy(), p1n, p2n)
        X = X4[:3] / (X4[3] + 1e-8)
        X_np = X.T.astype(np.float32)

        frames.append(X_np)
        pts3d_all.append(X_np)
        pts2d_cam0_all.append(np.asarray(p1n.T, dtype=np.float32))
        pts2d_cam2_all.append(np.asarray(p2n.T, dtype=np.float32))

        c2 = -R_np.T @ t_np
        cam2_centers.append(c2.flatten().reshape(1, 3))

        sk = np.full((17, 3), np.nan, dtype=np.float32)
        ids_np = _to_np(match["ids"])
        for j, jid in enumerate(ids_np):
            if 0 <= jid < 17:
                sk[jid] = X_np[j]
        skeleton_frames.append(sk)

    if not frames:
        raise SystemExit("No reconstructed frames available after RT matching.")

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
