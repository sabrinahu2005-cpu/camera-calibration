import argparse
from pathlib import Path
import numpy as np


# ----------------------------
# Utils
# ----------------------------
def _finite_xyz(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return pts
    m = np.isfinite(pts).all(axis=1)
    return pts[m]


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
    span *= (1.0 + 2.0 * pad)
    h = span / 2.0
    return (
        (float(center[0] - h), float(center[0] + h)),
        (float(center[1] - h), float(center[1] + h)),
        (float(center[2] - h), float(center[2] + h)),
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to per_frame_3d.npz")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--out", default="interactive_fixed_axes.html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import plotly.graph_objects as go

    data = np.load(args.npz, allow_pickle=True)
    frames = list(data["frames"])
    cam2s = list(data["cam2"]) if "cam2" in data else [None] * len(frames)

    n_total = len(frames)
    if n_total == 0:
        raise SystemExit("NPZ has no frames")

    start = max(0, int(args.start))
    end = n_total - 1 if int(args.end) < 0 else min(n_total - 1, int(args.end))
    step = max(1, int(args.step))
    sel = list(range(start, end + 1, step))
    if not sel:
        raise SystemExit("No frames selected. Check --start/--end/--step")

    rng = np.random.default_rng(args.seed)

    # ----------------------------
    # Preprocess + GLOBAL bounds
    # ----------------------------
    pts_list = []
    c2_list = []

    min_xyz = np.array([np.inf, np.inf, np.inf], np.float32)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], np.float32)

    for i in sel:
        pts = _finite_xyz(frames[i])
        pts = _sample_pts(pts, args.sample, rng)
        pts_list.append(pts)

        if i < len(cam2s) and cam2s[i] is not None:
            c2 = _finite_xyz(cam2s[i])[:1]
        else:
            c2 = np.zeros((0, 3), np.float32)
        c2_list.append(c2)

        if pts.shape[0]:
            min_xyz = np.minimum(min_xyz, pts.min(axis=0))
            max_xyz = np.maximum(max_xyz, pts.max(axis=0))
        if c2.shape[0]:
            min_xyz = np.minimum(min_xyz, c2.min(axis=0))
            max_xyz = np.maximum(max_xyz, c2.max(axis=0))

    # include cam1 origin
    min_xyz = np.minimum(min_xyz, np.zeros(3, np.float32))
    max_xyz = np.maximum(max_xyz, np.zeros(3, np.float32))

    xr, yr, zr = _make_cube_ranges(min_xyz, max_xyz, float(args.pad))

    # ----------------------------
    # Frames (DATA ONLY)
    # ----------------------------
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
                    x=[0], y=[0], z=[0],
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

    # ----------------------------
    # FIGURE (FIXED SPACE + FIXED CAMERA)
    # ----------------------------
    fig = go.Figure(
        data=frames_plotly[0].data,
        frames=frames_plotly,
        layout=go.Layout(
            title="Interactive 3D (fixed axes + fixed camera)",
            uirevision="lock",  # keep user interaction stable
            scene=dict(
                # ✅ XYZ 顯示比例固定 1:1:1（每格一樣大）
                aspectmode="cube",

                # ✅ 固定 range（同一個 span）
                xaxis=dict(
                    range=[xr[0], xr[1]],
                    autorange=True,
                    showticklabels=False,
                    title="X",
                    dtick=1,
                    showbackground=True,
                    backgroundcolor="rgb(240,240,240)",
                    gridcolor="white",
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    range=[yr[0], yr[1]],
                    autorange=True,
                    showticklabels=False,
                    title="Y",
                    dtick=1,
                    showbackground=True,
                    backgroundcolor="rgb(240,240,240)",
                    gridcolor="white",
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    range=[zr[0], zr[1]],
                    autorange=True,
                    showticklabels=False,
                    title="Z",
                    dtick=1,
                    showbackground=True,
                    backgroundcolor="rgb(240,240,240)",
                    gridcolor="white",
                    zerolinecolor="white",
                ),

                # ✅ 視角固定，切 frame 不會跳
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    print(f"[OK] wrote {out.resolve()}")


if __name__ == "__main__":
    main()
