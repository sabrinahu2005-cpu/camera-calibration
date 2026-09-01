# 一幀

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_lib


# =========================
# IO
# =========================
def load_k(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_d(path: Path | None) -> np.ndarray:
    return np.loadtxt(path) if path is not None else np.zeros((5, 1), dtype=np.float32)


def get_aruco_dict(name: str):
    aruco = cv2.aruco
    return cv2.aruco.getPredefinedDictionary(getattr(aruco, name))


# =========================
# Detection
# =========================
def detect(frame, aruco_dict) -> dict[int, np.ndarray]:
    if frame is None:
        return {}
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)
    if ids is None:
        return {}
    return {int(i): c for i, c in zip(ids.flatten(), corners)}


# =========================
# PnP
# =========================
def solve_pnp(corner, size, K, D, max_err=2.0):
    half = size / 2.0

    obj = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    img = corner.reshape(4, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, D,
        flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img, axis=1)))

    if err > max_err:
        return None

    return R, tvec.reshape(3), err


# =========================
# Fusion
# =========================
def fuse(Rs, ts, ws):
    ws = np.asarray(ws, dtype=float)
    ws = ws / (ws.sum() + 1e-8)

    t = np.sum([w * ti.reshape(3) for w, ti in zip(ws, ts)], axis=0)
    R = R_lib.from_matrix(Rs).mean(weights=ws).as_matrix()
    return R, t


def estimate(marker_dict, ids, K, D, size, max_err=2.0):
    Rs, ts, ws, errs = [], [], [], []

    for i in ids:
        out = solve_pnp(marker_dict[i], size, K, D, max_err)
        if out is None:
            continue

        R, t, e = out

        area = float(cv2.contourArea(marker_dict[i].reshape(4, 1, 2)))
        w = area / (e + 1e-6)

        Rs.append(R)
        ts.append(t)
        ws.append(w)
        errs.append(e)

    if not Rs:
        return None, None, None

    Rf, tf = fuse(Rs, ts, ws)
    return Rf, tf, float(np.mean(errs))


# =========================
# Relative pose
# =========================
def relative(R1, t1, R2, t2):
    R = R2 @ R1.T
    t = t2 - R @ t1
    return R, t


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam0",           required=True)
    ap.add_argument("--cam2",           required=True)
    ap.add_argument("--k1",             required=True)
    ap.add_argument("--k2",             required=True)
    ap.add_argument("--d1",             default=None)
    ap.add_argument("--d2",             default=None)
    ap.add_argument("--marker_size",    type=float, default=0.194)
    ap.add_argument("--stride",         type=int,   default=1)
    ap.add_argument("--dict",           default="DICT_6X6_50")
    ap.add_argument("--max_reproj_err", type=float, default=2.0)
    ap.add_argument("--out",            default="rt.txt")
    args = ap.parse_args()

    K1 = load_k(Path(args.k1))
    K2 = load_k(Path(args.k2))
    D1 = load_d(Path(args.d1) if args.d1 else None)
    D2 = load_d(Path(args.d2) if args.d2 else None)

    aruco = get_aruco_dict(args.dict)
    cap0  = cv2.VideoCapture(args.cam0)
    cap2  = cv2.VideoCapture(args.cam2)

    fps = cap0.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    out_path = Path(args.out)
    print(f"[INFO] Output file: {out_path.resolve()}")

    rows = []
    i = 0

    while True:
        ok0 = cap0.grab()
        ok2 = cap2.grab()
        if not ok0 or not ok2:
            break

        if i % args.stride == 0:
            _, f0 = cap0.retrieve()
            _, f2 = cap2.retrieve()

            d0 = detect(f0, aruco)
            d2 = detect(f2, aruco)
            common = set(d0) & set(d2)

            if common:
                R1, t1, e1 = estimate(d0, common, K1, D1, args.marker_size, args.max_reproj_err)
                R2, t2, e2 = estimate(d2, common, K2, D2, args.marker_size, args.max_reproj_err)

                if R1 is not None and R2 is not None:
                    R, t = relative(R1, t1, R2, t2)

                    q = R_lib.from_matrix(R).as_quat()         # [qx, qy, qz, qw]
                    q = q / (np.linalg.norm(q) + 1e-8)         # normalize

                    timestamp = i / fps

                    rows.append([
                        timestamp,
                        t[0], t[1], t[2],
                        q[0], q[1], q[2], q[3]
                    ])

                    print(
                        f"frame {i:05d}  "
                        f"t={timestamp:.3f}s  "
                        f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]  "
                        f"q=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}]"
                    )

        i += 1

    cap0.release()
    cap2.release()

    if not rows:
        print("No valid frames")
        return

    with open(out_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for r in rows:
            f.write(
                f"{r[0]:.1f} "
                f"{r[1]:.6f} {r[2]:.6f} {r[3]:.6f} "
                f"{r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n"
            )

    print(f"\nDone. {len(rows)} frames written → {out_path.resolve()}")


if __name__ == "__main__":
    main()

    








    




