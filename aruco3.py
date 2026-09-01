# 五幀

from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque
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
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


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
# Smoothing（高斯中心窗口，支援任意窗口大小）
# 修正：改用 PoseSmoother class，避免 function attribute 共享狀態
# 導致 main loop 與 tail flush 之間出現 quaternion sign flip 跳變
# =========================
def make_gaussian_weights(n: int) -> np.ndarray:
    """產生長度為 n 的高斯權重（以中心為峰值）。"""
    if n == 1:
        return np.array([1.0])
    sigma = (n - 1) / 4.0
    x = np.arange(n) - (n - 1) / 2.0
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


class PoseSmoother:
    """
    封裝高斯平滑邏輯與 quaternion sign flip 狀態。
    main loop 與 tail flush 共用同一個 instance，
    確保 last_q 在兩段之間連續，不產生符號跳變。
    """
    def __init__(self):
        self.last_q: np.ndarray | None = None

    def smooth(self, buf: list) -> tuple[np.ndarray, np.ndarray]:
        """
        輸入：buf = [(timestamp, t_raw, R_raw), ...]
        輸出：(t_smooth, q_smooth)，q 為 [qx, qy, qz, qw]
        """
        ts_list = [item[1] for item in buf]
        R_list  = [item[2] for item in buf]
        weights = make_gaussian_weights(len(buf))

        # Translation: weighted mean
        t_smooth = np.sum([w * t for w, t in zip(weights, ts_list)], axis=0)

        # Rotation: SO(3) weighted mean (Fréchet mean on rotation manifold)
        R_smooth = R_lib.from_matrix(R_list).mean(weights=weights).as_matrix()

        # 轉成 quaternion 輸出
        q = R_lib.from_matrix(R_smooth).as_quat()   # [qx, qy, qz, qw]
        q = q / (np.linalg.norm(q) + 1e-8)

        # Sign flip fix：確保與上一幀同半球，避免不連續跳變
        if self.last_q is not None and np.dot(q, self.last_q) < 0:
            q = -q
        self.last_q = q

        return t_smooth, q


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
    ap.add_argument("--smooth",         type=int,   default=5)
    ap.add_argument("--out",            default="rt.txt")
    args = ap.parse_args()

    K1 = load_k(Path(args.k1))
    K2 = load_k(Path(args.k2))
    D1 = load_d(Path(args.d1) if args.d1 else None)
    D2 = load_d(Path(args.d2) if args.d2 else None)

    aruco    = get_aruco_dict(args.dict)
    cap0     = cv2.VideoCapture(args.cam0)
    cap2     = cv2.VideoCapture(args.cam2)
    fps      = cap0.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    win_size = args.smooth
    # Non-overlapping block smoothing：
    # 每累積 win_size 幀算一次，結果寫給這整段每一幀
    block_buf: list = []   # 當前 block 的原始幀

    smoother = PoseSmoother()

    out_path = Path(args.out)
    print(f"[INFO] Output file  : {out_path.resolve()}")
    print(f"[INFO] Smooth window: {win_size} frames (Gaussian, non-overlapping block)")

    rows: list = []
    i = 0

    def flush_block(block: list):
        """計算 block 的平滑值，寫給 block 內每一幀。"""
        t_sm, q_sm = smoother.smooth(block)
        for ts, _, _ in block:
            rows.append([ts,
                         t_sm[0], t_sm[1], t_sm[2],
                         q_sm[0], q_sm[1], q_sm[2], q_sm[3]])
        print(f"[block] {len(block)} frames  "
              f"t=[{t_sm[0]:.3f},{t_sm[1]:.3f},{t_sm[2]:.3f}]  "
              f"q=[{q_sm[0]:.3f},{q_sm[1]:.3f},{q_sm[2]:.3f},{q_sm[3]:.3f}]")

    while True:
        if not cap0.grab() or not cap2.grab():
            break

        if i % args.stride == 0:
            _, f0 = cap0.retrieve()
            _, f2 = cap2.retrieve()

            d0 = detect(f0, aruco)
            d2 = detect(f2, aruco)
            common = set(d0) & set(d2)

            if len(common) >= 1:
                if i % 100 == 0:
                    print(f"[INFO] frame {i:05d}  markers: {len(common)}  ids: {sorted(common)}")

                R1, t1, _ = estimate(d0, common, K1, D1, args.marker_size, args.max_reproj_err)
                R2, t2, _ = estimate(d2, common, K2, D2, args.marker_size, args.max_reproj_err)

                if R1 is not None and R2 is not None:
                    R_raw, t_raw = relative(R1, t1, R2, t2)
                    timestamp    = i / fps

                    block_buf.append((timestamp, t_raw, R_raw))

                    # 滿 win_size 幀就 flush 這個 block
                    if len(block_buf) == win_size:
                        flush_block(block_buf)
                        block_buf.clear()

        i += 1

    cap0.release()
    cap2.release()

    # flush 尾巴：最後不足 win_size 的剩餘幀
    if block_buf:
        print(f"[INFO] Flushing tail block ({len(block_buf)} frames)")
        flush_block(block_buf)
        block_buf.clear()

    if not rows:
        print("No valid frames")
        return

    # 依 timestamp 排序（正常情況已有序，保險起見）
    rows.sort(key=lambda r: r[0])

    with open(out_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for r in rows:
            f.write(f"{r[0]:.6f} "
                    f"{r[1]:.6f} {r[2]:.6f} {r[3]:.6f} "
                    f"{r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n")

    print(f"\nDone. {len(rows)} frames written → {out_path.resolve()}")


if __name__ == "__main__":
    main()
