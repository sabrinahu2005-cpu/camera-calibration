import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

BASE_DIR = Path(__file__).resolve().parent


def build_parser():
    # 讀取配對點檔並輸出 K1/K2 的命令列參數。
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default=str(BASE_DIR / "data" / "cam_match.txt"))
    parser.add_argument("--out_k1", default=str(BASE_DIR / "data" / "K1.txt"))
    parser.add_argument("--out_k2", default=str(BASE_DIR / "data" / "K2.txt"))
    parser.add_argument("--img_w", type=int, default=1280)
    parser.add_argument("--img_h", type=int, default=720)
    parser.add_argument("--mode", choices=["simple", "expert"], default="simple")
    parser.add_argument("--n_start", type=int, default=5)
    parser.add_argument("--min_pairs", type=int, default=8)
    return parser


def parse_corr_pair(item):
    # 將單一關節配對整理成 [lx, ly, rx, ry]。
    if not isinstance(item, list) or len(item) < 2:
        return None
    try:
        lx, ly = float(item[0][0]), float(item[0][1])
        rx, ry = float(item[1][0]), float(item[1][1])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (lx, ly, rx, ry)):
        return None
    return [lx, ly, rx, ry]


def load_matches(match_path: Path, min_pairs: int):
    # 讀 cam_match.txt，保留有效點數足夠的 frame。
    matches = []
    with match_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, json_str = line.split(" ", 1)
                pairs = json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                continue

            corr = [parsed for item in pairs if (parsed := parse_corr_pair(item)) is not None]
            if len(corr) >= min_pairs:
                matches.append(np.array(corr, dtype=float))
    return matches


def save_matrix(path: Path, mat: np.ndarray):
    # 將矩陣輸出成文字檔。
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, mat, fmt="%.8f")


def normalize_points(pts):
    # 對 2D 點做 Hartley normalization，提升 F matrix 穩定性。
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean
    dist = np.sqrt(np.sum(pts_centered**2, axis=1))
    scale = np.sqrt(2) / (np.mean(dist) + 1e-8)

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1],
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def compute_fundamental_matrix(corr):
    # 以 eight-point algorithm 估計 Fundamental Matrix。
    pts1 = corr[:, :2]
    pts2 = corr[:, 2:]

    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    rows = []
    for (x1, y1), (x2, y2) in zip(pts1_n, pts2_n):
        rows.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    A = np.array(rows)

    _, _, vt = np.linalg.svd(A)
    F = vt[-1].reshape(3, 3)

    U, S, vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ vt
    F = T2.T @ F @ T1
    return F / (np.linalg.norm(F) + 1e-8)


def sampson_distance(F, pts1, pts2):
    # 計算 epipolar geometry 的 Sampson error。
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    err = np.sum(pts2_h * Fx1, axis=1) ** 2
    return np.mean(err / (denom + 1e-8))


def apply_undistort(pts, fx, k1, img_w, img_h):
    # 用單一 k1 徑向模型做去畸變。
    if k1 == 0:
        return pts
    cx, cy = img_w / 2, img_h / 2
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fx
    r2 = x**2 + y**2
    x_u = x * (1 + k1 * r2)
    y_u = y * (1 + k1 * r2)
    return np.stack([x_u * fx + cx, y_u * fx + cy], axis=1)


def essential_loss_hetero(params, matches_all, img_w, img_h, use_k1=False):
    # 以 Essential Matrix 條件與 Sampson error 組成 loss。
    fx_f = params[0]
    fx_m = params[1]
    k1_m = params[2] if use_k1 else 0.0

    cx, cy = img_w / 2, img_h / 2
    K_f = np.array([[fx_f, 0, cx], [0, fx_f, cy], [0, 0, 1]])
    K_m = np.array([[fx_m, 0, cx], [0, fx_m, cy], [0, 0, 1]])

    total_loss = 0.0
    for corr in matches_all:
        pts_f = corr[:, :2]
        pts_m = corr[:, 2:]
        pts_m_u = apply_undistort(pts_m, fx_m, k1_m, img_w, img_h)

        F = compute_fundamental_matrix(np.hstack([pts_f, pts_m_u]))
        E = K_m.T @ F @ K_f
        _, S, _ = np.linalg.svd(E)
        loss_E = ((S[0] - S[1]) / S[0]) ** 2 + (S[2] / S[0]) ** 2
        loss_geo = sampson_distance(F, pts_f, pts_m_u)
        total_loss += loss_E + 0.05 * np.log1p(loss_geo)

    return total_loss / len(matches_all)


def calibrate_hetero(matches_final, img_w=1920, img_h=1080, mode="simple", n_start=5):
    # 多次隨機起點優化，估計 fixed/moving 兩台相機的 K。
    use_k1 = mode == "expert"
    matches_clean = [np.array(m) for m in matches_final if len(m) >= 8]
    if not matches_clean:
        raise ValueError("沒有足夠的有效對應點 (>=8)")

    best_res = None
    min_loss = np.inf

    for i in range(n_start):
        if use_k1:
            init_guess = [
                np.random.uniform(800, 1500),
                np.random.uniform(800, 1500),
                np.random.uniform(-0.1, 0.1),
            ]
        else:
            init_guess = [
                np.random.uniform(800, 1500),
                np.random.uniform(800, 1500),
            ]

        res = minimize(
            essential_loss_hetero,
            init_guess,
            args=(matches_clean, img_w, img_h, use_k1),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4},
        )
        if res.fun < min_loss:
            min_loss = res.fun
            best_res = res
        print(f"[Multi-start {i + 1}/{n_start}] Loss: {res.fun:.6f}")

    fx_f, fx_m = best_res.x[0], best_res.x[1]
    k1_m = best_res.x[2] if use_k1 else 0.0
    cx, cy = img_w / 2, img_h / 2
    K_fixed = np.array([[fx_f, 0, cx], [0, fx_f, cy], [0, 0, 1]])
    K_moving = np.array([[fx_m, 0, cx], [0, fx_m, cy], [0, 0, 1]])
    return K_fixed, K_moving, k1_m, best_res.success


def main():
    # 讀取配對點後，直接在同一支程式內估 K1/K2 並輸出。
    args = build_parser().parse_args()
    match_path = Path(args.match)
    if not match_path.exists():
        raise SystemExit(f"match file not found: {match_path}")

    matches = load_matches(match_path, args.min_pairs)
    if not matches:
        raise SystemExit("no valid matches (>= min_pairs) found")

    k1, k2, k1_m, ok = calibrate_hetero(
        matches,
        img_w=args.img_w,
        img_h=args.img_h,
        mode=args.mode,
        n_start=args.n_start,
    )

    out_k1 = Path(args.out_k1)
    out_k2 = Path(args.out_k2)
    save_matrix(out_k1, k1)
    save_matrix(out_k2, k2)

    print("K1 saved:", out_k1)
    print("K2 saved:", out_k2)
    print("k1 (moving) =", k1_m)
    print("converged =", ok)


if __name__ == "__main__":
    main()
