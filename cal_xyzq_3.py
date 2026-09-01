import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_tool


def build_parser():
    # 定義輸入 match 檔、相機內參，以及輸出位姿檔案路徑。
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="data/cam_match_1.txt", help="輸入的 match 檔案")
    parser.add_argument("--k1", required=True, help="相機 1 內參矩陣 (例如 data/K1.txt)")
    parser.add_argument("--k2", required=True, help="相機 2 內參矩陣 (例如 data/K2.txt)")
    parser.add_argument("--out", default="data/rt_quaternion.txt", help="輸出的四元數外參檔案路徑")
    parser.add_argument("--threshold", type=float, default=0.5, help="可見度門檻")
    parser.add_argument("--min_pairs", type=int, default=8, help="至少需要多少組對應點才估計")
    return parser


def load_intrinsics(k1_path: str, k2_path: str):
    # 讀取兩台相機的內參矩陣；若格式不對就直接中止。
    try:
        return np.loadtxt(k1_path), np.loadtxt(k2_path)
    except Exception as exc:
        raise SystemExit(f"[ERROR] 讀取 K 矩陣失敗: {exc}")


def parse_match_line(line: str, threshold: float):
    # 從一行 match 資料中取出 timestamp 與通過可見度門檻的對應點。
    parts = line.strip().split(" ", 1)
    if len(parts) < 2:
        return None, None, None

    timestamp_str = parts[0]
    pairs = json.loads(parts[1])
    pts1, pts2 = [], []
    for p in pairs:
        if p[2] >= threshold and p[3] >= threshold:
            pts1.append(p[0])
            pts2.append(p[1])
    return timestamp_str, pts1, pts2


def estimate_pose_from_points(pts1, pts2, K1, K2, min_pairs: int):
    # 用一組左右相機對應點估計 Essential Matrix，再恢復相對姿態 R/t。
    if len(pts1) < min_pairs:
        return None

    pts1_np = np.expand_dims(np.array(pts1, dtype=np.float64), axis=1)
    pts2_np = np.expand_dims(np.array(pts2, dtype=np.float64), axis=1)
    pts1_norm = cv2.undistortPoints(pts1_np, K1, None)
    pts2_norm = cv2.undistortPoints(pts2_np, K2, None)

    E, _ = cv2.findEssentialMat(
        pts1_norm,
        pts2_norm,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.001,
    )
    if E is None or E.shape != (3, 3):
        return None

    _, R, t, _ = cv2.recoverPose(E, pts1_norm, pts2_norm, np.eye(3))
    qx, qy, qz, qw = R_tool.from_matrix(R).as_quat()
    tx, ty, tz = t.flatten()
    return tx, ty, tz, qx, qy, qz, qw


def main():
    # 主流程：讀 match 與 K1/K2，逐幀估計相對位姿，輸出四元數格式結果。
    args = build_parser().parse_args()
    K1, K2 = load_intrinsics(args.k1, args.k2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_frames = 0

    with open(args.match, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("# timestamp tx ty tz qx qy qz qw\n")

        for line in f_in:
            if not line.strip():
                continue

            timestamp_str, pts1, pts2 = parse_match_line(line, args.threshold)
            if timestamp_str is None:
                continue

            pose = estimate_pose_from_points(pts1, pts2, K1, K2, args.min_pairs)
            if pose is None:
                continue

            tx, ty, tz, qx, qy, qz, qw = pose
            f_out.write(f"{timestamp_str} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            valid_frames += 1

    print("---")
    print("[DONE] Stage 1 完成")
    print(f"[INFO] 成功估計 {valid_frames} 筆外參")
    print(f"[SAVE] 結果已輸出到 {out_path}")


if __name__ == "__main__":
    main()
