#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_k_matrix(path: Path) -> np.ndarray:
    k = np.loadtxt(path, dtype=float)
    if k.shape != (3, 3):
        raise ValueError(f"K matrix must be 3x3, got {k.shape} from {path}")
    return k


def resolve_k_matrix(args, camera_label: str) -> np.ndarray:
    if args.camera == "both":
        if camera_label == "cam0":
            if not args.k_cam0:
                raise ValueError("--camera both requires --k_cam0")
            return load_k_matrix(Path(args.k_cam0))
        if camera_label == "cam2":
            if not args.k_cam2:
                raise ValueError("--camera both requires --k_cam2")
            return load_k_matrix(Path(args.k_cam2))
        raise ValueError(f"Unsupported camera label: {camera_label}")

    if args.k_est:
        return load_k_matrix(Path(args.k_est))
    if camera_label == "cam0" and args.k_cam0:
        return load_k_matrix(Path(args.k_cam0))
    if camera_label == "cam2" and args.k_cam2:
        return load_k_matrix(Path(args.k_cam2))
    raise ValueError("Single-camera mode requires --k_est, or the matching --k_cam0 / --k_cam2")


def load_xyzq_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = [float(x) for x in s.split()]
            if len(vals) < 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = vals[:8]
            rows.append(
                {
                    "ts": float(ts),
                    "t": np.array([tx, ty, tz], dtype=np.float64),
                    "q": np.array([qx, qy, qz, qw], dtype=np.float64),
                }
            )
    if not rows:
        raise ValueError(f"No valid xyzq rows found in {path}")
    return rows


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("Zero quaternion is invalid")
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_match_timestamp(path: Path, frame_idx: int) -> float:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i != frame_idx:
                continue
            parts = line.strip().split(maxsplit=1)
            if not parts:
                break
            return float(parts[0])
    raise ValueError(f"Could not load timestamp for frame_idx={frame_idx} from {path}")


def select_pose(rows: list[dict], frame_idx: int, align_mode: str, match_path: Path | None) -> dict:
    if align_mode == "index":
        if not (0 <= frame_idx < len(rows)):
            raise IndexError(f"frame_idx {frame_idx} out of range for xyzq length {len(rows)}")
        return rows[frame_idx]

    if match_path is None:
        raise ValueError("align_mode=timestamp requires --match")
    target_ts = load_match_timestamp(match_path, frame_idx)
    return min(rows, key=lambda row: abs(row["ts"] - target_ts))


def load_point_array(path: Path, point_dim: int, name: str, frame_idx: int | None) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim == 2 and arr.shape[1] == point_dim:
        return arr
    if arr.ndim == 3 and arr.shape[2] == point_dim:
        if frame_idx is None:
            raise ValueError(f"{name} has shape {arr.shape}; --frame_idx is required")
        if not (0 <= frame_idx < arr.shape[0]):
            raise IndexError(f"{name} frame_idx {frame_idx} out of range for shape {arr.shape}")
        return arr[frame_idx]
    raise ValueError(f"{name} must have shape (N, {point_dim}) or (F, N, {point_dim}), got {arr.shape}")


def infer_frame_count(path: Path, point_dim: int, name: str) -> int:
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim == 2 and arr.shape[1] == point_dim:
        return 1
    if arr.ndim == 3 and arr.shape[2] == point_dim:
        return int(arr.shape[0])
    raise ValueError(f"{name} must have shape (N, {point_dim}) or (F, N, {point_dim}), got {arr.shape}")


def orthogonalize_rotation(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    r_ortho = u @ vt
    if np.linalg.det(r_ortho) < 0:
        u[:, -1] *= -1
        r_ortho = u @ vt
    return r_ortho


def rotation_error_deg(R_est: np.ndarray, R_aruco: np.ndarray) -> float:
    r_rel = R_aruco.T @ R_est
    trace_val = np.trace(r_rel)
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def translation_error(t_est: np.ndarray, t_aruco: np.ndarray) -> tuple[float, float]:
    t_est = t_est.reshape(3)
    t_aruco = t_aruco.reshape(3)
    distance = float(np.linalg.norm(t_est - t_aruco))
    norm_est = float(np.linalg.norm(t_est))
    norm_aruco = float(np.linalg.norm(t_aruco))
    if norm_est == 0.0 or norm_aruco == 0.0:
        return distance, float("nan")
    cos_ang = float(np.dot(t_est, t_aruco) / (norm_est * norm_aruco))
    direction_deg = math.degrees(math.acos(np.clip(cos_ang, -1.0, 1.0)))
    return distance, direction_deg


def project_points(K_est: np.ndarray, R_est: np.ndarray, t_est: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R_est)
    pts2d, _ = cv2.projectPoints(
        pts3d.astype(np.float64),
        rvec.astype(np.float64),
        t_est.reshape(3, 1).astype(np.float64),
        K_est.astype(np.float64),
        None,
    )
    return pts2d.reshape(-1, 2)


def reprojection_rmse(K_est: np.ndarray, R_est: np.ndarray, t_est: np.ndarray, pts3d: np.ndarray, pts2d_obs: np.ndarray) -> float:
    pts2d_pred = project_points(K_est, R_est, t_est, pts3d)
    diff = pts2d_pred - pts2d_obs
    return float(math.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def format_value(val: float) -> str:
    if math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.6f}"


def render_table(rows: list[list[str]]) -> str:
    headers = ["Metric", "Value"]
    w0 = max(len(headers[0]), *(len(str(r[0])) for r in rows))
    w1 = max(len(headers[1]), *(len(str(r[1])) for r in rows))

    def row(a: str, b: str) -> str:
        return f"| {a.ljust(w0)} | {b.ljust(w1)} |"

    lines = [row(headers[0], headers[1]), f"| {'-' * w0} | {'-' * w1} |"]
    lines.extend(row(str(a), str(b)) for a, b in rows)
    return "\n".join(lines)


def build_combined_report(shared_metrics: dict, reprojection_by_camera: dict[str, float], points_used_by_camera: dict[str, float | int]) -> str:
    rows = [
        ["Rotation Error (deg)", format_value(shared_metrics["rotation_error_deg"])],
        ["Translation Distance", format_value(shared_metrics["translation_distance"])],
        ["Translation Direction (deg)", format_value(shared_metrics["translation_direction_deg"])],
    ]
    for camera_label in ["cam0", "cam2"]:
        if camera_label in reprojection_by_camera:
            rows.append([f"Reprojection RMSE {camera_label} (px)", format_value(reprojection_by_camera[camera_label])])
    for camera_label in ["cam0", "cam2"]:
        if camera_label in points_used_by_camera:
            rows.append([f"Points Used {camera_label}", format_value(points_used_by_camera[camera_label])])
    return render_table(rows)


def build_payload(args, pts2d_obs: np.ndarray, frame_idx: int, camera_label: str) -> dict:
    k_est = resolve_k_matrix(args, camera_label)
    est_pose = select_pose(load_xyzq_rows(Path(args.xyzq_est)), frame_idx, args.align_mode, Path(args.match) if args.match else None)
    aruco_pose = select_pose(load_xyzq_rows(Path(args.xyzq_aruco)), frame_idx, args.align_mode, Path(args.match) if args.match else None)
    pts3d = load_point_array(Path(args.pts3d_npy), 3, "pts3d", frame_idx)

    return {
        "K_est": k_est.tolist(),
        "R_est": quat_xyzw_to_rot(est_pose["q"]).tolist(),
        "R_aruco": quat_xyzw_to_rot(aruco_pose["q"]).tolist(),
        "t_est": est_pose["t"].reshape(3).tolist(),
        "t_aruco": aruco_pose["t"].reshape(3).tolist(),
        "pts3d": pts3d.tolist(),
        "pts2d_obs": pts2d_obs.tolist(),
    }


def evaluate_payload(payload: dict) -> str:
    metrics = compute_metrics(payload)
    rows = [
        ["Rotation Error (deg)", format_value(metrics["rotation_error_deg"])],
        ["Translation Distance", format_value(metrics["translation_distance"])],
        ["Translation Direction (deg)", format_value(metrics["translation_direction_deg"])],
        ["Reprojection RMSE (px)", format_value(metrics["reprojection_rmse_px"])],
        ["Points Used", str(int(metrics["points_used"]))],
    ]
    return render_table(rows)


def compute_metrics(payload: dict) -> dict:
    K_est = np.asarray(payload["K_est"], dtype=float)
    R_est = orthogonalize_rotation(np.asarray(payload["R_est"], dtype=float))
    R_aruco = orthogonalize_rotation(np.asarray(payload["R_aruco"], dtype=float))
    t_est = np.asarray(payload["t_est"], dtype=float).reshape(3)
    t_aruco = np.asarray(payload["t_aruco"], dtype=float).reshape(3)
    pts3d = np.asarray(payload["pts3d"], dtype=float)
    pts2d_obs = np.asarray(payload["pts2d_obs"], dtype=float)

    if pts3d.ndim != 2 or pts3d.shape[1] != 3:
        raise ValueError(f"pts3d must have shape (N, 3), got {pts3d.shape}")
    if pts2d_obs.ndim != 2 or pts2d_obs.shape[1] != 2:
        raise ValueError(f"pts2d_obs must have shape (N, 2), got {pts2d_obs.shape}")
    if pts3d.shape[0] != pts2d_obs.shape[0]:
        raise ValueError("pts3d and pts2d_obs must have the same number of points")

    rot_err = rotation_error_deg(R_est, R_aruco)
    trans_dist, trans_dir = translation_error(t_est, t_aruco)
    rmse = reprojection_rmse(K_est, R_est, t_est, pts3d, pts2d_obs)
    return {
        "rotation_error_deg": float(rot_err),
        "translation_distance": float(trans_dist),
        "translation_direction_deg": float(trans_dir),
        "reprojection_rmse_px": float(rmse),
        "points_used": int(pts3d.shape[0]),
    }


def summarize_metrics(rows: list[dict]) -> dict:
    def mean_of(key: str) -> float:
        vals = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "rotation_error_deg": mean_of("rotation_error_deg"),
        "translation_distance": mean_of("translation_distance"),
        "translation_direction_deg": mean_of("translation_direction_deg"),
        "reprojection_rmse_px": mean_of("reprojection_rmse_px"),
        "points_used": mean_of("points_used"),
        "frames_evaluated": len(rows),
    }


def build_summary_report(summary: dict) -> str:
    rows = [
        ["Frames Evaluated", str(int(summary["frames_evaluated"]))],
        ["Avg Rotation Error (deg)", format_value(summary["rotation_error_deg"])],
        ["Avg Translation Distance", format_value(summary["translation_distance"])],
        ["Avg Translation Direction (deg)", format_value(summary["translation_direction_deg"])],
        ["Avg Reprojection RMSE (px)", format_value(summary["reprojection_rmse_px"])],
        ["Avg Points Used", format_value(summary["points_used"])],
    ]
    return render_table(rows)


def build_combined_summary_report(summary_by_camera: dict[str, dict]) -> str:
    first_summary = next(iter(summary_by_camera.values()))
    reprojection_by_camera = {
        camera_label: summary["reprojection_rmse_px"]
        for camera_label, summary in summary_by_camera.items()
    }
    points_used_by_camera = {
        camera_label: summary["points_used"]
        for camera_label, summary in summary_by_camera.items()
    }
    rows = [
        ["Frames Evaluated", str(int(first_summary["frames_evaluated"]))],
        ["Avg Rotation Error (deg)", format_value(first_summary["rotation_error_deg"])],
        ["Avg Translation Distance", format_value(first_summary["translation_distance"])],
        ["Avg Translation Direction (deg)", format_value(first_summary["translation_direction_deg"])],
    ]
    for camera_label in ["cam0", "cam2"]:
        if camera_label in reprojection_by_camera:
            rows.append([f"Avg Reprojection RMSE {camera_label} (px)", format_value(reprojection_by_camera[camera_label])])
    for camera_label in ["cam0", "cam2"]:
        if camera_label in points_used_by_camera:
            rows.append([f"Avg Points Used {camera_label}", format_value(points_used_by_camera[camera_label])])
    return render_table(rows)


def resolve_pts2d(args, frame_idx: int, camera_label: str) -> np.ndarray:
    if args.pts2d_obs_npy:
        return load_point_array(Path(args.pts2d_obs_npy), 2, "pts2d_obs", frame_idx)
    if camera_label == "cam0" and args.pts2d_cam0_npy:
        return load_point_array(Path(args.pts2d_cam0_npy), 2, "pts2d_cam0", frame_idx)
    if camera_label == "cam2" and args.pts2d_cam2_npy:
        return load_point_array(Path(args.pts2d_cam2_npy), 2, "pts2d_cam2", frame_idx)
    raise ValueError(f"No 2D observation file provided for {camera_label}")


def infer_batch_frame_count(args) -> int:
    counts = [infer_frame_count(Path(args.pts3d_npy), 3, "pts3d")]
    if args.pts2d_obs_npy:
        counts.append(infer_frame_count(Path(args.pts2d_obs_npy), 2, "pts2d_obs"))
    if args.pts2d_cam0_npy:
        counts.append(infer_frame_count(Path(args.pts2d_cam0_npy), 2, "pts2d_cam0"))
    if args.pts2d_cam2_npy:
        counts.append(infer_frame_count(Path(args.pts2d_cam2_npy), 2, "pts2d_cam2"))
    return max(counts)


def build_batch_reports(args) -> tuple[str, dict]:
    frame_count = infer_batch_frame_count(args)
    camera_labels = ["cam0", "cam2"] if args.camera == "both" else [args.camera]
    all_rows = []

    for frame_idx in range(frame_count):
        for camera_label in camera_labels:
            payload = build_payload(args, resolve_pts2d(args, frame_idx, camera_label), frame_idx=frame_idx, camera_label=camera_label)
            metrics = compute_metrics(payload)
            metrics["frame_idx"] = frame_idx
            metrics["camera"] = camera_label
            all_rows.append(metrics)

    summary_by_camera = {}
    for camera_label in camera_labels:
        camera_rows = [row for row in all_rows if row["camera"] == camera_label]
        summary = summarize_metrics(camera_rows)
        summary_by_camera[camera_label] = summary
    if len(camera_labels) == 1:
        report_text = build_summary_report(summary_by_camera[camera_labels[0]])
    else:
        report_text = build_combined_summary_report(summary_by_camera)
    return report_text, {"summary": summary_by_camera, "frames": all_rows}


def save_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "frame_idx",
        "camera",
        "rotation_error_deg",
        "translation_distance",
        "translation_direction_deg",
        "reprojection_rmse_px",
        "points_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Standalone xyzq calibration evaluator")
    parser.add_argument("--k_est", help="Single intrinsic matrix txt for one camera")
    parser.add_argument("--k_cam0", help="Intrinsic matrix for cam0")
    parser.add_argument("--k_cam2", help="Intrinsic matrix for cam2")
    parser.add_argument("--xyzq_est", required=True, help="Estimated xyzq txt: timestamp tx ty tz qx qy qz qw")
    parser.add_argument("--xyzq_aruco", required=True, help="Reference xyzq txt: timestamp tx ty tz qx qy qz qw")
    parser.add_argument("--pts3d_npy", required=True, help="pts3d npy, shape (N,3) or (F,N,3)")
    parser.add_argument("--pts2d_obs_npy", help="Generic pts2d_obs npy, shape (N,2) or (F,N,2)")
    parser.add_argument("--pts2d_cam0_npy", help="cam0 pts2d npy")
    parser.add_argument("--pts2d_cam2_npy", help="cam2 pts2d npy")
    parser.add_argument("--camera", choices=["cam0", "cam2", "both"], default="cam0", help="Which 2D observation set to evaluate")
    parser.add_argument("--frame_idx", type=int, help="0-based frame index. Omit to evaluate the whole file and average")
    parser.add_argument("--align_mode", choices=["index", "timestamp"], default="index", help="How to align xyzq rows to frame_idx")
    parser.add_argument("--match", help="Required only when align_mode=timestamp")
    parser.add_argument("--save_json", help="Optional path to save normalized input JSON")
    parser.add_argument("--csv_output", help="Optional per-frame CSV output for whole-file mode")
    parser.add_argument("--output", default="results/report.txt", help="Output report path")
    args = parser.parse_args()

    if args.align_mode == "timestamp" and not args.match:
        raise ValueError("--align_mode timestamp requires --match")

    if args.camera == "both":
        if not args.pts2d_cam0_npy or not args.pts2d_cam2_npy:
            raise ValueError("--camera both requires --pts2d_cam0_npy and --pts2d_cam2_npy")
        if not args.k_cam0 or not args.k_cam2:
            raise ValueError("--camera both requires --k_cam0 and --k_cam2")
    else:
        if not (args.k_est or args.k_cam0 or args.k_cam2):
            raise ValueError("Single-camera mode requires --k_est or the matching --k_cam0 / --k_cam2")

    if args.frame_idx is None:
        report_text, batch_payload = build_batch_reports(args)
        if args.save_json:
            save_path = Path(args.save_json)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(batch_payload, indent=2), encoding="utf-8")
        if args.csv_output:
            save_csv(Path(args.csv_output), batch_payload["frames"])
    else:
        reports: list[tuple[str, dict, str]] = []
        camera_labels = ["cam0", "cam2"] if args.camera == "both" else [args.camera]
        for camera_label in camera_labels:
            payload = build_payload(args, resolve_pts2d(args, args.frame_idx, camera_label), frame_idx=args.frame_idx, camera_label=camera_label)
            reports.append((camera_label, payload, evaluate_payload(payload)))
        if len(reports) == 1:
            report_text = reports[0][2]
        else:
            metrics_by_camera = {
                label: compute_metrics(payload)
                for label, payload, _report in reports
            }
            shared_metrics = metrics_by_camera["cam0"]
            reprojection_by_camera = {
                label: metrics["reprojection_rmse_px"]
                for label, metrics in metrics_by_camera.items()
            }
            points_used_by_camera = {
                label: metrics["points_used"]
                for label, metrics in metrics_by_camera.items()
            }
            report_text = build_combined_report(shared_metrics, reprojection_by_camera, points_used_by_camera)
        if args.save_json:
            save_path = Path(args.save_json)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payloads = {label: payload for label, payload, _report in reports}
            save_path.write_text(json.dumps(payloads if len(reports) > 1 else next(iter(payloads.values())), indent=2), encoding="utf-8")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text + "\n", encoding="utf-8")

    print(report_text)
    print(f"\n[SAVE] report -> {output_path}")
    if args.save_json:
        print(f"[SAVE] json -> {args.save_json}")
    if args.csv_output and args.frame_idx is None:
        print(f"[SAVE] csv -> {args.csv_output}")


if __name__ == "__main__":
    main()
