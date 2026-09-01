from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class StereoSample:
    name: str
    frame_idx: int
    timestamp: float
    frame1: np.ndarray
    frame2: np.ndarray
    corners1: np.ndarray
    corners2: np.ndarray
    image_size: tuple[int, int]


def progress(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _open_capture(source: Path):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open capture source: {source}")
    return cap


def _make_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def _find_chessboard(gray: np.ndarray, pattern: tuple[int, int], use_sb: bool):
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.astype(np.float32)


def _image_size(frame: np.ndarray) -> tuple[int, int]:
    return int(frame.shape[1]), int(frame.shape[0])


def _find_in_pair(
    name: str,
    frame_idx: int,
    timestamp: float,
    frame1: np.ndarray,
    frame2: np.ndarray,
    pattern: tuple[int, int],
    use_sb: bool,
) -> StereoSample | None:
    if frame1 is None or frame2 is None:
        return None
    size1 = _image_size(frame1)
    size2 = _image_size(frame2)
    if size1 != size2:
        raise SystemExit(
            f"Image sizes differ for pair {name}: cam1={size1}, cam2={size2}. "
            "Use the same recording resolution for both cameras."
        )

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    found1, corners1 = _find_chessboard(gray1, pattern, use_sb)
    found2, corners2 = _find_chessboard(gray2, pattern, use_sb)
    if not (found1 and found2):
        return None

    return StereoSample(
        name=name,
        frame_idx=int(frame_idx),
        timestamp=float(timestamp),
        frame1=frame1,
        frame2=frame2,
        corners1=corners1,
        corners2=corners2,
        image_size=size1,
    )


def _iter_video_pairs(
    cam1_video: Path,
    cam2_video: Path,
    frame_step: int,
    start_frame: int,
    max_frames: int,
) -> Iterable[tuple[str, int, float, np.ndarray, np.ndarray]]:
    cap1 = _open_capture(cam1_video)
    cap2 = _open_capture(cam2_video)
    try:
        fps = float(cap1.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = float(cap2.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        if start_frame > 0:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        frame_idx = int(start_frame)
        yielded = 0
        step = max(1, int(frame_step))
        while True:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()
            if not (ok1 and ok2):
                break
            if (frame_idx - start_frame) % step == 0:
                yield f"frame_{frame_idx:06d}", frame_idx, frame_idx / fps, frame1, frame2
                yielded += 1
                if max_frames > 0 and yielded >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap1.release()
        cap2.release()


def _stack_preview(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    if h1 != h2:
        scale = h1 / max(1, h2)
        frame2 = cv2.resize(frame2, (int(w2 * scale), h1))
    return cv2.hconcat([frame1, frame2])


def _collect_from_videos(args, pattern: tuple[int, int]) -> list[StereoSample]:
    pair_iter = _iter_video_pairs(
        Path(args.cam1_video),
        Path(args.cam2_video),
        args.frame_step,
        args.start_frame,
        args.max_frames,
    )
    samples: list[StereoSample] = []
    debug_dir = Path(args.debug_draw_dir) if args.debug_draw_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for name, frame_idx, timestamp, frame1, frame2 in pair_iter:
        total += 1
        sample = _find_in_pair(name, frame_idx, timestamp, frame1, frame2, pattern, args.use_sb)
        if sample is None:
            if args.verbose:
                progress(f"rejected {name}: chessboard not found in both cameras")
            continue
        samples.append(sample)
        if debug_dir:
            d1 = sample.frame1.copy()
            d2 = sample.frame2.copy()
            cv2.drawChessboardCorners(d1, pattern, sample.corners1, True)
            cv2.drawChessboardCorners(d2, pattern, sample.corners2, True)
            cv2.imwrite(str(debug_dir / f"{len(samples):04d}_{name}.jpg"), _stack_preview(d1, d2))
        if len(samples) >= int(args.max_samples):
            break

    progress(f"source=videos, checked_pairs={total}, accepted_pairs={len(samples)}")
    return samples


def _load_k(path: str, label: str) -> np.ndarray:
    path_obj = Path(path)
    if not path_obj.exists():
        candidates = ", ".join(str(p) for p in sorted(path_obj.parent.glob("K*.txt")))
        hint = f" Available K files: {candidates}" if candidates else ""
        raise SystemExit(f"{label} file not found: {path}.{hint}")
    K = np.loadtxt(path_obj, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise SystemExit(f"{label} must be a 3x3 matrix, got shape {K.shape}: {path}")
    return K


def _load_dist(path: str, label: str) -> np.ndarray:
    path_obj = Path(path)
    if not path_obj.exists():
        candidates = ", ".join(str(p) for p in sorted(path_obj.parent.glob("dist*.txt")))
        hint = f" Available dist files: {candidates}" if candidates else ""
        raise SystemExit(f"{label} file not found: {path}.{hint}")
    dist = np.loadtxt(path_obj, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.shape[0] < 4:
        raise SystemExit(f"{label} must contain at least 4 values, got {dist.shape[0]}: {path}")
    return dist


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), np.finfo(np.float64).eps)


def _solve_board_pose(objp: np.ndarray, corners: np.ndarray, K: np.ndarray, dist: np.ndarray, label: str):
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(objp, dtype=np.float32),
        np.asarray(corners, dtype=np.float32),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError(f"solvePnP failed for {label}")
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.reshape(3).astype(np.float64)


def _relative_rt_from_board_poses(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray):
    R_rel = np.asarray(R2, dtype=np.float64) @ np.asarray(R1, dtype=np.float64).T
    t_rel = np.asarray(t2, dtype=np.float64).reshape(3) - R_rel @ np.asarray(t1, dtype=np.float64).reshape(3)
    return R_rel, t_rel


def _pose_reprojection_stats(objp: np.ndarray, corners: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    projected, _ = cv2.projectPoints(
        np.asarray(objp, dtype=np.float32),
        rvec,
        np.asarray(t, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    observed = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    err = np.linalg.norm(projected - observed, axis=1)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
    }


def _estimate_per_frame_rt(samples: list[StereoSample], objp: np.ndarray, K1, dist1, K2, dist2):
    rows = []
    reproj_rows = []
    skipped = 0
    for sample in samples:
        try:
            R1, t1 = _solve_board_pose(objp, sample.corners1, K1, dist1, f"{sample.name} cam1")
            R2, t2 = _solve_board_pose(objp, sample.corners2, K2, dist2, f"{sample.name} cam2")
        except RuntimeError as exc:
            skipped += 1
            progress(f"skipped {sample.name}: {exc}")
            continue

        R_rel, t_rel = _relative_rt_from_board_poses(R1, t1, R2, t2)
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(R_rel)
        tx, ty, tz = t_rel
        rows.append((sample.timestamp, tx, ty, tz, qx, qy, qz, qw))

        err1 = _pose_reprojection_stats(objp, sample.corners1, R1, t1, K1, dist1)
        err2 = _pose_reprojection_stats(objp, sample.corners2, R2, t2, K2, dist2)
        reproj_rows.append(
            {
                "frame": sample.frame_idx,
                "timestamp": sample.timestamp,
                "name": sample.name,
                "cam1_rmse": err1["rmse"],
                "cam1_mean": err1["mean"],
                "cam1_median": err1["median"],
                "cam1_max": err1["max"],
                "cam2_rmse": err2["rmse"],
                "cam2_mean": err2["mean"],
                "cam2_median": err2["median"],
                "cam2_max": err2["max"],
                "both_rmse": float(np.sqrt((err1["rmse"] ** 2 + err2["rmse"] ** 2) / 2.0)),
            }
        )

    if not rows:
        raise SystemExit("No per-frame RT rows estimated.")
    if skipped:
        progress(f"per-frame solvePnP skipped={skipped}")
    return rows, reproj_rows


def _write_rt_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in rows:
            timestamp, tx, ty, tz, qx, qy, qz, qw = row
            f.write(
                f"{timestamp:.9f} {tx:.10f} {ty:.10f} {tz:.10f} "
                f"{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}\n"
            )


def _print_reprojection_summary(reproj_rows) -> None:
    if not reproj_rows:
        return
    vals = np.asarray([row["both_rmse"] for row in reproj_rows], dtype=np.float64)
    print(
        "[RESULT] reprojection both_rmse px: "
        f"mean={np.mean(vals):.6f}, median={np.median(vals):.6f}, max={np.max(vals):.6f}",
        flush=True,
    )


def _resolve_out_rt(args) -> Path:
    if args.out_rt:
        return Path(args.out_rt)
    if args.tag:
        tag_path = Path(args.tag)
        if tag_path.parent != Path(".") or tag_path.suffix:
            return tag_path if tag_path.suffix else tag_path.with_suffix(".txt")
        return Path(args.out_dir) / f"rt_chessboard_{args.tag}.txt"
    return Path(args.out_dir) / f"rt_chessboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def _write_outputs(args, rows, reproj_rows) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rt = _resolve_out_rt(args)

    _write_rt_rows(out_rt, rows)
    progress(f"RT saved -> {out_rt}")
    print(f"[RESULT] per-frame RT rows: {len(rows)}", flush=True)
    _print_reprojection_summary(reproj_rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stereo chessboard calibration for the 3D pipeline. "
            "Reads two chessboard videos plus existing K/dist files, then outputs "
            "per-frame RT quaternion rows: timestamp tx ty tz qx qy qz qw."
        )
    )

    parser.add_argument("--cam1_video", required=True, help="cam1 chessboard video")
    parser.add_argument("--cam2_video", required=True, help="cam2 chessboard video")
    parser.add_argument("--k1", required=True, help="input cam1 intrinsic matrix txt")
    parser.add_argument("--k2", required=True, help="input cam2 intrinsic matrix txt")
    parser.add_argument("--dist1", required=True, help="input cam1 distortion txt")
    parser.add_argument("--dist2", required=True, help="input cam2 distortion txt")
    parser.add_argument("--frame_step", type=int, default=5, help="sample every N video frames")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0, help="max sampled video frames checked; 0 means all")

    parser.add_argument("--cols", type=int, default=9, help="inner chessboard corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner chessboard corners per column")
    parser.add_argument("--square_size", type=float, default=2.0, help="square size, e.g. cm")
    parser.add_argument("--min_samples", type=int, default=12)
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--use_sb", action="store_true", default=True, help="use findChessboardCornersSB when available")
    parser.add_argument("--no_use_sb", action="store_false", dest="use_sb")

    parser.add_argument("--out_dir", default=str(Path("output")))
    parser.add_argument("--tag", default="", help="tag name, or output path stem like .\\output\\people_d_chessboard")
    parser.add_argument("--out_rt", default="")
    parser.add_argument("--debug_draw_dir", default="", help="optional folder for accepted pair overlays")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pattern = (int(args.cols), int(args.rows))
    objp = _make_object_points(args.cols, args.rows, args.square_size)

    progress(
        f"pattern={args.cols}x{args.rows} inner corners, square_size={args.square_size}, "
        f"min_samples={args.min_samples}"
    )
    samples = _collect_from_videos(args, pattern)
    if len(samples) < int(args.min_samples):
        raise SystemExit(
            f"Need at least {args.min_samples} accepted stereo pairs, got {len(samples)}. "
            "Record more board poses or lower --min_samples."
        )

    K1 = _load_k(args.k1, "K1")
    K2 = _load_k(args.k2, "K2")
    dist1 = _load_dist(args.dist1, "dist1")
    dist2 = _load_dist(args.dist2, "dist2")

    progress("estimating per-frame cam1->cam2 RT from chessboard solvePnP")
    rows, reproj_rows = _estimate_per_frame_rt(samples, objp, K1, dist1, K2, dist2)
    _write_outputs(args, rows, reproj_rows)


if __name__ == "__main__":
    main()
