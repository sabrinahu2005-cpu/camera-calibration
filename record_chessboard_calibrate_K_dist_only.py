from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def _backend_id(name: str) -> int:
    name = name.lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def _open_camera(index: int, backend: str, width: int, height: int, fps: float):
    cap = cv2.VideoCapture(index, _backend_id(backend))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index={index}")
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
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


def _mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist) -> float:
    total_sq = 0.0
    count = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_sq += float(err * err)
        count += int(len(projected))
    if count == 0:
        return float("nan")
    return float(np.sqrt(total_sq / count))


def _calibrate(objpoints, imgpoints, image_size):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_err = _mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)
    return float(ret), K, dist.reshape(-1), mean_err


def _save_k_dist(out_k: str, out_dist: str, K: np.ndarray, dist: np.ndarray, dist_count: int):
    out_k_path = Path(out_k)
    out_dist_path = Path(out_dist)
    out_k_path.parent.mkdir(parents=True, exist_ok=True)
    out_dist_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_k_path, K, fmt="%.8f")

    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.shape[0] < dist_count:
        dist = np.pad(dist, (0, dist_count - dist.shape[0]))
    dist_to_save = dist[:dist_count]
    np.savetxt(out_dist_path, dist_to_save.reshape(1, -1), fmt="%.8f")

    print(f"[SAVE] K -> {out_k_path}", flush=True)
    print(f"[SAVE] dist -> {out_dist_path}", flush=True)
    print("[RESULT] K:", flush=True)
    print(K, flush=True)
    print("[RESULT] dist:", " ".join(f"{x:.8f}" for x in dist_to_save), flush=True)


def _show_preview(window_name: str, frame: np.ndarray) -> bool:
    # 某些 OpenCV 安裝版本沒有編入 HighGUI，這時 imshow 會直接丟例外。
    try:
        cv2.imshow(window_name, frame)
        return True
    except cv2.error:
        return False


def _safe_destroy_windows():
    # 沒有 HighGUI 時，destroyAllWindows 也會失敗；這裡直接忽略即可。
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live chessboard camera calibration. Output only K.txt and dist.txt."
    )
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)

    parser.add_argument("--cols", type=int, default=9, help="inner chessboard corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner chessboard corners per column")
    parser.add_argument("--square_size", type=float, default=2.0, help="square size, e.g. cm or mm")

    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means until q/c/max_samples")
    parser.add_argument("--sample_every", type=float, default=0.5, help="seconds between automatic accepted samples")
    parser.add_argument("--min_samples", type=int, default=15)
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--no_preview", action="store_true")
    parser.add_argument("--use_sb", action="store_true", default=True)
    parser.add_argument("--no_use_sb", action="store_false", dest="use_sb")

    parser.add_argument("--out_k", default="data/K.txt", help="output intrinsic matrix txt")
    parser.add_argument("--out_dist", default="data/dist.txt", help="output distortion txt")
    parser.add_argument("--dist_count", type=int, choices=[4, 5], default=5, help="save 4 values k1 k2 p1 p2, or 5 values k1 k2 p1 p2 k3")
    args = parser.parse_args()

    cap = _open_camera(args.cam, args.backend, args.width, args.height, args.fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)

    pattern = (int(args.cols), int(args.rows))
    objp = _make_object_points(args.cols, args.rows, args.square_size)
    objpoints = []
    imgpoints = []

    last_accept_t = -1e9
    frame_idx = 0
    start_t = time.time()
    last_found = False
    preview_enabled = not args.no_preview
    preview_warned = False
    preview_window_name = "Chessboard intrinsic calibration"

    if preview_enabled:
        try:
            cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_window_name, max(640, actual_w), max(360, actual_h))
            print(f"[INFO] preview window created: {preview_window_name}", flush=True)
        except cv2.error:
            preview_enabled = False
            print(
                "[WARN] OpenCV could not create a preview window; continuing without GUI preview.",
                flush=True,
            )

    print(f"[INFO] camera index={args.cam} backend={args.backend}", flush=True)
    print(f"[INFO] actual={actual_w}x{actual_h}@{actual_fps:g}", flush=True)
    print(f"[INFO] pattern inner corners={args.cols}x{args.rows}, square_size={args.square_size}", flush=True)
    print("[KEYS] q/c=calibrate and save K/dist, s=force sample, ESC=quit without calibration", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] failed to read frame", flush=True)
                break

            now = time.time()
            elapsed = now - start_t
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = _find_chessboard(gray, pattern, args.use_sb)
            last_found = bool(found)

            key = 255
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, found)
            status = "FOUND" if found else "NO BOARD"
            color = (0, 220, 0) if found else (0, 0, 255)
            cv2.putText(display, f"{status} samples={len(objpoints)} frame={frame_idx}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, "q/c calibrate, s sample, ESC quit", (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            if preview_enabled:
                shown = _show_preview(preview_window_name, display)
                if shown:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    preview_enabled = False
                    if not preview_warned:
                        print(
                            "[WARN] OpenCV preview window is unavailable in this build; "
                            "continuing without GUI preview. Use a non-headless OpenCV install if you want a window.",
                            flush=True,
                        )
                        preview_warned = True

            should_auto_sample = (
                found
                and len(objpoints) < args.max_samples
                and (now - last_accept_t) >= float(args.sample_every)
            )
            force_sample = key == ord("s")

            if found and (should_auto_sample or force_sample):
                objpoints.append(objp.copy())
                imgpoints.append(corners.copy())
                last_accept_t = now
                print(f"[SAMPLE] #{len(objpoints)} frame={frame_idx} elapsed={elapsed:.1f}s", flush=True)
            elif force_sample and not found:
                print(f"[SAMPLE] rejected frame={frame_idx}: chessboard not found", flush=True)

            if key in (ord("q"), ord("c")):
                break
            if key == 27:
                objpoints = []
                imgpoints = []
                print("[INFO] ESC pressed; quit without calibration", flush=True)
                break
            if args.duration > 0 and elapsed >= float(args.duration):
                break
            if len(objpoints) >= int(args.max_samples):
                break

            frame_idx += 1
    finally:
        cap.release()
        _safe_destroy_windows()

    print(f"[INFO] last_detection={last_found}, accepted_samples={len(objpoints)}", flush=True)
    if len(objpoints) < int(args.min_samples):
        raise SystemExit(
            f"Need at least {args.min_samples} accepted samples, got {len(objpoints)}. "
            "Record more angles/distances or lower --min_samples."
        )

    image_size = (actual_w, actual_h)
    ret, K, dist, mean_err = _calibrate(objpoints, imgpoints, image_size)
    _save_k_dist(args.out_k, args.out_dist, K, dist, args.dist_count)
    print(f"[RESULT] OpenCV RMS={ret:.6f}, mean reprojection={mean_err:.6f}px", flush=True)


if __name__ == "__main__":
    main()
