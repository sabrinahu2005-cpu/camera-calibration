from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_tool


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BASE_DIR / "one_program_test"
TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# offline 測試模式輸出重導向：所有輸出檔案統一存入 test/，檔名加前綴。
# 伺服器/樹莓派即時模式不設定此值，行為不受影響。
OUTPUT_PREFIX = ""


def test_output_dir() -> Path:
    d = Path("test")
    d.mkdir(parents=True, exist_ok=True)
    return d

MP_TO_COCO17 = [
    (0, 0, "nose"),
    (1, 2, "left_eye"),
    (2, 5, "right_eye"),
    (3, 7, "left_ear"),
    (4, 8, "right_ear"),
    (5, 11, "left_shoulder"),
    (6, 12, "right_shoulder"),
    (7, 13, "left_elbow"),
    (8, 14, "right_elbow"),
    (9, 15, "left_wrist"),
    (10, 16, "right_wrist"),
    (11, 23, "left_hip"),
    (12, 24, "right_hip"),
    (13, 25, "left_knee"),
    (14, 26, "right_knee"),
    (15, 27, "left_ankle"),
    (16, 28, "right_ankle"),
]

SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 11),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

# ===== IMU framing (same direct capture logic as pose_fin.py) =====
FRAME_HEAD = 0xFC
FRAME_LEN = 120
PAYLOAD_GYRO_OFF = 0
PAYLOAD_ACCEL_OFF = 68

ACC_ABS_MAX = 80.0
GYRO_ABS_MAX = 30.0
ACC_NORM_MAX = 80.0
GYRO_NORM_MAX = 50.0


def import_websockets():
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: websockets\n"
            "Install it on both PC and Pi with: python -m pip install websockets"
        ) from exc
    return websockets


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "raw": root / "raw",
        "json": root / "json",
        "video": root / "video",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sanitize_tag(tag: str) -> str:
    cleaned = TAG_SAFE_RE.sub("_", str(tag).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or time.strftime("%Y%m%d_%H%M%S")


def output_paths(root: Path, cam_id: str, tag: str) -> tuple[Path, Path, Path]:
    tag = sanitize_tag(tag)
    paths = ensure_dirs(root)
    raw_video = paths["raw"] / f"{cam_id}_{tag}.mp4"
    pose_json = paths["json"] / f"{cam_id}_{tag}.jsonl"
    pose_video = paths["video"] / f"{cam_id}_{tag}.mp4"
    return raw_video, pose_json, pose_video


def discover_v4l2_capture_indices() -> list[int]:
    dev_root = Path("/dev")
    if not dev_root.exists():
        return []
    indices = []
    for node in sorted(dev_root.glob("video*")):
        match = re.fullmatch(r"video(\d+)", node.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > 20:
            continue
        if shutil.which("v4l2-ctl"):
            proc = subprocess.run(
                ["v4l2-ctl", "-d", str(node), "--all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if "video capture" not in (proc.stdout or "").lower():
                continue
        indices.append(idx)
    return indices


def camera_candidates(cam_index: int) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    seen = set()

    def add(index: int, backend: int, label: str) -> None:
        key = (int(index), int(backend))
        if key not in seen:
            seen.add(key)
            candidates.append((int(index), int(backend), label))

    add(cam_index, cv2.CAP_ANY, f"index={cam_index} via ANY")
    if hasattr(cv2, "CAP_V4L2"):
        add(cam_index, cv2.CAP_V4L2, f"index={cam_index} via V4L2")
        for idx in discover_v4l2_capture_indices():
            add(idx, cv2.CAP_V4L2, f"index={idx} via V4L2")
    return candidates


def open_camera(cam_index: int, fps: float, width: int, height: int):
    tried = []
    for index, backend, label in camera_candidates(cam_index):
        tried.append(label)
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FPS, fps)
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[camera] opened {label}", flush=True)
            return cap, frame
        cap.release()
    tried_text = ", ".join(tried) if tried else f"index={cam_index}"
    raise RuntimeError(f"Cannot open camera. Tried: {tried_text}")


def create_writer(path: Path, fps: float, frame_shape) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


def draw_pose(frame_bgr, landmarks, visibility_threshold: float):
    height, width = frame_bgr.shape[:2]
    annot = frame_bgr.copy()
    keypoints17 = []
    coco_points: dict[int, tuple[float, float, float]] = {}

    if not landmarks:
        return annot, keypoints17

    lms = landmarks.landmark
    for coco_id, mp_idx, name in MP_TO_COCO17:
        lm = lms[mp_idx]
        x = float(lm.x * width)
        y = float(lm.y * height)
        vis = float(lm.visibility)
        coco_points[coco_id] = (x, y, vis)
        keypoints17.append({"id": coco_id, "name": name, "x": x, "y": y, "visibility": vis})

    for a, b in SKELETON_EDGES:
        if a not in coco_points or b not in coco_points:
            continue
        x1, y1, v1 = coco_points[a]
        x2, y2, v2 = coco_points[b]
        if v1 >= visibility_threshold and v2 >= visibility_threshold:
            cv2.line(annot, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)

    for x, y, vis in coco_points.values():
        if vis >= visibility_threshold:
            cv2.circle(annot, (int(x), int(y)), 4, (0, 255, 0), -1)

    return annot, keypoints17


def f32le(b: bytes) -> float:
    return struct.unpack("<f", b)[0]


def read_exact(ser, n: int) -> bytes:
    data = ser.read(n)
    if len(data) != n:
        raise TimeoutError(f"timeout while reading {n} bytes (got {len(data)})")
    return data


def sync_to_head(ser) -> None:
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("timeout waiting for frame head (0xFC)")
        if b[0] == FRAME_HEAD:
            return


def read_one_frame(ser) -> bytes:
    sync_to_head(ser)
    rest = read_exact(ser, FRAME_LEN - 1)
    return bytes([FRAME_HEAD]) + rest


def _finite(v: float) -> bool:
    return (v == v) and (v != float("inf")) and (v != float("-inf"))


def _norm3(x: float, y: float, z: float) -> float:
    return (x * x + y * y + z * z) ** 0.5


def remap_imu_to_camera(gx: float, gy: float, gz: float, ax: float, ay: float, az: float):
    """Same mounting remap as pose_fin.py."""
    gx2 = gx
    gy2 = -gz
    gz2 = gy

    ax2 = ax
    ay2 = -az
    az2 = ay
    return gx2, gy2, gz2, ax2, ay2, az2


def parse_imu_from_frame(frame: bytes):
    if len(frame) != FRAME_LEN:
        return None

    payload = frame[7:-1]
    if len(payload) < PAYLOAD_ACCEL_OFF + 12:
        return None

    gx = f32le(payload[PAYLOAD_GYRO_OFF:PAYLOAD_GYRO_OFF + 4])
    gy = f32le(payload[PAYLOAD_GYRO_OFF + 4:PAYLOAD_GYRO_OFF + 8])
    gz = f32le(payload[PAYLOAD_GYRO_OFF + 8:PAYLOAD_GYRO_OFF + 12])

    ax = f32le(payload[PAYLOAD_ACCEL_OFF:PAYLOAD_ACCEL_OFF + 4])
    ay = f32le(payload[PAYLOAD_ACCEL_OFF + 4:PAYLOAD_ACCEL_OFF + 8])
    az = f32le(payload[PAYLOAD_ACCEL_OFF + 8:PAYLOAD_ACCEL_OFF + 12])

    for v in (gx, gy, gz, ax, ay, az):
        if not _finite(v):
            return None

    if (abs(ax) > ACC_ABS_MAX or abs(ay) > ACC_ABS_MAX or abs(az) > ACC_ABS_MAX):
        return None
    if (abs(gx) > GYRO_ABS_MAX or abs(gy) > GYRO_ABS_MAX or abs(gz) > GYRO_ABS_MAX):
        return None

    if _norm3(ax, ay, az) > ACC_NORM_MAX:
        return None
    if _norm3(gx, gy, gz) > GYRO_NORM_MAX:
        return None

    gx, gy, gz, ax, ay, az = remap_imu_to_camera(gx, gy, gz, ax, ay, az)

    for v in (gx, gy, gz, ax, ay, az):
        if not _finite(v):
            return None

    return gx, gy, gz, ax, ay, az


def imu_thread_worker(
    port: str,
    baud: int,
    timeout: float,
    out_queue: queue.Queue,
    stop_evt: threading.Event,
):
    """Read direct IMU frames from serial and push samples into a queue."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pyserial (required for IMU fusion)") from exc

    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
    )

    try:
        while not stop_evt.is_set():
            try:
                frame = read_one_frame(ser)
                imu = parse_imu_from_frame(frame)
                if imu is None:
                    continue
                gx, gy, gz, ax, ay, az = imu
                out_queue.put({
                    "timestamp_wall": time.time(),
                    "gx": gx,
                    "gy": gy,
                    "gz": gz,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                })
            except TimeoutError:
                continue
            except Exception:
                continue
    finally:
        try:
            ser.close()
        except Exception:
            pass


class CapturePoseWorker:
    def __init__(
        self,
        *,
        cam_id: str,
        tag: str,
        root: Path,
        cam_index: int,
        fps: float,
        width: int,
        height: int,
        vis: float,
        start_at_wall: float,
        secs: float,
        stop_event: threading.Event,
        record_queue: queue.Queue | None,
        start_gate: threading.Event | None = None,
        start_at_box: dict[str, float] | None = None,
    ):
        self.cam_id = cam_id
        self.tag = tag
        self.root = root
        self.cam_index = cam_index
        self.fps = fps
        self.width = width
        self.height = height
        self.vis = vis
        self.start_at_wall = start_at_wall
        self.secs = secs
        self.stop_event = stop_event
        self.record_queue = record_queue
        self.start_gate = start_gate
        self.start_at_box = start_at_box
        self.ready_event = threading.Event()
        self.done_event = threading.Event()
        self.error: BaseException | None = None
        self.ready_payload: dict[str, Any] | None = None
        self.raw_video, self.pose_json, self.pose_video = output_paths(root, cam_id, tag)
        self.thread = threading.Thread(target=self._run, name=f"{cam_id}-capture-pose", daemon=True)

    def start(self):
        self.thread.start()

    def join(self, timeout: float | None = None):
        self.thread.join(timeout=timeout)

    def _run(self):
        cap = None
        raw_writer = None
        pose_writer = None
        try:
            import mediapipe as mp

            cap, first_frame = open_camera(self.cam_index, self.fps, self.width, self.height)
            raw_writer = create_writer(self.raw_video, self.fps, first_frame.shape)
            pose_writer = create_writer(self.pose_video, self.fps, first_frame.shape)
            actual_h, actual_w = first_frame.shape[:2]
            self.ready_payload = {
                "type": "ready",
                "cam": self.cam_id,
                "width": actual_w,
                "height": actual_h,
                "fps": self.fps,
                "raw_video": str(self.raw_video),
                "pose_json": str(self.pose_json),
                "pose_video": str(self.pose_video),
            }
            self.ready_event.set()

            if self.start_gate is not None:
                while not self.start_gate.is_set() and not self.stop_event.is_set():
                    time.sleep(0.01)
                if self.start_at_box is not None:
                    self.start_at_wall = float(self.start_at_box["start_at_wall"])

            delay = self.start_at_wall - time.time()
            if delay > 0:
                deadline = time.time() + delay
                while time.time() < deadline and not self.stop_event.is_set():
                    time.sleep(min(0.01, deadline - time.time()))

            target_frames = None if self.secs <= 0 else max(1, int(round(self.secs * self.fps)))
            dt = 1.0 / self.fps
            frame_idx = 0
            last_frame = first_frame
            mp_pose = mp.solutions.pose

            with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
                with self.pose_json.open("w", encoding="utf-8") as fjson:
                    start_wall = self.start_at_wall
                    start_mono = time.monotonic()
                    while not self.stop_event.is_set():
                        if target_frames is not None and frame_idx >= target_frames:
                            break

                        slot = start_mono + frame_idx * dt
                        while time.monotonic() < slot and not self.stop_event.is_set():
                            time.sleep(min(0.005, slot - time.monotonic()))

                        ok, frame = cap.read()
                        if not ok or frame is None:
                            frame = last_frame.copy()
                        else:
                            last_frame = frame

                        raw_writer.write(frame)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)
                        annot, keypoints17 = draw_pose(frame, result.pose_landmarks, self.vis)
                        pose_writer.write(annot)

                        timestamp = frame_idx * dt
                        record = {
                            "cam": self.cam_id,
                            "tag": self.tag,
                            "frame": frame_idx,
                            "timestamp": round(timestamp, 9),
                            "timestamp_wall": round(start_wall + timestamp, 9),
                            "keypoints17": keypoints17,
                        }
                        line = json.dumps(record, ensure_ascii=False)
                        fjson.write(line + "\n")
                        fjson.flush()
                        if self.record_queue is not None:
                            self.record_queue.put(record)
                        frame_idx += 1
        except BaseException as exc:
            self.error = exc
            self.ready_event.set()
        finally:
            if cap is not None:
                cap.release()
            if raw_writer is not None:
                raw_writer.release()
            if pose_writer is not None:
                pose_writer.release()
            self.done_event.set()


def load_intrinsics(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def keypoint_dict(record: dict[str, Any], vis_thresh: float) -> dict[int, tuple[float, float, float]]:
    out = {}
    for item in record.get("keypoints17") or []:
        try:
            kid = int(item["id"])
            x = float(item["x"])
            y = float(item["y"])
            vis = float(item["visibility"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y) and vis >= vis_thresh:
            out[kid] = (x, y, vis)
    return out


def matched_points(cam0_record: dict[str, Any], cam2_record: dict[str, Any], vis_thresh: float):
    left = keypoint_dict(cam0_record, vis_thresh)
    right = keypoint_dict(cam2_record, vis_thresh)
    pts1, pts2, ids = [], [], []
    for kid in sorted(set(left) & set(right)):
        pts1.append([left[kid][0], left[kid][1]])
        pts2.append([right[kid][0], right[kid][1]])
        ids.append(kid)
    return pts1, pts2, ids


def raw_keypoint_dict(record: dict[str, Any]) -> dict[int, tuple[float, float, float]]:
    out = {}
    for item in record.get("keypoints17") or []:
        try:
            kid = int(item["id"])
            out[kid] = (float(item["x"]), float(item["y"]), float(item["visibility"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def make_pairs17(cam0_record: dict[str, Any], cam2_record: dict[str, Any]):
    left = raw_keypoint_dict(cam0_record)
    right = raw_keypoint_dict(cam2_record)
    pairs = []
    for kid in range(17):
        lx, ly, lv = left.get(kid, (float("nan"), float("nan"), float("nan")))
        rx, ry, rv = right.get(kid, (float("nan"), float("nan"), float("nan")))
        pairs.append([[lx, ly], [rx, ry], lv, rv])
    return pairs


def compute_reprojection_error(
    pts1,
    pts2,
    K_cam0: np.ndarray,
    K_cam2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
):
    pts1_np = np.asarray(pts1, dtype=np.float64)
    pts2_np = np.asarray(pts2, dtype=np.float64)
    if pts1_np.shape[0] < 2:
        return None

    p1 = K_cam0 @ np.hstack([np.eye(3), np.zeros((3, 1), dtype=np.float64)])
    p2 = K_cam2 @ np.hstack([R, t.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(p1, p2, pts1_np.T, pts2_np.T)
    w = pts4[3]
    valid = np.abs(w) > 1e-9
    if not np.any(valid):
        return None

    pts3 = (pts4[:3, valid] / w[valid]).T
    obs1 = pts1_np[valid]
    obs2 = pts2_np[valid]

    proj1, _ = cv2.projectPoints(
        pts3,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K_cam0,
        None,
    )
    rvec2, _ = cv2.Rodrigues(R)
    proj2, _ = cv2.projectPoints(pts3, rvec2, t.reshape(3, 1), K_cam2, None)
    proj1 = proj1.reshape(-1, 2)
    proj2 = proj2.reshape(-1, 2)

    err1 = np.linalg.norm(proj1 - obs1, axis=1)
    err2 = np.linalg.norm(proj2 - obs2, axis=1)
    rmse1 = float(np.sqrt(np.mean(err1**2)))
    rmse2 = float(np.sqrt(np.mean(err2**2)))
    rmse_mean = float(np.sqrt(np.mean(np.concatenate([err1, err2]) ** 2)))
    return {
        "reproj_rmse_cam0": rmse1,
        "reproj_rmse_cam2": rmse2,
        "reproj_rmse_mean": rmse_mean,
        "reproj_points": int(pts3.shape[0]),
    }


def stats_min_max_mean(values: list[float] | np.ndarray) -> dict[str, float | str]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": "", "max": "", "mean": ""}
    return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}


def load_xyzq_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_rt_txt(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                ts, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
            except ValueError:
                continue
            rows.append({"timestamp": ts, "x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    return rows


def write_rt_txt_from_xyzq_jsonl(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in load_xyzq_jsonl(src):
            try:
                f.write(
                    f"{float(row['timestamp']):.9f} "
                    f"{float(row['x']):.10f} {float(row['y']):.10f} {float(row['z']):.10f} "
                    f"{float(row['qx']):.10f} {float(row['qy']):.10f} {float(row['qz']):.10f} {float(row['qw']):.10f}\n"
                )
            except (KeyError, TypeError, ValueError):
                continue


def write_match_txt_from_jsonl(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs = row.get("pairs17")
            if pairs is None:
                continue
            fout.write(f"{float(row['timestamp']):.9f} {json.dumps(pairs, ensure_ascii=False)}\n")


def reproj_summary_from_xyzq(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping = {
        "cam0": "reproj_rmse_cam0",
        "cam2": "reproj_rmse_cam2",
        "combined": "reproj_rmse_mean",
    }
    for label, key in mapping.items():
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        st = stats_min_max_mean(vals)
        out[f"{prefix}_{label}_reproj_min"] = st["min"]
        out[f"{prefix}_{label}_reproj_max"] = st["max"]
        out[f"{prefix}_{label}_reproj_mean"] = st["mean"]
    return out


def quat_rotation_error_deg(q1, q2) -> float:
    r1 = R_tool.from_quat(np.asarray(q1, dtype=np.float64))
    r2 = R_tool.from_quat(np.asarray(q2, dtype=np.float64))
    return float((r2.inv() * r1).magnitude() * 180.0 / math.pi)


def normalized_translation_error(t1, t2) -> float:
    a = np.asarray(t1, dtype=np.float64).reshape(3)
    b = np.asarray(t2, dtype=np.float64).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(a / na - b / nb))


def compare_pose_series(human: list[dict[str, Any]], ref: list[dict[str, Any]], tolerance: float, prefix: str):
    out: dict[str, Any] = {}
    if not human or not ref:
        return out
    ref_sorted = sorted(ref, key=lambda r: float(r["timestamp"]))
    ref_ts = np.asarray([float(r["timestamp"]) for r in ref_sorted], dtype=np.float64)
    t_errs = []
    r_errs = []
    for h in human:
        try:
            ts = float(h["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        idx = int(np.searchsorted(ref_ts, ts))
        candidates = []
        if idx < len(ref_sorted):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = min(candidates, key=lambda i: abs(ref_ts[i] - ts), default=None)
        if best is None or abs(ref_ts[best] - ts) > tolerance:
            continue
        r = ref_sorted[best]
        try:
            t_errs.append(normalized_translation_error([h["x"], h["y"], h["z"]], [r["x"], r["y"], r["z"]]))
            r_errs.append(quat_rotation_error_deg([h["qx"], h["qy"], h["qz"], h["qw"]], [r["qx"], r["qy"], r["qz"], r["qw"]]))
        except (KeyError, TypeError, ValueError):
            continue
    t_st = stats_min_max_mean(t_errs)
    r_st = stats_min_max_mean(r_errs)
    out[f"{prefix}_matched_count"] = int(np.isfinite(np.asarray(t_errs, dtype=float)).sum())
    out[f"{prefix}_translation_norm_error_min"] = t_st["min"]
    out[f"{prefix}_translation_norm_error_max"] = t_st["max"]
    out[f"{prefix}_translation_norm_error_mean"] = t_st["mean"]
    out[f"{prefix}_rotation_error_deg_min"] = r_st["min"]
    out[f"{prefix}_rotation_error_deg_max"] = r_st["max"]
    out[f"{prefix}_rotation_error_deg_mean"] = r_st["mean"]
    return out


def write_summary_csv(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    existing_rows.append({k: row.get(k, "") for k in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def split_essential_candidates(E: np.ndarray) -> list[np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return [E[i : i + 3, :] for i in range(0, E.shape[0], 3)]
    if E.ndim == 3 and E.shape[1:] == (3, 3):
        return [E[i] for i in range(E.shape[0])]
    return []


def estimate_xyzq(pts1, pts2, K_cam0: np.ndarray, K_cam2: np.ndarray, min_pairs: int):
    if len(pts1) < min_pairs:
        return None

    pts1_np = np.expand_dims(np.asarray(pts1, dtype=np.float64), axis=1)
    pts2_np = np.expand_dims(np.asarray(pts2, dtype=np.float64), axis=1)
    pts1_norm = cv2.undistortPoints(pts1_np, K_cam0, None)
    pts2_norm = cv2.undistortPoints(pts2_np, K_cam2, None)

    E, inlier_mask = cv2.findEssentialMat(
        pts1_norm,
        pts2_norm,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.001,
    )
    if E is None:
        return None

    candidates = split_essential_candidates(E)
    if not candidates:
        return None

    best = None
    best_score = None
    for candidate_idx, E_candidate in enumerate(candidates):
        try:
            inliers, R, t, pose_mask = cv2.recoverPose(
                E_candidate,
                pts1_norm,
                pts2_norm,
                np.eye(3),
                mask=None if inlier_mask is None else inlier_mask.copy(),
            )
        except cv2.error:
            continue
        reproj = compute_reprojection_error(pts1, pts2, K_cam0, K_cam2, R, t)
        if reproj is None:
            continue
        score = (float(reproj["reproj_rmse_mean"]), -int(inliers))
        if best_score is None or score < best_score:
            best_score = score
            best = (candidate_idx, inliers, R, t, reproj)

    if best is None:
        return None

    candidate_idx, inliers, R, t, reproj = best
    qx, qy, qz, qw = R_tool.from_matrix(R).as_quat()
    x, y, z = t.reshape(3)
    row = {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "qx": float(qx),
        "qy": float(qy),
        "qz": float(qz),
        "qw": float(qw),
        "inliers": int(inliers),
        "essential_candidates": int(len(candidates)),
        "essential_candidate_idx": int(candidate_idx),
    }
    row.update(reproj)
    return row


class XyzqSmoother:
    def __init__(self, window: int):
        self.window = max(1, int(window))
        self.rows: list[dict[str, Any]] = []
        self.last_q: np.ndarray | None = None

    def smooth(self, row: dict[str, Any]) -> dict[str, Any]:
        q = np.asarray([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=float)
        q = q / (np.linalg.norm(q) + 1e-12)
        if self.last_q is not None and float(np.dot(q, self.last_q)) < 0:
            q = -q
        row = dict(row)
        row["qx"], row["qy"], row["qz"], row["qw"] = [float(v) for v in q]
        self.rows.append(row)
        if len(self.rows) > self.window:
            self.rows.pop(0)

        t_vals = np.asarray([[r["x"], r["y"], r["z"]] for r in self.rows], dtype=float)
        q_vals = np.asarray([[r["qx"], r["qy"], r["qz"], r["qw"]] for r in self.rows], dtype=float)
        q_mean = R_tool.from_quat(q_vals).mean().as_quat()
        q_mean = q_mean / (np.linalg.norm(q_mean) + 1e-12)
        if self.last_q is not None and float(np.dot(q_mean, self.last_q)) < 0:
            q_mean = -q_mean
        self.last_q = q_mean

        out = dict(row)
        t_mean = np.mean(t_vals, axis=0)
        out["x"], out["y"], out["z"] = [float(v) for v in t_mean]
        out["qx"], out["qy"], out["qz"], out["qw"] = [float(v) for v in q_mean]
        out["smooth_window"] = len(self.rows)
        return out


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rv / theta
    half = 0.5 * theta
    s = np.sin(half)
    return _normalize_quat(np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)], dtype=np.float64))


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if abs(dot) > 0.9995:
        return _normalize_quat((1.0 - alpha) * q0 + alpha * q1)
    theta = np.arccos(dot)
    s0 = np.sin((1.0 - alpha) * theta) / np.sin(theta)
    s1 = np.sin(alpha * theta) / np.sin(theta)
    return _normalize_quat(s0 * q0 + s1 * q1)


class RealTimeXyzqEstimator:
    def __init__(
        self,
        *,
        K_cam0: np.ndarray,
        K_cam2: np.ndarray,
        out_xyzq: Path,
        out_match: Path,
        tolerance: float,
        vis_thresh: float,
        min_pairs: int,
        solve_batch_size: int,
        smooth_window: int,
        enable_imu_fusion: bool = False,
    ):
        self.K_cam0 = K_cam0
        self.K_cam2 = K_cam2
        self.tolerance = tolerance
        self.vis_thresh = vis_thresh
        self.min_pairs = min_pairs
        self.solve_batch_size = max(1, int(solve_batch_size))
        self.smoother = XyzqSmoother(smooth_window)
        self.enable_imu_fusion = bool(enable_imu_fusion)
        self.cam0_buf: list[dict[str, Any]] = []
        self.cam2_buf: list[dict[str, Any]] = []
        self.batch: list[dict[str, Any]] = []
        self.imu_samples: deque[dict[str, Any]] = deque()
        self.imu_window_sec = 1.0
        self.imu_alpha = 0.25
        self.last_pose_ts: float | None = None
        self.last_pose_q: np.ndarray | None = None
        out_xyzq.parent.mkdir(parents=True, exist_ok=True)
        out_match.parent.mkdir(parents=True, exist_ok=True)
        self.xyzq_f = out_xyzq.open("w", encoding="utf-8")
        self.match_f = out_match.open("w", encoding="utf-8")
        self.written = 0
        self.matched = 0
        self.skipped = 0
        self.reproj_rmse_values: list[float] = []
        self.reproj_rmse_cam0_values: list[float] = []
        self.reproj_rmse_cam2_values: list[float] = []

    def close(self):
        self.xyzq_f.close()
        self.match_f.close()

    def add_imu_sample(self, sample: dict[str, Any]):
        if not self.enable_imu_fusion:
            return
        try:
            ts = float(sample.get("timestamp_wall", sample.get("timestamp", time.time())))
            gx = float(sample.get("gx", 0.0))
            gy = float(sample.get("gy", 0.0))
            gz = float(sample.get("gz", 0.0))
        except (KeyError, TypeError, ValueError):
            return
        self.imu_samples.append({"timestamp_wall": ts, "gx": gx, "gy": gy, "gz": gz})
        while self.imu_samples and self.imu_samples[0]["timestamp_wall"] < ts - self.imu_window_sec:
            self.imu_samples.popleft()

    def _timestamp_for_row(self, row: dict[str, Any]) -> float:
        for key in ("timestamp_wall", "timestamp"):
            if key in row:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    pass
        return float("nan")

    def _apply_imu_fusion(self, row: dict[str, Any]) -> dict[str, Any]:
        if not self.enable_imu_fusion:
            return row
        row = dict(row)
        ts = self._timestamp_for_row(row)
        if not np.isfinite(ts):
            return row
        q_vis = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)
        if self.last_pose_ts is None or self.last_pose_q is None:
            self.last_pose_ts = ts
            self.last_pose_q = _normalize_quat(q_vis)
            return row

        imu_window = [s for s in self.imu_samples if self.last_pose_ts <= s["timestamp_wall"] <= ts]
        if imu_window:
            q_pred = self.last_pose_q.copy()
            prev_ts = self.last_pose_ts
            for sample in imu_window:
                dt = max(0.0, float(sample["timestamp_wall"] - prev_ts))
                if dt <= 0.0:
                    continue
                rv = np.array([sample["gx"], sample["gy"], sample["gz"]], dtype=np.float64) * dt
                q_pred = _quat_mul(q_pred, _quat_from_rotvec(rv))
                prev_ts = float(sample["timestamp_wall"])
            q_pred = _normalize_quat(q_pred)
            q_fused = _quat_slerp(q_pred, q_vis, self.imu_alpha)
        else:
            q_fused = q_vis

        row["qx"], row["qy"], row["qz"], row["qw"] = [float(v) for v in _normalize_quat(q_fused)]
        row["imu_fused"] = True
        self.last_pose_ts = ts
        self.last_pose_q = _normalize_quat(q_fused)
        return row

    def add_record(self, record: dict[str, Any]):
        cam = record.get("cam")
        if cam == "cam0":
            self._add_and_match(record, self.cam0_buf, self.cam2_buf, left_is_new=True)
        elif cam == "cam2":
            self._add_and_match(record, self.cam2_buf, self.cam0_buf, left_is_new=False)

    def _add_and_match(self, record, own_buf, other_buf, *, left_is_new: bool):
        ts = float(record.get("timestamp", 0.0))
        best_idx = None
        best_dt = None
        for idx, other in enumerate(other_buf):
            dt = abs(float(other.get("timestamp", 0.0)) - ts)
            if dt <= self.tolerance and (best_dt is None or dt < best_dt):
                best_idx = idx
                best_dt = dt

        if best_idx is None:
            own_buf.append(record)
            self._trim_buffers()
            return

        other = other_buf.pop(best_idx)
        cam0_record, cam2_record = (record, other) if left_is_new else (other, record)
        self._handle_match(cam0_record, cam2_record, float(best_dt or 0.0))

    def _trim_buffers(self):
        max_keep = 120
        del self.cam0_buf[:-max_keep]
        del self.cam2_buf[:-max_keep]

    def _handle_match(self, cam0_record, cam2_record, dt: float):
        pts1, pts2, ids = matched_points(cam0_record, cam2_record, self.vis_thresh)
        match_row = {
            "timestamp": cam0_record["timestamp"],
            "timestamp_cam2": cam2_record["timestamp"],
            "dt": dt,
            "frame_cam0": cam0_record.get("frame"),
            "frame_cam2": cam2_record.get("frame"),
            "joint_ids": ids,
            "points": len(ids),
            "pairs17": make_pairs17(cam0_record, cam2_record),
        }
        self.match_f.write(json.dumps(match_row, ensure_ascii=False) + "\n")
        self.match_f.flush()
        self.matched += 1

        if len(ids) < self.min_pairs:
            self.skipped += 1
            return

        self.batch.append(
            {
                "timestamp": float(cam0_record["timestamp"]),
                "timestamp_cam2": float(cam2_record["timestamp"]),
                "pts1": pts1,
                "pts2": pts2,
                "ids": ids,
            }
        )
        if len(self.batch) >= self.solve_batch_size:
            self._solve_batch()

    def _solve_batch(self):
        batch = self.batch[: self.solve_batch_size]
        del self.batch[: self.solve_batch_size]
        pts1, pts2 = [], []
        for item in batch:
            pts1.extend(item["pts1"])
            pts2.extend(item["pts2"])
        pose = estimate_xyzq(pts1, pts2, self.K_cam0, self.K_cam2, self.min_pairs)
        if pose is None:
            self.skipped += len(batch)
            return

        ts_values = [item["timestamp"] for item in batch]
        row = {
            "timestamp": float(ts_values[-1]),
            "timestamp_start": float(ts_values[0]),
            "timestamp_end": float(ts_values[-1]),
            "batch_size": len(batch),
            "points": len(pts1),
            **pose,
        }
        row = self.smoother.smooth(row)
        row = self._apply_imu_fusion(row)
        self.xyzq_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.xyzq_f.flush()
        self.written += 1
        if "reproj_rmse_mean" in row:
            self.reproj_rmse_values.append(float(row["reproj_rmse_mean"]))
            self.reproj_rmse_cam0_values.append(float(row["reproj_rmse_cam0"]))
            self.reproj_rmse_cam2_values.append(float(row["reproj_rmse_cam2"]))
        if getattr(self, "print_xyzq", True):
            err_text = ""
            if "reproj_rmse_mean" in row:
                err_text = (
                    f" reproj={row['reproj_rmse_mean']:.2f}px"
                    f" cam0={row['reproj_rmse_cam0']:.2f}px"
                    f" cam2={row['reproj_rmse_cam2']:.2f}px"
                )
            imu_text = " [IMU-FUSED]" if row.get("imu_fused") else ""
            print(
                "[xyzq] "
                f"ts={row['timestamp']:.3f} "
                f"batch={row['batch_size']} pts={row['points']} "
                f"x={row['x']:.4f} y={row['y']:.4f} z={row['z']:.4f} "
                f"q=({row['qx']:.4f},{row['qy']:.4f},{row['qz']:.4f},{row['qw']:.4f})"
                f"{err_text}{imu_text}",
                flush=True,
            )

    def average_reprojection_error(self):
        if not self.reproj_rmse_values:
            return None
        return {
            "mean": float(np.mean(np.asarray(self.reproj_rmse_values, dtype=float))),
            "cam0": float(np.mean(np.asarray(self.reproj_rmse_cam0_values, dtype=float))),
            "cam2": float(np.mean(np.asarray(self.reproj_rmse_cam2_values, dtype=float))),
        }


# ==========================================================================
# Offline video analysis (no camera/websocket needed)
#
# Accepts either:
#   (a) an already-labeled pose jsonl (MP_TO_COCO17 / keypoints17 format,
#       same schema CapturePoseWorker writes), via --camX-json, or
#   (b) a raw, unlabeled video file, via --camX-video, in which case
#       MediaPipe Pose is run frame-by-frame to produce that same schema.
#
# If both cam0 and cam2 sources are given, the existing RealTimeXyzqEstimator
# matching/triangulation/smoothing pipeline is reused unchanged (records are
# simply merge-sorted by timestamp and fed in as if they arrived live), so
# behaviour is identical to the live "server" path. If only one camera is
# given, this just performs pose extraction/labeling with no stereo step.
# ==========================================================================


def extract_pose_from_video(
    video_path: Path,
    cam_id: str,
    tag: str,
    out_json: Path,
    out_pose_video: Path | None,
    fps_override: float | None,
    vis_thresh: float,
) -> list[dict[str, Any]]:
    """Run MediaPipe Pose over every frame of an unlabeled video file and
    write MP_TO_COCO17-format records (same schema as CapturePoseWorker)."""
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_override) if fps_override else (float(src_fps) if src_fps and src_fps > 0 else 30.0)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    pose_writer: cv2.VideoWriter | None = None
    records: list[dict[str, Any]] = []
    mp_pose = mp.solutions.pose

    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose, \
                out_json.open("w", encoding="utf-8") as fjson:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if pose_writer is None and out_pose_video is not None:
                    pose_writer = create_writer(out_pose_video, fps, frame.shape)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                annot, keypoints17 = draw_pose(frame, result.pose_landmarks, vis_thresh)
                if pose_writer is not None:
                    pose_writer.write(annot)

                timestamp = frame_idx / fps
                record = {
                    "cam": cam_id,
                    "tag": tag,
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 9),
                    "timestamp_wall": round(timestamp, 9),
                    "keypoints17": keypoints17,
                }
                fjson.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                frame_idx += 1
    finally:
        cap.release()
        if pose_writer is not None:
            pose_writer.release()

    if not records:
        raise ValueError(f"No frames could be read from {video_path}")
    return records


def resolve_offline_records(
    video_path: str | None,
    json_path: str | None,
    cam_id: str,
    tag: str,
    out_root: Path,
    fps_override: float | None,
    vis_thresh: float,
    save_annotated: bool,
) -> list[dict[str, Any]]:
    """Pick the right source for one camera's pose records: a pre-labeled
    jsonl if given (fastest, no MediaPipe needed), otherwise run pose
    extraction on the raw video. Returns [] if neither is provided."""
    if json_path:
        records = load_xyzq_jsonl(Path(json_path))  # generic jsonl loader, works for any per-line record
        if not records:
            raise ValueError(f"No pose records found in {json_path}")
        for r in records:
            r.setdefault("cam", cam_id)
        print(f"[offline] {cam_id}: loaded {len(records)} pre-labeled frames from {json_path}", flush=True)
        return records

    if not video_path:
        return []

    video_path_p = Path(video_path)
    if not video_path_p.exists():
        raise FileNotFoundError(f"{cam_id} video not found: {video_path_p}")

    out_dir = test_output_dir()
    out_json = out_dir / f"{OUTPUT_PREFIX}{cam_id}_{tag}.jsonl"
    out_pose_video = (out_dir / f"{OUTPUT_PREFIX}{cam_id}_{tag}.mp4") if save_annotated else None

    print(f"[offline] {cam_id}: no pre-labeled JSON given, running MediaPipe pose extraction on {video_path_p}", flush=True)
    records = extract_pose_from_video(video_path_p, cam_id, tag, out_json, out_pose_video, fps_override, vis_thresh)
    print(f"[offline] {cam_id}: extracted {len(records)} frames -> {out_json}", flush=True)
    return records


def run_offline_postprocess(
    args,
    tag: str,
    estimator: "RealTimeXyzqEstimator",
    cam0_video: Path | None,
    cam2_video: Path | None,
    out_root: Path,
    data_dir: Path,
):
    xyzq_jsonl = data_dir / f"{OUTPUT_PREFIX}xyzq_{tag}.jsonl"
    match_jsonl = data_dir / f"{OUTPUT_PREFIX}cam_match_{tag}.jsonl"
    summary: dict[str, Any] = {
        "tag": tag,
        "mode": "offline",
        "xyzq_count": int(estimator.written),
    }
    human_rows = load_xyzq_jsonl(xyzq_jsonl)
    summary.update(reproj_summary_from_xyzq(human_rows, "human"))

    want_3d = args.make_3d_video or args.postprocess_all
    want_aruco = args.run_aruco or args.postprocess_all
    want_chess = args.run_chessboard or args.postprocess_all

    if want_3d:
        try:
            summary.update(run_make_3d_video(args, tag, match_jsonl, xyzq_jsonl, out_root))
        except Exception as exc:
            print(f"[offline][WARN] 3D video failed: {exc}", flush=True)

    if want_aruco or want_chess:
        if cam0_video is None or cam2_video is None:
            print(
                "[offline][WARN] --run-aruco/--run-chessboard need the raw cam0/cam2 video files "
                "(cannot run from pre-labeled JSON alone); skipping.",
                flush=True,
            )
        else:
            if want_aruco:
                try:
                    aruco_rt, aruco_stats = run_aruco_postprocess(args, tag, cam0_video, cam2_video, out_root)
                    summary.update(aruco_stats)
                    summary.update(compare_pose_series(human_rows, load_rt_txt(aruco_rt), float(args.tolerance), "human_vs_aruco"))
                    summary["aruco_rt"] = str(aruco_rt)
                except Exception as exc:
                    print(f"[offline][WARN] ArUco failed: {exc}", flush=True)
            if want_chess:
                try:
                    chess_rt, chess_stats = run_chessboard_postprocess(args, tag, cam0_video, cam2_video, out_root)
                    summary.update(chess_stats)
                    summary.update(compare_pose_series(human_rows, load_rt_txt(chess_rt), float(args.tolerance), "human_vs_chessboard"))
                    summary["chessboard_rt"] = str(chess_rt)
                except Exception as exc:
                    print(f"[offline][WARN] chessboard failed: {exc}", flush=True)

    summary_path = out_root / f"{OUTPUT_PREFIX}summary.csv"
    write_summary_csv(summary_path, summary)
    print(f"[offline] summary saved: {summary_path}", flush=True)


def run_offline(args):
    if not args.cam0_video and not args.cam0_json:
        raise ValueError("offline mode requires at least --cam0-video or --cam0-json")

    global OUTPUT_PREFIX
    OUTPUT_PREFIX = "testeval_"

    tag = sanitize_tag(args.tag or time.strftime("%Y%m%d_%H%M%S"))
    out_root = test_output_dir()   # 所有輸出統一存入 test/
    data_dir = out_root            # xyzq/match jsonl 也存在同一個 test/ 底下

    cam0_records = resolve_offline_records(
        args.cam0_video, args.cam0_json, "cam0", tag, out_root, args.fps, args.vis, args.save_annotated_video
    )
    cam2_records = (
        resolve_offline_records(
            args.cam2_video, args.cam2_json, "cam2", tag, out_root, args.fps, args.vis, args.save_annotated_video
        )
        if (args.cam2_video or args.cam2_json)
        else []
    )

    if not cam2_records:
        print(f"[offline] cam2 not provided -> pose-labeling only, no stereo xyzq. cam0 frames: {len(cam0_records)}", flush=True)
        print("[offline] done.", flush=True)
        return

    K_cam0 = load_intrinsics(Path(args.k1))
    K_cam2 = load_intrinsics(Path(args.k2))

    estimator = RealTimeXyzqEstimator(
        K_cam0=K_cam0,
        K_cam2=K_cam2,
        out_xyzq=data_dir / f"{OUTPUT_PREFIX}xyzq_{tag}.jsonl",
        out_match=data_dir / f"{OUTPUT_PREFIX}cam_match_{tag}.jsonl",
        tolerance=float(args.tolerance),
        vis_thresh=float(args.vis),
        min_pairs=int(args.min_pairs),
        solve_batch_size=int(args.solve_batch_size),
        smooth_window=int(args.smooth_window),
        enable_imu_fusion=False,  # no live IMU stream available offline
    )
    estimator.print_xyzq = not args.quiet  # avoid flooding stdout on long offline batches

    # Replay both cameras' records in timestamp order, exactly as the live
    # server loop would receive them frame-by-frame -> reuses all existing
    # matching/triangulation/smoothing logic unchanged.
    combined = sorted(cam0_records + cam2_records, key=lambda r: float(r.get("timestamp", 0.0)))
    for record in combined:
        estimator.add_record(record)
    estimator.close()

    print(
        f"[offline] xyzq written={estimator.written} matched={estimator.matched} skipped={estimator.skipped}",
        flush=True,
    )
    avg_reproj = estimator.average_reprojection_error()
    if avg_reproj is not None:
        print(
            "[offline] average reprojection RMSE="
            f"{avg_reproj['mean']:.2f}px cam0={avg_reproj['cam0']:.2f}px cam2={avg_reproj['cam2']:.2f}px",
            flush=True,
        )
    else:
        print("[offline] average reprojection RMSE=N/A (no successful pose solves)", flush=True)

    run_offline_postprocess(
        args,
        tag,
        estimator,
        Path(args.cam0_video) if args.cam0_video else None,
        Path(args.cam2_video) if args.cam2_video else None,
        out_root,
        data_dir,
    )


async def send_json(ws, payload: dict[str, Any]):
    await ws.send(json.dumps(payload, ensure_ascii=False))


async def countdown_until(start_at_wall: float):
    last_shown = None
    while True:
        remaining = start_at_wall - time.time()
        if remaining <= 0:
            print("[server] START", flush=True)
            return
        shown = max(1, int(math.ceil(remaining)))
        if shown != last_shown:
            print(f"[server] recording starts in {shown}...", flush=True)
            last_shown = shown
        await asyncio.sleep(min(0.1, remaining))


async def send_file(ws, path: Path, label: str, chunk_size: int = 1024 * 512):
    if not path.exists():
        await send_json(ws, {"type": "file_missing", "label": label, "path": str(path)})
        return
    await send_json(ws, {"type": "file_start", "label": label, "name": path.name, "size": path.stat().st_size})
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            await ws.send(chunk)
    await send_json(ws, {"type": "file_end", "label": label, "name": path.name})


def run_checked(cmd: list[str], cwd: Path):
    print("[post] " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_make_3d_video(args, tag: str, match_jsonl: Path, xyzq_jsonl: Path, out_root: Path):
    match_txt = out_root / "data" / f"{OUTPUT_PREFIX}cam_match_{tag}_legacy.txt"
    rt_txt = out_root / "data" / f"{OUTPUT_PREFIX}xyzq_{tag}_legacy.txt"
    out_dir = out_root / "output"
    out_html = out_dir / f"{OUTPUT_PREFIX}{tag}_3d.html"
    out_mp4 = out_dir / f"{OUTPUT_PREFIX}{tag}_3d.mp4"
    write_match_txt_from_jsonl(match_jsonl, match_txt)
    write_rt_txt_from_xyzq_jsonl(xyzq_jsonl, rt_txt)
    run_checked(
        [
            str(Path(args.python_exe)),
            str(BASE_DIR / "reconstruct_3d_quat_4_1.py"),
            "--match",
            str(match_txt),
            "--rt_quat",
            str(rt_txt),
            "--k1",
            str(args.k1),
            "--k2",
            str(args.k2),
            "--out",
            str(out_html),
            "--out_mp4",
            str(out_mp4),
            "--fps",
            str(int(float(args.fps))),
            "--max_reproj",
            str(args.reconstruct_max_reproj),
            "--min_keep",
            str(args.reconstruct_min_keep),
        ],
        BASE_DIR,
    )
    return {"3d_html": str(out_html), "3d_mp4": str(out_mp4)}


def aruco_reprojection_stats(cam0_video: Path, cam2_video: Path, args):
    K0 = load_intrinsics(Path(args.k1))
    K1 = load_intrinsics(Path(args.k2))
    d0 = np.loadtxt(args.d1).reshape(-1) if args.d1 else np.zeros(5)
    d1 = np.loadtxt(args.d2).reshape(-1) if args.d2 else np.zeros(5)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    cap0 = cv2.VideoCapture(str(cam0_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    half = float(args.aruco_marker_size) / 2.0
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    cam0_vals, cam2_vals = [], []
    try:
        i = 0
        while True:
            ok0, f0 = cap0.read()
            ok2, f2 = cap2.read()
            if not (ok0 and ok2):
                break
            if i % max(1, int(args.aruco_stride)) != 0:
                i += 1
                continue
            corners0, ids0, _ = cv2.aruco.detectMarkers(f0, aruco_dict)
            corners2, ids2, _ = cv2.aruco.detectMarkers(f2, aruco_dict)
            if ids0 is None or ids2 is None:
                i += 1
                continue
            dct0 = {int(mid): c for mid, c in zip(ids0.flatten(), corners0)}
            dct2 = {int(mid): c for mid, c in zip(ids2.flatten(), corners2)}
            for mid in sorted(set(dct0) & set(dct2)):
                for corner, K, dist, vals in ((dct0[mid], K0, d0, cam0_vals), (dct2[mid], K1, d1, cam2_vals)):
                    ok, rvec, tvec = cv2.solvePnP(obj, corner.reshape(4, 2).astype(np.float32), K, dist, flags=cv2.SOLVEPNP_SQPNP)
                    if not ok:
                        continue
                    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                    err = np.linalg.norm(proj.reshape(4, 2) - corner.reshape(4, 2), axis=1)
                    vals.append(float(np.sqrt(np.mean(err * err))))
            i += 1
    finally:
        cap0.release()
        cap2.release()
    combined = list(cam0_vals) + list(cam2_vals)
    out: dict[str, Any] = {}
    for label, vals in (("aruco_cam0", cam0_vals), ("aruco_cam2", cam2_vals), ("aruco_combined", combined)):
        st = stats_min_max_mean(vals)
        out[f"{label}_reproj_min"] = st["min"]
        out[f"{label}_reproj_max"] = st["max"]
        out[f"{label}_reproj_mean"] = st["mean"]
    return out


def run_aruco_postprocess(args, tag: str, cam0_video: Path, cam2_video: Path, out_root: Path):
    out_rt = out_root / "data" / f"{OUTPUT_PREFIX}rt_aruco_{tag}.txt"
    run_checked(
        [
            str(Path(args.python_exe)),
            str(BASE_DIR / "aruco3.py"),
            "--cam0",
            str(cam0_video),
            "--cam2",
            str(cam2_video),
            "--k1",
            str(args.k1),
            "--k2",
            str(args.k2),
            "--d1",
            str(args.d1),
            "--d2",
            str(args.d2),
            "--marker_size",
            str(args.aruco_marker_size),
            "--stride",
            str(args.aruco_stride),
            "--dict",
            str(args.aruco_dict),
            "--smooth",
            str(args.aruco_smooth),
            "--out",
            str(out_rt),
        ],
        BASE_DIR,
    )
    stats = aruco_reprojection_stats(cam0_video, cam2_video, args)
    return out_rt, stats


def run_chessboard_postprocess(args, tag: str, cam0_video: Path, cam2_video: Path, out_root: Path):
    out_rt = out_root / "data" / f"{OUTPUT_PREFIX}rt_chessboard_{tag}.txt"
    run_checked(
        [
            str(Path(args.python_exe)),
            str(BASE_DIR / "chessboard_stereo_calibrate.py"),
            "--cam1_video",
            str(cam0_video),
            "--cam2_video",
            str(cam2_video),
            "--k1",
            str(args.k1),
            "--k2",
            str(args.k2),
            "--dist1",
            str(args.d1),
            "--dist2",
            str(args.d2),
            "--out_dir",
            str(out_root / "data"),
            "--out_rt",
            str(out_rt),
        ],
        BASE_DIR,
    )
    stats = compute_chessboard_reprojection_stats(args, cam0_video, cam2_video)
    return out_rt, stats


def resolve_recorded_video(root: Path, cam_id: str, tag: str) -> Path:
    raw_dir = root / "raw"
    candidates = [
        raw_dir / f"{cam_id}_{tag}.mp4",
        raw_dir / f"{cam_id}_{tag}_raw.mp4",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    matches = sorted(raw_dir.glob(f"{cam_id}_{tag}*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return candidates[0]


def compute_chessboard_reprojection_stats(args, cam0_video: Path, cam2_video: Path):
    import chessboard_stereo_calibrate as chess

    class Obj:
        pass

    cargs = Obj()
    cargs.cam1_video = str(cam0_video)
    cargs.cam2_video = str(cam2_video)
    cargs.frame_step = 5
    cargs.start_frame = 0
    cargs.max_frames = 0
    cargs.max_samples = 80
    cargs.use_sb = True
    cargs.debug_draw_dir = ""
    cargs.verbose = False
    cargs.cols = 9
    cargs.rows = 6
    cargs.square_size = 2.9
    pattern = (cargs.cols, cargs.rows)
    samples = chess._collect_from_videos(cargs, pattern)
    if len(samples) < 1:
        return {}
    objp = chess._make_object_points(cargs.cols, cargs.rows, cargs.square_size)
    K0 = chess._load_k(args.k1, "K0")
    K1 = chess._load_k(args.k2, "K1")
    d0 = chess._load_dist(args.d1, "dist0")
    d1 = chess._load_dist(args.d2, "dist1")
    _, reproj_rows = chess._estimate_per_frame_rt(samples, objp, K0, d0, K1, d1)
    cam0 = [r["cam1_rmse"] for r in reproj_rows]
    cam2 = [r["cam2_rmse"] for r in reproj_rows]
    combined = [r["both_rmse"] for r in reproj_rows]
    out: dict[str, Any] = {}
    for label, vals in (("chessboard_cam0", cam0), ("chessboard_cam2", cam2), ("chessboard_combined", combined)):
        st = stats_min_max_mean(vals)
        out[f"{label}_reproj_min"] = st["min"]
        out[f"{label}_reproj_max"] = st["max"]
        out[f"{label}_reproj_mean"] = st["mean"]
    return out


def run_postprocess(args, tag: str, estimator: RealTimeXyzqEstimator):
    out_root = DEFAULT_OUTPUT_ROOT
    data_dir = Path(args.data_root).expanduser()
    cam0_video = resolve_recorded_video(Path(args.pc_root).expanduser(), "cam0", tag)
    cam2_video = resolve_recorded_video(Path(args.pi_local_root).expanduser(), "cam2", tag)
    xyzq_jsonl = data_dir / f"xyzq_{tag}.jsonl"
    match_jsonl = data_dir / f"cam_match_{tag}.jsonl"
    summary: dict[str, Any] = {
        "tag": tag,
        "duration_sec": float(args.secs),
        "xyzq_count": int(estimator.written),
    }
    human_rows = load_xyzq_jsonl(xyzq_jsonl)
    summary.update(reproj_summary_from_xyzq(human_rows, "human"))

    if args.make_3d_video or args.postprocess_all:
        try:
            summary.update(run_make_3d_video(args, tag, match_jsonl, xyzq_jsonl, out_root))
        except Exception as exc:
            print(f"[post][WARN] 3D video failed: {exc}", flush=True)

    if args.run_aruco or args.postprocess_all:
        try:
            aruco_rt, aruco_stats = run_aruco_postprocess(args, tag, cam0_video, cam2_video, out_root)
            summary.update(aruco_stats)
            summary.update(compare_pose_series(human_rows, load_rt_txt(aruco_rt), float(args.tolerance), "human_vs_aruco"))
            summary["aruco_rt"] = str(aruco_rt)
        except Exception as exc:
            print(f"[post][WARN] ArUco failed: {exc}", flush=True)

    if args.run_chessboard or args.postprocess_all:
        try:
            chess_rt, chess_stats = run_chessboard_postprocess(args, tag, cam0_video, cam2_video, out_root)
            summary.update(chess_stats)
            summary.update(compare_pose_series(human_rows, load_rt_txt(chess_rt), float(args.tolerance), "human_vs_chessboard"))
            summary["chessboard_rt"] = str(chess_rt)
        except Exception as exc:
            print(f"[post][WARN] chessboard failed: {exc}", flush=True)

    summary_path = out_root / "results" / "summary.csv"
    write_summary_csv(summary_path, summary)
    print(f"[post] summary saved: {summary_path}", flush=True)


async def pi_client(args):
    websockets = import_websockets()
    q: queue.Queue = queue.Queue()
    imu_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    start_gate = threading.Event()
    start_at_box: dict[str, float] = {}
    imu_error_box: dict[str, str | None] = {"message": None}

    async def pump_imu(ws):
        while not stop_event.is_set():
            if imu_error_box["message"] is not None:
                raise RuntimeError(f"IMU worker failed: {imu_error_box['message']}")
            try:
                sample = await asyncio.to_thread(imu_q.get, True, 0.1)
            except queue.Empty:
                continue
            await send_json(ws, {"type": "imu", "sample": sample})

    def imu_worker_runner():
        try:
            imu_thread_worker(args.imu_port, args.imu_baud, args.imu_timeout, imu_q, stop_event)
        except BaseException as exc:
            imu_error_box["message"] = str(exc)
            stop_event.set()
            raise

    async with websockets.connect(args.server) as ws:
        hello = json.loads(await ws.recv())
        if hello.get("type") != "hello":
            raise RuntimeError(f"Unexpected server message: {hello}")

        cfg = hello["config"]
        worker = CapturePoseWorker(
            cam_id="cam2",
            tag=cfg["tag"],
            root=Path(args.pi_root).expanduser(),
            cam_index=int(args.pi_cam),
            fps=float(cfg["fps"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            vis=float(cfg["vis"]),
            start_at_wall=0.0,
            secs=float(cfg["secs"]),
            stop_event=stop_event,
            record_queue=q,
            start_gate=start_gate,
            start_at_box=start_at_box,
        )
        worker.start()
        await asyncio.to_thread(worker.ready_event.wait)
        if worker.error:
            await send_json(ws, {"type": "error", "message": str(worker.error)})
            raise worker.error
        await send_json(ws, worker.ready_payload or {"type": "ready", "cam": "cam2"})

        imu_worker = None
        imu_task = None
        if bool(args.enable_imu_fusion):
            print(
                f"[IMU] fusion enabled on Pi: direct serial capture from {args.imu_port} @ {args.imu_baud} baud",
                flush=True,
            )
            imu_worker = threading.Thread(
                target=imu_worker_runner,
                daemon=True,
            )
            imu_worker.start()
            imu_task = asyncio.create_task(pump_imu(ws))

        start_msg = json.loads(await ws.recv())
        if start_msg.get("type") != "start":
            raise RuntimeError(f"Expected start message, got: {start_msg}")
        start_at_box["start_at_wall"] = float(start_msg["start_at_wall"])
        start_gate.set()

        async def receiver():
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "stop":
                    stop_event.set()
                    break

        recv_task = asyncio.create_task(receiver())
        while not worker.done_event.is_set() or not q.empty():
            try:
                record = await asyncio.to_thread(q.get, True, 0.1)
            except queue.Empty:
                continue
            await send_json(ws, {"type": "pose", "record": record})

        stop_event.set()
        worker.join()
        recv_task.cancel()
        if imu_task is not None:
            imu_task.cancel()
        if imu_worker is not None:
            imu_worker.join(timeout=1.0)
        await send_json(ws, {"type": "done", "cam": "cam2"})
        await send_file(ws, worker.raw_video, "pi_raw")
        await send_file(ws, worker.pose_video, "pi_pose")
        await send_json(ws, {"type": "all_done", "cam": "cam2"})


async def server_receiver(ws, estimator_queue: queue.Queue, pi_json_path: Path, pi_root: Path, estimator: RealTimeXyzqEstimator | None = None):
    pi_json_path.parent.mkdir(parents=True, exist_ok=True)
    file_out = None
    file_label = None
    all_done = asyncio.Event()
    pi_done = asyncio.Event()

    with pi_json_path.open("w", encoding="utf-8") as pi_json:
        async for raw in ws:
            if isinstance(raw, bytes):
                if file_out is not None:
                    file_out.write(raw)
                continue

            msg = json.loads(raw)
            typ = msg.get("type")
            if typ == "pose":
                record = msg["record"]
                pi_json.write(json.dumps(record, ensure_ascii=False) + "\n")
                pi_json.flush()
                estimator_queue.put(record)
            elif typ == "ready":
                continue
            elif typ == "imu":
                sample = msg.get("sample", {})
                if estimator is not None and getattr(estimator, "enable_imu_fusion", False):
                    estimator.add_imu_sample(sample)
            elif typ == "done":
                pi_done.set()
            elif typ == "file_start":
                label = str(msg["label"])
                name = Path(str(msg["name"])).name
                sub = "raw" if label == "pi_raw" else "video"
                out_path = ensure_dirs(pi_root)[sub] / name
                file_out = out_path.open("wb")
                file_label = label
                print(f"[server] receiving {label}: {out_path}", flush=True)
            elif typ == "file_end":
                if file_out is not None:
                    file_out.close()
                print(f"[server] received {file_label}", flush=True)
                file_out = None
                file_label = None
            elif typ == "all_done":
                all_done.set()
                break
            elif typ == "error":
                raise RuntimeError(f"Pi error: {msg.get('message')}")
    if file_out is not None:
        file_out.close()
    return pi_done, all_done


async def run_server_session(ws, args):
    now_tag = time.strftime("%Y%m%d_%H%M%S")
    raw_tag = f"{args.tag}_{now_tag}" if args.tag else now_tag
    tag = sanitize_tag(raw_tag)
    if tag != raw_tag:
        print(f"[server][WARN] sanitized tag {raw_tag!r} -> {tag!r}", flush=True)
    config = {
        "tag": tag,
        "fps": float(args.fps),
        "width": int(args.width),
        "height": int(args.height),
        "vis": float(args.vis),
        "secs": float(args.secs),
    }
    await send_json(ws, {"type": "hello", "config": config})

    pc_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    pc_start_gate = threading.Event()
    pc_start_at_box: dict[str, float] = {}
    pc_worker = CapturePoseWorker(
        cam_id="cam0",
        tag=tag,
        root=Path(args.pc_root).expanduser(),
        cam_index=int(args.pc_cam),
        fps=float(args.fps),
        width=int(args.width),
        height=int(args.height),
        vis=float(args.vis),
        start_at_wall=0.0,
        secs=float(args.secs),
        stop_event=stop_event,
        record_queue=pc_queue,
        start_gate=pc_start_gate,
        start_at_box=pc_start_at_box,
    )
    pc_worker.start()
    await asyncio.to_thread(pc_worker.ready_event.wait)
    if pc_worker.error:
        raise pc_worker.error

    pi_ready_raw = await ws.recv()
    pi_ready = json.loads(pi_ready_raw)
    if pi_ready.get("type") != "ready":
        raise RuntimeError(f"Expected Pi ready message, got: {pi_ready}")

    print(f"[server] tag={tag}", flush=True)
    print(f"[server] PC ready: {pc_worker.ready_payload}", flush=True)
    print(f"[server] Pi ready: {pi_ready}", flush=True)

    data_dir = Path(args.data_root).expanduser()
    estimator = RealTimeXyzqEstimator(
        K_cam0=load_intrinsics(Path(args.k1)),
        K_cam2=load_intrinsics(Path(args.k2)),
        out_xyzq=data_dir / f"xyzq_{tag}.jsonl",
        out_match=data_dir / f"cam_match_{tag}.jsonl",
        tolerance=float(args.tolerance),
        vis_thresh=float(args.vis),
        min_pairs=int(args.min_pairs),
        solve_batch_size=int(args.solve_batch_size),
        smooth_window=int(args.smooth_window),
        enable_imu_fusion=bool(args.enable_imu_fusion),
    )
    if estimator.enable_imu_fusion:
        print(
            "[IMU] fusion enabled on server: rotation will be blended with direct IMU samples (translation stays visual)",
            flush=True,
        )

    pi_root = Path(args.pi_local_root).expanduser()
    _, pi_json, _ = output_paths(pi_root, "cam2", tag)
    receiver_task = asyncio.create_task(server_receiver(ws, pc_queue, pi_json, pi_root, estimator))

    async def pc_queue_pump():
        while not pc_worker.done_event.is_set() or not pc_queue.empty():
            try:
                record = await asyncio.to_thread(pc_queue.get, True, 0.1)
            except queue.Empty:
                continue
            estimator.add_record(record)

    async def timeout_stop():
        if float(args.secs) <= 0:
            return
        await asyncio.sleep(float(args.secs) + 0.5)
        stop_event.set()
        await send_json(ws, {"type": "stop"})

    async def manual_stop():
        if float(args.secs) > 0 and not args.manual_stop:
            return
        try:
            await asyncio.to_thread(input, "Press Enter to stop recording...\n")
        except EOFError:
            return
        stop_event.set()
        await send_json(ws, {"type": "stop"})

    pump_task = asyncio.create_task(pc_queue_pump())
    timeout_task = None
    manual_task = None
    try:
        start_at = time.time() + float(args.start_delay)
        pc_start_at_box["start_at_wall"] = start_at
        pc_start_gate.set()
        await send_json(ws, {"type": "start", "start_at_wall": start_at})
        print(f"[server] start_at_wall={start_at:.3f}", flush=True)
        await countdown_until(start_at)

        timeout_task = asyncio.create_task(timeout_stop())
        manual_task = asyncio.create_task(manual_stop())
        await receiver_task
    finally:
        stop_event.set()
        if not receiver_task.done():
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
        await pump_task
        if timeout_task is not None:
            timeout_task.cancel()
        if manual_task is not None:
            manual_task.cancel()
        pc_worker.join()
        estimator.close()

    print(
        f"[server] xyzq written={estimator.written} matched={estimator.matched} skipped={estimator.skipped}",
        flush=True,
    )
    avg_reproj = estimator.average_reprojection_error()
    if avg_reproj is not None:
        print(
            "[server] average reprojection RMSE="
            f"{avg_reproj['mean']:.2f}px "
            f"cam0={avg_reproj['cam0']:.2f}px "
            f"cam2={avg_reproj['cam2']:.2f}px",
            flush=True,
        )
    else:
        print("[server] average reprojection RMSE=N/A", flush=True)
    print(f"[server] outputs: {data_dir / f'xyzq_{tag}.jsonl'}", flush=True)
    await asyncio.to_thread(run_postprocess, args, tag, estimator)


async def server_main(args):
    websockets = import_websockets()
    print(f"[server] waiting on ws://{args.host}:{args.port}", flush=True)
    done_event = asyncio.Event()

    async def handler(ws, *_):
        try:
            await run_server_session(ws, args)
        finally:
            if not args.keep_server:
                done_event.set()

    async with websockets.serve(handler, args.host, int(args.port), max_size=None):
        if args.keep_server:
            await asyncio.Future()
        else:
            await done_event.wait()


def build_parser():
    parser = argparse.ArgumentParser(description="Realtime PC/Pi pose streaming and xyzq estimation over WebSocket.")
    sub = parser.add_subparsers(dest="mode", required=True)

    server = sub.add_parser("server")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--tag", default=None)
    server.add_argument("--pc-root", default=str(DEFAULT_OUTPUT_ROOT / "pc_pose_video"))
    server.add_argument("--pi-local-root", default=str(DEFAULT_OUTPUT_ROOT / "pi_pose_video"))
    server.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT / "data"))
    server.add_argument("--pc-cam", type=int, default=0)
    server.add_argument("--fps", type=float, default=10.0)
    server.add_argument("--secs", type=float, default=0.0, help="0 means manual stop.")
    server.add_argument("--manual-stop", action="store_true", help="Allow Enter-to-stop even when --secs is set.")
    server.add_argument("--keep-server", action="store_true", help="Keep listening after one recording session.")
    server.add_argument("--start-delay", type=float, default=3.0)
    server.add_argument("--width", type=int, default=0)
    server.add_argument("--enable-imu-fusion", action="store_true", help="Enable optional IMU-assisted rotation fusion for realtime xyzq.")
    server.add_argument("--height", type=int, default=0)
    server.add_argument("--vis", type=float, default=0.5)
    server.add_argument("--tolerance", type=float, default=0.03)
    server.add_argument("--min-pairs", type=int, default=8)
    server.add_argument("--solve-batch-size", type=int, choices=[1, 5], default=1)
    server.add_argument("--smooth-window", type=int, default=5)
    server.add_argument("--k1", default=str(BASE_DIR / "data" / "K0.txt"), help="PC cam0 intrinsics, usually data/K0.txt.")
    server.add_argument("--k2", default=str(BASE_DIR / "data" / "K1.txt"), help="Pi cam2 intrinsics, usually data/K1.txt.")
    server.add_argument("--d1", default=str(BASE_DIR / "data" / "dist0.txt"), help="PC cam0 distortion, usually data/dist0.txt.")
    server.add_argument("--d2", default=str(BASE_DIR / "data" / "dist1.txt"), help="Pi cam2 distortion, usually data/dist1.txt.")
    server.add_argument("--make-3d-video", action="store_true", help="After recording, render 3D MP4 and HTML.")
    server.add_argument("--run-aruco", action="store_true", help="After recording, run aruco3.py reference extrinsics.")
    server.add_argument("--run-chessboard", action="store_true", help="After recording, run chessboard_stereo_calibrate.py reference extrinsics.")
    server.add_argument("--write-summary", action="store_true", help="Deprecated: summary is always written.")
    server.add_argument("--postprocess-all", action="store_true", help="Enable 3D video, ArUco, and chessboard. Summary is always written.")
    server.add_argument("--python-exe", default=str(Path(".venv") / "Scripts" / "python.exe"), help="Python executable for postprocess scripts.")
    server.add_argument("--aruco-marker-size", type=float, default=0.194)
    server.add_argument("--aruco-dict", default="DICT_6X6_50")
    server.add_argument("--aruco-stride", type=int, default=1)
    server.add_argument("--aruco-smooth", type=int, default=5)
    server.add_argument("--reconstruct-max-reproj", type=float, default=80.0)
    server.add_argument("--reconstruct-min-keep", type=int, default=4)

    pi = sub.add_parser("pi")
    pi.add_argument("--server", required=True, help="Example: ws://192.168.1.10:8765")
    pi.add_argument("--pi-root", default="~/one_program_test/pi_pose_video")
    pi.add_argument("--pi-cam", type=int, default=0)
    pi.add_argument("--enable-imu-fusion", action="store_true", help="Enable direct serial IMU capture and realtime fusion on the Pi client.")
    pi.add_argument("--imu-port", default="/dev/ttyUSB0", help="Serial port for the IMU device.")
    pi.add_argument("--imu-baud", type=int, default=921600, help="IMU serial baud rate.")
    pi.add_argument("--imu-timeout", type=float, default=0.2, help="IMU serial read timeout in seconds.")

    offline = sub.add_parser(
        "offline",
        help="Analyze pre-recorded video(s) with no camera/websocket needed. "
        "Works with an already-labeled MP_TO_COCO17 pose jsonl (--camX-json) "
        "or a raw unlabeled video (--camX-video, auto pose extraction via MediaPipe).",
    )
    offline.add_argument("--cam0-video", help="cam0 video file. Required unless --cam0-json is given.")
    offline.add_argument("--cam0-json", help="Pre-labeled cam0 pose jsonl (MP_TO_COCO17/keypoints17 schema). Skips MediaPipe extraction for cam0 if given.")
    offline.add_argument("--cam2-video", help="cam2 video file. Omit for single-camera pose-labeling-only mode (no stereo xyzq).")
    offline.add_argument("--cam2-json", help="Pre-labeled cam2 pose jsonl. Skips MediaPipe extraction for cam2 if given.")
    offline.add_argument("--tag", default=None, help="Run tag used to name output files. Default: current timestamp.")
    offline.add_argument("--fps", type=float, default=None, help="Override fps used for timestamp calc during pose extraction. Default: read from the video file itself (CAP_PROP_FPS).")
    offline.add_argument("--vis", type=float, default=0.5, help="Visibility threshold for keypoints and for stereo matching.")
    offline.add_argument("--tolerance", type=float, default=0.03, help="Max |dt| (seconds) to consider a cam0/cam2 frame pair matched.")
    offline.add_argument("--min-pairs", type=int, default=8, help="Minimum matched keypoints required to attempt pose estimation for a frame pair/batch.")
    offline.add_argument("--solve-batch-size", type=int, default=1, help="How many matched frame pairs to pool together per Essential-Matrix solve (accumulates more points, fewer but smoother poses).")
    offline.add_argument("--smooth-window", type=int, default=5, help="Sliding-window size for pose smoothing.")
    offline.add_argument("--save-annotated-video", action="store_true", help="When a camera's pose is extracted from a raw video (no --camX-json given), also save a pose-overlay mp4 alongside the jsonl.")
    offline.add_argument("--quiet", action="store_true", help="Suppress the per-frame [xyzq] console line (recommended for long offline batches).")
    offline.add_argument("--k1", default=str(BASE_DIR / "data" / "K0.txt"), help="cam0 intrinsics.")
    offline.add_argument("--k2", default=str(BASE_DIR / "data" / "K1.txt"), help="cam2 intrinsics.")
    offline.add_argument("--d1", default=str(BASE_DIR / "data" / "dist0.txt"), help="cam0 distortion (only used by --run-aruco/--run-chessboard postprocess steps).")
    offline.add_argument("--d2", default=str(BASE_DIR / "data" / "dist1.txt"), help="cam2 distortion (only used by --run-aruco/--run-chessboard postprocess steps).")
    offline.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT / "data"), help="Where xyzq_*.jsonl / cam_match_*.jsonl are written.")
    offline.add_argument("--out-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root for json/video outputs and results/summary.csv.")
    offline.add_argument("--make-3d-video", action="store_true", help="After estimation, render 3D MP4 and HTML.")
    offline.add_argument("--run-aruco", action="store_true", help="After estimation, run aruco3.py reference extrinsics (requires --cam0-video and --cam2-video, not just JSON).")
    offline.add_argument("--run-chessboard", action="store_true", help="After estimation, run chessboard_stereo_calibrate.py reference extrinsics (requires --cam0-video and --cam2-video, not just JSON).")
    offline.add_argument("--postprocess-all", action="store_true", help="Enable 3D video, ArUco, and chessboard postprocess. Summary is always written.")
    offline.add_argument("--python-exe", default=str(Path(".venv") / "Scripts" / "python.exe"), help="Python executable for postprocess scripts.")
    offline.add_argument("--aruco-marker-size", type=float, default=0.194)
    offline.add_argument("--aruco-dict", default="DICT_6X6_50")
    offline.add_argument("--aruco-stride", type=int, default=1)
    offline.add_argument("--aruco-smooth", type=int, default=5)
    offline.add_argument("--reconstruct-max-reproj", type=float, default=80.0)
    offline.add_argument("--reconstruct-min-keep", type=int, default=4)

    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "server":
        asyncio.run(server_main(args))
    elif args.mode == "pi":
        asyncio.run(pi_client(args))
    elif args.mode == "offline":
        run_offline(args)


if __name__ == "__main__":
    main()