import argparse
import csv
import json
import os
import re
import shutil
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).name
SCRIPT_STEM = Path(__file__).stem

FRAME_HEAD = 0xFC
FRAME_LEN = 120
PAYLOAD_GYRO_OFF = 0
PAYLOAD_ACCEL_OFF = 68

ACC_ABS_MAX = 80.0
GYRO_ABS_MAX = 30.0
ACC_NORM_MAX = 80.0
GYRO_NORM_MAX = 50.0


# MediaPipe 33 點到 COCO17 關鍵點的對應表，供離線 pose 輸出格式轉換使用。
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

# COCO17 骨架連線定義，供標註預覽影片時畫線使用。
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
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def q_remote_path(value: str) -> str:
    # 遠端路徑要經過 shell quote，避免空白或特殊字元破壞 SSH 命令。
    if value.startswith("~/"):
        return "~/" + shlex.quote(value[2:])
    return shlex.quote(value)


def ssh_base(args):
    # 組出所有 SSH 命令都會共用的前綴參數。
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.ssh_timeout}",
        f"{args.pi_user}@{args.pi_ip}",
    ]


def scp_base(args):
    # 組出所有 SCP 命令都會共用的前綴參數。
    return [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.ssh_timeout}",
        "-C",
    ]


def format_remote_cmd(cmd):
    # 讓 list 形式的命令也能轉成可讀字串，用於錯誤訊息。
    if isinstance(cmd, (list, tuple)):
        return " ".join(shlex.quote(str(token)) for token in cmd)
    return str(cmd)


def format_remote_fail_hint(args, action: str, cmd, stderr: str) -> str:
    # 依錯誤內容產生較有指向性的 Pi 偵錯提示。
    endpoint = f"{args.pi_user}@{args.pi_ip}"
    err_lower = (stderr or "").lower()
    is_space_or_write = (
        "no space left" in err_lower
        or "disk quota exceeded" in err_lower
        or "write remote" in err_lower
        or "failed to upload file" in err_lower
        or "read-only file system" in err_lower
    )
    msg = [
        f"{action} failed: {format_remote_cmd(cmd)}",
        f"Target Pi: {endpoint}",
    ]
    if stderr:
        msg.append(f"Remote error: {stderr}")
    if is_space_or_write:
        msg.extend(
            [
                "Quick checks:",
                f"  1) ssh {endpoint} \"df -h ~; df -i ~\"",
                f"  2) ssh {endpoint} \"ls -ld ~/pose_code; touch ~/pose_code/.write_test && rm -f ~/pose_code/.write_test\"",
                "  3) free some space on Pi if filesystem is full",
            ]
        )
    else:
        msg.extend(
            [
                "Quick checks:",
                f"  1) ping {args.pi_ip}",
                f"  2) ssh -o BatchMode=yes -o ConnectTimeout={args.ssh_timeout} {endpoint} \"echo ok\"",
                "  3) verify Pi power / Wi-Fi / IP address",
            ]
        )
    return "\n".join(msg)


def run_remote_with_retry(args, cmd, *, check=True, input_text=None, action="Remote command", timeout_s=None):
    # 對 SSH/SCP 這類遠端命令提供統一的 retry、timeout 與錯誤包裝。
    retries = max(0, int(getattr(args, "ssh_retries", 0)))
    attempts = retries + 1 if check else 1
    retry_delay = float(getattr(args, "ssh_retry_delay", 1.5))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return subprocess.run(
                cmd,
                check=check,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Missing `ssh`/`scp` executable in PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            timeout_desc = f"Timed out after {timeout_s:.1f}s" if timeout_s is not None else "Timed out"
            stderr = (exc.stderr or "")
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += timeout_desc
            if attempt < attempts:
                print(
                    f"[WARN] {action} failed ({attempt}/{attempts}): {timeout_desc}. "
                    f"Retrying in {retry_delay:.1f}s...",
                    flush=True,
                )
                time.sleep(retry_delay)
                continue
            if check:
                raise RuntimeError(format_remote_fail_hint(args, action, cmd, stderr.strip())) from exc
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=stderr,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < attempts:
                stderr = (exc.stderr or "").strip()
                short_err = stderr or f"exit code {exc.returncode}"
                print(
                    f"[WARN] {action} failed ({attempt}/{attempts}): {short_err}. "
                    f"Retrying in {retry_delay:.1f}s...",
                    flush=True,
                )
                time.sleep(retry_delay)
                continue
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(format_remote_fail_hint(args, action, cmd, stderr)) from exc
    if last_error is not None:
        stderr = (last_error.stderr or "").strip()
        raise RuntimeError(format_remote_fail_hint(args, action, cmd, stderr))
    raise RuntimeError(f"{action} failed unexpectedly.")


def run_ssh(args, command: str, *, check=True, input_text=None, action="SSH command", timeout_s=None):
    # 把完整遠端命令當成單一 SSH 參數傳入，避免遠端 shell 把它拆壞。
    # 否則 `bash -lc <cmd>` 內部內容可能在遠端被錯誤切分。
    remote_cmd = f"bash -lc {shlex.quote(command)}"
    return run_remote_with_retry(
        args,
        ssh_base(args) + [remote_cmd],
        check=check,
        input_text=input_text,
        action=action,
        timeout_s=timeout_s,
    )


def run_scp(args, src: str, dst: str, *, check=True, action="SCP command", timeout_s=None):
    # SCP 僅負責補上共用前綴與統一的 retry/timeout 行為。
    return run_remote_with_retry(
        args,
        scp_base(args) + [src, dst],
        check=check,
        action=action,
        timeout_s=timeout_s,
    )


def get_remote_available_kb(args, path: str = "~") -> int:
    # 使用 POSIX `df -Pk`，讓輸出在裝置名稱很長時仍然好解析。
    # 這裡不用 shell pipeline，避免 `df` 失敗時被管線掩蓋掉。
    target = "~" if path == "~" else q_remote_path(path)
    proc = run_ssh(args, f"df -Pk {target}", action="Check Pi disk space")
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("Cannot parse Pi disk space: empty `df` output.")
    parts = lines[-1].split()
    if len(parts) < 4:
        raise RuntimeError(f"Cannot parse Pi disk space from line: {lines[-1]}")
    try:
        return int(parts[3])
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse Pi available KB from line: {lines[-1]}") from exc


def ensure_dirs(root: Path):
    # 統一建立這支程式會用到的輸出資料夾結構。
    paths = {
        "root": root,
        "raw": root / "raw",
        "timestamps": root / "timestamps",
        "video": root / "video",
        "json": root / "json",
        "imu": root / "imu",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_capture_paths(paths, cam_id: str, tag: str):
    # 原始錄影檔固定寫到共用輸出目錄 `raw/`。
    raw_video = paths["raw"] / f"{cam_id}_{tag}_raw.mp4"
    # 每幀時間戳另外存 CSV，後續 pose 分析可以直接重用。
    timestamp_csv = paths["timestamps"] / f"{cam_id}_{tag}.csv"
    return raw_video, timestamp_csv


def build_imu_path(paths, cam_id: str, tag: str):
    return paths["imu"] / f"{cam_id}_{tag}_imu.csv"


def build_pose_paths(root_paths, cam_id: str, tag: str):
    # Pose 會讀取前面錄影流程產生的原始影片。
    raw_video = root_paths["raw"] / f"{cam_id}_{tag}_raw.mp4"
    # Pose 也需要錄影時輸出的 CSV 時間資訊。
    timestamp_csv = root_paths["timestamps"] / f"{cam_id}_{tag}.csv"
    # JSONL 會一幀寫一筆姿態資料，方便後續處理。
    out_json = root_paths["json"] / f"{cam_id}_{tag}.jsonl"
    # 標註後的預覽影片會寫到 `video/` 目錄。
    out_video = root_paths["video"] / f"{cam_id}_{tag}.mp4"
    return raw_video, timestamp_csv, out_json, out_video


def wait_for_file(path: Path, timeout_s: float, label: str):
    # 輪詢等待本機檔案出現，常用於 ready/start 同步控制檔。
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {label}: {path}")


def wait_for_remote_file(args, path: str, timeout_s: float, label: str):
    # 透過 SSH 在 Pi 上輪詢檔案是否存在。
    deadline = time.monotonic() + timeout_s
    quoted = q_remote_path(path)
    poll_timeout = float(getattr(args, "ssh_poll_timeout", 8.0))
    while time.monotonic() < deadline:
        proc = run_ssh(
            args,
            f"test -f {quoted}",
            check=False,
            action=f"Poll remote file ({label})",
            timeout_s=poll_timeout,
        )
        if proc.returncode == 0:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {label}: {path}")


def wait_for_pi_ready(args, ready_path: str, pi_log_path: Path, timeout_s: float):
    # 先看遠端 ready 檔，必要時退回看 Pi log 中的 ready 標記。
    deadline = time.monotonic() + timeout_s
    quoted = q_remote_path(ready_path)
    poll_timeout = float(getattr(args, "ssh_poll_timeout", 8.0))
    ready_marker = "[cam2] ready:"
    while time.monotonic() < deadline:
        proc = run_ssh(
            args,
            f"test -f {quoted}",
            check=False,
            action="Poll Pi ready file",
            timeout_s=poll_timeout,
        )
        if proc.returncode == 0:
            return "remote_file"

        if pi_log_path.exists():
            text = pi_log_path.read_text(encoding="utf-8", errors="replace")
            if ready_marker in text:
                return "pi_log"

        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for Pi camera ready: {ready_path}")


def wait_for_pc_ready(args, ready_path: Path, pc_log_path: Path, pc_proc, timeout_s: float):
    # PC 端先看 ready 檔，若檔案還沒落盤，就退回從 PC log 尋找 ready 標記。
    deadline = time.monotonic() + timeout_s
    ready_marker = "[cam0] ready:"
    while time.monotonic() < deadline:
        if ready_path.exists():
            return "local_file"

        if pc_proc.poll() is not None:
            raise RuntimeError(
                f"PC capture exited before ready (code={pc_proc.returncode}). See {pc_log_path}"
            )

        if pc_log_path.exists():
            text = pc_log_path.read_text(encoding="utf-8", errors="replace")
            if ready_marker in text:
                return "pc_log"

        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for PC camera ready: {ready_path}")


def write_json(path: Path, data):
    # 寫 JSON 前先確保父資料夾存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def read_start_file(path: Path):
    # start 檔目前只需要取出同步起錄的 wall-clock 時間。
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data["start_wall_ns"])


def discover_v4l2_capture_nodes():
    # 先把 `/dev/videoN` 這類數字節點收集起來，並固定用升冪探測。
    nodes = []
    for node in Path("/dev").glob("video*"):
        m = re.fullmatch(r"video(\d+)", node.name)
        if m:
            nodes.append((int(m.group(1)), str(node)))
    nodes.sort(key=lambda item: item[0])

    # 若系統沒有 `v4l2-ctl`，退回只嘗試通常較像實體相機的低編號節點。
    if not shutil.which("v4l2-ctl"):
        return [idx for idx, _ in nodes if idx <= 9]

    # 逐一探測節點，只保留有回報至少一種像素格式的裝置。
    capture_like = []
    for idx, dev in nodes:
        if idx > 9:
            continue

        all_proc = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        all_text = all_proc.stdout or ""
        if "video capture" not in all_text.lower():
            continue

        fmt_proc = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        fmt_text = fmt_proc.stdout or ""
        if re.search(r"^\s*\[\d+\]:", fmt_text, flags=re.MULTILINE):
            capture_like.append(idx)
    return capture_like


def build_capture_candidates(requested_cam: int, backend: str):
    # 候選開啟方式依序從最符合預期到較寬鬆的 fallback。
    candidates = []
    # 避免把等價的來源與 backend 組合重複加入。
    seen = set()

    def add(source, api_pref, label):
        key = (str(source), int(api_pref))
        if key in seen:
            return
        seen.add(key)
        candidates.append((source, api_pref, label))

    # Windows 端先試 DSHOW，再 fallback 到 OpenCV 預設 backend；
    # 有些機器上的 ANY 會在 read() 卡住，所以不要把它放在第一順位。
    if os.name == "nt":
        if backend in {"auto", "dshow"}:
            add(requested_cam, cv2.CAP_DSHOW, f"index={requested_cam} via DSHOW")
        if backend in {"auto", "default"}:
            add(requested_cam, cv2.CAP_ANY, f"index={requested_cam} via ANY")
        return candidates

    # Linux 先試數字 index，再試明確的 `/dev/videoN` 路徑。
    if os.name != "nt":
        add(requested_cam, cv2.CAP_V4L2, f"index={requested_cam} via V4L2")
        add(f"/dev/video{requested_cam}", cv2.CAP_V4L2, f"/dev/video{requested_cam} via V4L2")
        add(requested_cam, cv2.CAP_ANY, f"index={requested_cam} via ANY")
        add(f"/dev/video{requested_cam}", cv2.CAP_ANY, f"/dev/video{requested_cam} via ANY")

        capture_indices = discover_v4l2_capture_nodes()
        for idx in capture_indices:
            add(idx, cv2.CAP_V4L2, f"index={idx} via V4L2")
            add(f"/dev/video{idx}", cv2.CAP_V4L2, f"/dev/video{idx} via V4L2")
            add(idx, cv2.CAP_ANY, f"index={idx} via ANY")
            add(f"/dev/video{idx}", cv2.CAP_ANY, f"/dev/video{idx} via ANY")
        return candidates

    # Windows 在未強制 DSHOW 時，退回 OpenCV 的預設 backend。
    add(requested_cam, cv2.CAP_ANY, f"index={requested_cam} via ANY")
    return candidates


def open_capture_device(args, cam_id: str, requested_cam: int, fps: float):
    # 先建立完整探測清單，讓後面的開啟迴圈維持單純。
    capture_candidates = build_capture_candidates(requested_cam, args.backend)

    cap = None
    last_frame = None
    selected_cam = None
    selected_label = None
    tried_labels = []
    for source, api_pref, label in capture_candidates:
        tried_labels.append(label)
        print(f"[{cam_id}] probing camera candidate: {label}", flush=True)
        probe = cv2.VideoCapture(source, api_pref)
        if not probe.isOpened():
            print(f"[{cam_id}] candidate not opened: {label}", flush=True)
            probe.release()
            continue

        # 在暖機讀取前先套用使用者要求的解析度與 FPS。
        probe.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.w))
        probe.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.h))
        probe.set(cv2.CAP_PROP_FPS, fps)

        ok = False
        probe_frame = None
        for _ in range(int(args.warmup_frames)):
            ret, frame = probe.read()
            if ret and frame is not None:
                ok = True
                probe_frame = frame
                print(f"[{cam_id}] candidate produced first frame: {label}", flush=True)
                break
            time.sleep(0.01)

        if ok and probe_frame is not None:
            cap = probe
            last_frame = probe_frame
            selected_cam = source
            selected_label = label
            break
        probe.release()

    if cap is None or last_frame is None:
        tried_text = ", ".join(tried_labels[:12])
        if len(tried_labels) > 12:
            tried_text += f", ... ({len(tried_labels)} total)"
        raise RuntimeError(f"Cannot open camera index={requested_cam}. Tried: {tried_text}")
    if selected_label and selected_label != f"index={requested_cam} via V4L2" and selected_label != f"index={requested_cam} via DSHOW":
        print(f"[{cam_id}] requested camera index={requested_cam}, using fallback {selected_label}", flush=True)

    return cap, last_frame, selected_cam


def build_local_capture_cmd(args, tag, root, ready_file: Path, start_file: Path):
    # 本機相機錄影直接重用這支腳本的 `capture` 模式。
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "capture-pc",
        "--cam-id",
        "cam0",
        "--cam",
        str(args.pc_cam),
        "--tag",
        tag,
        "--secs",
        str(args.secs),
        "--fps",
        str(args.fps),
        "--w",
        str(args.w),
        "--h",
        str(args.h),
        "--root",
        str(root),
        "--ready-file",
        str(ready_file),
        "--start-file",
        str(start_file),
        "--warmup-frames",
        str(getattr(args, "warmup_frames", 5)),
        "--backend",
        str(getattr(args, "pc_backend", "auto")),
    ]
    if getattr(args, "enable_imu", False):
        cmd.extend(["--enable-imu", "--imu-port", str(args.imu_port), "--imu-baud", str(args.imu_baud), "--imu-timeout", str(args.imu_timeout)])
    return cmd


def build_remote_capture_cmd(args, tag, ready_file: str, start_file: str):
    # 遠端 Pi 在執行錄影前，先啟用它自己的虛擬環境。
    cmd_parts = [
            "source",
            f"{q_remote_path(args.pi_venv)}/bin/activate",
            "&&",
            "cd",
            q_remote_path(args.pi_workdir),
            "&&",
            "python3",
            shlex.quote(SCRIPT_NAME),
            "capture-pi",
            "--cam-id",
            "cam2",
            "--cam",
            shlex.quote(str(args.pi_cam)),
            "--tag",
            shlex.quote(tag),
            "--secs",
            shlex.quote(str(args.secs)),
            "--fps",
            shlex.quote(str(args.fps)),
            "--w",
            shlex.quote(str(args.w)),
            "--h",
            shlex.quote(str(args.h)),
            "--root",
            q_remote_path(args.pi_root),
            "--ready-file",
            q_remote_path(ready_file),
            "--start-file",
            q_remote_path(start_file),
                "--warmup-frames",
                shlex.quote(str(getattr(args, "warmup_frames", 5))),
            "--backend",
            "default",
    ]
    if getattr(args, "enable_imu", False):
        cmd_parts.extend(
            [
                "--enable-imu",
                "--imu-port",
                q_remote_path(str(args.imu_port)),
                "--imu-baud",
                shlex.quote(str(args.imu_baud)),
                "--imu-timeout",
                shlex.quote(str(args.imu_timeout)),
            ]
        )
    return " ".join(cmd_parts)


def build_pose_cmd(args, cam_id, tag, root_paths):
    # 依照統一命名規則組出 pose 的輸入與輸出檔案位置。
    raw_video, timestamp_csv, out_json, out_video = build_pose_paths(root_paths, cam_id, tag)
    # 直接重用這支腳本的 `pose` 模式，避免分散邏輯。
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "pose",
        "--cam-id",
        cam_id,
        "--tag",
        tag,
        "--video",
        str(raw_video),
        "--timestamps",
        str(timestamp_csv),
        "--out-json",
        str(out_json),
        "--out-video",
        str(out_video),
        "--fps",
        str(args.fps),
        "--vis",
        str(args.vis),
    ]


def prepare_run_paths(args):
    # PC 輸出與從 Pi 拉回來的輸出都共用同一套目錄結構。
    pc_paths = ensure_dirs(Path(args.pc_root))
    pi_paths = ensure_dirs(Path(args.pi_local_root))
    # 同步 log 放在腳本旁邊，出問題時比較好查。
    log_dir = BASE_DIR / "sync_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return pc_paths, pi_paths, log_dir


def prepare_sync_files(tag: str, log_dir: Path):
    # 本機 ready/start 檔用來協調 PC 端錄影子程序。
    pc_ready = log_dir / f"{SCRIPT_STEM}_pc_ready_{tag}.json"
    pc_start = log_dir / f"{SCRIPT_STEM}_pc_start_{tag}.json"
    # 遠端 ready/start 檔用來協調 Pi 端錄影子程序。
    pi_ready = f"/tmp/{SCRIPT_STEM}_pi_ready_{tag}.json"
    pi_start = f"/tmp/{SCRIPT_STEM}_pi_start_{tag}.json"
    return pc_ready, pc_start, pi_ready, pi_start


def cleanup_sync_files(args, pc_ready: Path, pc_start: Path, pi_ready: str, pi_start: str):
    # 先清掉本機舊控制檔，避免上一輪執行殘留狀態。
    for path in (pc_ready, pc_start):
        if path.exists():
            path.unlink()
    # Pi 端的舊控制檔也一併清掉，避免誤判 ready/start。
    run_ssh(args, f"rm -f {q_remote_path(pi_ready)} {q_remote_path(pi_start)}", check=False)


def wait_for_ready_and_start(args, tag: str, pc_ready: Path, pc_start: Path, pi_ready: str, pi_log: Path, pc_proc, pc_log: Path):
    # 先等本機 ready，再等 Pi ready，兩邊都就緒後才發 start 訊號。
    pc_ready_source = wait_for_pc_ready(args, pc_ready, pc_log, pc_proc, float(args.ready_timeout))
    if pc_ready_source == "pc_log":
        print("[WARN] PC ready detected from log marker (ready file was not observed yet).", flush=True)
    pi_ready_source = wait_for_pi_ready(args, pi_ready, pi_log, float(args.ready_timeout))
    if pi_ready_source == "pi_log":
        print("[WARN] Pi ready detected from log marker (remote file polling was unavailable).", flush=True)

    # 使用未來時間點做同步起錄，讓兩邊依同一份 payload 開始。
    start_wall_ns = time.time_ns() + int(float(args.start_delay) * 1e9)
    payload = {"start_wall_ns": start_wall_ns, "tag": tag}
    return payload


def print_countdown(seconds: float):
    # 在真正送出 start 訊號前，先在 terminal 顯示倒數。
    whole = int(round(float(seconds)))
    if whole <= 0:
        return
    for n in range(whole, 0, -1):
        print(f"[RUN] Starting in {n}...", flush=True)
        time.sleep(1)


def wait_for_capture_processes(pc_proc, pi_proc, log_dir: Path, tag: str):
    # 等待兩個錄影子程序結束，並把非零退出碼轉成明確錯誤。
    pc_code = pc_proc.wait()
    pi_code = pi_proc.wait()
    if pc_code != 0:
        raise RuntimeError(f"PC capture failed. See {log_dir / f'{SCRIPT_STEM}_pc_{tag}.log'}")
    if pi_code != 0:
        raise RuntimeError(f"Pi capture failed. See {log_dir / f'{SCRIPT_STEM}_pi_{tag}.log'}")


def terminate_capture_processes(pc_proc, pi_proc):
    # 只在程序還活著時才送終止訊號，避免多餘例外。
    if pc_proc.poll() is None:
        pc_proc.terminate()
    if pi_proc.poll() is None:
        pi_proc.terminate()


def build_timeout_error(exc: TimeoutError, pi_log: Path):
    # Timeout 時把 Pi log 最後幾行附上，方便直接看卡在哪裡。
    pi_tail = ""
    if pi_log.exists():
        pi_tail = "\n".join(pi_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
    extra = f"\nPi log tail ({pi_log}):\n{pi_tail}" if pi_tail else f"\nPi log: {pi_log}"
    return RuntimeError(f"{exc}{extra}")


def run_pose_pipeline(args, tag: str, pc_paths, pi_paths):
    # 若未跳過 pose，就依序處理 PC 與 Pi 兩份影片。
    if args.skip_pose:
        return
    print("[RUN] Running offline pose for PC video...")
    run_pose_subprocess(args, "cam0", tag, pc_paths)
    print("[RUN] Running offline pose for Pi video...")
    run_pose_subprocess(args, "cam2", tag, pi_paths)


def print_run_outputs(tag: str, pc_paths, pi_paths, skip_pose: bool):
    # 最後統一列出本次 run 產生的主要輸出檔案位置。
    print("[DONE] Outputs:")
    print(f"  PC raw : {pc_paths['raw'] / f'cam0_{tag}_raw.mp4'}")
    print(f"  Pi raw : {pi_paths['raw'] / f'cam2_{tag}_raw.mp4'}")
    if not skip_pose:
        print(f"  PC json: {pc_paths['json'] / f'cam0_{tag}.jsonl'}")
        print(f"  Pi json: {pi_paths['json'] / f'cam2_{tag}.jsonl'}")


def add_size_args(parser):
    parser.add_argument("--w", type=int, default=1280)
    parser.add_argument("--h", type=int, default=720)


def add_timing_args(parser, *, secs_required: bool, fps_default, include_vis: bool):
    parser.add_argument("--secs", type=float, required=secs_required, default=None if secs_required else 10.0)
    parser.add_argument("--fps", type=float, required=secs_required, default=fps_default)
    if include_vis:
        parser.add_argument("--vis", type=float, default=0.5)


def add_capture_worker_args(parser):
    parser.add_argument("--cam-id", required=True)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--tag", required=True)
    add_timing_args(parser, secs_required=True, fps_default=None, include_vis=False)
    add_size_args(parser)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--start-file", required=True)
    parser.add_argument("--backend", choices=["auto", "default", "dshow"], default="auto")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--start-timeout", type=float, default=120.0)
    parser.add_argument("--enable-imu", action="store_true")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-baud", type=int, default=921600)
    parser.add_argument("--imu-timeout", type=float, default=0.2)


class SyncedClock:
    def __init__(self, base_wall_ns: int, base_mono_ns: int):
        self.base_wall_ns = base_wall_ns
        self.base_mono_ns = base_mono_ns

    def now_ns(self) -> int:
        return self.base_wall_ns + (time.monotonic_ns() - self.base_mono_ns)


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


def read_one_imu_frame(ser) -> bytes:
    sync_to_head(ser)
    return bytes([FRAME_HEAD]) + read_exact(ser, FRAME_LEN - 1)


def finite(v: float) -> bool:
    return (v == v) and (v != float("inf")) and (v != float("-inf"))


def norm3(x: float, y: float, z: float) -> float:
    return (x * x + y * y + z * z) ** 0.5


def remap_imu_to_camera(gx: float, gy: float, gz: float, ax: float, ay: float, az: float):
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

    gx = f32le(payload[PAYLOAD_GYRO_OFF : PAYLOAD_GYRO_OFF + 4])
    gy = f32le(payload[PAYLOAD_GYRO_OFF + 4 : PAYLOAD_GYRO_OFF + 8])
    gz = f32le(payload[PAYLOAD_GYRO_OFF + 8 : PAYLOAD_GYRO_OFF + 12])
    ax = f32le(payload[PAYLOAD_ACCEL_OFF : PAYLOAD_ACCEL_OFF + 4])
    ay = f32le(payload[PAYLOAD_ACCEL_OFF + 4 : PAYLOAD_ACCEL_OFF + 8])
    az = f32le(payload[PAYLOAD_ACCEL_OFF + 8 : PAYLOAD_ACCEL_OFF + 12])

    for v in (gx, gy, gz, ax, ay, az):
        if not finite(v):
            return None
    if abs(ax) > ACC_ABS_MAX or abs(ay) > ACC_ABS_MAX or abs(az) > ACC_ABS_MAX:
        return None
    if abs(gx) > GYRO_ABS_MAX or abs(gy) > GYRO_ABS_MAX or abs(gz) > GYRO_ABS_MAX:
        return None
    if norm3(ax, ay, az) > ACC_NORM_MAX or norm3(gx, gy, gz) > GYRO_NORM_MAX:
        return None

    gx, gy, gz, ax, ay, az = remap_imu_to_camera(gx, gy, gz, ax, ay, az)
    for v in (gx, gy, gz, ax, ay, az):
        if not finite(v):
            return None
    return gx, gy, gz, ax, ay, az


def imu_thread_worker(port: str, baud: int, timeout: float, out_imu_csv: str, clock: SyncedClock, stop_evt: threading.Event):
    try:
        import serial
    except ImportError:
        print("[IMU] pyserial not installed; IMU recording disabled.", flush=True)
        return

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"[IMU] failed to open {port}: {exc}", flush=True)
        return

    last_t_ns = None
    with open(out_imu_csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        while not stop_evt.is_set():
            try:
                frame = read_one_imu_frame(ser)
                imu = parse_imu_from_frame(frame)
                if imu is None:
                    continue
                gx, gy, gz, ax, ay, az = imu
                t_ns = clock.now_ns()
                if last_t_ns is not None and t_ns <= last_t_ns:
                    continue
                last_t_ns = t_ns
                fcsv.write(f"{t_ns},{gx:.8f},{gy:.8f},{gz:.8f},{ax:.8f},{ay:.8f},{az:.8f}\n")
                if ((t_ns // 10_000_000) % 50) == 0:
                    fcsv.flush()
            except TimeoutError:
                continue
            except Exception:
                continue
    ser.close()


def create_capture_writer(raw_video: Path, cap, fps: float):
    # 依實際第一幀大小建立 VideoWriter，避免尺寸與相機輸出不一致。
    height, width = cap.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {raw_video}")
    return writer, width, height


def write_ready_file(ready_file: Path, cam_id: str, tag: str, width: int, height: int, fps: float, target_frames: int, selected_cam):
    # 相機完成暖機後，把實際參數寫進 ready 檔供 controller 判定同步狀態。
    ready_data = {
        "cam": cam_id,
        "tag": tag,
        "ready_wall_ns": time.time_ns(),
        "width": width,
        "height": height,
        "fps": fps,
        "target_frames": target_frames,
        "cam_index": str(selected_cam),
    }
    write_json(ready_file, ready_data)
    print(f"[{cam_id}] ready: {width}x{height}, fps={fps}, frames={target_frames}", flush=True)


def wait_until_capture_start(start_file: Path, start_timeout: float):
    # 等 controller 寫入 start 檔後，再精準等到指定的同步起錄時間點。
    wait_for_file(start_file, start_timeout, "start file")
    start_wall_ns = read_start_file(start_file)
    while True:
        remain = (start_wall_ns - time.time_ns()) / 1e9
        if remain <= 0:
            break
        time.sleep(min(remain, 0.01))
    return start_wall_ns, time.monotonic()


def capture_one_frame(cap, last_frame):
    # 嘗試讀一幀；若讀取失敗，就沿用上一幀並標記 duplicated。
    ret, frame = cap.read()
    duplicated = False
    if ret and frame is not None:
        return frame, frame, duplicated
    duplicated = True
    return last_frame, last_frame, duplicated


def record_capture_frames(cap, writer, timestamp_csv: Path, last_frame, target_frames: int, dt: float, start_mono: float, start_wall_ns: int):
    # 依固定時間槽寫影片與 timestamp，保持輸出幀數穩定。
    duplicate_frames = 0
    late_frames = 0

    with timestamp_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer_csv = csv.DictWriter(
            fcsv,
            fieldnames=["frame", "timestamp_wall", "timestamp", "duplicated", "late"],
        )
        writer_csv.writeheader()

        for frame_idx in range(target_frames):
            slot_mono = start_mono + frame_idx * dt
            while True:
                remain = slot_mono - time.monotonic()
                if remain <= 0:
                    break
                time.sleep(min(remain, 0.005))

            late = time.monotonic() - slot_mono
            if late > dt:
                late_frames += 1

            frame, last_frame, duplicated = capture_one_frame(cap, last_frame)
            if duplicated:
                duplicate_frames += 1

            writer.write(frame)
            timestamp_wall = start_wall_ns / 1e9 + frame_idx * dt
            writer_csv.writerow(
                {
                    "frame": frame_idx,
                    "timestamp_wall": f"{timestamp_wall:.9f}",
                    "timestamp": f"{frame_idx * dt:.9f}",
                    "duplicated": int(duplicated),
                    "late": f"{max(0.0, late):.6f}",
                }
            )

    return duplicate_frames, late_frames


def finish_capture(cam_id: str, cap, writer, raw_video: Path, timestamp_csv: Path, duplicate_frames: int, late_frames: int):
    # 釋放資源並統一輸出本次錄影摘要。
    cap.release()
    writer.release()
    print(
        f"[{cam_id}] saved raw={raw_video} timestamps={timestamp_csv} "
        f"duplicates={duplicate_frames} late_slots={late_frames}",
        flush=True,
    )


def run_capture_worker(args, role: str):
    # `capture` 模式負責單機錄影，不處理 PC/Pi 雙機協調。
    root = Path(args.root).expanduser()
    paths = ensure_dirs(root)
    cam_id = args.cam_id
    tag = args.tag
    fps = float(args.fps)
    target_frames = max(1, int(round(float(args.secs) * fps)))
    dt = 1.0 / fps

    raw_video, timestamp_csv = build_capture_paths(paths, cam_id, tag)
    ready_file = Path(args.ready_file).expanduser()
    start_file = Path(args.start_file).expanduser()

    print(
        f"[{cam_id}] capture start role={role} cam={args.cam} backend={getattr(args, 'backend', 'default')} "
        f"ready={ready_file} start={start_file}",
        flush=True,
    )

    if ready_file.exists():
        ready_file.unlink()

    requested_cam = int(args.cam)
    cap, last_frame, selected_cam = open_capture_device(args, cam_id, requested_cam, fps)
    print(f"[{cam_id}] camera opened: selected={selected_cam}", flush=True)
    imu_stop_evt = None
    imu_thread = None

    try:
        writer, width, height = create_capture_writer(raw_video, last_frame, fps)
        print(f"[{cam_id}] writer ready: {raw_video}", flush=True)
        write_ready_file(ready_file, cam_id, tag, width, height, fps, target_frames, selected_cam)
        start_wall_ns, start_mono = wait_until_capture_start(start_file, float(args.start_timeout))
        print(f"[{cam_id}] start file received: {start_file}", flush=True)
        if role == "pi" and getattr(args, "enable_imu", False):
            imu_csv = build_imu_path(paths, cam_id, tag)
            imu_stop_evt = threading.Event()
            imu_clock = SyncedClock(start_wall_ns, time.monotonic_ns())
            imu_thread = threading.Thread(
                target=imu_thread_worker,
                args=(args.imu_port, args.imu_baud, args.imu_timeout, str(imu_csv), imu_clock, imu_stop_evt),
                daemon=True,
            )
            imu_thread.start()
        duplicate_frames, late_frames = record_capture_frames(
            cap,
            writer,
            timestamp_csv,
            last_frame,
            target_frames,
            dt,
            start_mono,
            start_wall_ns,
        )
        if imu_stop_evt is not None:
            imu_stop_evt.set()
        if imu_thread is not None:
            imu_thread.join(timeout=2.0)
        finish_capture(cam_id, cap, writer, raw_video, timestamp_csv, duplicate_frames, late_frames)
    except Exception:
        if imu_stop_evt is not None:
            imu_stop_evt.set()
        if imu_thread is not None:
            imu_thread.join(timeout=2.0)
        cap.release()
        raise


def capture_mode(args):
    # 舊的 capture 模式保留相容性，預設仍走共用 worker。
    run_capture_worker(args, role="capture")


def capture_pc_mode(args):
    # PC 專用 worker；之後若要加 PC 特有處理，可從這裡延伸。
    run_capture_worker(args, role="pc")


def capture_pi_mode(args):
    # Pi 專用 worker；之後可在這裡加上 IMU 記錄等 Pi 特有流程。
    run_capture_worker(args, role="pi")


def load_timestamps(path: Path):
    # 把 CSV 時間戳讀成 frame index -> row 的對照表。
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            try:
                idx = int(row["frame"])
                rows[idx] = row
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def draw_pose(frame_bgr, landmarks, visibility_threshold: float):
    # 把 MediaPipe landmark 轉成 COCO17，並順便畫出預覽骨架。
    height, width = frame_bgr.shape[:2]
    annot = frame_bgr.copy()
    keypoints17 = []
    coco_points = {}

    if not landmarks:
        return annot, keypoints17

    lms = landmarks.landmark
    for coco_id, mp_idx, name in MP_TO_COCO17:
        lm = lms[mp_idx]
        x = lm.x * width
        y = lm.y * height
        vis = float(lm.visibility)
        coco_points[coco_id] = (x, y, vis)
        keypoints17.append(
            {
                "id": coco_id,
                "name": name,
                "x": float(x),
                "y": float(y),
                "visibility": vis,
            }
        )

    for a, b in SKELETON_EDGES:
        x1, y1, v1 = coco_points[a]
        x2, y2, v2 = coco_points[b]
        if v1 < visibility_threshold or v2 < visibility_threshold:
            continue
        cv2.line(annot, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)

    for coco_id, (x, y, vis) in coco_points.items():
        if vis < visibility_threshold:
            continue
        cv2.circle(annot, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(
            annot,
            f"{coco_id}:{vis:.2f}",
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annot, keypoints17


def pose_mode(args):
    # `pose` 模式負責讀影片、跑姿態估計、輸出 JSONL 與標註影片。
    import mediapipe as mp

    video_path = Path(args.video).expanduser()
    timestamp_path = Path(args.timestamps).expanduser()
    out_json = Path(args.out_json).expanduser()
    out_video = Path(args.out_video).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    timestamps = load_timestamps(timestamp_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(args.fps) if args.fps else float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter: {out_video}")

    mp_pose = mp.solutions.pose
    frame_idx = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose, out_json.open(
        "w", encoding="utf-8"
    ) as fjson:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            annot, keypoints17 = draw_pose(frame_bgr, result.pose_landmarks, float(args.vis))
            writer.write(annot)

            row = timestamps.get(frame_idx, {})
            timestamp = float(row.get("timestamp", frame_idx / fps))
            timestamp_wall = row.get("timestamp_wall")
            record = {
                "cam": args.cam_id,
                "tag": args.tag,
                "frame": frame_idx,
                "timestamp": round(timestamp, 9),
                "timestamp_wall": float(timestamp_wall) if timestamp_wall else None,
                "keypoints17": keypoints17,
            }
            fjson.write(json.dumps(record, ensure_ascii=False) + "\n")
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"[pose] {args.cam_id}: frames={frame_idx} json={out_json} video={out_video}", flush=True)


def copy_self_to_pi(args):
    # 每次 run 前都把目前腳本同步到 Pi，避免兩端版本不一致。
    remote_workdir = q_remote_path(args.pi_workdir)
    run_ssh(args, f"mkdir -p {remote_workdir}", action="Create Pi workdir")
    local_file = Path(__file__).resolve()
    need_kb = max(1, (local_file.stat().st_size + 1023) // 1024)
    avail_kb = get_remote_available_kb(args, "~")
    if avail_kb < need_kb:
        raise RuntimeError(
            "Pi disk is full or almost full for upload.\n"
            f"Required >= {need_kb} KB, available {avail_kb} KB under ~.\n"
            "Run on Pi: `df -h ~` and remove files (for example old videos/logs), then retry."
        )
    run_scp(
        args,
        str(local_file),
        f"{args.pi_user}@{args.pi_ip}:{args.pi_workdir.rstrip('/')}/{SCRIPT_NAME}",
        action=f"Copy {SCRIPT_NAME} to Pi",
    )


def start_pc_capture(args, tag, paths, ready_file: Path, start_file: Path, log_file: Path):
    # 組指令的邏輯集中管理，之後 CLI 參數調整比較不會漏掉。
    cmd = build_local_capture_cmd(args, tag, paths["root"], ready_file, start_file)
    return subprocess.Popen(
        cmd,
        stdout=log_file.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )


def start_pi_capture(args, tag, ready_file: str, start_file: str, log_file: Path):
    # 遠端錄影也走同樣模式，只是改成透過 SSH 在 Pi 上啟動。
    remote_cmd = build_remote_capture_cmd(args, tag, ready_file, start_file)
    remote_shell_cmd = f"bash -lc {shlex.quote(remote_cmd)}"
    return subprocess.Popen(
        ssh_base(args) + [remote_shell_cmd],
        stdout=log_file.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )


def pull_pi_capture(args, tag, pi_paths):
    # 從 Pi 拉回 raw 影片與對應 timestamps，供本機後續 pose 使用。
    pi_paths["raw"].mkdir(parents=True, exist_ok=True)
    pi_paths["timestamps"].mkdir(parents=True, exist_ok=True)
    remote_raw = f"{args.pi_root.rstrip('/')}/raw/cam2_{tag}_raw.mp4"
    remote_ts = f"{args.pi_root.rstrip('/')}/timestamps/cam2_{tag}.csv"
    run_scp(
        args,
        f"{args.pi_user}@{args.pi_ip}:{remote_raw}",
        str(pi_paths["raw"]),
        action="Download Pi raw video",
    )
    run_scp(
        args,
        f"{args.pi_user}@{args.pi_ip}:{remote_ts}",
        str(pi_paths["timestamps"]),
        action="Download Pi timestamps",
    )
    if getattr(args, "enable_imu", False):
        pi_paths["imu"].mkdir(parents=True, exist_ok=True)
        remote_imu = f"{args.pi_root.rstrip('/')}/imu/cam2_{tag}_imu.csv"
        run_scp(
            args,
            f"{args.pi_user}@{args.pi_ip}:{remote_imu}",
            str(pi_paths["imu"]),
            action="Download Pi IMU CSV",
        )


def run_pose_subprocess(args, cam_id, tag, root_paths):
    # 透過共用的路徑 helper 組指令，避免命名規則散落各處。
    cmd = build_pose_cmd(args, cam_id, tag, root_paths)
    subprocess.run(cmd, check=True)


def run_controller(args):
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    # 先準備好所有輸出資料夾，再啟動任何子程序。
    pc_paths, pi_paths, log_dir = prepare_run_paths(args)
    # 依這次 tag 產生對應的同步控制檔路徑。
    pc_ready, pc_start, pi_ready, pi_start = prepare_sync_files(tag, log_dir)
    # 本機與 Pi 都先清掉舊檔，確保這次 run 乾淨開始。
    cleanup_sync_files(args, pc_ready, pc_start, pi_ready, pi_start)

    print(f"[RUN] tag={tag}")
    print(f"[RUN] Copying {SCRIPT_NAME} to Pi...")
    copy_self_to_pi(args)

    print("[RUN] Starting both cameras and waiting for ready files...")
    pc_log = log_dir / f"{SCRIPT_STEM}_pc_{tag}.log"
    pi_log = log_dir / f"{SCRIPT_STEM}_pi_{tag}.log"
    pc_proc = start_pc_capture(args, tag, pc_paths, pc_ready, pc_start, pc_log)
    pi_proc = start_pi_capture(args, tag, pi_ready, pi_start, pi_log)

    try:
        payload = wait_for_ready_and_start(args, tag, pc_ready, pc_start, pi_ready, pi_log, pc_proc, pc_log)
        print_countdown(float(args.start_delay))
        write_json(pc_start, payload)
        run_ssh(args, f"cat > {q_remote_path(pi_start)}", input_text=json.dumps(payload), action="Send start signal to Pi")
        print(f"[RUN] Both cameras ready. Start signal sent.")

        wait_for_capture_processes(pc_proc, pi_proc, log_dir, tag)

    except KeyboardInterrupt:
        pc_proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        pi_proc.terminate()
        raise
    except TimeoutError as exc:
        try:
            terminate_capture_processes(pc_proc, pi_proc)
        except Exception:
            pass
        raise build_timeout_error(exc, pi_log) from exc

    print("[RUN] Pulling Pi raw video, timestamps, and IMU data...")
    pull_pi_capture(args, tag, pi_paths)

    run_pose_pipeline(args, tag, pc_paths, pi_paths)
    print_run_outputs(tag, pc_paths, pi_paths, args.skip_pose)
    if getattr(args, "enable_imu", False):
        print(f"  Pi imu : {pi_paths['imu'] / f'cam2_{tag}_imu.csv'}")


def build_parser():
    # 三個子命令共用同一份 parser：run / capture / pose。
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    run = sub.add_parser("run")
    run.add_argument("--pi_user", default="shc899085")
    run.add_argument("--pi_ip", default="192.168.1.100")
    run.add_argument("--pi_venv", default="~/pose_env")
    run.add_argument("--pi_workdir", default="~/pose_code")
    run.add_argument("--pi_root", default="~/pi_pose_video")
    run.add_argument("--pi_local_root", default=str(BASE_DIR / "pi_pose_video"))
    run.add_argument("--pi_cam", type=int, default=0)
    run.add_argument("--pc_root", default=str(BASE_DIR / "pc_pose_video"))
    run.add_argument("--pc_cam", type=int, default=0)
    run.add_argument("--pc-backend", choices=["auto", "default", "dshow"], default="auto")
    run.add_argument("--tag", default=None)
    add_timing_args(run, secs_required=False, fps_default=10.0, include_vis=True)
    add_size_args(run)
    run.add_argument("--start-delay", type=float, default=3.0)
    run.add_argument("--ready-timeout", type=float, default=30.0)
    run.add_argument("--warmup-frames", type=int, default=5)
    run.add_argument("--ssh-timeout", type=int, default=5)
    run.add_argument("--ssh-poll-timeout", type=float, default=8.0, help="Per-attempt timeout for SSH polling commands.")
    run.add_argument("--ssh-retries", type=int, default=2, help="Retries for SSH/SCP commands after first failure.")
    run.add_argument("--ssh-retry-delay", type=float, default=1.5, help="Seconds to wait between SSH/SCP retries.")
    run.add_argument("--skip-pose", action="store_true")
    run.add_argument("--enable-imu", action="store_true")
    run.add_argument("--imu-port", default="/dev/ttyUSB0")
    run.add_argument("--imu-baud", type=int, default=921600)
    run.add_argument("--imu-timeout", type=float, default=0.2)

    capture = sub.add_parser("capture")
    add_capture_worker_args(capture)

    capture_pc = sub.add_parser("capture-pc")
    add_capture_worker_args(capture_pc)

    capture_pi = sub.add_parser("capture-pi")
    add_capture_worker_args(capture_pi)

    pose = sub.add_parser("pose")
    pose.add_argument("--cam-id", required=True)
    pose.add_argument("--tag", required=True)
    pose.add_argument("--video", required=True)
    pose.add_argument("--timestamps", required=True)
    pose.add_argument("--out-json", required=True)
    pose.add_argument("--out-video", required=True)
    pose.add_argument("--fps", type=float, default=0.0)
    pose.add_argument("--vis", type=float, default=0.5)

    return parser


def main():
    # 沒有明確給子命令時，預設走 `run` 主流程。
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"run", "capture", "capture-pc", "capture-pi", "pose"}:
        argv = ["run"] + argv
    args = parser.parse_args(argv)

    try:
        if args.mode == "capture":
            capture_mode(args)
        elif args.mode == "capture-pc":
            capture_pc_mode(args)
        elif args.mode == "capture-pi":
            capture_pi_mode(args)
        elif args.mode == "pose":
            pose_mode(args)
        else:
            run_controller(args)
    except (RuntimeError, TimeoutError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
