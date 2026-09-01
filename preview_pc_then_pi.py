from __future__ import annotations

import argparse
import shlex
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


REMOTE_PREVIEW_CODE = r"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2


def discover_v4l2_capture_nodes() -> list[int]:
    dev_root = Path("/dev")
    nodes = []
    for node in dev_root.glob("video*"):
        match = re.fullmatch(r"video(\d+)", node.name)
        if match:
            nodes.append((int(match.group(1)), str(node)))
    nodes.sort(key=lambda item: item[0])

    if not shutil.which("v4l2-ctl"):
        return [idx for idx, _ in nodes if idx <= 9]

    capture_indices = []
    for idx, dev in nodes:
        if idx > 9:
            continue
        all_proc = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if "video capture" not in (all_proc.stdout or "").lower():
            continue
        fmt_proc = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if re.search(r"^\s*\[\d+\]:", fmt_proc.stdout or "", flags=re.MULTILINE):
            capture_indices.append(idx)
    return capture_indices


def build_candidates(requested_cam: int, backend: str):
    candidates = []
    seen = set()

    def add(source, api_pref, label):
        key = (str(source), int(api_pref))
        if key not in seen:
            seen.add(key)
            candidates.append((source, api_pref, label))

    if backend in {"auto", "v4l2"}:
        add(requested_cam, cv2.CAP_V4L2, f"index={requested_cam} via V4L2")
    if backend in {"auto", "any"}:
        add(requested_cam, cv2.CAP_ANY, f"index={requested_cam} via ANY")

    for idx in discover_v4l2_capture_nodes():
        if backend in {"auto", "v4l2"}:
            add(idx, cv2.CAP_V4L2, f"index={idx} via V4L2")
        if backend in {"auto", "any"}:
            add(idx, cv2.CAP_ANY, f"index={idx} via ANY")
    return candidates


def open_camera(args):
    tried = []
    for source, api_pref, label in build_candidates(args.cam, args.backend):
        tried.append(label)
        print(f"[PI] probing {label}", file=sys.stderr, flush=True)
        cap = cv2.VideoCapture(source, api_pref)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
        cap.set(cv2.CAP_PROP_FPS, float(args.fps))

        for _ in range(max(1, int(args.warmup_frames))):
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[PI] opened {label}", file=sys.stderr, flush=True)
                return cap, frame, label
            time.sleep(0.02)
        cap.release()

    tried_text = ", ".join(tried[:12])
    if len(tried) > 12:
        tried_text += f", ... ({len(tried)} total)"
    raise SystemExit(f"Cannot open Pi camera. Tried: {tried_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "v4l2", "any"], default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args()

    cap, first_frame, selected = open_camera(args)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    try:
        frame = first_frame
        while True:
            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                payload = encoded.tobytes()
                sys.stdout.buffer.write(struct.pack(">I", len(payload)))
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[PI] failed to read frame", file=sys.stderr, flush=True)
                time.sleep(0.03)
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        print(f"[PI] preview stopped ({selected})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
"""


def _backend_id(name: str) -> int:
    name = name.lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def _q_remote_path(value: str) -> str:
    if value.startswith("~/"):
        return "~/" + shlex.quote(value[2:])
    return shlex.quote(value)


def _read_exact(pipe, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            raise EOFError("stream ended")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _drain_stderr(pipe) -> None:
    for raw_line in iter(pipe.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if line:
            print(line, flush=True)


def _put_overlay(frame, text: str, subtext: str) -> None:
    cv2.putText(
        frame,
        text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        subtext,
        (12, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def preview_pc(args) -> bool:
    cap = cv2.VideoCapture(args.pc_cam, _backend_id(args.pc_backend))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open PC camera index={args.pc_cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    title = "PC camera preview"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[PC] failed to read frame", flush=True)
                return False

            _put_overlay(frame, "PC camera", "Press s to switch to Pi, q/ESC to quit")
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.destroyWindow(title)
                return True
            if key in (ord("q"), 27):
                cv2.destroyWindow(title)
                return False
    finally:
        cap.release()


def build_pi_command(args):
    remote_args = [
        "--cam",
        str(args.pi_cam),
        "--backend",
        args.pi_backend,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--jpeg-quality",
        str(args.jpeg_quality),
        "--warmup-frames",
        str(args.warmup_frames),
    ]
    remote_python = "python3 -u -c " + shlex.quote(REMOTE_PREVIEW_CODE)
    remote_python += " " + " ".join(shlex.quote(item) for item in remote_args)

    if args.pi_venv:
        remote_cmd = f"source {_q_remote_path(args.pi_venv)}/bin/activate && {remote_python}"
    else:
        remote_cmd = remote_python

    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.ssh_timeout}",
        f"{args.pi_user}@{args.pi_ip}",
        f"bash -lc {shlex.quote(remote_cmd)}",
    ]


def preview_pi(args) -> None:
    cmd = build_pi_command(args)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
    stderr_thread.start()

    title = "Pi camera preview"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    last_tick = time.time()
    preview_fps = 0.0

    try:
        while True:
            header = _read_exact(proc.stdout, 4)
            frame_size = struct.unpack(">I", header)[0]
            payload = _read_exact(proc.stdout, frame_size)

            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            now = time.time()
            delta = now - last_tick
            if delta > 0:
                current_fps = 1.0 / delta
                preview_fps = current_fps if preview_fps <= 0 else preview_fps * 0.9 + current_fps * 0.1
            last_tick = now

            _put_overlay(frame, "Pi camera", f"preview_fps={preview_fps:.1f}  Press s to stop")
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
            if key in (ord("q"), 27):
                break
    except EOFError:
        code = proc.poll()
        raise RuntimeError(f"Pi preview stream ended unexpectedly. ssh exit code={code}") from None
    finally:
        cv2.destroyWindow(title)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview PC camera first, then Pi camera. Press s to advance/stop."
    )
    parser.add_argument("--pc-cam", type=int, default=0)
    parser.add_argument("--pc-backend", choices=["any", "dshow", "msmf"], default="dshow")
    parser.add_argument("--pi-user", default="shc899085")
    parser.add_argument("--pi-ip", default="192.168.1.100")
    parser.add_argument("--pi-venv", default="~/pose_env")
    parser.add_argument("--pi-cam", type=int, default=0)
    parser.add_argument("--pi-backend", choices=["auto", "v4l2", "any"], default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--ssh-timeout", type=int, default=5)
    args = parser.parse_args()

    print("[PC] preview started. Press s to switch to Pi.", flush=True)
    if not preview_pc(args):
        cv2.destroyAllWindows()
        return

    print("[PI] preview starting over SSH. Press s to stop.", flush=True)
    try:
        preview_pi(args)
    finally:
        cv2.destroyAllWindows()
    print("[DONE] preview finished.", flush=True)


if __name__ == "__main__":
    main()
