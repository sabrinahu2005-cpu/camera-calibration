#!/usr/bin/env python3
# encoding: utf-8
"""Keyboard-hold front/back/left/right teleop for Turtle/TurboPi-like mecanum chassis.

This script is intended to be launched on the ROS 2 side and controlled from a
terminal. Press and hold W/A/S/D (or arrow keys) to move, release to stop.
It keeps publishing cmd_vel at a fixed rate and always sends zero velocity on
exit so the chassis does not keep drifting.

Examples:
    python drive_lr.py
    python drive_lr.py --speed 0.3 --hz 20
"""

from __future__ import annotations

import argparse
import select
import signal
import sys
import threading
import time
from typing import Optional

import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class StableLrDrive(Node):
    def __init__(self, topic: str, speed: float, hz: float, release_timeout: float):
        super().__init__("stable_lr_drive")
        self.topic = topic
        self.pub = self.create_publisher(Twist, topic, 1)
        self.speed = float(speed)
        self.hz = max(1.0, float(hz))
        self.release_timeout = max(0.05, float(release_timeout))
        self._stop_evt = threading.Event()
        self._last_key_ts = 0.0
        self._active_direction: Optional[str] = None

    def _make_twist(self, direction: Optional[str]) -> Twist:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        if direction == "left":
            twist.linear.y = self.speed
        elif direction == "right":
            twist.linear.y = -self.speed
        elif direction == "forward":
            twist.linear.x = self.speed
        elif direction == "backward":
            twist.linear.x = -self.speed
        elif direction == "diag_lf":
            twist.linear.x = self.speed
            twist.linear.y = self.speed
        elif direction == "diag_lb":
            twist.linear.x = -self.speed
            twist.linear.y = self.speed
        elif direction == "diag_rf":
            twist.linear.x = self.speed
            twist.linear.y = -self.speed
        elif direction == "diag_rb":
            twist.linear.x = -self.speed
            twist.linear.y = -self.speed
        return twist

    def _publish(self, direction: Optional[str]) -> None:
        self.pub.publish(self._make_twist(direction))

    def _set_active_direction(self, direction: str) -> None:
        if direction not in ("left", "right", "forward", "backward"):
            raise ValueError("direction must be one of: left, right, forward, backward")
        self._active_direction = direction
        self._last_key_ts = time.monotonic()

    def stop(self) -> None:
        if self._stop_evt.is_set():
            return
        self._stop_evt.set()
        self._active_direction = None
        self._publish(None)
        self.get_logger().info("stop and publish zero cmd_vel")

    def run_hold(self) -> None:
        self.get_logger().info(
            f"keyboard-hold drive ready: press and hold A/D (or left/right arrow), speed={self.speed:.3f}, hz={self.hz:.1f}, topic={self.topic}"
        )
        settings = None
        try:
            settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            period = 1.0 / self.hz
            next_tick = time.monotonic()
            while rclpy.ok() and not self._stop_evt.is_set():
                now = time.monotonic()

                while select.select([sys.stdin], [], [], 0.0)[0]:
                    key = sys.stdin.read(1)
                    if not key:
                        break
                    if key == "\x03":
                        raise KeyboardInterrupt
                    if key == " ":
                        self._active_direction = None
                        self._publish(None)
                        self.get_logger().info("space pressed -> stop")
                        continue

                    if key == "\x1b":
                        seq = key
                        for _ in range(2):
                            if select.select([sys.stdin], [], [], 0.0)[0]:
                                seq += sys.stdin.read(1)
                        if seq == "\x1b[A":
                            self._set_active_direction("forward")
                            self.get_logger().info("forward")
                        elif seq == "\x1b[B":
                            self._set_active_direction("backward")
                            self.get_logger().info("backward")
                        elif seq == "\x1b[D":
                            self._set_active_direction("left")
                            self.get_logger().info("left")
                        elif seq == "\x1b[C":
                            self._set_active_direction("right")
                            self.get_logger().info("right")
                        continue

                    lowered = key.lower()
                    if lowered == "a":
                        self._set_active_direction("left")
                        self.get_logger().info("left")
                    elif lowered == "d":
                        self._set_active_direction("right")
                        self.get_logger().info("right")
                    elif lowered == "w":
                        self._set_active_direction("forward")
                        self.get_logger().info("forward")
                    elif lowered == "s":
                        self._set_active_direction("backward")
                        self.get_logger().info("backward")

                if self._active_direction is not None and (now - self._last_key_ts) <= self.release_timeout:
                    self._publish(self._active_direction)
                else:
                    self._active_direction = None
                    self._publish(None)

                next_tick += period
                sleep_sec = max(0.0, next_tick - time.monotonic())
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
        finally:
            if settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            self.stop()


def _print_help() -> None:
    print(
        """Keyboard-hold drive hints:
  W or Up Arrow    : move forward while held
  S or Down Arrow  : move backward while held
  A or Left Arrow  : move left while held
  D or Right Arrow : move right while held
  SPACE            : stop
  Ctrl-C           : quit

This version also supports a simple diagnostic mode via CLI arguments for testing
individual directions without holding the keyboard.
""",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard-hold or diagnostic ROS 2 drive helper.")
    parser.add_argument("--speed", type=float, default=0.5, help="linear speed magnitude (m/s)")
    parser.add_argument("--hz", type=float, default=20.0, help="publish rate")
    parser.add_argument(
        "--release-timeout",
        type=float,
        default=0.18,
        help="seconds after the last key repeat before treating the key as released",
    )
    parser.add_argument("--topic", default="cmd_vel", help="velocity topic")
    parser.add_argument(
        "--diag",
        choices=["forward", "backward", "left", "right", "diag_lf", "diag_lb", "diag_rf", "diag_rb"],
        default=None,
        help="run a fixed diagnostic command once for 2 seconds instead of keyboard hold",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rclpy.init(args=None)
    node = StableLrDrive(args.topic, args.speed, args.hz, args.release_timeout)

    def _handle_signal(signum, frame):
        del signum, frame
        node.get_logger().info("signal received, stopping")
        node.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass

    try:
        if args.diag is not None:
            print(f"[diag] sending {args.diag} for 2.0s", flush=True)
            node._publish(args.diag)
            time.sleep(2.0)
            node._publish(None)
        else:
            _print_help()
            node.run_hold()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
