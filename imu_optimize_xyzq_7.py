import argparse
import csv
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Use IMU integration to smooth/optimize per-frame xyzq trajectory."
    )
    parser.add_argument(
        "--xyzq_in",
        default="data/rt_quaternion.txt",
        help="Input xyzq txt: timestamp tx ty tz qx qy qz qw",
    )
    parser.add_argument(
        "--imu_csv",
        required=True,
        help="IMU csv from record_control_1.py (cam2_*_imu.csv)",
    )
    parser.add_argument(
        "--pi_timestamps",
        required=True,
        help="Pi timestamp csv (cam2_*.csv) used to map relative seconds -> wall ns",
    )
    parser.add_argument(
        "--out",
        default="data/rt_quaternion_imu.txt",
        help="Output optimized xyzq txt",
    )
    parser.add_argument(
        "--rot_meas_weight",
        type=float,
        default=0.35,
        help="Vision measurement weight for quaternion update, in [0,1]",
    )
    parser.add_argument(
        "--trans_meas_weight",
        type=float,
        default=0.80,
        help="Vision measurement weight for translation update, in [0,1]",
    )
    parser.add_argument(
        "--vel_feedback",
        type=float,
        default=0.05,
        help="Velocity feedback gain from translation residual",
    )
    parser.add_argument(
        "--gravity_window_sec",
        type=float,
        default=1.0,
        help="Seconds at IMU start used to estimate gravity vector",
    )
    parser.add_argument(
        "--max_imu_dt",
        type=float,
        default=0.02,
        help="Clamp each IMU integration step to avoid unstable spikes",
    )
    parser.add_argument(
        "--normalize_t",
        action="store_true",
        help="Normalize output translation to unit vector (recommended for recoverPose t)",
    )
    return parser


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rv))
    if theta <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rv / theta
    half = 0.5 * theta
    s = np.sin(half)
    return quat_normalize(np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)], dtype=np.float64))


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:3]


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = clamp01(alpha)
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return quat_normalize((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize((s0 * q0) + (s1 * q1))


def quat_angle_deg(q_prev: np.ndarray, q_next: np.ndarray) -> float:
    dq = quat_mul(quat_conj(quat_normalize(q_prev)), quat_normalize(q_next))
    w = max(-1.0, min(1.0, float(abs(dq[3]))))
    return float(np.degrees(2.0 * np.arccos(w)))


def load_xyzq(path: Path):
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
                    "q": quat_normalize(np.array([qx, qy, qz, qw], dtype=np.float64)),
                }
            )
    if not rows:
        raise ValueError(f"No valid xyzq rows found: {path}")
    return rows


def load_pi_timestamp_mapping(path: Path):
    rel_ts = []
    wall_ns = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rel = float(row["timestamp"])
                wall_sec = float(row["timestamp_wall"])
            except Exception:
                continue
            rel_ts.append(rel)
            wall_ns.append(int(round(wall_sec * 1e9)))
    if len(rel_ts) < 2:
        raise ValueError(f"Need at least 2 rows in pi timestamp csv: {path}")
    rel_ts_np = np.asarray(rel_ts, dtype=np.float64)
    wall_ns_np = np.asarray(wall_ns, dtype=np.int64)
    order = np.argsort(rel_ts_np)
    return rel_ts_np[order], wall_ns_np[order]


def map_rel_ts_to_wall_ns(rel_query: np.ndarray, rel_ts_ref: np.ndarray, wall_ns_ref: np.ndarray) -> np.ndarray:
    q = np.asarray(rel_query, dtype=np.float64)
    q_clip = np.clip(q, rel_ts_ref[0], rel_ts_ref[-1])
    wall_interp = np.interp(q_clip, rel_ts_ref, wall_ns_ref.astype(np.float64))
    return np.rint(wall_interp).astype(np.int64)


def load_imu_csv(path: Path):
    data = np.genfromtxt(str(path), delimiter=",", comments="#", dtype=np.float64)
    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)
    if data.size == 0 or data.shape[1] < 7:
        raise ValueError(f"Invalid IMU csv (need >=7 columns): {path}")
    t_ns = np.rint(data[:, 0]).astype(np.int64)
    w = data[:, 1:4].astype(np.float64)
    a = data[:, 4:7].astype(np.float64)
    order = np.argsort(t_ns)
    return t_ns[order], w[order], a[order]


def estimate_gravity_sensor_frame(imu_t_ns: np.ndarray, imu_a: np.ndarray, window_sec: float) -> np.ndarray:
    if imu_t_ns.size == 0:
        return np.array([0.0, 0.0, 9.81], dtype=np.float64)
    t0 = imu_t_ns[0]
    tend = t0 + int(max(0.1, float(window_sec)) * 1e9)
    mask = imu_t_ns <= tend
    if not np.any(mask):
        return np.median(imu_a, axis=0)
    return np.median(imu_a[mask], axis=0)


def integrate_imu_segment(
    imu_t_ns: np.ndarray,
    imu_w: np.ndarray,
    imu_a: np.ndarray,
    t_start_ns: int,
    t_end_ns: int,
    q_init: np.ndarray,
    v_init: np.ndarray,
    t_init: np.ndarray,
    g_sensor: np.ndarray,
    max_imu_dt: float,
):
    q = quat_normalize(q_init.copy())
    v = v_init.copy()
    t = t_init.copy()
    if t_end_ns <= t_start_ns:
        return q, v, t

    lo = int(np.searchsorted(imu_t_ns, t_start_ns, side="left"))
    hi = int(np.searchsorted(imu_t_ns, t_end_ns, side="right"))
    if hi - lo <= 0:
        return q, v, t

    prev_ns = t_start_ns
    max_dt = max(1e-4, float(max_imu_dt))
    for i in range(lo, hi):
        cur_ns = int(imu_t_ns[i])
        dt = (cur_ns - prev_ns) * 1e-9
        prev_ns = cur_ns
        if dt <= 0.0:
            continue
        dt = min(dt, max_dt)
        omega = imu_w[i]
        dq = quat_from_rotvec(omega * dt)
        q = quat_normalize(quat_mul(q, dq))
        a_lin_s = imu_a[i] - g_sensor
        a_lin_w = quat_rotate_vec(q, a_lin_s)
        t = t + v * dt + 0.5 * a_lin_w * dt * dt
        v = v + a_lin_w * dt

    tail_dt = (t_end_ns - prev_ns) * 1e-9
    if tail_dt > 0:
        tail_dt = min(tail_dt, max_dt)
        t = t + v * tail_dt
    return q, v, t


def compute_step_stats(ts: np.ndarray, t_arr: np.ndarray, q_arr: np.ndarray):
    if len(ts) < 2:
        return 0.0, 0.0
    angs = []
    dists = []
    for i in range(1, len(ts)):
        angs.append(quat_angle_deg(q_arr[i - 1], q_arr[i]))
        dists.append(float(np.linalg.norm(t_arr[i] - t_arr[i - 1])))
    return float(np.mean(angs)), float(np.mean(dists))


def main():
    args = build_parser().parse_args()
    rot_w = clamp01(args.rot_meas_weight)
    trans_w = clamp01(args.trans_meas_weight)
    vel_fb = float(args.vel_feedback)

    xyz_rows = load_xyzq(Path(args.xyzq_in))
    rel_ts = np.array([r["ts"] for r in xyz_rows], dtype=np.float64)
    vis_t = np.array([r["t"] for r in xyz_rows], dtype=np.float64)
    vis_q = np.array([quat_normalize(r["q"]) for r in xyz_rows], dtype=np.float64)

    pi_rel_ts, pi_wall_ns = load_pi_timestamp_mapping(Path(args.pi_timestamps))
    frame_wall_ns = map_rel_ts_to_wall_ns(rel_ts, pi_rel_ts, pi_wall_ns)

    imu_t_ns, imu_w, imu_a = load_imu_csv(Path(args.imu_csv))
    g_sensor = estimate_gravity_sensor_frame(imu_t_ns, imu_a, args.gravity_window_sec)

    fused_t = np.zeros_like(vis_t)
    fused_q = np.zeros_like(vis_q)
    fused_v = np.zeros_like(vis_t)

    fused_t[0] = vis_t[0]
    fused_q[0] = vis_q[0]
    fused_v[0] = np.zeros(3, dtype=np.float64)

    for i in range(1, len(rel_ts)):
        dt_frame = max(1e-6, float(rel_ts[i] - rel_ts[i - 1]))
        q_pred, v_pred, t_pred = integrate_imu_segment(
            imu_t_ns=imu_t_ns,
            imu_w=imu_w,
            imu_a=imu_a,
            t_start_ns=int(frame_wall_ns[i - 1]),
            t_end_ns=int(frame_wall_ns[i]),
            q_init=fused_q[i - 1],
            v_init=fused_v[i - 1],
            t_init=fused_t[i - 1],
            g_sensor=g_sensor,
            max_imu_dt=float(args.max_imu_dt),
        )
        q_upd = quat_slerp(q_pred, vis_q[i], rot_w)
        t_upd = (1.0 - trans_w) * t_pred + trans_w * vis_t[i]
        v_upd = v_pred + vel_fb * (t_upd - t_pred) / dt_frame

        fused_q[i] = quat_normalize(q_upd)
        fused_t[i] = t_upd
        fused_v[i] = v_upd

    if args.normalize_t:
        n = np.linalg.norm(fused_t, axis=1, keepdims=True)
        n = np.where(n > 1e-9, n, 1.0)
        fused_t = fused_t / n

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(len(rel_ts)):
            q = quat_normalize(fused_q[i])
            t = fused_t[i]
            f.write(
                f"{rel_ts[i]:.9f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )

    vis_ang, vis_step = compute_step_stats(rel_ts, vis_t, vis_q)
    out_ang, out_step = compute_step_stats(rel_ts, fused_t, fused_q)
    print("[imu_optimize_xyzq] done")
    print(f"[input]  {args.xyzq_in}")
    print(f"[imu]    {args.imu_csv}")
    print(f"[tsmap]  {args.pi_timestamps}")
    print(f"[output] {out_path}")
    print(f"[weights] rot_meas={rot_w:.3f} trans_meas={trans_w:.3f} vel_feedback={vel_fb:.3f}")
    print(f"[gravity_sensor] {g_sensor[0]:.6f} {g_sensor[1]:.6f} {g_sensor[2]:.6f}")
    print(f"[jitter] angle_deg mean: {vis_ang:.6f} -> {out_ang:.6f}")
    print(f"[jitter] trans_step mean: {vis_step:.6f} -> {out_step:.6f}")


if __name__ == "__main__":
    main()

