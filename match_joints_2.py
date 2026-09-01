import argparse
import bisect
import json
import math
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def build_parser():
    # 定義命令列參數：兩邊 pose 結果、輸出檔，以及時間對齊相關設定。
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", default=str(BASE_DIR / "pc_pose_video" / "json" / "cam0.jsonl"))
    parser.add_argument("--cam2", default=str(BASE_DIR / "pi_pose_video" / "json" / "cam2.jsonl"))
    parser.add_argument("--out", default=str(BASE_DIR / "data" / "cam_match.txt"))
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--cam0_offset", type=float, default=0.0)
    parser.add_argument("--cam2_offset", type=float, default=0.0)
    parser.add_argument("--vis_thresh", type=float, default=0.1, help="Minimum keypoint visibility")
    return parser


def read_jsonl_group_by_timestamp(path, *, time_offset=0.0):
    # 逐行讀 jsonl，依 timestamp 分組，必要時可先套用整體時間位移。
    groups = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = float(obj["timestamp"]) + float(time_offset)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            groups[ts].append(obj)
    return groups


def extract_keypoints(obj, visibility_thresh=0.1):
    # 從單幀資料取出 keypoints17，並過濾掉可見度太低的點。
    kp_dict = {}
    for item in obj.get("keypoints17") or []:
        try:
            kid = int(item.get("id"))
            x = float(item.get("x"))
            y = float(item.get("y"))
            v = float(item.get("visibility"))
        except (TypeError, ValueError):
            continue
        if v >= visibility_thresh:
            kp_dict[kid] = (x, y, v)
    return kp_dict


def nan_or_value(v):
    # 缺值統一輸出成 NaN，讓後續文字檔格式固定。
    return float("nan") if v is None or (isinstance(v, float) and math.isnan(v)) else v


def format_pair_line(timestamp, cam0_obj, cam2_obj, visibility_thresh=0.1):
    # 把一對 frame 整理成一行文字：timestamp + 17 個左右相機關節配對。
    cam0_kp = extract_keypoints(cam0_obj, visibility_thresh) if cam0_obj else {}
    cam2_kp = extract_keypoints(cam2_obj, visibility_thresh) if cam2_obj else {}
    pairs = []
    for kid in range(17):
        lx, ly, lvis = cam0_kp.get(kid, (None, None, None))
        rx, ry, rvis = cam2_kp.get(kid, (None, None, None))
        pairs.append(
            [
                [nan_or_value(lx), nan_or_value(ly)],
                [nan_or_value(rx), nan_or_value(ry)],
                nan_or_value(lvis),
                nan_or_value(rvis),
            ]
        )
    return f"{timestamp} {json.dumps(pairs, ensure_ascii=False)}"


def write_matches(timestamp, left_list, right_list, out_f, visibility_thresh=0.1):
    # 同一個 timestamp 若兩邊有多筆資料，就逐筆配對寫出。
    for cam0_obj, cam2_obj in zip_longest(left_list, right_list):
        out_f.write(format_pair_line(timestamp, cam0_obj, cam2_obj, visibility_thresh) + "\n")


def find_best_timestamp(cam2_ts, ts0, tolerance):
    # 在 cam2 的時間序列中找出距離 ts0 最近、且落在容許誤差內的 timestamp。
    idx = bisect.bisect_left(cam2_ts, ts0)
    candidates = []
    if idx < len(cam2_ts):
        candidates.append(cam2_ts[idx])
    if idx > 0:
        candidates.append(cam2_ts[idx - 1])

    best_ts = None
    best_dt = None
    for ts2 in candidates:
        dt = abs(ts2 - ts0)
        if dt <= tolerance and (best_dt is None or dt < best_dt):
            best_dt = dt
            best_ts = ts2
    return best_ts


def exact_timestamp_match(cam0_groups, cam2_groups, out_f, visibility_thresh=0.1):
    # 嚴格模式：只有完全相同的 timestamp 才配對。
    for ts in sorted(set(cam0_groups) | set(cam2_groups)):
        write_matches(ts, cam0_groups.get(ts, []), cam2_groups.get(ts, []), out_f, visibility_thresh)


def tolerance_timestamp_match(cam0_groups, cam2_groups, tolerance, out_f, visibility_thresh=0.1):
    # 寬鬆模式：對每個 cam0 timestamp 找 cam2 最接近的 timestamp 來配對。
    cam2_ts = sorted(cam2_groups)
    for ts0 in sorted(cam0_groups):
        best_ts = find_best_timestamp(cam2_ts, ts0, tolerance)
        write_matches(
            ts0,
            cam0_groups.get(ts0, []),
            cam2_groups.get(best_ts, []) if best_ts is not None else [],
            out_f,
            visibility_thresh,
        )


def main():
    # 主流程：讀兩邊 jsonl，依指定策略配對，再輸出 cam_match.txt。
    args = build_parser().parse_args()
    cam0_path = Path(args.cam0).resolve()
    cam2_path = Path(args.cam2).resolve()
    out_path = Path(args.out).resolve()
    print(f"[match_joints] cam0: {cam0_path}")
    print(f"[match_joints] cam2: {cam2_path}")
    print(f"[match_joints] out : {out_path}")

    cam0_groups = read_jsonl_group_by_timestamp(cam0_path, time_offset=args.cam0_offset)
    cam2_groups = read_jsonl_group_by_timestamp(cam2_path, time_offset=args.cam2_offset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        if args.tolerance <= 0:
            exact_timestamp_match(cam0_groups, cam2_groups, out_f, args.vis_thresh)
        else:
            tolerance_timestamp_match(cam0_groups, cam2_groups, args.tolerance, out_f, args.vis_thresh)


if __name__ == "__main__":
    main()
