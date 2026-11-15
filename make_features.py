"""
Turn raw MediaPipe landmarks into 16 normalised features per frame.

    python make_features.py keypoints.csv --out features.csv

What it does, in order:

1. Drops frames with no pose. Those are 'none' by rule, not by model.

2. Corrects the aspect ratio. MediaPipe gives x and y both in 0-1, but the image
   is 16:9, so a horizontal distance of 0.1 is much larger than a vertical one.
   X is multiplied by width/height so both axes are in units of image height.

3. Detects direction of travel per clip from the hip trajectory, and mirrors X
   so every bowler travels left-to-right. No handedness column needed.

4. Normalises: hip midpoint becomes the origin, torso length becomes the unit.
   After this a frame carries no information about where the bowler was in the
   image or how far from the camera.

5. Builds 16 features using ORDER STATISTICS rather than left/right:
   "leading leg" is whichever ankle is further forward, "higher arm" is whichever
   wrist is higher. Handedness drops out entirely, so a left-arm bowler needs no
   correction and there is no mirror flag to get backwards.

Angles are in degrees, distances in torso lengths. Both are scale-free, so a
bowler 11 m away and one 8 m away give the same numbers for the same pose.
"""

import argparse
import csv
import math
from collections import defaultdict

import numpy as np

# MediaPipe landmark indices
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

FEATURES = [
    "knee_lead", "knee_trail",
    "elbow_high", "elbow_low",
    "ankle_sep_x", "ankle_diff_y",
    "low_ankle_y", "high_ankle_y",
    "trunk_lean",
    "arm_high_elev", "arm_low_elev",
    "wrist_high_vs_head",
    "shoulder_width", "hip_width",
    "shoulder_hip_offset", "lead_ankle_fwd",
]


def joint_angle(a, b, c):
    """Interior angle at b, in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def signed_from_vertical(v):
    """Angle of vector v from straight up, positive toward +x (direction of travel).

    Image coordinates: y grows downward, so 'up' is (0, -1).
    0 = straight up, +90 = horizontal forward, 180 = straight down.
    """
    dx, dy = float(v[0]), float(v[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return float("nan")
    return math.degrees(math.atan2(dx, -dy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keypoints")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--min-torso", type=float, default=0.02,
                    help="skip frames whose torso is shorter than this (image heights)")
    ap.add_argument("--size-ratio", type=float, default=0.7,
                    help="exclude a clip whose median torso is below this fraction "
                         "of the bowler's own median; catches pose locked onto a bystander")
    args = ap.parse_args()

    # ---------------------------------------------------------- load
    by_video = defaultdict(list)
    n_in = n_undetected = 0
    with open(args.keypoints, newline="") as fh:
        for r in csv.DictReader(fh):
            n_in += 1
            if r["detected"] != "1":
                n_undetected += 1
                continue
            by_video[r["video"]].append(r)
    print(f"{n_in} rows in, {n_undetected} with no pose dropped, "
          f"{n_in - n_undetected} kept across {len(by_video)} clips")

    # ------------------------------------------- direction of travel per clip
    direction = {}
    for vid, rows in by_video.items():
        frames, hipx = [], []
        for r in rows:
            W, H = float(r["width"]), float(r["height"])
            ar = W / H
            hx = (float(r[f"x{L_HIP}"]) + float(r[f"x{R_HIP}"])) / 2 * ar
            frames.append(int(r["frame"]))
            hipx.append(hx)
        slope = np.polyfit(frames, hipx, 1)[0] if len(frames) > 2 else 1.0
        direction[vid] = "L2R" if slope >= 0 else "R2L"

    n_r2l = sum(1 for d in direction.values() if d == "R2L")
    print(f"direction of travel: {len(direction)-n_r2l} clips L2R, {n_r2l} R2L (mirrored)")

    # ------------------------------------------- is the pose on the right person?
    # MediaPipe returns one pose per frame and can lock onto a bystander. Such a
    # clip gives a body much smaller than that bowler's usual, because the wrong
    # person is further away. Compare each clip's median torso against the
    # bowler's own median across clips.
    torso_med, bowler_of = {}, {}
    for vid, rows in by_video.items():
        ar = float(rows[0]["width"]) / float(rows[0]["height"])
        t = []
        for r in rows:
            hx = (float(r[f"x{L_HIP}"]) + float(r[f"x{R_HIP}"])) / 2 * ar
            hy = (float(r[f"y{L_HIP}"]) + float(r[f"y{R_HIP}"])) / 2
            sx = (float(r[f"x{L_SH}"]) + float(r[f"x{R_SH}"])) / 2 * ar
            sy = (float(r[f"y{L_SH}"]) + float(r[f"y{R_SH}"])) / 2
            t.append(math.hypot(sx - hx, sy - hy))
        torso_med[vid] = float(np.median(t))
        bowler_of[vid] = rows[0]["bowler"]

    per_bowler = defaultdict(list)
    for vid, m in torso_med.items():
        per_bowler[bowler_of[vid]].append(m)
    bowler_med = {b: float(np.median(v)) for b, v in per_bowler.items()}

    wrong_person = []
    for vid in list(by_video):
        ratio = torso_med[vid] / bowler_med[bowler_of[vid]]
        if ratio < args.size_ratio:
            wrong_person.append((bowler_of[vid], vid, ratio))
    if wrong_person:
        print(f"\n  WARNING, pose is probably on the wrong person in "
              f"{len(wrong_person)} clip(s) -- body far smaller than this "
              f"bowler's usual. Excluded:")
        for b, vid, ratio in sorted(wrong_person, key=lambda z: z[2]):
            print(f"    {b:12s} {vid}  torso {ratio:.0%} of normal")
        for _, vid, _ in wrong_person:
            del by_video[vid]
        print("  (open those clips and look for a bystander behind the bowler)")

    by_bowler_dir = defaultdict(set)
    for vid, rows in by_video.items():
        by_bowler_dir[rows[0]["bowler"]].add(direction[vid])
    mixed = {b: d for b, d in by_bowler_dir.items() if len(d) > 1}
    if mixed:
        print("\n  NOTE, these bowlers have clips going both ways "
              "(shot from both sides, or two people in one folder):")
        for b, d in mixed.items():
            print(f"    {b}: {sorted(d)}")

    # ---------------------------------------------------------- features
    out_rows = []
    skipped = 0
    for vid, rows in by_video.items():
        flip = direction[vid] == "R2L"
        for r in rows:
            W, H = float(r["width"]), float(r["height"])
            ar = W / H

            P = np.empty((33, 2))
            for i in range(33):
                x = float(r[f"x{i}"]) * ar
                P[i] = (-x if flip else x, float(r[f"y{i}"]))

            hip_c = (P[L_HIP] + P[R_HIP]) / 2
            sh_c = (P[L_SH] + P[R_SH]) / 2
            torso = float(np.linalg.norm(sh_c - hip_c))
            if torso < args.min_torso:
                skipped += 1
                continue

            Q = (P - hip_c) / torso          # origin at hips, unit = torso length

            # leading / trailing leg by how far forward the ankle is
            if Q[L_AN][0] >= Q[R_AN][0]:
                lead = (L_HIP, L_KN, L_AN)
                trail = (R_HIP, R_KN, R_AN)
            else:
                lead = (R_HIP, R_KN, R_AN)
                trail = (L_HIP, L_KN, L_AN)

            # higher / lower arm by wrist height (smaller y is higher)
            if Q[L_WR][1] <= Q[R_WR][1]:
                high = (L_SH, L_EL, L_WR)
                low = (R_SH, R_EL, R_WR)
            else:
                high = (R_SH, R_EL, R_WR)
                low = (L_SH, L_EL, L_WR)

            a_lead, a_trail = Q[lead[2]], Q[trail[2]]
            lower_ankle = a_lead if a_lead[1] >= a_trail[1] else a_trail
            upper_ankle = a_trail if a_lead[1] >= a_trail[1] else a_lead

            f = {
                "knee_lead":  joint_angle(Q[lead[0]], Q[lead[1]], Q[lead[2]]),
                "knee_trail": joint_angle(Q[trail[0]], Q[trail[1]], Q[trail[2]]),
                "elbow_high": joint_angle(Q[high[0]], Q[high[1]], Q[high[2]]),
                "elbow_low":  joint_angle(Q[low[0]], Q[low[1]], Q[low[2]]),
                "ankle_sep_x":  float(a_lead[0] - a_trail[0]),
                "ankle_diff_y": float(a_lead[1] - a_trail[1]),
                "low_ankle_y":  float(lower_ankle[1]),
                "high_ankle_y": float(upper_ankle[1]),
                "trunk_lean":   signed_from_vertical(Q[L_SH] / 2 + Q[R_SH] / 2),
                "arm_high_elev": signed_from_vertical(Q[high[2]] - Q[high[0]]),
                "arm_low_elev":  signed_from_vertical(Q[low[2]] - Q[low[0]]),
                "wrist_high_vs_head": float(Q[high[2]][1] - Q[NOSE][1]),
                "shoulder_width": abs(float(Q[L_SH][0] - Q[R_SH][0])),
                "hip_width":      abs(float(Q[L_HIP][0] - Q[R_HIP][0])),
                "shoulder_hip_offset": float((Q[L_SH][0] + Q[R_SH][0]) / 2),
                "lead_ankle_fwd": float(a_lead[0]),
            }
            if any(math.isnan(v) for v in f.values()):
                skipped += 1
                continue

            out_rows.append(dict(
                video=r["video"], bowler=r["bowler"], frame=int(r["frame"]),
                phase=r["phase"], direction=direction[vid],
                is_bfc_event=r["is_bfc_event"], is_ffc_event=r["is_ffc_event"],
                is_release_event=r["is_release_event"],
                torso_px=round(torso * H, 1),
                **{k: round(v, 5) for k, v in f.items()},
            ))

    if skipped:
        print(f"{skipped} frames skipped (degenerate pose)")

    cols = ["video", "bowler", "frame", "phase", "direction",
            "is_bfc_event", "is_ffc_event", "is_release_event", "torso_px"] + FEATURES
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n{len(out_rows)} rows written to {args.out}")

    # ------------------------------------------------- sanity: do they separate?
    phases = ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]
    show = ["ankle_sep_x", "high_ankle_y", "knee_lead", "arm_high_elev",
            "shoulder_width", "trunk_lean"]
    print("\nmean feature value by phase -- these should differ between classes\n")
    print(f"  {'feature':20s}" + "".join(f"{p[:9]:>11s}" for p in phases))
    for name in show:
        line = f"  {name:20s}"
        for p in phases:
            vals = [r[name] for r in out_rows if r["phase"] == p]
            line += f"{np.mean(vals):11.2f}" if vals else f"{'-':>11s}"
        print(line)

    print("\n  {:20s}".format("n frames")
          + "".join(f"{sum(1 for r in out_rows if r['phase']==p):11d}" for p in phases))

    sizes = [r["torso_px"] for r in out_rows]
    print(f"\ntorso length in pixels: min {min(sizes):.0f}, "
          f"median {np.median(sizes):.0f}, max {max(sizes):.0f}")
    print("(a very small minimum means some frames had the bowler far away; "
          "those keypoints are the least precise)")


if __name__ == "__main__":
    main()
