"""
One video in, one delivery report out. The inference path for the web app.

    py -3.13 analyze.py clip.mp4
    py -3.13 analyze.py clip.mp4 --json

The offline scripts assume hand labels: run_pose.py needs frame_labels.csv to
know which frames to process, and measure_deliveries.py reads event frames from
a CSV. At inference there are no labels, so this module runs the whole chain on
an unseen clip:

    MediaPipe over every frame
      -> the sixteen per-frame features, identical to make_features.py
      -> window stacking, identical to stack_windows.py
      -> the saved forest predicts a phase per frame
      -> phase blocks give back foot contact, front foot contact, release
      -> geometry at those three frames, reusing measure_deliveries.measure

The feature code is duplicated from make_features.py rather than imported
because that logic lives inside its main(). If you change a feature definition
there, change it here too -- FEATURE_NAMES below is checked against the saved
model's column list on load, so a mismatch raises rather than silently
producing nonsense.

NOT ASSESSED: whether the delivery is a legal bowling action. That judgement
was made by hand for every delivery in the study and is not automated.
"""
import argparse, json, math, os, sys
from collections import defaultdict

import numpy as np

try:
    import cv2, joblib
except ImportError:
    sys.exit("pip install opencv-python joblib")

NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

FEATURE_NAMES = ["knee_lead","knee_trail","elbow_high","elbow_low",
                 "ankle_sep_x","ankle_diff_y","low_ankle_y","high_ankle_y",
                 "trunk_lean","arm_high_elev","arm_low_elev","wrist_high_vs_head",
                 "shoulder_width","hip_width","shoulder_hip_offset","lead_ankle_fwd"]

# Worthington (2010), elite male fast bowlers. mean, sd.
REF = {
    "front_knee_ffc":     (164.1, 6.1),
    "front_knee_release": (167.3, 18.8),
    "knee_flexion":       (17.5, 11.2),
    "knee_extension":     (11.9, 7.4),
    "plant_angle":        (36.3, 3.9),
    "foot_strike":        (-8.2, 13.2),
    "lean_back_bfc":      (9.0, 5.2),
    "stride_pct_height":  (78.5, 7.2),
    "release_pct_height": (112.8, 4.1),
}
TRUNK_RATIO = 0.288          # Winter (1990), shoulder 0.818H - hip 0.530H


# ----------------------------------------------------------------- geometry
def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def signed_from_vertical(v):
    return math.degrees(math.atan2(float(v[0]), -float(v[1])))


# --------------------------------------------------------------------- pose
def run_pose(path, task="models/pose_landmarker_heavy.task", max_width=1280,
             progress=None):
    """[(frame_index, 33x3 array of x, y, visibility) ...] plus (w, h, fps)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out, W, H = [], None, None
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h, w = fr.shape[:2]
            if w > max_width:
                fr = cv2.resize(fr, (max_width, int(h * max_width / w)),
                                interpolation=cv2.INTER_AREA)
                h, w = fr.shape[:2]
            W, H = w, h
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(i * 1000 / fps))
            if res.pose_landmarks:
                P = np.array([[p.x, p.y, p.visibility]
                              for p in res.pose_landmarks[0]], dtype=np.float64)
                out.append((i, P))
            i += 1
            if progress and total and i % 15 == 0:
                progress(i / total)
    cap.release()
    return out, (W, H, fps)


# ----------------------------------------------------------------- features
def features_for(poses, W, H, min_torso=0.02):
    """Per-frame features, matching make_features.py exactly."""
    ar = W / H
    # direction of travel from the hip trajectory, so handedness never enters
    fr = [f for f, _ in poses]
    hx = [float(P[L_HIP][0] + P[R_HIP][0]) / 2 * ar for _, P in poses]
    slope = np.polyfit(fr, hx, 1)[0] if len(fr) > 2 else 1.0
    flip = slope < 0

    rows = []
    for f, Praw in poses:
        P = np.empty((33, 2))
        for i in range(33):
            x = float(Praw[i][0]) * ar
            P[i] = (-x if flip else x, float(Praw[i][1]))
        hip_c = (P[L_HIP] + P[R_HIP]) / 2
        sh_c = (P[L_SH] + P[R_SH]) / 2
        torso = float(np.linalg.norm(sh_c - hip_c))
        if torso < min_torso:
            continue
        Q = (P - hip_c) / torso

        if Q[L_AN][0] >= Q[R_AN][0]:
            lead, trail = (L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)
        else:
            lead, trail = (R_HIP, R_KN, R_AN), (L_HIP, L_KN, L_AN)
        if Q[L_WR][1] <= Q[R_WR][1]:
            high, low = (L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)
        else:
            high, low = (R_SH, R_EL, R_WR), (L_SH, L_EL, L_WR)

        a_lead, a_trail = Q[lead[2]], Q[trail[2]]
        lower = a_lead if a_lead[1] >= a_trail[1] else a_trail
        upper = a_trail if a_lead[1] >= a_trail[1] else a_lead

        d = {
            "knee_lead":  joint_angle(Q[lead[0]], Q[lead[1]], Q[lead[2]]),
            "knee_trail": joint_angle(Q[trail[0]], Q[trail[1]], Q[trail[2]]),
            "elbow_high": joint_angle(Q[high[0]], Q[high[1]], Q[high[2]]),
            "elbow_low":  joint_angle(Q[low[0]], Q[low[1]], Q[low[2]]),
            "ankle_sep_x":  float(a_lead[0] - a_trail[0]),
            "ankle_diff_y": float(a_lead[1] - a_trail[1]),
            "low_ankle_y":  float(lower[1]),
            "high_ankle_y": float(upper[1]),
            "trunk_lean":   signed_from_vertical(Q[L_SH] / 2 + Q[R_SH] / 2),
            "arm_high_elev": signed_from_vertical(Q[high[2]] - Q[high[0]]),
            "arm_low_elev":  signed_from_vertical(Q[low[2]] - Q[low[0]]),
            "wrist_high_vs_head": float(Q[high[2]][1] - Q[NOSE][1]),
            "shoulder_width": abs(float(Q[L_SH][0] - Q[R_SH][0])),
            "hip_width":      abs(float(Q[L_HIP][0] - Q[R_HIP][0])),
            "shoulder_hip_offset": float((Q[L_SH][0] + Q[R_SH][0]) / 2),
            "lead_ankle_fwd": float(a_lead[0]),
        }
        if any(math.isnan(v) for v in d.values()):
            continue
        rows.append((f, d, P))          # P kept unflipped-scaled for measurement
    return rows, flip


def stack(rows, k):
    """Window stacking, matching stack_windows.py."""
    L = len(rows)
    out = []
    for i, (_, d, _) in enumerate(rows):
        v = dict(d)
        for f in FEATURE_NAMES:
            here = d[f]
            for off in range(-k, k + 1):
                if off == 0:
                    continue
                j = min(max(i + off, 0), L - 1)
                v[f"{f}__{'m' if off < 0 else 'p'}{abs(off)}"] = rows[j][1][f]
            pm1 = rows[max(i - 1, 0)][1][f]
            pp1 = rows[min(i + 1, L - 1)][1][f]
            pm3 = rows[max(i - 3, 0)][1][f]
            v[f"{f}__d1"] = here - pm1
            v[f"{f}__d3"] = here - pm3
            v[f"{f}__v"] = (pp1 - pm1) / 2
        out.append(v)
    return out


# ------------------------------------------------------------------ events
def median_filter(a, k):
    if k < 3 or k % 2 == 0:
        return a
    h, n = k // 2, len(a)
    return np.array([np.median(a[max(0, i - h):min(n, i + h + 1)], axis=0)
                     for i in range(n)])


def blocks_of(labels, want):
    """[(start, end)] index ranges where labels == want."""
    out, s = [], None
    for i, v in enumerate(labels):
        if v == want and s is None:
            s = i
        elif v != want and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(labels) - 1))
    return out


def extract_events(frames, labels, prob, phases, smooth=9):
    """Back foot contact, front foot contact and release as frame numbers.

    The phases are ordered: a bfc_phase block must precede an ffc_phase block.
    Where several blocks exist, take the pair maximising summed confidence
    subject to that ordering, as extract_events.py does offline.
    """
    P = median_filter(prob, smooth)
    lab = [phases[i] for i in P.argmax(axis=1)]
    bi, fi = phases.index("bfc_phase"), phases.index("ffc_phase")
    bb, fb = blocks_of(lab, "bfc_phase"), blocks_of(lab, "ffc_phase")
    if not bb or not fb:
        return None, "could not find both foot contacts"

    best, score = None, -1
    for b in bb:
        for f in fb:
            if b[0] >= f[0]:
                continue
            s = (P[b[0]:b[1] + 1, bi].sum() + P[f[0]:f[1] + 1, fi].sum())
            if s > score:
                score, best = s, (b, f)
    if best is None:
        return None, "back foot contact was not found before front foot contact"
    b, f = best
    # the event instant is the first frame of each block; release is the last
    # frame of the ffc block, where the ball leaves the hand
    return {"bfc": frames[b[0]], "ffc": frames[f[0]],
            "release": frames[f[1]]}, ""


# ------------------------------------------------------------------ report
def band(z):
    a = abs(z)
    return "typical" if a <= 1 else ("outside the usual range" if a <= 2
                                     else "well outside the usual range")


def analyse(path, model_path="models/phase_model.joblib", height_cm=None,
            progress=None):
    bundle = joblib.load(model_path)
    model, feats, phases, k = (bundle["model"], bundle["features"],
                               list(bundle["phases"]), bundle["k"])

    poses, (W, H, fps) = run_pose(path, progress=progress)
    if len(poses) < 30:
        return {"ok": False, "error": "no person detected for long enough -- "
                "check the bowler is fully in frame"}

    rows, flip = features_for(poses, W, H)
    if len(rows) < 30:
        return {"ok": False, "error": "pose was too unstable to measure"}

    X = np.array([[r[f] for f in feats] for r in stack(rows, k)],
                 dtype=np.float32)
    prob = model.predict_proba(X)
    frames = [f for f, _, _ in rows]
    ev, err = extract_events(frames, None, prob, phases)
    if ev is None:
        return {"ok": False, "error": err}

    # geometry, reusing the offline measurement code unchanged
    import measure_deliveries as md
    byf = {}
    for f, _, P in rows:
        byf[f] = np.array([[P[i][0], P[i][1]] for i in range(33)])
    m, note = md.measure_clip(byf, ev, 1.0, tol=3)
    if m is None:
        return {"ok": False, "error": note or "could not measure the delivery"}

    out = {k2: v for k2, v in m.items() if not k2.startswith("_")}
    for src, dst in (("stride_torso", "stride_pct_height"),
                     ("release_hgt_torso", "release_pct_height")):
        v = out.get(src)
        out[dst] = (v * TRUNK_RATIO * 100) if v is not None and not math.isnan(v) else None

    comp = {}
    for key, (mu, sd) in REF.items():
        v = out.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        z = (v - mu) / sd
        comp[key] = {"value": round(v, 1), "ref_mean": mu, "ref_sd": sd,
                     "z": round(z, 2), "band": band(z)}

    return {"ok": True, "events": ev, "fps": round(fps, 1),
            "frames_analysed": len(rows), "direction": "R2L" if flip else "L2R",
            "measurements": {a: (round(b, 2) if isinstance(b, float)
                                 and not math.isnan(b) else None)
                             for a, b in out.items()},
            "comparison": comp, "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/phase_model.joblib")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = analyse(a.video, a.model)
    if a.json:
        print(json.dumps(r, indent=2)); return
    if not r["ok"]:
        sys.exit("could not analyse: " + r["error"])
    print(f"\n{os.path.basename(a.video)}   {r['frames_analysed']} frames, "
          f"{r['fps']:g} fps, travelling {r['direction']}")
    print(f"events: back foot {r['events']['bfc']}, front foot "
          f"{r['events']['ffc']}, release {r['events']['release']}\n")
    print(f"  {'measurement':<22}{'you':>9}{'elite':>16}{'':>6}")
    for k2, c in r["comparison"].items():
        print(f"  {k2:<22}{c['value']:>9.1f}{c['ref_mean']:>10.1f} +-{c['ref_sd']:>4.1f}"
              f"   {c['z']:+.1f} sd  {c['band']}")
    print("\nlegality of the bowling action is NOT assessed.")


if __name__ == "__main__":
    main()
