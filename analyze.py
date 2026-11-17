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


def extract_events(frames, labels, prob, phases, smooth=9, info=None):
    """Back foot contact, front foot contact and release as frame numbers.

    The phases are ordered: a bfc_phase block must precede an ffc_phase block.
    Where several blocks exist, take the pair maximising summed confidence
    subject to that ordering, as extract_events.py does offline.

    Pass a dict as `info` to get back what the confidence estimate needs: the
    smoothed probabilities and how fragmented each phase was.
    """
    P = median_filter(prob, smooth)
    if info is not None:
        info["P"] = P
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
    if info is not None:
        info["n_blocks"] = {"bfc": len(bb), "ffc": len(fb), "release": len(fb)}
    # the event instant is the first frame of each block; release is the last
    # frame of the ffc block, where the ball leaves the hand
    return {"bfc": frames[b[0]], "ffc": frames[f[0]],
            "release": frames[f[1]]}, ""


# ------------------------------------------------------------------ report
#
# Three safeguards sit between the geometry and the page, because a single
# clip is a much weaker measurement than the 262-delivery cohort:
#
#   plausibility  every value is also scored against our own cohort. Anything
#                 past 3 cohort sd is almost always a mis-detected event, not
#                 an extraordinary bowler, so it is reported as such instead of
#                 as a finding. Two or more values past 2 sd together trigger
#                 the same warning: a coherent set of extremes is the
#                 signature of wrong event frames.
#   confidence    each measurement inherits the reliability of the event
#                 frames it is computed at. Back foot contact starts low (59%
#                 within 3 frames in validation); a fragmented phase block, an
#                 unsure classifier edge, pose drop-out or hidden legs lower it.
#   explanation   rule-based, citation-bound text. It says how a value differs
#                 from the elite mean and what the published association is.
#                 It never prescribes a change -- an association across
#                 bowlers is not a prediction about one bowler.

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT_PATH = os.path.join(HERE, "models", "cohort_stats.json")

# key, label, unit -- the order the report prints in
MEASURES = [
    ("front_knee_ffc", "Front knee at foot contact", "°"),
    ("front_knee_release", "Front knee at release", "°"),
    ("knee_flexion", "Knee flexion after contact", "°"),
    ("knee_extension", "Knee extension before release", "°"),
    ("plant_angle", "Plant angle", "°"),
    ("foot_strike", "Foot strike angle", "°"),
    ("lean_back_bfc", "Trunk lean at back foot contact", "°"),
    ("trunk_lean_ffc", "Trunk lean at front foot contact", "°"),
    ("trunk_lean_release", "Trunk lean at release", "°"),
    ("arm_angle_release", "Bowling arm against the trunk", "°"),
    ("stride_pct_height", "Delivery stride", "% height"),
    ("release_pct_height", "Release height", "% height"),
]
LABEL = {k: l for k, l, _ in MEASURES}

# which event frames each measurement is computed at
EVENT_DEPS = {
    "front_knee_ffc": ("ffc",), "plant_angle": ("ffc",),
    "foot_strike": ("ffc",), "trunk_lean_ffc": ("ffc",),
    "front_knee_release": ("release",), "trunk_lean_release": ("release",),
    "arm_angle_release": ("release",), "release_pct_height": ("release",),
    "knee_flexion": ("ffc", "release"), "knee_extension": ("ffc", "release"),
    "lean_back_bfc": ("bfc",), "stride_pct_height": ("bfc", "ffc"),
}
EVENT_NAME = {"bfc": "back foot contact", "ffc": "front foot contact",
              "release": "ball release"}
# leave-one-bowler-out share of deliveries within 3 frames of a hand mark
EVENT_HIT_RATE = {"bfc": 59.0, "ffc": 91.8, "release": 98.5}
# starting confidence from those rates: 2 high, 1 moderate, 0 low
EVENT_BASE = {"bfc": 0, "ffc": 2, "release": 2}
LEVELS = ["low", "moderate", "high"]

IMPLAUSIBLE_SD = 3.0
UNUSUAL_SD = 2.0
LEG_LANDMARKS = list(range(23, 33))


def build_cohort(csv_path="measurements.csv", out_path=COHORT_PATH):
    """Our own cohort's mean and sd per measurement, for plausibility checks.

    Written to models/ so it ships with the model (the Docker image excludes
    CSVs). Rebuild whenever measurements.csv changes.
    """
    import csv
    vals, bowlers = defaultdict(list), set()
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            bowlers.add(r.get("bowler"))
            for key, _, _ in MEASURES:
                try:
                    v = float(r.get(key, ""))
                except ValueError:
                    continue
                if not math.isnan(v):
                    vals[key].append(v)
    stats = {k: {"mean": round(float(np.mean(v)), 3),
                 "sd": round(float(np.std(v, ddof=1)), 3), "n": len(v)}
             for k, v in vals.items() if len(v) > 2}
    with open(out_path, "w") as fh:
        json.dump({"source": os.path.basename(csv_path),
                   "bowlers": len(bowlers), "stats": stats}, fh, indent=2)
    return stats


def load_cohort(path=COHORT_PATH):
    try:
        with open(path) as fh:
            return json.load(fh)["stats"]
    except (OSError, ValueError, KeyError):
        return {}


COHORT = load_cohort()


def band(z):
    a = abs(z)
    return "typical" if a <= 1 else ("outside the usual range" if a <= 2
                                     else "well outside the usual range")


# --------------------------------------------------------------- confidence
def event_quality(st, ev, checked=False):
    """How far to trust each event frame on this clip.

    {event: {"level": "high"|"moderate"|"low"|"checked", "reasons": [...]}}
    """
    frames, P, phases = st["frames"], st["P"], st["phases"]
    out = {}
    for e in ("bfc", "ffc", "release"):
        f = ev.get(e)
        if checked:
            out[e] = {"level": "checked", "reasons": ["frame checked by you"]}
            continue
        if f is None or not frames:
            out[e] = {"level": "low", "reasons": ["not detected"]}
            continue
        i = min(range(len(frames)), key=lambda j: abs(frames[j] - f))
        lvl = EVENT_BASE[e]
        reasons = []
        if e == "bfc":
            reasons.append(f"back foot contact is found within 3 frames on "
                           f"only {EVENT_HIT_RATE['bfc']:.0f}% of clips")
        if abs(frames[i] - f) > 0:
            lvl -= 1
            reasons.append("no pose on that exact frame")
        lo, hi = max(0, i - 3), min(len(frames) - 1, i + 3)
        if any(frames[j + 1] - frames[j] > 1 for j in range(lo, hi)):
            lvl -= 1
            reasons.append("pose dropped out around this instant")
        vis = st["vis"].get(frames[i])
        if vis is not None and float(np.mean(vis[LEG_LANDMARKS])) < 0.6:
            lvl -= 1
            reasons.append("legs poorly visible")
        if st["auto_events"].get(e) == f:        # classifier evidence only
            p = phases.index("bfc_phase" if e == "bfc" else "ffc_phase")
            if float(P[i, p]) < 0.6:              # applies to its own frame
                lvl -= 1
                reasons.append("the classifier was unsure where the phase "
                               + ("ends" if e == "release" else "begins"))
            nb = st.get("n_blocks", {}).get(e, 1)
            if nb > 1:
                lvl -= 1
                reasons.append(f"the phase was split into {nb} pieces")
        out[e] = {"level": LEVELS[max(0, lvl)], "reasons": reasons}
    return out


def measure_confidence(key, quality):
    deps = EVENT_DEPS.get(key, ())
    if not deps:
        return "moderate", ""
    if all(quality[e]["level"] == "checked" for e in deps):
        return "checked", "event frames checked by you"
    worst = min(deps, key=lambda e: LEVELS.index(quality[e]["level"])
                if quality[e]["level"] in LEVELS else 3)
    lvl = quality[worst]["level"]
    why = ""
    if lvl != "high":
        why = (f"{EVENT_NAME[worst]} uncertain: "
               + "; ".join(quality[worst]["reasons"]))
    return lvl, why


# -------------------------------------------------------------- explanation
# Only claims already made, with citations, in the project report. Nothing
# here says what a bowler should change.
_ASSOC = {
    "front_knee_release":
        "In Worthington's cohort of 20 elite bowlers, front knee angle at "
        "release was one of four technique parameters that together explained "
        "73.6% of the variance in release speed.",
    "plant_angle":
        "Worthington found that braking impulse, rather than peak force, "
        "characterised the fastest bowlers in his cohort (r = 0.574), and that "
        "they generated it with a larger plant angle.",
}
_DESCRIBE = {
    "knee_extension":
        "Portus et al. (2004) classify front-knee technique by this behaviour: "
        "a knee that bends and re-extends before release is a flexor-extender, "
        "one that bends and stays bent is a flexor.",
    "knee_flexion":
        "Portus et al. (2004) classify front-knee technique by how far the knee "
        "bends after landing and whether it recovers before release.",
    "foot_strike":
        "This varies widely between individuals, and the reference bowlers all "
        "wore spiked shoes indoors, so footwear and surface shift it.",
    "stride_pct_height":
        "Converted from torso lengths with an anthropometric constant, so it is "
        "approximate.",
    "release_pct_height":
        "Measured wrist-above-ankle, so it reads roughly 10 cm low against the "
        "laboratory definition.",
}


def explain(key, z):
    """One rule-based paragraph for a measurement > 1 sd from elite, else None."""
    if z is None or abs(z) <= 1:
        return None
    head = (f"Your {LABEL[key].lower()} is {abs(z):.1f} sd "
            f"{'below' if z < 0 else 'above'} the elite mean.")
    if key in _ASSOC:
        return (f"{head} {_ASSOC[key]} This is an association across bowlers, "
                "not a prediction about you.")
    tail = _DESCRIBE.get(key, "")
    return (f"{head} {tail} Worthington reports this as a descriptive "
            "reference; being different from it is not by itself a fault."
            ).replace("  ", " ")


# ------------------------------------------------------------------ stages
def prepare(path, model_path="models/phase_model.joblib", progress=None):
    """Pose, features, phase probabilities and the auto-detected events.

    Returns (state, error). The state lets measure_at() re-measure at
    corrected event frames without running MediaPipe again.
    """
    bundle = joblib.load(model_path)
    model, feats, phases, k = (bundle["model"], bundle["features"],
                               list(bundle["phases"]), bundle["k"])

    poses, (W, H, fps) = run_pose(path, progress=progress)
    if len(poses) < 30:
        return None, ("no person detected for long enough -- check the bowler "
                      "is fully in frame")

    rows, flip = features_for(poses, W, H)
    if len(rows) < 30:
        return None, "pose was too unstable to measure"

    X = np.array([[r[f] for f in feats] for r in stack(rows, k)],
                 dtype=np.float32)
    prob = model.predict_proba(X)
    frames = [f for f, _, _ in rows]
    info = {}
    ev, err = extract_events(frames, None, prob, phases, info=info)
    if ev is None:
        return None, err

    byf = {f: np.array([[P[i][0], P[i][1]] for i in range(33)])
           for f, _, P in rows}
    st = {"frames": frames, "byf": byf, "P": info["P"], "phases": phases,
          "n_blocks": info.get("n_blocks", {}), "fps": float(fps),
          "W": W, "H": H, "flip": flip, "auto_events": dict(ev),
          "poses": {f: P for f, P in poses},
          "vis": {f: P[:, 2] for f, P in poses}}
    return st, ""


def measure_at(st, ev, checked=False):
    """Every measurement, comparison and safeguard at the given event frames."""
    import measure_deliveries as md
    if not (ev["bfc"] < ev["ffc"] < ev["release"]):
        return {"ok": False, "error": "back foot contact must come before "
                "front foot contact, and that before release"}
    m, note = md.measure_clip(st["byf"], ev, 1.0, tol=3)
    if m is None:
        return {"ok": False, "error": note or "could not measure the delivery"}

    vals = {k2: v for k2, v in m.items() if not k2.startswith("_")}
    for src, dst in (("stride_torso", "stride_pct_height"),
                     ("release_hgt_torso", "release_pct_height")):
        v = vals.get(src)
        vals[dst] = (v * TRUNK_RATIO * 100) if v is not None and not math.isnan(v) else None

    quality = event_quality(st, ev, checked)
    rows, comp = {}, {}
    for key, label, unit in MEASURES:
        v = vals.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        row = {"label": label, "unit": unit, "value": round(v, 1)}
        z = None
        if key in REF:
            mu, sd = REF[key]
            z = (v - mu) / sd
            row.update(ref_mean=mu, ref_sd=sd, z=round(z, 2), band=band(z))
            comp[key] = {"value": round(v, 1), "ref_mean": mu, "ref_sd": sd,
                         "z": round(z, 2), "band": band(z)}
        coh = COHORT.get(key)
        plaus = "ok"
        if coh and coh["sd"] > 0:
            cz = (v - coh["mean"]) / coh["sd"]
            row.update(cohort_mean=round(coh["mean"], 1), cohort_z=round(cz, 2))
            plaus = ("implausible" if abs(cz) > IMPLAUSIBLE_SD else
                     "unusual" if abs(cz) > UNUSUAL_SD else "ok")
        row["plausibility"] = plaus
        row["confidence"], row["confidence_reason"] = measure_confidence(key, quality)
        # an implausible value on unchecked frames is a detection question,
        # not a finding: no coaching text is attached to it
        trusted = checked or plaus != "implausible"
        row["trusted"] = trusted
        row["explanation"] = explain(key, z) if trusted else None
        rows[key] = row

    return {"ok": True, "events": dict(ev), "checked": checked,
            "auto_events": dict(st["auto_events"]),
            "fps": round(st["fps"], 1), "frames_analysed": len(st["frames"]),
            "direction": "R2L" if st["flip"] else "L2R",
            "quality": quality, "rows": rows,
            "warning": pattern_warning(rows, checked),
            "measurements": {a: (round(b, 2) if isinstance(b, float)
                                 and not math.isnan(b) else None)
                             for a, b in vals.items()},
            "comparison": comp, "note": note}


def pattern_warning(rows, checked):
    """A coherent set of extremes usually means the event frames are wrong."""
    bad = [k for k, r in rows.items() if r["plausibility"] == "implausible"]
    odd = [k for k, r in rows.items() if r["plausibility"] != "ok"]
    if not bad and len(odd) < 2:
        return None
    hits = defaultdict(int)
    for k in odd:
        for e in EVENT_DEPS.get(k, ()):
            hits[e] += 1
    top = max(hits.values())
    suspects = [e for e in ("bfc", "ffc", "release") if hits.get(e) == top]
    names = " and ".join(EVENT_NAME[e] for e in suspects)
    listed = ", ".join(LABEL[k].lower() for k in odd)
    lead = (f"{len(odd)} values sit far outside our own 262-delivery cohort "
            f"({listed})." if len(odd) > 1 else
            f"{LABEL[odd[0]]} sits far outside our own 262-delivery cohort.")
    if checked:
        text = (f"{lead} You have checked the frames, so "
                f"{'these are' if len(odd) > 1 else 'it is'} reported, but "
                "MediaPipe's joint-angle error on fast bowling has not been "
                "validated -- treat with care.")
    else:
        why = ("Extremes that move together are the signature" if len(odd) > 1
               else "A value this extreme is usually the sign")
        text = (f"{lead} {why} of a wrongly detected event, most likely "
                f"{names}. Check the frames below before reading anything "
                "into these numbers.")
    return {"flagged": odd, "implausible": bad, "suspects": suspects,
            "text": text}


def summarise(results):
    """Mean and spread over several deliveries by the same bowler.

    Values on unchecked frames that fail the plausibility check are left out
    of the mean, since they are almost certainly detection errors.
    """
    good = [r for r in results if r.get("ok")]
    out = {}
    for key, label, unit in MEASURES:
        use, dropped = [], 0
        for r in good:
            row = r["rows"].get(key)
            if not row:
                continue
            if row["trusted"]:
                use.append(row["value"])
            else:
                dropped += 1
        if not use:
            continue
        mean = float(np.mean(use))
        sd = float(np.std(use, ddof=1)) if len(use) > 1 else None
        row = {"label": label, "unit": unit, "value": round(mean, 1),
               "spread": round(sd, 1) if sd is not None else None,
               "n": len(use), "dropped": dropped}
        z = None
        if key in REF:
            mu, rsd = REF[key]
            z = (mean - mu) / rsd
            row.update(ref_mean=mu, ref_sd=rsd, z=round(z, 2), band=band(z))
        row["explanation"] = explain(key, z)
        out[key] = row
    return out


def analyse(path, model_path="models/phase_model.joblib", height_cm=None,
            progress=None):
    st, err = prepare(path, model_path, progress=progress)
    if st is None:
        return {"ok": False, "error": err}
    return measure_at(st, st["auto_events"])


# ----------------------------------------------------------------- frames
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32),
         (0, 11), (0, 12)]


def grab_frames(path, st, pad=25, height=360, max_width=1280):
    """JPEG bytes, skeleton drawn on, for every frame from pad before back foot
    contact to pad after release -- enough to nudge any event by hand."""
    ev = st["auto_events"]
    lo, hi = max(0, ev["bfc"] - pad), ev["release"] + pad
    # one crop box for the whole strip, around everywhere the bowler goes, so
    # the feet are big enough to judge contact and frames stay comparable
    pts = [P[:, :2] for f, P in st["poses"].items() if lo <= f <= hi]
    box = None
    if pts:
        A = np.concatenate(pts)
        (x0, y0), (x1, y1) = A.min(axis=0), A.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hh = (y1 - y0) * 1.25
        aspect = (st["W"] or 16) / (st["H"] or 9)
        ww = max((x1 - x0) * 1.15, hh * 16 / 9 / aspect)
        hh = max(hh, ww * 9 / 16 * aspect)
        box = (max(0.0, cx - ww / 2), max(0.0, cy - hh / 2),
               min(1.0, cx + ww / 2), min(1.0, cy + hh / 2))
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    out, i = {}, lo
    while i <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > max_width:                     # same scale run_pose used
            fr = cv2.resize(fr, (max_width, int(h * max_width / w)))
            h, w = fr.shape[:2]
        P = st["poses"].get(i)
        if P is not None:
            pts = [(int(x * w), int(y * h)) for x, y, _ in P]
            for a, b in EDGES:
                cv2.line(fr, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
            for (x, y), v in zip(pts, P[:, 2]):
                cv2.circle(fr, (x, y), 4, (0, 255, 0) if v > 0.5 else (0, 0, 255),
                           -1, cv2.LINE_AA)
        if box:
            fr = fr[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
            h, w = fr.shape[:2]
        s = height / h
        fr = cv2.resize(fr, (int(w * s), height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out[i] = buf.tobytes()
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--model", default="models/phase_model.joblib")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--build-cohort", metavar="CSV",
                    help="rebuild models/cohort_stats.json from measurements.csv")
    a = ap.parse_args()
    if a.build_cohort:
        s = build_cohort(a.build_cohort)
        print(f"wrote {COHORT_PATH}: {len(s)} measurements"); return
    if not a.video:
        ap.error("give a video")
    r = analyse(a.video, a.model)
    if a.json:
        print(json.dumps(r, indent=2)); return
    if not r["ok"]:
        sys.exit("could not analyse: " + r["error"])
    print(f"\n{os.path.basename(a.video)}   {r['frames_analysed']} frames, "
          f"{r['fps']:g} fps, travelling {r['direction']}")
    print("events: " + "   ".join(
        f"{EVENT_NAME[e]} {r['events'][e]} ({r['quality'][e]['level']})"
        for e in ("bfc", "ffc", "release")) + "\n")
    print(f"  {'measurement':<34}{'you':>8}{'elite':>16}{'':>12}  confidence")
    for k2, row in r["rows"].items():
        if not row["trusted"]:
            diff = "check frames"
        elif "z" in row:
            diff = f"{row['z']:+.1f} sd"
        else:
            diff = ""
        elite = (f"{row['ref_mean']:.1f} +-{row['ref_sd']:>4.1f}"
                 if "ref_mean" in row else "")
        print(f"  {row['label']:<34}{row['value']:>8.1f}{elite:>16}{diff:>12}"
              f"  {row['confidence']}")
    if r["warning"]:
        print("\nWARNING: " + r["warning"]["text"])
    print("\nlegality of the bowling action is NOT assessed.")


if __name__ == "__main__":
    main()
