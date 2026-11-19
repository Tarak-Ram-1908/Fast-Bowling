"""
Measure the delivery: joint angles and distances at the three event instants.

    python measure_deliveries.py --out measurements.csv
    python measure_deliveries.py --at true        # measure at YOUR marks
    python measure_deliveries.py --at both        # both, and the difference

This is the output the project exists to produce. Everything before it --
labelling, pose, features, the classifier, event extraction -- was to arrive at
three frame numbers per delivery. This reads the pose at those frames and
computes the quantities Worthington reports.

WHAT IS MEASURED, and why each one is honest from a single side-on camera.

Scale-free, directly comparable to Worthington:

  front_knee_ffc        angle at the front knee at front foot contact.
  front_knee_release    the same at ball release. Straight = 180 deg, and a
                        knee past straight reads above 180 as in Worthington
                        (his range at release reaches 186.2).
  knee_flexion          how far it bent from front foot contact to its most
                        bent. NOT compared with Worthington: he measures from
                        front foot FLAT, a few frames later, which this system
                        does not detect, so ours reads several degrees higher.
  knee_extension        how far it straightened again, most bent to release.
  plant_angle           front hip to front ankle, against the downward
                        vertical, at front foot contact. Worthington defines
                        this as a TWO-dimensional angle on a vertical plane --
                        the plane a side-on camera sees -- so this is the one
                        measurement here that needs no reinterpretation at all.
  foot_strike           front ankle to foot index, against the horizontal.
                        Positive is dorsi-flexed (heel first), negative is
                        plantar-flexed (toe first).
  trunk_lean_ffc        hip midpoint to shoulder midpoint, against vertical,
  trunk_lean_release    positive leaning forward down the pitch.
  lean_back_bfc         back ankle to shoulder midpoint at back foot contact.
  arm_angle_release     upper arm (shoulder to elbow) against the line of the
                        trunk. Zero and sign differ from Worthington's 3D
                        shoulder angle, so it is never compared with it.

In torso lengths, because the camera has no metric scale:

  stride_torso          ankle separation along the pitch at front foot
                        contact -- Worthington's definition (thesis 4.8.2).
  stride_contacts_torso back foot position at back foot contact to front foot
                        position at front foot contact. Reported, never
                        compared: it is not what Worthington measured.
  release_hgt_torso     wrist height above the lower ankle at release.

Multiply those two by the bowler's torso length in cm to get centimetres, or
fill height_cm in legality.csv and this prints them as a percentage of
standing height the way Worthington does.

THE SENSITIVITY CHECK. With --at both, every quantity is computed twice: once
at the frames you marked by hand, once at the frames the classifier predicted.
The difference is what the event timing error actually COSTS a measurement,
in degrees. That number matters more than the frame error itself -- a two
frame miss on a nearly-straight knee is worth a fraction of a degree, and the
same miss during rapid flexion is worth several. Report it.

Only deliveries marked legal in legality.csv are measured. Worthington's
reference values describe legal fast bowling, and nothing else belongs beside
them. --include-illegal measures the rest as a separate group.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np

# MediaPipe landmarks
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Worthington (2010), Chapter 5: twenty elite fast bowlers, mean +- SD
WORTHINGTON = {
    "front_knee_ffc":     (164.1, 6.1,  "deg"),
    "front_knee_release": (167.3, 18.8, "deg"),
    "knee_extension":     (11.9,  7.4,  "deg"),
    "plant_angle":        (36.3,  3.9,  "deg"),
    "foot_strike":        (-8.24, 13.2, "deg"),
    "lean_back_bfc":      (9.02,  5.22, "deg"),
    "stride_pct_height":  (78.5,  7.2,  "%"),
    "release_pct_height": (112.8, 4.1,  "%"),
}

MEASURES = ["front_knee_ffc", "front_knee_release", "knee_flexion",
            "knee_extension", "plant_angle", "foot_strike",
            "trunk_lean_ffc", "trunk_lean_release", "lean_back_bfc",
            "arm_angle_release", "stride_torso", "stride_contacts_torso",
            "release_hgt_torso"]

# derived from the two torso-relative lengths, for the Worthington comparison
DERIVED = ["stride_pct_height", "release_pct_height"]


# ------------------------------------------------------------------ geometry

def joint_angle(a, b, c):
    """Interior angle at b, degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    return math.degrees(math.acos(
        max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))))


def knee_angle(hip, knee, ankle):
    """Knee angle in the sagittal plane, 180 = straight, >180 = past straight.

    The interior angle alone cannot exceed 180, so a hyper-extended knee would
    fold back and read as bent -- a bias toward flexion exactly where legs are
    near straight. Flexion moves the knee forward of the hip-ankle line (the
    bowler faces +x after normalising), so the side of that line the knee is
    on gives the sign.
    """
    inner = joint_angle(hip, knee, ankle)
    d = ankle - hip
    forward = float(np.dot(knee - hip, np.array([d[1], -d[0]])))
    return inner if forward >= 0 else 360.0 - inner


def from_vertical(v):
    """Angle of v from straight up, positive toward +x (down the pitch).

    Image y grows downward, so 'up' is (0, -1).
    """
    dx, dy = float(v[0]), float(v[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return float("nan")
    return math.degrees(math.atan2(dx, -dy))


def from_horizontal(v):
    """Angle of v from the +x axis, positive upward in the image."""
    dx, dy = float(v[0]), float(v[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return float("nan")
    return math.degrees(math.atan2(-dy, dx))


# ------------------------------------------------------------------ loading

def load_keypoints(path):
    """{video: {frame: P}} with P a 33x2 array in units of image height,
    plus the direction each clip travels."""
    raw = defaultdict(dict)
    hipx = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["detected"] != "1":
                continue
            ar = float(r["width"]) / float(r["height"])
            P = np.empty((33, 2))
            ok = True
            for i in range(33):
                try:
                    P[i] = (float(r[f"x{i}"]) * ar, float(r[f"y{i}"]))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            f = int(r["frame"])
            raw[r["video"]][f] = P
            hipx[r["video"]].append((f, (P[L_HIP][0] + P[R_HIP][0]) / 2))

    direction = {}
    for vid, pts in hipx.items():
        if len(pts) > 2:
            fs, xs = zip(*sorted(pts))
            slope = np.polyfit(fs, xs, 1)[0]
        else:
            slope = 1.0
        direction[vid] = 1.0 if slope >= 0 else -1.0
    return raw, direction


def load_events(events_csv, labels_csv):
    """(predicted, true) as {video: {bfc, ffc, release}}."""
    pred, true = {}, {}
    with open(events_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k in ("bfc", "ffc", "release"):
                try:
                    d[k] = int(r[f"pred_{k}"])
                except (ValueError, KeyError):
                    d[k] = None
            pred[r["video"]] = d
    with open(labels_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k in ("bfc", "ffc", "release"):
                try:
                    d[k] = int(r[k])
                except (ValueError, KeyError):
                    d[k] = None
            true[r["video"]] = d
    return pred, true


def load_legality(path):
    """{video: row}, carrying any per-bowler columns typed in by hand."""
    out = {}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                out[r["video"]] = r
    except FileNotFoundError:
        pass
    return out


def heights_by_bowler(leg, path="bowlers.csv"):
    """{bowler: cm}. Prefers bowlers.csv, one row per bowler; falls back to a
    height_cm column inside legality.csv if that file is absent."""
    import os
    src = []
    if os.path.exists(path):
        with open(path, newline="") as fh:
            src = list(csv.DictReader(fh))
    else:
        src = list(leg.values())
    seen = defaultdict(set)
    for r in src:
        v = (r.get("height_cm") or "").strip()
        if v:
            try:
                seen[r["bowler"]].add(float(v))
            except ValueError:
                pass
    out, bad = {}, []
    for b, vals in seen.items():
        if len(vals) == 1:
            out[b] = vals.pop()
        else:
            bad.append((b, sorted(vals)))
    return out, bad


# ------------------------------------------------------------------ measuring

def nearest(frames, want, tol):
    """The sampled frame closest to `want`, or None past tolerance."""
    if want is None or not frames:
        return None
    f = min(frames, key=lambda x: abs(x - want))
    return f if abs(f - want) <= tol else None


def from_down(v):
    """Angle of v from straight DOWN, positive toward +x (down the pitch).

    Image y grows downward, so 'down' is (0, +1). A front leg reaching out
    in front of the hip gives a positive angle -- Worthington's plant angle.
    """
    dx, dy = float(v[0]), float(v[1])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return float("nan")
    return math.degrees(math.atan2(dx, dy))


def measure(P, flip):
    """Everything computable from a single frame, after normalising."""
    Q = P.copy()
    Q[:, 0] *= flip                       # every bowler now travels +x

    hip_c = (Q[L_HIP] + Q[R_HIP]) / 2
    sh_c = (Q[L_SH] + Q[R_SH]) / 2
    torso = float(np.linalg.norm(sh_c - hip_c))
    if torso < 0.02:
        return None

    # leading leg = whichever ankle is further down the pitch; at front foot
    # contact that is the front leg, at back foot contact the back leg is the
    # trailing one. Order statistics, so handedness never enters.
    if Q[L_AN][0] >= Q[R_AN][0]:
        lead, trail = (L_HIP, L_KN, L_AN, L_FOOT), (R_HIP, R_KN, R_AN, R_FOOT)
    else:
        lead, trail = (R_HIP, R_KN, R_AN, R_FOOT), (L_HIP, L_KN, L_AN, L_FOOT)

    # bowling arm = the higher wrist; its upper arm is shoulder to elbow
    high = (L_SH, L_EL, L_WR) if Q[L_WR][1] <= Q[R_WR][1] else (R_SH, R_EL, R_WR)

    trunk = from_vertical(sh_c - hip_c)
    ground = float(max(Q[L_AN][1], Q[R_AN][1]))      # lower ankle = the floor

    return {
        "torso": torso,
        "knee": knee_angle(Q[lead[0]], Q[lead[1]], Q[lead[2]]),
        "plant": from_down(Q[lead[2]] - Q[lead[0]]),
        "foot_strike": from_horizontal(Q[lead[3]] - Q[lead[2]]),
        "trunk": trunk,
        # positive = leaning away from the batsman, as Worthington reports it.
        # Measured from the PLANTED foot, which is the lower ankle, not the
        # rearward one. At back foot contact the two ankles can be at nearly
        # the same x, so the lead/trail order statistic flips between adjacent
        # frames and the angle swings by over 150 degrees. Height off the
        # ground does not flip: the planted foot is the one on the floor.
        "lean_back": -from_vertical(
            sh_c - (Q[L_AN] if Q[L_AN][1] >= Q[R_AN][1] else Q[R_AN])),
        "lean_back_trail": -from_vertical(sh_c - Q[trail[2]]),
        "arm_vs_trunk": from_vertical(Q[high[1]] - Q[high[0]]) - trunk,
        "stride": abs(float(Q[lead[2]][0] - Q[trail[2]][0])) / torso,
        "lead_ankle_x": float(Q[lead[2]][0]),
        "trail_ankle_x": float(Q[trail[2]][0]),
        "release_hgt": (ground - float(Q[high[2]][1])) / torso,
    }


def measure_clip(frames, ev, flip, tol):
    """The twelve quantities for one delivery at one set of event frames."""
    idx = sorted(frames)
    f_bfc = nearest(idx, ev.get("bfc"), tol)
    f_ffc = nearest(idx, ev.get("ffc"), tol)
    f_rel = nearest(idx, ev.get("release"), tol)
    if f_ffc is None or f_rel is None:
        return None, "no pose at front foot contact or release"

    m_ffc = measure(frames[f_ffc], flip)
    m_rel = measure(frames[f_rel], flip)
    if m_ffc is None or m_rel is None:
        return None, "degenerate pose at an event frame"
    m_bfc = measure(frames[f_bfc], flip) if f_bfc is not None else None

    # the knee through the contact phase, for flexion and extension
    knees = []
    for f in idx:
        if f_ffc <= f <= f_rel:
            mm = measure(frames[f], flip)
            if mm and not math.isnan(mm["knee"]):
                knees.append((mm["knee"], f))
    knee_min, f_min = min(knees) if knees else (float("nan"), None)

    out = {
        "front_knee_ffc": m_ffc["knee"],
        "front_knee_release": m_rel["knee"],
        "knee_flexion": m_ffc["knee"] - knee_min,
        "knee_extension": m_rel["knee"] - knee_min,
        "plant_angle": m_ffc["plant"],
        "foot_strike": m_ffc["foot_strike"],
        "trunk_lean_ffc": m_ffc["trunk"],
        "trunk_lean_release": m_rel["trunk"],
        "lean_back_bfc": m_bfc["lean_back"] if m_bfc else float("nan"),
        "arm_angle_release": m_rel["arm_vs_trunk"],
        # Worthington (thesis 4.8.2): "the distance between the ankle joint
        # centre of each leg ... at the instant of front foot contact". One
        # frame, both ankles. From the side only the along-pitch component is
        # visible; the across-pitch part of a typical stride adds about 2 cm.
        "stride_torso": m_ffc["stride"],
        # Back foot where it planted to front foot where it planted. Longer,
        # because by front foot contact the back foot has pushed off and
        # moved forward. Not Worthington's definition, so never compared.
        "stride_contacts_torso": (abs(m_ffc["lead_ankle_x"] - m_bfc["trail_ankle_x"])
                                  / m_ffc["torso"]) if m_bfc else float("nan"),
        "release_hgt_torso": m_rel["release_hgt"],
        "_torso_ffc": m_ffc["torso"],
        # the frames actually measured, for drawing them
        "_f_bfc": f_bfc, "_f_ffc": f_ffc, "_f_rel": f_rel, "_f_min": f_min,
    }
    note = "" if m_bfc else "no pose at back foot contact"
    return out, note


# ------------------------------------------------------------------ reporting

def summarise(rows, key, label):
    vals = [r[key] for r in rows
            if r.get(key) is not None and not math.isnan(r[key])]
    if not vals:
        return None
    a = np.array(vals)
    return (label, len(a), a.mean(), a.std(), a.min(), a.max())


def print_table(rows, title, keys):
    print(f"\n{title}  (n = {len(rows)} deliveries)\n")
    print(f"  {'measurement':22s}{'n':>5s}{'mean':>9s}{'sd':>8s}"
          f"{'min':>9s}{'max':>9s}   Worthington")
    for k in keys:
        s = summarise(rows, k, k)
        if not s:
            continue
        _, n, mu, sd, lo, hi = s
        ref = ""
        if k in WORTHINGTON:
            wm, wsd, unit = WORTHINGTON[k]
            z = (mu - wm) / wsd if wsd else 0.0
            ref = f"   {wm:6.1f} +- {wsd:4.1f}   ({z:+.1f} sd)"
        print(f"  {k:22s}{n:5d}{mu:9.1f}{sd:8.1f}{lo:9.1f}{hi:9.1f}{ref}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", default="keypoints.csv")
    ap.add_argument("--events", default="events.csv")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--legality", default="legality.csv")
    ap.add_argument("--out", default="measurements.csv")
    ap.add_argument("--at", default="both", choices=["pred", "true", "both"],
                    help="measure at the predicted event frames, at your own "
                         "marks, or both and report the difference")
    ap.add_argument("--trunk-ratio", type=float, default=0.288,
                    help="hip joint to shoulder joint as a fraction of "
                         "standing height. 0.288 follows Winter (1990): "
                         "shoulder 0.818H, hip 0.530H. This is what turns a "
                         "torso-relative length into a percentage of standing "
                         "height without measuring anybody")
    ap.add_argument("--tol", type=int, default=3,
                    help="how far the nearest sampled frame may be from the "
                         "event frame")
    ap.add_argument("--include-illegal", action="store_true",
                    help="also measure the deliveries marked illegal, "
                         "reported as their own group")
    args = ap.parse_args()

    print("reading pose ...", flush=True)
    kp, direction = load_keypoints(args.keypoints)
    pred, true = load_events(args.events, args.labels)
    leg = load_legality(args.legality)
    heights, conflicting = heights_by_bowler(leg)

    if not leg:
        sys.exit(f"{args.legality} not found -- run label_legality.py first")
    if conflicting:
        print("\n  WARNING, height_cm disagrees with itself for:")
        for b, vals in conflicting:
            print(f"    {b}: {', '.join(str(v) for v in vals)}")
        print("  those bowlers get no percentage-of-height figures.\n")

    legal = {v for v, r in leg.items() if r.get("legal") == "1"}
    illegal = {v for v, r in leg.items() if r.get("legal") == "0"}
    print(f"{len(legal)} legal deliveries, {len(illegal)} illegal, "
          f"{len(kp)} clips with pose")

    rows, skipped = [], defaultdict(int)
    for vid in sorted(kp):
        if vid not in leg:
            skipped["not in legality.csv"] += 1
            continue
        is_legal = vid in legal
        if not is_legal and not args.include_illegal:
            continue
        frames = kp[vid]
        flip = direction[vid]
        bowler = leg[vid]["bowler"]

        got = {}
        for which, src in (("true", true), ("pred", pred)):
            if args.at in (which, "both") and vid in src:
                m, note = measure_clip(frames, src[vid], flip, args.tol)
                if m is None:
                    skipped[note] += 1
                else:
                    got[which] = m

        primary = got.get("pred") or got.get("true")
        if primary is None:
            continue

        row = {"video": vid, "bowler": bowler,
               "legal": 1 if is_legal else 0,
               "measured_at": "pred" if "pred" in got else "true"}
        for k in MEASURES:
            row[k] = round(primary[k], 2) if not math.isnan(primary[k]) else ""
        # what the event timing error costs, in the units of each measurement
        if "true" in got and "pred" in got:
            for k in MEASURES:
                a, b = got["true"][k], got["pred"][k]
                row[f"d_{k}"] = (round(b - a, 2)
                                 if not (math.isnan(a) or math.isnan(b)) else "")
        # Percentage of standing height needs no measured height at all.
        # A length in torso units times (torso / standing height) is a length
        # in standing heights, and that trunk ratio is an anthropometric
        # constant, not a per-person measurement. See --trunk-ratio.
        for src, dst in (("stride_torso", "stride_pct_height"),
                         ("release_hgt_torso", "release_pct_height")):
            v = primary[src]
            row[dst] = (round(v * args.trunk_ratio * 100, 2)
                        if not math.isnan(v) else None)
        rows.append(row)

    for k in MEASURES + DERIVED:
        for r in rows:
            if r.get(k) == "":
                r[k] = None
            elif r.get(k) is not None:
                r[k] = float(r[k])

    if skipped:
        print("\nskipped:")
        for k, n in sorted(skipped.items(), key=lambda z: -z[1]):
            print(f"  {n:4d}  {k}")

    lg = [r for r in rows if r["legal"] == 1]
    il = [r for r in rows if r["legal"] == 0]
    print_table(lg, "LEGAL DELIVERIES", MEASURES + DERIVED)
    if il:
        print_table(il, "ILLEGAL DELIVERIES (reported separately, "
                        "NOT comparable to Worthington)", MEASURES + DERIVED)

    # ---------------------------------------------------- per bowler
    print("\nper-bowler means, legal deliveries only\n")
    cols = ["front_knee_ffc", "front_knee_release", "plant_angle",
            "foot_strike", "trunk_lean_release", "stride_torso"]
    print(f"  {'bowler':13s}{'n':>4s}" + "".join(f"{c[:11]:>13s}" for c in cols))
    byb = defaultdict(list)
    for r in lg:
        byb[r["bowler"]].append(r)
    for b in sorted(byb):
        rs = byb[b]
        line = f"  {b:13s}{len(rs):4d}"
        for c in cols:
            v = [x[c] for x in rs if x.get(c) is not None]
            line += f"{np.mean(v):13.1f}" if v else f"{'-':>13s}"
        print(line)

    # ---------------------------------------------------- sensitivity
    if args.at == "both":
        d = [r for r in rows if any(k.startswith("d_") for k in r)]
        if d:
            print("\n\nWHAT THE EVENT TIMING ERROR COSTS\n")
            print("  measuring at the predicted frames instead of your marks "
                  "changes\n  each quantity by this much:\n")
            print(f"  {'measurement':22s}{'n':>5s}{'mean |diff|':>13s}"
                  f"{'worst':>9s}")
            for k in MEASURES:
                v = [abs(r[f"d_{k}"]) for r in d
                     if isinstance(r.get(f"d_{k}"), float)]
                if v:
                    print(f"  {k:22s}{len(v):5d}{np.mean(v):13.2f}"
                          f"{max(v):9.2f}")
            print("\n  A quantity near the top of this list is one your "
                  "measurement\n  inherits the event error from. One near "
                  "zero is immune to it.")

    # ---------------------------------------------------- save
    base = ["video", "bowler", "legal", "measured_at"]
    cols_out = base + MEASURES + DERIVED + [
        f"d_{k}" for k in MEASURES if any(f"d_{k}" in r for r in rows)]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c, ""))
                        for c in cols_out})
    print(f"\n{len(rows)} deliveries written to {args.out}")

    print(f"""
stride_pct_height and release_pct_height were converted from torso lengths
using a trunk ratio of {args.trunk_ratio:.3f} (Winter 1990), NOT from measured
heights. Say so in the report. As a check on the constant: Worthington's mean
stride of 78.5% of height and mean release height of 112.8% correspond to
{78.5/(args.trunk_ratio*100):.2f} and {112.8/(args.trunk_ratio*100):.2f} torso
lengths, which is what an elite bowler's own measurements come out at.

Two caveats on release height. It is the WRIST above the lower ankle, not the
ball above the ground, so expect it to read roughly 10 cm low. And the lower
ankle is a stand-in for ground level, which is good to a centimetre or two.""")
    if heights:
        print(f"\nheight_cm is filled in for {len(heights)} bowler(s). It is "
              f"not used above,\nbut it is what you need for absolute "
              f"centimetres rather than percentages.")


if __name__ == "__main__":
    main()
