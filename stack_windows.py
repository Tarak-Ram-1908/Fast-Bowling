"""
Add temporal context to features.csv.

    py -3.13 stack_windows.py features.csv --out feat_win.csv --k 5
    py -3.13 stack_windows.py feat_noab.csv --out feat_noab_win.csv --k 5

WHY
---
Every one of the sixteen features describes a single frame in isolation: where
an ankle is, how far apart the feet are, how high a wrist sits. Not one of them
encodes motion.

Foot contact is not a position. It is the instant a descending foot stops
descending. A planted ankle and an ankle 8 ms before planting look nearly
identical in a still frame -- which is exactly the pattern in the results:
ball release, where the arm is in a visually distinct configuration, lands
within 3 frames on 95% of deliveries, while back foot contact manages 51%.

This script gives each frame its neighbours. Frame i gets the features of
frames i-k ... i+k, plus explicit first differences, so the classifier can
finally tell a falling ankle from a stationary one.

WHAT IT WRITES
--------------
For each base feature f and each offset d in -k..k (excluding 0):
    f__m5, f__m4, ... f__p4, f__p5     the value at frame i+d
plus, for each base feature:
    f__d1      f(i) - f(i-1)           instantaneous change
    f__d3      f(i) - f(i-3)           change over 25 ms
    f__v       (f(i+1) - f(i-1)) / 2   central difference, a velocity

Windows are built per clip and clamped at clip edges, so no frame ever borrows
from a different delivery.

With k=5 the 16 base features become 16 * (1 + 10 + 3) = 224 columns. The
random forest handles that comfortably at this sample size; the logistic
regression may not, so compare both.
"""
import argparse, csv, sys
from collections import defaultdict

META = {"video", "bowler", "frame", "phase", "direction", "torso_px",
        "is_bfc_event", "is_ffc_event", "is_release_event",
        "width", "height", "legal", "usable"}

ap = argparse.ArgumentParser()
ap.add_argument("features")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=5, help="half-window in frames")
ap.add_argument("--no-diffs", action="store_true", help="skip the difference columns")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.features, newline="")))
if not rows:
    sys.exit("empty input")
base = [c for c in rows[0] if c not in META]
if not base:
    sys.exit("no feature columns found")

# group by clip, keep frame order
clips = defaultdict(list)
for r in rows:
    clips[r["video"]].append(r)
for v in clips:
    clips[v].sort(key=lambda r: int(r["frame"]))

def val(rec, f):
    try: return float(rec[f])
    except (TypeError, ValueError): return 0.0

offs = [d for d in range(-a.k, a.k + 1) if d != 0]
cols = list(rows[0].keys())
for f in base:
    for d in offs:
        cols.append(f"{f}__{'m' if d < 0 else 'p'}{abs(d)}")
    if not a.no_diffs:
        cols += [f"{f}__d1", f"{f}__d3", f"{f}__v"]

n = 0
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for v, seq in clips.items():
        L = len(seq)
        for i, r in enumerate(seq):
            out = dict(r)
            for f in base:
                here = val(r, f)
                for d in offs:
                    j = min(max(i + d, 0), L - 1)          # clamp at clip edges
                    out[f"{f}__{'m' if d < 0 else 'p'}{abs(d)}"] = f"{val(seq[j], f):.5f}"
                if not a.no_diffs:
                    pm1 = val(seq[max(i - 1, 0)], f)
                    pp1 = val(seq[min(i + 1, L - 1)], f)
                    pm3 = val(seq[max(i - 3, 0)], f)
                    out[f"{f}__d1"] = f"{here - pm1:.5f}"
                    out[f"{f}__d3"] = f"{here - pm3:.5f}"
                    out[f"{f}__v"]  = f"{(pp1 - pm1) / 2:.5f}"
            w.writerow(out); n += 1

print(f"{len(rows)} rows, {len(clips)} clips, {len(base)} base features")
print(f"-> {n} rows, {len(cols) - len([c for c in rows[0] if c in META])} feature columns, k={a.k}")
print(f"written: {a.out}")
print(f"\nnext:  py -3.13 train_classifier.py {a.out} --folds 24 --auto-features --out predictions.csv")
