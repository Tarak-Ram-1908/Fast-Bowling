"""
Front-leg technique classification, after Portus et al. (2004).

    py -3.13 classify_technique.py measurements.csv

This does NOT label injury risk, and it deliberately does not try to.  What it
does is apply the published front-leg classification to measurements you
already have, and then state plainly which injury-related variables a side-on
camera cannot recover.  That second part is a finding, not a failure.


THE RULE (Portus et al. 2004, Sports Biomechanics 3(2), 263-284)
----------------------------------------------------------------
Measured from front foot contact to ball release, with a 10 degree threshold
on each of flexion and extension:

    flexor            flexion >= 10,  extension <  10
    flexor-extender   flexion >= 10,  extension >= 10
    extender          flexion <  10,  extension >= 10
    constant brace    flexion <  10,  extension <  10

Portus classified 42 high-performance bowlers as 25 flexors, 9 constant brace,
4 extenders and 4 flexor-extenders, and noted that the extender and
flexor-extender groups (n = 4 each) had too little statistical power to support
injury conclusions.


WHAT THIS CANNOT TELL YOU
-------------------------
The strongest technique-injury association in the literature is shoulder
counter-rotation: Portus found 41 +- 10 degrees in the lumbar stress fracture
group against 19 +- 10 in the uninjured group (F = 4.5, P = 0.01).  The
trunk classification that depends on it -- side-on, semi-open, front-on,
mixed -- is defined on hip and shoulder segment angles in the TRANSVERSE
plane, measured at back foot contact.

A single side-on camera cannot recover transverse plane rotation.  Colyer et
al. (2018) put the constraint exactly: 2D analysis recovers three degrees of
freedom from a minimum of two known points, while six degrees of freedom need
at least three non-collinear points, which one view cannot supply.  The same
review concludes that transverse plane rotations are the hardest quantity for
markerless systems even with many cameras, and warns that projecting onto 2D
planes makes movement in one plane register as movement in another -- citing
cricket as its example.

So this script reports the front-leg classification, which is sagittal and
therefore honestly measurable, and reports the trunk classification as
not determinable from this data.  Saying so is worth more than guessing.
"""

import argparse
import collections
import csv
import math
import statistics
import sys

THRESHOLD = 10.0

PORTUS_42 = {"flexor": 25, "constant brace": 9,
             "extender": 4, "flexor-extender": 4}


def classify(flexion, extension, thr=THRESHOLD):
    if flexion is None or extension is None:
        return None
    f = flexion >= thr
    e = extension >= thr
    if f and not e:
        return "flexor"
    if f and e:
        return "flexor-extender"
    if e:
        return "extender"
    return "constant brace"


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def bar(n, total, width=34):
    return "#" * int(round(width * n / total)) if total else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurements", nargs="?", default="measurements.csv")
    ap.add_argument("--out", default="technique.csv")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--legal-only", action="store_true",
                    help="use only deliveries marked legal")
    args = ap.parse_args()

    try:
        rows = list(csv.DictReader(open(args.measurements, newline="")))
    except FileNotFoundError:
        sys.exit(f"not found: {args.measurements}")

    for c in ("knee_flexion", "knee_extension"):
        if c not in rows[0]:
            sys.exit(f"{args.measurements} has no '{c}' column -- "
                     "run measure_deliveries.py first")

    used, skipped = [], 0
    for r in rows:
        if args.legal_only and r.get("legal", "1") == "0":
            skipped += 1
            continue
        fl, ex = num(r.get("knee_flexion")), num(r.get("knee_extension"))
        cls = classify(fl, ex, args.threshold)
        if cls is None:
            skipped += 1
            continue
        r["_flexion"], r["_extension"], r["_class"] = fl, ex, cls
        used.append(r)

    if not used:
        sys.exit("nothing to classify")

    print(f"{len(used)} deliveries classified, {skipped} skipped "
          f"(threshold {args.threshold:g} deg)\n")

    # ------------------------------------------------- per delivery
    counts = collections.Counter(r["_class"] for r in used)
    print("front-leg technique, per delivery")
    for k in ("flexor", "flexor-extender", "extender", "constant brace"):
        n = counts[k]
        print(f"  {k:<16} {n:4d}  {n / len(used):5.1%}  {bar(n, len(used))}")

    # ------------------------------------------------- per bowler
    byb = collections.defaultdict(list)
    for r in used:
        byb[r["bowler"]].append(r)

    print("\nper bowler  (modal class, and whether the bowler is consistent)")
    modal = {}
    for b in sorted(byb):
        cs = collections.Counter(x["_class"] for x in byb[b])
        top, n = cs.most_common(1)[0]
        modal[b] = top
        share = n / len(byb[b])
        mark = "" if share >= 0.7 else "   <-- varies between deliveries"
        fl = statistics.median(x["_flexion"] for x in byb[b])
        ex = statistics.median(x["_extension"] for x in byb[b])
        print(f"  {b:<14} {top:<16} {n}/{len(byb[b])}  "
              f"flex {fl:5.1f}  ext {ex:5.1f}{mark}")

    nb = len(modal)
    print(f"\nmodal class across the {nb} bowlers with legal deliveries")
    mc = collections.Counter(modal.values())
    for k in ("flexor", "flexor-extender", "extender", "constant brace"):
        print(f"  {k:<16} {mc[k]:3d}  {mc[k] / nb:5.1%}   "
              f"(Portus 42 elite: {PORTUS_42[k]}, {PORTUS_42[k] / 42:.1%})")

    small = [k for k in mc if 0 < mc[k] <= 4]
    if small:
        print("\n  note: " + ", ".join(small) +
              " have four or fewer bowlers, the same condition under which"
              "\n  Portus declined to draw injury conclusions from these groups.")

    # ------------------------------------------------- write
    fields = ["video", "bowler", "legal", "knee_flexion", "knee_extension",
              "front_leg_class", "trunk_class"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in used:
            w.writerow({
                "video": r["video"], "bowler": r["bowler"],
                "legal": r.get("legal", ""),
                "knee_flexion": f"{r['_flexion']:.2f}",
                "knee_extension": f"{r['_extension']:.2f}",
                "front_leg_class": r["_class"],
                "trunk_class": "not determinable from a single side-on view",
            })
    print(f"\nwritten: {args.out}")

    print("""
NOT REPORTED, AND WHY
  Shoulder counter-rotation, hip-shoulder separation, and the side-on /
  semi-open / front-on / mixed trunk classification are all transverse plane
  quantities measured at back foot contact. One side-on camera cannot recover
  them. Counter-rotation is the variable Portus found separating the lumbar
  stress fracture group (41 +- 10 deg) from the uninjured group (19 +- 10 deg),
  so the strongest published technique-injury link is out of reach here.
  A second camera placed behind the bowler would supply it.""")


if __name__ == "__main__":
    main()
