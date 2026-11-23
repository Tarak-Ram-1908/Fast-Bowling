"""
Turn five boundary marks per clip into a per-frame label table.

Reads labels.csv (from label_deliveries.py) and writes frame_labels.csv with one
row per frame you intend to keep.

Phase definitions follow Worthington (2010), Figure 1.1:
    run_up          runup_start .. bfc-1      (includes the jump or bound,
                                               and every run-up foot landing)
    bfc_phase       bfc        .. ffc-1
    ffc_phase       ffc        .. release
    follow_through  release+1  .. ft_end
    none            everything outside runup_start .. ft_end

Sampling: bfc_phase and ffc_phase are rare, so every frame is kept. run_up and
follow_through are long and repetitive, so they are subsampled -- except near a
boundary, where every frame is kept because that is where the decision is hard.

Usage
    python expand_labels.py labels.csv --out frame_labels.csv
    python expand_labels.py labels.csv --keep-all        # no subsampling
"""

import argparse
import csv
from collections import Counter

MARKS = ["runup_start", "bfc", "ffc", "release", "ft_end"]


def phase_for(f, m):
    if f < m["runup_start"] or f > m["ft_end"]:
        return "none"
    if f < m["bfc"]:
        return "run_up"
    if f < m["ffc"]:
        return "bfc_phase"
    if f <= m["release"]:
        return "ffc_phase"
    return "follow_through"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", default="frame_labels.csv")
    ap.add_argument("--runup-stride", type=int, default=30,
                    help="keep 1 run-up frame in N away from boundaries")
    ap.add_argument("--ft-stride", type=int, default=12,
                    help="keep 1 follow-through frame in N away from boundaries")
    ap.add_argument("--none-stride", type=int, default=40)
    ap.add_argument("--guard", type=int, default=12,
                    help="keep every frame within this many of a boundary")
    ap.add_argument("--keep-all", action="store_true")
    args = ap.parse_args()

    out_rows = []
    counts = Counter()
    skipped = []

    with open(args.labels, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("usable", "1") == "0":
                skipped.append((r["video"], "flagged unusable"))
                continue
            if any(r.get(m) in (None, "") for m in MARKS):
                skipped.append((r["video"], "incomplete marks"))
                continue

            m = {k: int(r[k]) for k in MARKS}
            if [m[k] for k in MARKS] != sorted(m[k] for k in MARKS):
                skipped.append((r["video"], "marks out of order"))
                continue

            total = int(r["n_frames"])
            bounds = set(m.values())

            for f in range(total):
                ph = phase_for(f, m)
                near = any(abs(f - b) <= args.guard for b in bounds)
                if not args.keep_all and not near:
                    if ph == "run_up" and f % args.runup_stride:
                        continue
                    if ph == "follow_through" and f % args.ft_stride:
                        continue
                    if ph == "none" and f % args.none_stride:
                        continue
                out_rows.append({
                    "video": r["video"],
                    "bowler": r["bowler"],
                    "frame": f,
                    "phase": ph,
                    "is_bfc_event": int(f == m["bfc"]),
                    "is_ffc_event": int(f == m["ffc"]),
                    "is_release_event": int(f == m["release"]),
                })
                counts[ph] += 1

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    bowlers = {r["bowler"] for r in out_rows}
    clips = {r["video"] for r in out_rows}
    print(f"{len(out_rows)} frames from {len(clips)} clips, {len(bowlers)} bowlers -> {args.out}\n")
    for ph in ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]:
        print(f"  {ph:16s} {counts[ph]:6d}")
    if skipped:
        print("\nskipped:")
        for v, why in skipped:
            print(f"  {v}: {why}")


if __name__ == "__main__":
    main()
