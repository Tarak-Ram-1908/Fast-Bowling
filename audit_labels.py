"""
Find clips that were shot but never labelled, and rows that point at nothing.

    python audit_labels.py videos --labels labels.csv

Walks the video folders, compares against labels.csv, and reports both
directions of mismatch. Nothing is written -- it only tells you what to do.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

VIDEO_EXT = (".mp4", ".mov", ".m4v")


def walk(root):
    """{filename: bowler folder} for every clip under root."""
    found = {}
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(("_", "."))]
        if os.path.basename(dirpath).startswith(("_", ".")):
            continue
        for f in sorted(files):
            if f.lower().endswith(VIDEO_EXT):
                found[f] = os.path.basename(dirpath)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_root")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--show", type=int, default=20,
                    help="how many missing-file rows to list")
    args = ap.parse_args()

    disk = walk(args.video_root)
    rows = {}
    with open(args.labels, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["video"]] = r

    print(f"{len(disk)} clips on disk, {len(rows)} rows in {args.labels}\n")

    missing = {v: b for v, b in disk.items() if v not in rows}
    ghost = {v: r["bowler"] for v, r in rows.items() if v not in disk}

    if missing:
        print(f"NOT LABELLED -- {len(missing)} clip(s) on disk with no row:\n")
        by = defaultdict(list)
        for v, b in missing.items():
            by[b].append(v)
        for b in sorted(by):
            print(f"  {b}")
            for v in sorted(by[b]):
                print(f"      {v}")
        print("\n  label them with, one bowler at a time:")
        for b in sorted(by):
            print(f"      python label_deliveries.py \"{b}\" --out {args.labels}")
        print("  (it skips clips that already have a row, so you will only")
        print("   be shown the ones listed above)")
    else:
        print("every clip on disk has a row -- nothing missing")

    if ghost:
        print(f"\nNO SUCH FILE -- {len(ghost)} row(s) point at a clip that is "
              f"not in {args.video_root}:")
        for v in sorted(ghost)[:args.show]:
            print(f"  {ghost[v]:12s} {v}")
        if len(ghost) > args.show:
            print(f"  ... and {len(ghost)-args.show} more")
        print("  (renamed, moved, deleted -- or you pointed this script at "
              "the wrong folder)")

    print("\nper bowler\n")
    print(f"  {'bowler':12s}{'on disk':>9s}{'labelled':>10s}"
          f"{'usable':>8s}{'to do':>7s}")
    bowlers = set(disk.values()) | {r["bowler"] for r in rows.values()}
    d_count = Counter(disk.values())
    l_count = Counter(r["bowler"] for r in rows.values())
    u_count = Counter(r["bowler"] for r in rows.values() if r["usable"] == "1")
    m_count = Counter(missing.values())
    for b in sorted(bowlers):
        flag = "  <--" if m_count[b] else ""
        print(f"  {b:12s}{d_count[b]:9d}{l_count[b]:10d}"
              f"{u_count[b]:8d}{m_count[b]:7d}{flag}")
    print(f"  {'TOTAL':12s}{len(disk):9d}{len(rows):10d}"
          f"{sum(u_count.values()):8d}{len(missing):7d}")

    if missing:
        print("\nafter labelling, rerun the pipeline from expand_labels.py "
              "so the new\nclips reach features.csv and predictions.csv.")


if __name__ == "__main__":
    main()
