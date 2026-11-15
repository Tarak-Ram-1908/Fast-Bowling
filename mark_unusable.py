"""
Mark a clip unusable in labels.csv, with a reason.

    python mark_unusable.py VID20260920090138.mp4 "pose tracked a background person, not the bowler"

Blanks nothing and deletes nothing: it sets usable=0 and writes the reason into
the notes column, so the exclusion is visible in the data rather than buried in
a script's behaviour. expand_labels.py then skips the clip.
"""

import csv
import os
import sys

FIELDS = ["video", "bowler", "n_frames", "fps", "runup_start", "bfc", "ffc",
          "release", "ft_end", "usable", "notes"]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    video, reason = sys.argv[1], sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else "labels.csv"

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    hit = None
    for r in rows:
        if r["video"] == video:
            hit = r
            break
    if hit is None:
        print(f"{video} is not in {path}")
        print("  (check the exact filename, including the .mp4)")
        sys.exit(1)

    was = hit["usable"]
    hit["usable"] = "0"
    hit["notes"] = reason

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

    print(f"{video}  ({hit['bowler']})")
    print(f"  usable: {was} -> 0")
    print(f"  notes:  {reason}")

    n = sum(1 for r in rows if r["usable"] == "1")
    print(f"\n{n} usable clips remain of {len(rows)}")
    print("rerun expand_labels.py if you want frame_labels.csv updated too")


if __name__ == "__main__":
    main()
