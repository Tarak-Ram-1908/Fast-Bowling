"""
Event labelling for bowling deliveries -- second batch.

Marks five frames per delivery:
    1  runup_start   first frame the bowler is fully in shot and moving
    2  bfc           back foot lands, delivery stride
    3  ffc           front foot lands, delivery stride
    4  release       ball leaves the hand
    5  ft_end        follow-through finished / bowler leaves frame

Writes the same labels.csv as label_deliveries.py, so expand_labels.py and
make_features.py need no changes.


WHAT IS DIFFERENT FROM label_deliveries.py
------------------------------------------
1. LABEL BACKWARDS.  The tool now asks for release first, then ffc, then bfc.
   This is not cosmetic.  BFC sits at 53.8% within 3 frames while FFC sits at
   89.7%, and the reason is that the last run-up stride and the delivery
   stride look almost identical going forwards -- so the label itself is
   inconsistent, and no amount of extra data fixes an inconsistent target.
   Going backwards removes the ambiguity: release is unmistakable, the front
   foot plant before it is unmistakable, and BFC is then simply the previous
   contact by the other foot.  There is nothing to judge.

   The HUD shows the next mark it wants.  Press the number keys in any order
   you like -- the prompt is a nudge, not a constraint.

2. FRAMES ARE JPEG-BUFFERED.  The old tool held every decoded frame in RAM.
   At 960 px that is ~1.5 MB a frame, so an 878-frame clip wanted 1.4 GB.
   These are 4K clips and some run long.  Frames are now kept JPEG-encoded
   (~15 MB a clip) and decoded on demand in about 2 ms, which you will not
   notice.

3. DROPPED FRAMES ARE SURFACED.  The audit found 27 clips with dropped
   frames.  The tool reads real packet timestamps, shows a red DROPS banner
   on affected clips, marks the gaps on the timeline in red, and refuses to
   let a mark sit silently inside one.  A mark adjacent to a gap means the
   true instant may be in the missing frames.

4. SEEDING.  With --seed it pre-fills bfc/ffc/release from model predictions
   so you correct rather than label from scratch.  See below.


USAGE
    pip install opencv-python

    # label the ten new bowlers, skipping anything already in labels.csv
    python label_events.py videos/ --out labels.csv

    # restrict to particular bowlers
    python label_events.py videos/ --only Charan,Jagan,Jaswanth

    # revisit clips you have already done (e.g. to redo bfc on the old 15)
    python label_events.py videos/ --only Ram,Junior1 --redo --marks bfc

    # pre-fill from model predictions, then just fix what is wrong
    python label_events.py videos/ --seed events_pred.csv


KEYS
    d / a      forward / back 1 frame
    w / s      forward / back 10 frames
    e / q      forward / back 60 frames
    SPACE      play / pause at quarter speed
    z          hold-free 3x zoom toggle, follows the mouse
    1 2 3 4 5  set that mark at the current frame
    [ / ]      jump to previous / next mark
    u          undo the mark you set most recently
    c          clear all marks for this clip
    x          flag clip unusable
    n / p      save and go to next / previous clip
    ESC        save and quit
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import cv2

MARKS = ["runup_start", "bfc", "ffc", "release", "ft_end"]
FIELDS = ["video", "bowler", "n_frames", "fps", *MARKS, "usable", "notes"]

# The order the HUD asks for them in. Release is unambiguous, so it anchors
# everything that follows.
ASK_ORDER = ["release", "ffc", "bfc", "runup_start", "ft_end"]

COLORS = {
    "runup_start": (170, 170, 170),
    "bfc":         (200, 120, 60),
    "ffc":         (90, 190, 120),
    "release":     (60, 170, 230),
    "ft_end":      (120, 110, 220),
}

RULE = {
    "release":     "last frame the ball is still in the hand",
    "ffc":         "front foot first touches down, the stride before release",
    "bfc":         "previous contact, other foot -- do not judge, just step back",
    "runup_start": "bowler fully in shot and moving",
    "ft_end":      "follow-through done / leaving frame",
}


# --------------------------------------------------------------- gap detection
def frame_gaps(path):
    """Indices after which frames are missing, from real packet timestamps.

    Returns (set_of_indices, n_missing). Empty on any failure -- a missing
    ffprobe must not stop you labelling.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=90)
    except Exception:
        return set(), 0
    ts = sorted(float(x) for x in r.stdout.replace(",", " ").split()
                if x.replace(".", "", 1).isdigit())
    if len(ts) < 10:
        return set(), 0
    d = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    med = sorted(d)[len(d) // 2]
    if med <= 0:
        return set(), 0
    gaps, missing = set(), 0
    for i, x in enumerate(d):
        if x > med * 1.5:
            gaps.add(i)
            missing += round(x / med) - 1
    return gaps, missing


# --------------------------------------------------------------------- storage
def load_csv(path, key="video"):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get(key):
                    rows[r[key]] = r
    return rows


def save_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, path)


def find_videos(path, only=None):
    if os.path.isfile(path):
        return [os.path.abspath(path)]
    out = []
    for root, _, files in os.walk(path):
        bowler = os.path.basename(root)
        if only and bowler not in only:
            continue
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".mov", ".m4v")):
                out.append(os.path.abspath(os.path.join(root, f)))
    return sorted(out)


def decode_buffered(path, width, quality=85):
    """Decode once, keep frames JPEG-encoded. ~15 MB a clip instead of ~1.4 GB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    buf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > width:
            fr = cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok2, jpg = cv2.imencode(".jpg", fr, enc)
        if ok2:
            buf.append(jpg)
    cap.release()
    return buf, fps


def frame_at(buf, i):
    return cv2.imdecode(buf[i], cv2.IMREAD_COLOR)


# ------------------------------------------------------------------------ draw
def draw(img, i, total, fps, marks, usable, name, idx, nvids,
         gaps, missing, zoom, mouse, seeded):
    h, w = img.shape[:2]

    if zoom:
        mx, my = mouse
        side = max(w // 6, 60)
        x0 = max(0, min(w - 2 * side, mx - side))
        y0 = max(0, min(h - 2 * side, my - side))
        crop = img[y0:y0 + 2 * side, x0:x0 + 2 * side]
        if crop.size:
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

    pad = 104
    c = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    t = i / fps if fps else 0.0
    cv2.putText(c, f"[{idx + 1}/{nvids}] {name}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(c, f"frame {i}/{total - 1}   t={t:6.3f}s   fps={fps:.0f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

    if not usable:
        cv2.putText(c, "UNUSABLE", (w - 160, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
    if missing:
        cv2.putText(c, f"DROPS: {missing} frames", (w - 240, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 255), 2, cv2.LINE_AA)
    if zoom:
        cv2.putText(c, "ZOOM 3x", (w - 110, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    # ---- timeline
    y0 = h + 12
    cv2.rectangle(c, (10, y0), (w - 10, y0 + 12), (60, 60, 60), -1)
    span = max(total - 1, 1)
    for g in gaps:                                   # dropped-frame locations
        gx = 10 + int((w - 20) * g / span)
        cv2.line(c, (gx, y0), (gx, y0 + 12), (60, 60, 220), 1)
    cur = 10 + int((w - 20) * i / span)
    cv2.line(c, (cur, y0 - 5), (cur, y0 + 17), (255, 255, 255), 1)
    for m in MARKS:
        v = marks.get(m)
        if v is None:
            continue
        x = 10 + int((w - 20) * int(v) / span)
        cv2.line(c, (x, y0), (x, y0 + 12), COLORS[m], 2)

    # ---- which mark is wanted next
    nxt = next((m for m in ASK_ORDER if marks.get(m) is None), None)
    if nxt:
        k = MARKS.index(nxt) + 1
        cv2.putText(c, f"NEXT  [{k}] {nxt}: {RULE[nxt]}", (10, h + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[nxt], 1, cv2.LINE_AA)
    else:
        cv2.putText(c, "all five marked -- n for next clip", (10, h + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 230, 120), 1, cv2.LINE_AA)

    parts = []
    for m in MARKS:
        v = marks.get(m)
        tag = "" if v is None else ("*" if m in seeded else "")
        parts.append(f"{m}={v if v is not None else '-'}{tag}")
    done = all(marks.get(m) is not None for m in MARKS)
    cv2.putText(c, "   ".join(parts), (10, h + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (255, 255, 255) if done else (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(c, "d/a 1  w/s 10  e/q 60  SPACE play  z zoom  1-5 mark  [/] marks  "
                   "u undo  c clear  x unusable  n/p  ESC",
                (10, h + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    return c


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="labels.csv")
    ap.add_argument("--seed", help="CSV with video,bfc,ffc,release to pre-fill from")
    ap.add_argument("--only", help="comma-separated bowler folder names")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--redo", action="store_true", help="revisit clips already labelled")
    ap.add_argument("--marks", help="with --redo, clear only these marks (e.g. bfc)")
    ap.add_argument("--no-gaps", action="store_true", help="skip the ffprobe gap check")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    videos = find_videos(args.path, only)
    if not videos:
        sys.exit(f"no videos found under {args.path}"
                 + (f" for {sorted(only)}" if only else ""))

    rows = load_csv(args.out)
    seed = load_csv(args.seed) if args.seed else {}
    if args.seed:
        print(f"seed file: {len(seed)} clips")

    if args.redo:
        todo = videos
    else:
        todo = [v for v in videos if os.path.basename(v) not in rows]
        if not todo:
            print(f"every clip is already in {args.out} -- use --redo to revisit")
            return

    clear_marks = [m.strip() for m in args.marks.split(",")] if args.marks else None
    if clear_marks:
        bad = [m for m in clear_marks if m not in MARKS]
        if bad:
            sys.exit(f"unknown mark(s): {bad}; valid: {MARKS}")

    print(f"{len(todo)} clips to label")
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    mouse = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
    cv2.setMouseCallback("label", on_mouse)

    idx = 0
    while 0 <= idx < len(todo):
        path = todo[idx]
        name = os.path.basename(path)
        bowler = os.path.basename(os.path.dirname(path))

        print(f"loading {name} ...", flush=True)
        buf, fps = decode_buffered(path, args.width)
        if not buf:
            print("  could not decode, skipping")
            idx += 1
            continue
        total = len(buf)
        if fps <= 0:
            fps = 120.0

        gaps, missing = (set(), 0) if args.no_gaps else frame_gaps(path)
        if missing:
            print(f"  {missing} dropped frames in this clip")

        prev = rows.get(name, {})
        marks, seeded = {}, set()
        for m in MARKS:
            v = prev.get(m)
            if clear_marks and m in clear_marks:
                v = None
            marks[m] = int(v) if v not in (None, "", "-") else None

        # seed anything still unset
        s = seed.get(name)
        if s:
            for m in ("bfc", "ffc", "release"):
                if marks[m] is None and s.get(m) not in (None, "", "-"):
                    try:
                        val = int(float(s[m]))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= val < total:
                        marks[m] = val
                        seeded.add(m)

        usable = prev.get("usable", "1") != "0"
        history = []
        i = 0
        playing = False
        zoom = False
        action = None

        while True:
            img = frame_at(buf, i)
            cv2.imshow("label", draw(img, i, total, fps, marks, usable, name,
                                     idx, len(todo), gaps, missing, zoom,
                                     tuple(mouse), seeded))
            k = cv2.waitKey(8 if playing else 0) & 0xFF

            if playing and k == 255:
                i += 1
                if i >= total - 1:
                    i, playing = total - 1, False
                time.sleep(1 / max(fps / 4, 1))
                continue

            if k == ord("d"):
                i = min(i + 1, total - 1); playing = False
            elif k == ord("a"):
                i = max(i - 1, 0); playing = False
            elif k == ord("w"):
                i = min(i + 10, total - 1); playing = False
            elif k == ord("s"):
                i = max(i - 10, 0); playing = False
            elif k == ord("e"):
                i = min(i + 60, total - 1); playing = False
            elif k == ord("q"):
                i = max(i - 60, 0); playing = False
            elif k == ord(" "):
                playing = not playing
            elif k == ord("z"):
                zoom = not zoom
            elif k == ord("]"):
                nxt = sorted(v for v in marks.values() if v is not None and v > i)
                if nxt:
                    i = nxt[0]
            elif k == ord("["):
                prv = sorted(v for v in marks.values() if v is not None and v < i)
                if prv:
                    i = prv[-1]
            elif ord("1") <= k <= ord("5"):
                m = MARKS[k - ord("1")]
                history.append((m, marks[m]))
                marks[m] = i
                seeded.discard(m)
                playing = False
            elif k == ord("u") and history:
                m, old = history.pop()
                if m == "__all__":
                    marks = old
                else:
                    marks[m] = old
            elif k == ord("c"):
                history.append(("__all__", dict(marks)))
                marks = {m: None for m in MARKS}
                seeded.clear()
            elif k == ord("x"):
                usable = not usable
            elif k == ord("n"):
                action = "next"; break
            elif k == ord("p"):
                action = "prev"; break
            elif k == 27:
                action = "quit"; break

        # ------------------------------------------------------------ checks
        if usable:
            miss = [m for m in MARKS if marks[m] is None]
            if miss and action in ("next", "quit"):
                print(f"  WARNING {name}: no mark for {', '.join(miss)}")

            order = [(m, marks[m]) for m in MARKS if marks[m] is not None]
            vals = [v for _, v in order]
            if vals != sorted(vals):
                print(f"  WARNING {name}: marks out of order -> "
                      + ", ".join(f"{m}={v}" for m, v in order))

            b, f, r = marks["bfc"], marks["ffc"], marks["release"]
            if b is not None and f is not None and b >= f:
                print(f"  WARNING {name}: bfc({b}) must come before ffc({f})")
            if f is not None and r is not None and f >= r:
                print(f"  WARNING {name}: ffc({f}) must come before release({r})")

            if seeded:
                print(f"  NOTE {name}: accepted model seed for {', '.join(sorted(seeded))}")

            for m in ("bfc", "ffc", "release"):
                v = marks[m]
                if v is None:
                    continue
                near = [g for g in gaps if abs(g - v) <= 2]
                if near:
                    print(f"  WARNING {name}: {m}={v} sits next to a dropped-frame "
                          f"gap -- the true instant may be in the missing frames")

        rows[name] = {
            "video": name,
            "bowler": bowler,
            "n_frames": total,
            "fps": f"{fps:.3f}",
            **{m: ("" if marks[m] is None else marks[m]) for m in MARKS},
            "usable": "1" if usable else "0",
            "notes": prev.get("notes", ""),
        }
        save_csv(args.out, rows)

        if action == "quit":
            break
        idx += 1 if action != "prev" else -1
        idx = max(idx, 0)

    cv2.destroyAllWindows()
    done = sum(1 for r in rows.values()
               if r.get("usable") != "0" and all(r.get(m) not in (None, "") for m in MARKS))
    print(f"saved {args.out}  --  {len(rows)} clips, {done} fully marked")


if __name__ == "__main__":
    main()
