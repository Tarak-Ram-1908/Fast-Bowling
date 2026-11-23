"""
Mark each delivery legal or illegal, one clip at a time.

    pip install opencv-python
    python label_legality.py . --out legality.csv
    python label_legality.py . --out legality.csv --only Abhiram
    python label_legality.py . --out legality.csv --redo

This runs BEFORE the biomechanical analysis. A delivery that is not a legal
bowling action does not belong in a comparison against Worthington's reference
values, because those values describe twenty screened legal fast bowlers. So
legality is a gate: label it here, then let every later script filter on it.

Navigation is identical to label_deliveries.py, so your fingers already know it.

    d / a        forward / back 1 frame
    w / s        forward / back 10 frames
    e / q        forward / back 60 frames
    SPACE        play the delivery at quarter speed (press any key to stop)
    z            toggle 2x zoom; the zoomed view follows your mouse
    1            LEGAL
    2            ILLEGAL (throw)
    3            UNCLEAR -- cannot tell from this footage
    u            clear the label on this clip
    n / p        next / previous clip
    ESC          save and quit

WHAT YOU ARE JUDGING. The ICC rule is that the elbow may not extend by more
than 15 degrees between the moment the upper arm reaches horizontal and the
moment the ball is released. Fifteen degrees is not reliably visible by eye at
120 fps from side-on -- that is why real adjudication happens in a laboratory.
So judge what is actually visible: a clear bend-then-straighten of the elbow
through the delivery swing. Anything you are not sure about is a 3, not a
guess. The proportion you mark unclear is itself a result worth reporting.

Press SPACE before deciding. A throw is a motion, and a single frame will not
show it to you.
"""

import argparse
import csv
import os
import sys

import cv2

FIELDS = ["video", "bowler", "legal", "label", "notes"]
NAMES = {1: "legal", 0: "illegal", -1: "unclear"}
COLORS = {1: (120, 230, 120), 0: (90, 90, 255), -1: (90, 200, 255)}


# ------------------------------------------------------------------ data

def load_events(path):
    """bowler and usable for every clip; ffc / release where they parse.

    The usable flag must be read INDEPENDENTLY of the event frames. A clip
    rejected during labelling often has blank event columns, and parsing the
    whole row inside one try/except meant such a row vanished from this dict
    entirely -- taking its usable=0 with it, so the clip was offered for
    judging anyway. Bowler and usable always land; the window is optional.
    """
    ev = {}
    if not os.path.exists(path):
        return ev
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not r.get("video"):
                continue
            e = {"bowler": r.get("bowler", ""), "usable": r.get("usable", "1")}
            try:
                e["ffc"] = int(r["ffc"])
                e["release"] = int(r["release"])
            except (ValueError, KeyError, TypeError):
                pass                      # no window; open mid-clip instead
            ev[r["video"]] = e
    return ev


def find_clips(root):
    """Every delivery under root, skipping _Calibration and friends.

    dirnames is pruned in place rather than testing the folder name on the way
    past, so a subfolder INSIDE an underscore folder is skipped too. Testing
    only the current folder let calibration clips in through a nested
    directory, which is how this came to offer more clips than labels.csv has
    deliveries."""
    out = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(("_", "."))]
        if os.path.basename(dirpath).startswith(("_", ".")):
            continue
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".mov", ".m4v")):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def load_out(path):
    """Existing labels, plus any columns someone added by hand.

    You are free to keep per-bowler attributes (height_cm, handedness ...) in
    this same file. Everything this tool did not write is carried through
    untouched on every save, so typing a height into the spreadsheet and then
    relabelling a clip will not wipe it.
    """
    rows, extra = {}, []
    if os.path.exists(path):
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            extra = [c for c in (rd.fieldnames or []) if c not in FIELDS]
            for r in rd:
                rows[r["video"]] = r
    return rows, extra


def save_out(path, rows, extra):
    cols = FIELDS + [c for c in extra if c not in FIELDS]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in cols})
    os.replace(tmp, path)


def check_consistency(rows, extra):
    """A per-bowler attribute kept on per-delivery rows can disagree with
    itself. Say so rather than letting one typo change a bowler's height on
    four of his sixteen deliveries."""
    bad = []
    for col in extra:
        seen = {}
        for r in rows.values():
            v = (r.get(col) or "").strip()
            if v:
                seen.setdefault(r["bowler"], set()).add(v)
        for b, vals in seen.items():
            if len(vals) > 1:
                bad.append((col, b, sorted(vals)))
    return bad


# ------------------------------------------------------------------ decode

def screen_size(default=(1600, 900)):
    """The display, so the window never ends up taller than the screen."""
    try:
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        wh = (r.winfo_screenwidth(), r.winfo_screenheight())
        r.destroy()
        return wh
    except Exception:
        return default


def decode_window(path, lo, hi, max_width, max_height, quality=92):
    """Frames [lo, hi] into memory, downscaled and JPEG-compressed.

    Holding a whole delivery as raw BGR is what used to force a small window:
    190 frames at 1600 px is about 650 MB uncompressed. Re-encoded to JPEG it
    is nearer 45 MB, and decoding one frame for display costs a few
    milliseconds, which is invisible next to a 33 ms frame interval. So the
    picture can be as large as the screen allows.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0
    buf, i = [], 0
    enc = [cv2.IMWRITE_JPEG_QUALITY, quality]
    while i <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= lo:
            h, w = fr.shape[:2]
            sc = min(max_width / w, max_height / h, 1.0)
            if sc < 1.0:
                fr = cv2.resize(fr, (int(w * sc), int(h * sc)),
                                interpolation=cv2.INTER_AREA)
            ok2, jpg = cv2.imencode(".jpg", fr, enc)
            if ok2:
                buf.append(jpg)
        i += 1
    cap.release()
    return buf, lo


def frame_at(buf, i):
    return cv2.imdecode(buf[i], cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ drawing

def render(frame, zoom, cursor, head, sub, state):
    c = frame.copy()
    H, W = c.shape[:2]

    if zoom:
        mx, my = cursor
        bw, bh = W // 3, H // 3
        x0 = max(0, min(W - bw, mx - bw // 2))
        y0 = max(0, min(H - bh, my - bh // 2))
        crop = c[y0:y0 + bh, x0:x0 + bw]
        if crop.size:
            c = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.putText(c, "ZOOM 3x", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # The bar is drawn INSIDE the picture. Adding it underneath made the
    # window taller than the screen, which simply hid it.
    H, W = c.shape[:2]
    bar = 96
    y = H - bar
    band = c[y:H, 0:W]
    cv2.rectangle(band, (0, 0), (W, bar), (14, 14, 14), -1)
    cv2.addWeighted(band, 0.82, c[y:H, 0:W], 0.18, 0, c[y:H, 0:W])
    cv2.line(c, (0, y), (W, y), (70, 70, 70), 1)

    cv2.putText(c, head, (14, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(c, sub, (14, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                (150, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(c, "SPACE play   a/d 1   s/w 10   q/e 60   z zoom   "
                   "1 legal   2 illegal   3 unclear   u clear   n/p   ESC",
                (14, y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (165, 165, 165), 1, cv2.LINE_AA)
    if state is not None:
        txt = NAMES[state].upper()
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(c, txt, (W - tw - 24, y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[state], 2,
                    cv2.LINE_AA)
    return c


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_root", help="folder holding the bowler folders")
    ap.add_argument("--out", default="legality.csv")
    ap.add_argument("--labels", default="labels.csv",
                    help="used only to open each clip at the delivery")
    ap.add_argument("--only", default=None, help="one bowler's clips")
    ap.add_argument("--redo", action="store_true", help="revisit labelled clips")
    ap.add_argument("--attrs", default="",
                    help="comma-separated extra columns to create in the "
                         "output if absent, e.g. height_cm,handedness. They "
                         "are yours to fill in; this tool never writes them")
    ap.add_argument("--max-width", type=int, default=0,
                    help="displayed width in pixels; 0 fits the screen")
    ap.add_argument("--max-height", type=int, default=0,
                    help="displayed height in pixels; 0 fits the screen")
    ap.add_argument("--pre", type=int, default=120,
                    help="frames before front foot contact to decode")
    ap.add_argument("--post", type=int, default=60,
                    help="frames after release to decode")
    ap.add_argument("--fps", type=float, default=120.0)
    args = ap.parse_args()

    sw, sh = screen_size()
    max_w = args.max_width or max(640, sw - 60)
    max_h = args.max_height or max(480, sh - 140)   # title bar + taskbar
    print(f"screen {sw}x{sh}, drawing at up to {max_w}x{max_h}")

    events = load_events(args.labels)
    rows, extra = load_out(args.out)
    for c in [x.strip() for x in args.attrs.split(",") if x.strip()]:
        if c not in extra and c not in FIELDS:
            extra.append(c)

    # Drop anything judged that labels.csv says is unusable. Earlier versions
    # let a few through; this clears them out rather than leaving you to find
    # them by hand.
    stale = [v for v in rows if events.get(v, {}).get("usable") == "0"]
    if stale:
        print(f"removing {len(stale)} judged clip(s) that labels.csv marks "
              f"unusable:")
        for v in sorted(stale):
            print(f"  {events[v]['bowler']:12s} {v}  "
                  f"(was {rows[v].get('label', '?')})")
        for v in stale:
            del rows[v]
        save_out(args.out, rows, extra)
        print()
    clips = find_clips(args.video_root)
    if not clips:
        sys.exit(f"no videos found under {args.video_root}")

    todo = []
    for p in clips:
        name = os.path.basename(p)
        bowler = events.get(name, {}).get("bowler") \
            or os.path.basename(os.path.dirname(p))
        if args.only and args.only.lower() not in bowler.lower():
            continue
        if events.get(name, {}).get("usable") == "0":
            continue
        if name in rows and not args.redo:
            continue
        todo.append((p, name, bowler))

    if not todo:
        print(f"nothing to do -- every clip is already in {args.out} "
              f"(use --redo to revisit)")
        return

    print(f"{len(todo)} clips to judge\n")
    print("Press SPACE first on every clip. You cannot see a throw in a "
          "still frame.\n")
    cv2.namedWindow("legality", cv2.WINDOW_AUTOSIZE)
    cursor = [0, 0]

    def on_mouse(ev, x, y, flags, _):
        cursor[0], cursor[1] = x, y

    cv2.setMouseCallback("legality", on_mouse)

    i = 0
    while 0 <= i < len(todo):
        path, name, bowler = todo[i]
        ev = events.get(name)
        if ev and "ffc" in ev:
            lo = max(0, ev["ffc"] - args.pre)
            hi = ev["release"] + args.post
            opened_at = ev["ffc"] - lo
        else:
            lo, hi, opened_at = 0, 10 ** 6, 0

        frames, first = decode_window(path, lo, hi, max_w, max_h)
        if not frames:
            print(f"  could not read {name}")
            i += 1
            continue
        if not ev:
            opened_at = len(frames) // 2

        f = max(0, min(len(frames) - 1, opened_at))
        zoom = False
        prev = rows.get(name)
        state = int(prev["legal"]) if prev and prev["legal"] != "" else None
        action = None

        while True:
            head = (f"[{i+1}/{len(todo)}] {bowler}   {name}   "
                    f"frame {first + f}")
            sub = ("judge the elbow through the swing" if state is None
                   else f"marked {NAMES[state]} -- n for next")
            cv2.imshow("legality", render(frame_at(frames, f), zoom,
                                          tuple(cursor), head, sub, state))
            k = cv2.waitKey(20) & 0xFF

            if k == ord("d"):
                f = min(len(frames) - 1, f + 1)
            elif k == ord("a"):
                f = max(0, f - 1)
            elif k == ord("w"):
                f = min(len(frames) - 1, f + 10)
            elif k == ord("s"):
                f = max(0, f - 10)
            elif k == ord("e"):
                f = min(len(frames) - 1, f + 60)
            elif k == ord("q"):
                f = max(0, f - 60)
            elif k == ord("z"):
                zoom = not zoom
            elif k == 32:                      # SPACE: play it
                delay = max(1, int(4000 / args.fps))
                for j in range(len(frames)):
                    cv2.imshow("legality",
                               render(frame_at(frames, j), zoom, tuple(cursor),
                                      head, "playing...", state))
                    if (cv2.waitKey(delay) & 0xFF) != 255:
                        f = j
                        break
                else:
                    f = len(frames) - 1
            elif k in (ord("1"), ord("2"), ord("3")):
                state = {ord("1"): 1, ord("2"): 0, ord("3"): -1}[k]
                row = dict(rows.get(name) or prev or {})
                row.update({"video": name, "bowler": bowler,
                            "legal": state, "label": NAMES[state]})
                row.setdefault("notes", "")
                rows[name] = row
                save_out(args.out, rows, extra)
                print(f"  {bowler:12s} {name}  {NAMES[state]}")
            elif k == ord("u"):
                state = None
                if name in rows:          # clear the verdict, keep the row
                    rows[name]["legal"] = ""
                    rows[name]["label"] = ""
                    save_out(args.out, rows, extra)
            elif k == ord("n"):
                action = "next"; break
            elif k == ord("p"):
                action = "prev"; break
            elif k == 27:
                action = "quit"; break

        if action == "quit":
            break
        i += 1 if action != "prev" else -1
        i = max(i, 0)

    cv2.destroyAllWindows()
    save_out(args.out, rows, extra)

    # -------------------------------------------------------- what you have
    print(f"\nsaved {args.out}")
    by_bowler = {}
    for r in rows.values():
        if not r.get("label"):
            continue
        d = by_bowler.setdefault(r["bowler"], {"legal": 0, "illegal": 0,
                                               "unclear": 0})
        d[r["label"]] += 1
    if by_bowler:
        print(f"\n  {'bowler':14s}{'legal':>7s}{'illegal':>9s}"
              f"{'unclear':>9s}{'% legal':>9s}")
        tot = {"legal": 0, "illegal": 0, "unclear": 0}
        for b in sorted(by_bowler):
            d = by_bowler[b]
            n = sum(d.values())
            for k in tot:
                tot[k] += d[k]
            pct = d["legal"] / n if n else 0
            print(f"  {b:14s}{d['legal']:7d}{d['illegal']:9d}"
                  f"{d['unclear']:9d}{pct:8.0%}")
        n = sum(tot.values())
        print(f"  {'ALL':14s}{tot['legal']:7d}{tot['illegal']:9d}"
              f"{tot['unclear']:9d}{tot['legal']/n:8.0%}")
        bad = check_consistency(rows, extra)
        if bad:
            print("\n  WARNING, a per-bowler column disagrees with itself:")
            for col, b, vals in bad:
                print(f"    {b}: {col} = {', '.join(vals)}")
            print("  fill those columns DOWN so every row of a bowler matches.")
        if tot["unclear"]:
            print(f"\n  {tot['unclear']/n:.0%} unclear -- report this figure. "
                  f"It is the honest limit of\n  judging a 15-degree rule by "
                  f"eye from side-on video at {args.fps:.0f} fps.")


if __name__ == "__main__":
    main()
