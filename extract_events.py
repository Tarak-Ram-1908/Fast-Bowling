"""
Turn per-frame phase predictions into event instants, and measure the error.

    python extract_events.py predictions.csv --out events.csv

This is the step the project is actually judged on. A per-frame accuracy of 78%
sounds mediocre, but the measurements do not need every frame -- they need three
single frames per delivery:

    back foot contact   the first frame of the bfc_phase block
    front foot contact  the first frame of the ffc_phase block
    ball release        the last frame of the ffc_phase block

So a block can be 15% wrong in its middle and still start within two frames of
the truth. The question is how far off the EDGES are, in frames and in
milliseconds, and that is what this script reports.

How it works, per clip:

1. Median-filters the forest's class probabilities over a window of frames.
   A single frame flipping to ffc_phase in the middle of bfc_phase is noise;
   the filter removes it without moving the block edges, which a mode filter
   on the hard labels would not do as cleanly.

2. Takes argmax to get a smoothed label per frame.

3. Finds contiguous blocks, then picks the bfc block and the ffc block that
   maximise combined confidence SUBJECT TO the bfc block starting first. That
   constraint is free information -- back foot contact always precedes front
   foot contact, in every legal and illegal action -- and it rescues clips where
   a fragment of bfc_phase appears after release.

4. Reads the event frames off the block edges and compares with your marks.

Note on gaps: frame_labels.csv subsamples run_up, follow_through and none, so
frame numbers are not consecutive everywhere. Rows more than --max-gap frames
apart are treated as separate segments, so the filter never smooths across a
hole. Inside both contact phases and within the guard band of every boundary
every frame is present, which is exactly where precision is needed.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

PHASES = ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]
I_BFC, I_FFC = PHASES.index("bfc_phase"), PHASES.index("ffc_phase")


def prob_columns(header, model):
    """The probability columns for one model, or the legacy p_* if that is all
    predictions.csv has."""
    named = [f"p_{model}_{p}" for p in PHASES]
    if all(c in header for c in named):
        return named
    legacy = [f"p_{p}" for p in PHASES]
    if all(c in header for c in legacy):
        if model != "forest":
            raise SystemExit(
                f"predictions.csv has no per-model probabilities, so --model "
                f"{model} is not available.\nRerun train_classifier.py to "
                f"write them, then try again.")
        return legacy
    raise SystemExit("predictions.csv has no probability columns at all -- "
                     "rerun train_classifier.py")


# ------------------------------------------------------------------ smoothing

def median_filter(a, k):
    """Centred running median along axis 0, edges handled by clamping."""
    if k <= 1 or len(a) < 3:
        return a.copy()
    k = min(k if k % 2 else k + 1, len(a) if len(a) % 2 else len(a) - 1)
    if k < 3:
        return a.copy()
    h = k // 2
    idx = np.clip(np.arange(len(a))[:, None] + np.arange(-h, h + 1)[None, :],
                  0, len(a) - 1)
    return np.median(a[idx], axis=1)


def segments(frames, max_gap):
    """Split row indices into runs whose frame numbers are near-consecutive."""
    out, start = [], 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > max_gap:
            out.append((start, i))
            start = i
    out.append((start, len(frames)))
    return out


def blocks_of(labels):
    """[(label, start_index, end_index_inclusive), ...] for contiguous runs."""
    out = []
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((labels[s], s, i - 1))
            s = i
    return out


# ------------------------------------------------------------------ one clip

def pick_blocks(labels, prob):
    """The bfc and ffc blocks that maximise confidence with bfc starting first.

    Returns (bfc_block, ffc_block, note) where a block is (start, end) in row
    indices, or None if that class never appears.
    """
    bs = [(s, e) for lab, s, e in blocks_of(labels) if lab == "bfc_phase"]
    fs = [(s, e) for lab, s, e in blocks_of(labels) if lab == "ffc_phase"]

    def score(b, j):
        return float(prob[b[0]:b[1] + 1, j].sum())

    if not bs or not fs:
        bb = max(bs, key=lambda b: score(b, I_BFC)) if bs else None
        ff = max(fs, key=lambda b: score(b, I_FFC)) if fs else None
        missing = "no bfc block" if not bs else "no ffc block"
        return bb, ff, missing

    best, note = None, ""
    for b in bs:
        for f in fs:
            if b[0] < f[0]:
                v = score(b, I_BFC) + score(f, I_FFC)
                if best is None or v > best[0]:
                    best = (v, b, f)
    if best is None:
        # every ffc fragment precedes every bfc fragment: take the strongest of
        # each anyway and flag it, rather than silently inventing an order
        bb = max(bs, key=lambda b: score(b, I_BFC))
        ff = max(fs, key=lambda b: score(b, I_FFC))
        return bb, ff, "ORDER VIOLATED (ffc before bfc)"
    if len(bs) > 1 or len(fs) > 1:
        note = f"fragmented ({len(bs)} bfc, {len(fs)} ffc blocks)"
    return best[1], best[2], note


def process_clip(rows, smooth, max_gap, pcols):
    frames = np.array([int(r["frame"]) for r in rows])
    order = np.argsort(frames)
    frames = frames[order]
    rows = [rows[i] for i in order]
    prob = np.array([[float(r[c]) for c in pcols] for r in rows])

    sm = np.empty_like(prob)
    for s, e in segments(frames, max_gap):
        sm[s:e] = median_filter(prob[s:e], smooth)
    labels = [PHASES[j] for j in sm.argmax(axis=1)]

    bb, ff, note = pick_blocks(labels, sm)
    pred = {"bfc": None, "ffc": None, "release": None}
    if bb is not None:
        pred["bfc"] = int(frames[bb[0]])
    if ff is not None:
        pred["ffc"] = int(frames[ff[0]])
        pred["release"] = int(frames[ff[1]])

    truth = {"bfc": None, "ffc": None, "release": None}
    for r in rows:
        if r["is_bfc_event"] == "1":
            truth["bfc"] = int(r["frame"])
        if r["is_ffc_event"] == "1":
            truth["ffc"] = int(r["frame"])
        if r["is_release_event"] == "1":
            truth["release"] = int(r["frame"])

    return pred, truth, note, rows[0]["bowler"]


# ------------------------------------------------------------------ reporting

def report(errs, name, fps, tol):
    e = np.array([v for v in errs if v is not None])
    print(f"\n  {name}")
    if not len(e):
        print("    no clips")
        return
    a = np.abs(e)
    ms = 1000.0 / fps
    print(f"    n            {len(e)}")
    print(f"    mean |error| {a.mean():5.2f} frames   ({a.mean()*ms:5.1f} ms)")
    print(f"    median       {np.median(a):5.1f} frames   ({np.median(a)*ms:5.1f} ms)")
    print(f"    bias         {e.mean():+5.2f} frames   "
          f"({'late' if e.mean() > 0 else 'early'} on average)")
    for t in (1, 2, tol, 5, 10):
        print(f"    within {t:2d}     {(a <= t).mean():5.1%}")
    print(f"    worst        {a.max():.0f} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--out", default="events.csv")
    ap.add_argument("--fallback", default="auto",
                    help="which model to retry a clip with when the primary "
                         "finds no block: 'auto' picks another model present "
                         "in the file, 'none' disables it, or name one")
    ap.add_argument("--model", default="forest",
                    help="whose probabilities to extract events from -- any "
                         "model whose p_<model>_<phase> columns are in the "
                         "file (compare_models.py writes several). The "
                         "forest is more accurate per frame; the logistic "
                         "regression finds more ffc_phase frames, which is "
                         "what decides whether a block exists to read at all")
    ap.add_argument("--smooth", type=int, default=9,
                    help="median filter width in frames (odd)")
    ap.add_argument("--max-gap", type=int, default=3,
                    help="frame gap above which rows are a separate segment")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--tol", type=int, default=3,
                    help="the accuracy target, in frames")
    ap.add_argument("--show", type=int, default=12,
                    help="how many worst clips to list")
    args = ap.parse_args()

    by_video = defaultdict(list)
    with open(args.predictions, newline="") as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames or []
        pcols = prob_columns(header, args.model)
        for r in rd:
            by_video[r["video"]].append(r)

    # The two models fail differently. The forest is the more accurate of the
    # two when it finds a block, and simply finds nothing on actions unlike
    # anything in its training folds. The logistic regression almost always
    # finds something, a little less precisely. So: forest first, logreg only
    # for the clips the forest gave up on. Best timing where timing is
    # available, coverage everywhere else.
    present = sorted({c.split("_")[1] for c in header
                      if c.startswith("p_") and c.count("_") >= 2
                      and c.split("_", 2)[2] in PHASES})
    alts = [m for m in present if m != args.model]
    # prefer logreg as the rescuer: it over-predicts ffc_phase, so it finds a
    # block where a sharper model finds none. That is the whole point of the
    # fallback -- coverage, not timing.
    other = ("logreg" if "logreg" in alts
             else (alts[0] if alts else ("logreg" if args.model == "forest"
                                         else "forest")))
    fb_cols = None
    if args.fallback != "none":
        want = other if args.fallback == "auto" else args.fallback
        if want != args.model:
            try:
                fb_cols = prob_columns(header, want)
                other = want
            except SystemExit:
                fb_cols = None

    print(f"{len(by_video)} clips in {args.predictions}")
    print(f"model {args.model}"
          + (f", falling back to {other} when no block is found" if fb_cols
             else ", no fallback")
          + f", median filter {args.smooth} frames, "
            f"bfc-before-ffc constraint on, {args.fps:.0f} fps\n")

    out, errs = [], {k: [] for k in ("bfc", "ffc", "release")}
    per_bowler = defaultdict(lambda: defaultdict(list))
    notes = defaultdict(int)
    failed = []

    n_fallback = 0
    for vid in sorted(by_video):
        pred, truth, note, bowler = process_clip(by_video[vid], args.smooth,
                                                 args.max_gap, pcols)
        used = args.model
        if fb_cols and any(pred[k] is None for k in pred):
            alt, _, alt_note, _ = process_clip(by_video[vid], args.smooth,
                                               args.max_gap, fb_cols)
            # only take it if it is actually more complete
            if sum(v is not None for v in alt.values()) > \
               sum(v is not None for v in pred.values()):
                pred, used = alt, other
                note = (alt_note + "; " if alt_note else "") + f"via {other}"
                n_fallback += 1
        row = {"video": vid, "bowler": bowler, "model": used, "note": note}
        for k in ("bfc", "ffc", "release"):
            t, p = truth[k], pred[k]
            d = (p - t) if (t is not None and p is not None) else None
            row[f"true_{k}"] = t if t is not None else ""
            row[f"pred_{k}"] = p if p is not None else ""
            row[f"err_{k}"] = d if d is not None else ""
            errs[k].append(d)
            if d is not None:
                per_bowler[bowler][k].append(abs(d))
        out.append(row)
        if note:
            notes[note.split(" (")[0]] += 1
        if any(pred[k] is None for k in pred):
            failed.append((vid, bowler, note))

    # ------------------------------------------------------- overall accuracy
    print("=" * 66)
    print("EVENT TIMING ERROR   (predicted frame minus your mark)")
    print("=" * 66)
    report(errs["bfc"], "back foot contact", args.fps, args.tol)
    report(errs["ffc"], "front foot contact", args.fps, args.tol)
    report(errs["release"], "ball release", args.fps, args.tol)

    # ------------------------------------------------------- the headline
    print("\n" + "=" * 66)
    both = [(a, b) for a, b in zip(errs["bfc"], errs["ffc"])
            if a is not None and b is not None]
    if both:
        ok = sum(1 for a, b in both
                 if abs(a) <= args.tol and abs(b) <= args.tol)
        print(f"both contacts within {args.tol} frames: "
              f"{ok}/{len(both)} clips = {ok/len(both):.1%}")
        print(f"  ({args.tol} frames at {args.fps:.0f} fps = "
              f"{args.tol*1000/args.fps:.0f} ms)")
    if n_fallback:
        print(f"\n{n_fallback} clip(s) rescued by falling back to {other} "
              f"(column 'model' in {args.out} says which answered)")
    if failed:
        print(f"\n{len(failed)} clip(s) where a phase block was never found:")
        for vid, b, note in failed[:args.show]:
            print(f"  {b:12s} {vid}  {note}")
    if notes:
        print("\nblock-finding notes:")
        for k, v in sorted(notes.items(), key=lambda z: -z[1]):
            print(f"  {v:4d}  {k}")

    # ------------------------------------------------------- per bowler
    print("\nper-bowler mean |error| in frames\n")
    print(f"  {'bowler':14s}{'n':>4s}{'bfc':>8s}{'ffc':>8s}{'release':>10s}")
    rowsb = []
    for b, d in per_bowler.items():
        n = len(d["ffc"])
        m = [float(np.mean(d[k])) if d[k] else float("nan")
             for k in ("bfc", "ffc", "release")]
        rowsb.append((np.nanmean(m), b, n, m))
    for _, b, n, m in sorted(rowsb, key=lambda z: z[0]):
        print(f"  {b:14s}{n:4d}" + "".join(f"{v:8.1f}" for v in m[:2])
              + f"{m[2]:10.1f}")
    print("\n  (sorted best to worst -- a bowler at the bottom either has an")
    print("   unusual action or has marks that disagree with everyone else's)")

    # ------------------------------------------------------- worst clips
    print(f"\nworst {args.show} clips by front foot contact error\n")
    rank = sorted((r for r in out if r["err_ffc"] != ""),
                  key=lambda r: -abs(r["err_ffc"]))
    print(f"  {'bowler':12s}{'clip':34s}{'true':>6s}{'pred':>6s}{'err':>6s}")
    for r in rank[:args.show]:
        print(f"  {r['bowler']:12s}{r['video'][:33]:34s}"
              f"{r['true_ffc']:6d}{r['pred_ffc']:6d}{r['err_ffc']:+6d}"
              + (f"   {r['note']}" if r["note"] else ""))

    cols = ["video", "bowler", "model"] \
        + [f"{a}_{b}" for b in ("bfc", "ffc", "release")
           for a in ("true", "pred", "err")] + ["note"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\nper-clip events written to {args.out}")

    print("""
What to look at:

  The headline number is the fraction of clips where BOTH contacts land within
  your tolerance. That is the figure your measurements inherit: a front foot
  contact 2 frames late means the knee angle you report is the angle 17 ms after
  the real one, which on a flexing knee is a degree or two.

  Check the bias line. A consistent sign is a systematic offset you can correct
  for; scatter around zero cannot be corrected and is the real error.

  Then open the worst clips at the predicted frame and the marked frame and
  decide which one is right. Some of these will be the model finding a boundary
  you placed carelessly, which improves your labels rather than condemning the
  model.""")


if __name__ == "__main__":
    main()
