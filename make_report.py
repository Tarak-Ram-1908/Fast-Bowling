"""
Build every table and figure the semester report needs, from the CSVs the
pipeline already produced.

    py -3.13 make_report.py

Reads whatever it finds in the current folder and skips the rest, so it is safe
to run before the whole pipeline has been re-run:

    labels.csv              the human event marks
    legality.csv            legal / illegal per delivery
    bowlers.csv             per-bowler height, handedness, footwear, quality
    frame_labels.csv        per-frame phase labels
    predictions_all.csv     from compare_models.py (or predictions.csv)
    events_<model>.csv      from extract_events.py --out events_<model>.csv
    measurements.csv        from measure_deliveries.py
    technique.csv           from classify_technique.py

Writes:

    report/figures/*.png    300 dpi, ready to \\includegraphics
    report/tables/*.tex     booktabs tables, ready to \\input
    report/summary.txt      every number in one place

TO GET THE FULL SET, run this first:

    py -3.13 compare_models.py feat_noab_win.csv --folds 24
    py -3.13 extract_events.py predictions_all.csv --model forest  --out events_forest.csv
    py -3.13 extract_events.py predictions_all.csv --model logreg  --out events_logreg.csv
    py -3.13 extract_events.py predictions_all.csv --model hgb     --out events_hgb.csv
    py -3.13 make_report.py

A NOTE ON WHAT THE FIGURES ARGUE
--------------------------------
Figure 2 (the tolerance curves) is the most important one in the report. It is
the honest statement of what the system does: not "85% accurate", but "finds
ball release within N frames this often". Lead with it.

Figure 4 exists to make one specific comparison possible -- per-frame accuracy
against event accuracy. If a model wins one and loses the other, that is a
result about the metric, and it is more interesting than the ranking.
"""

import csv, glob, math, os, re, sys
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("pip install numpy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    sys.exit("pip install matplotlib")

# ---------------------------------------------------------------- palette
# Validated for colour-vision deficiency across all pairs (OKLab dE 9.2 worst
# CVD, 24.0 worst normal vision). Do not add a fourth categorical colour --
# yellow and orange together fail the all-pairs floor. Facet instead.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # ours / reference / third
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", S1])
# Diverging: two hues with a NEUTRAL GREY midpoint -- never a hue at zero,
# or the midpoint reads as a category of its own.
DIV = LinearSegmentedColormap.from_list("div", [S1, "#eceae4", S2])

PHASES = ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]
NICE = {"run_up": "Run-up", "bfc_phase": "Back-foot contact",
        "ffc_phase": "Front-foot contact", "follow_through": "Follow-through",
        "none": "None"}
EVENTS = [("release", "Ball release"), ("ffc", "Front-foot contact"),
          ("bfc", "Back-foot contact")]

# Worthington (2010), elite fast bowlers: (mean, sd)
REF = {"front_knee_ffc": (164.1, 6.1), "front_knee_release": (167.3, 18.8),
       "knee_extension": (11.9, 7.4),
       "plant_angle": (36.3, 3.9), "foot_strike": (-8.2, 13.2),
       "lean_back_bfc": (9.0, 5.2), "stride_pct_height": (78.5, 7.2),
       "release_pct_height": (112.8, 4.1)}
REF_NICE = {"front_knee_ffc": "Front knee at FFC",
            "front_knee_release": "Front knee at release",
            "knee_flexion": "Knee flexion", "knee_extension": "Knee extension",
            "plant_angle": "Plant angle", "foot_strike": "Foot strike",
            "lean_back_bfc": "Trunk lean at BFC",
            "stride_pct_height": "Stride (% height)",
            "release_pct_height": "Release height (% height)"}
REF_SHORT = {"front_knee_ffc": "Knee$_\\text{FFC}$",
             "front_knee_release": "Knee$_\\text{rel}$",
             "knee_flexion": "Flex", "knee_extension": "Ext",
             "plant_angle": "Plant", "foot_strike": "Foot",
             "lean_back_bfc": "Lean", "stride_pct_height": "Stride",
             "release_pct_height": "RelHt"}
# Portus et al. (2004), 42 elite bowlers
PORTUS = {"flexor": 25, "constant brace": 9, "extender": 4,
          "flexor-extender": 4}

FIG, TAB = "report/figures", "report/tables"
# The model the report speaks about. Every per-model figure uses THIS one, so
# the confusion matrix and the event table describe the same system. Picking
# the best per-frame scorer here instead would silently switch models halfway
# through the results section.
PRIMARY = "forest"
summary = []


def say(s=""):
    print(s)
    summary.append(s)


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        head = fh.read(4)
        if head[:2] == "PK":                       # an .xlsx saved as .csv
            say(f"  ! {path} is a spreadsheet, not CSV -- re-save as CSV")
            return None
        fh.seek(0)
        return list(csv.DictReader(fh))


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def style(ax, xlabel="", ylabel="", title="", grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold",
                     loc="left", pad=10)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    figure: {p}")


def tex_escape(v):
    """Escape a bare % so it cannot comment out the rest of a LaTeX line.

    REF_NICE holds names like "Stride (% height)". An unescaped % there eats
    the remainder of the row and the table fails with 'Misplaced \\noalign'.
    The lookbehind leaves an already-escaped \\% alone."""
    return re.sub(r"(?<!\\)%", r"\\%", str(v))


def table(name, header, rows, caption, label, align=None, size=None):
    os.makedirs(TAB, exist_ok=True)
    header = [tex_escape(h) for h in header]
    rows = [[tex_escape(c) for c in r] for r in rows]
    caption = tex_escape(caption)
    align = align or ("l" + "r" * (len(header) - 1))
    out = ["\\begin{table}[htbp]", "  \\centering",
           f"  \\caption{{{caption}}}", f"  \\label{{tab:{label}}}"]
    if size:
        out.append(f"  \\{size}")
    out += [f"  \\begin{{tabular}}{{{align}}}", "    \\toprule",
           "    " + " & ".join(header) + " \\\\", "    \\midrule"]
    out += ["    " + " & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    p = f"{TAB}/{name}.tex"
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"    table:  {p}")


# ============================================================ 1. dataset
def dataset():
    say("\n" + "=" * 70)
    say("1. DATASET")
    say("=" * 70)

    lab = read("labels.csv")
    leg = read("legality.csv")
    fl = read("frame_labels.csv")
    rows = []

    if lab:
        usable = [r for r in lab if r.get("usable", "1") != "0"]
        say(f"  clips marked      {len(lab)}")
        say(f"  usable            {len(usable)}")
        say(f"  bowlers           {len(set(r['bowler'] for r in lab))}")
        # "120" and "120.000" are the same rate; group numerically. The rest is
        # phone frame-rate drift -- the camera did not hold a steady 120 fps on
        # every clip. Report the spread, not 24 separate rows.
        rates = [num(r.get("fps")) for r in lab]
        rates = [v for v in rates if v]
        if rates:
            exact = sum(1 for v in rates if abs(v - 120) < 0.01)
            drift = [v for v in rates if abs(v - 120) >= 0.01]
            say(f"  frame rate        {exact} clips at 120 fps")
            if drift:
                say(f"                    {len(drift)} clips at "
                    f"{min(drift):.1f}-{max(drift):.1f} fps (capture drift, "
                    f"worst {abs(min(drift)-120)/120:.1%} off)")
        rows += [("Bowlers recorded", len(set(r['bowler'] for r in lab))),
                 ("Deliveries filmed", len(lab)),
                 ("Deliveries usable", len(usable))]

    if leg:
        ok = sum(1 for r in leg if r.get("legal") == "1")
        say(f"  legal deliveries  {ok} of {len(leg)}  ({ok/len(leg):.1%})")
        rows += [("Deliveries legal", ok)]

    if fl:
        say(f"  labelled frames   {len(fl)}")
        rows += [("Frames labelled", len(fl))]
        c = Counter(r["phase"] for r in fl)
        say("\n  phase balance")
        for p in PHASES:
            say(f"    {NICE[p]:<22}{c[p]:>7}  {c[p]/len(fl):>6.1%}")

        fig, ax = plt.subplots(figsize=(6.2, 2.9))
        ys = [c[p] for p in PHASES]
        bars = ax.barh([NICE[p] for p in PHASES], ys, color=S1, height=0.62)
        for b, v in zip(bars, ys):
            ax.text(b.get_width() + max(ys) * 0.015,
                    b.get_y() + b.get_height() / 2, f"{v:,}",
                    va="center", fontsize=8.5, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, max(ys) * 1.15)
        style(ax, xlabel="Frames", title="Phase label distribution", grid="x")
        save(fig, "fig1_class_balance")

    if rows:
        table("dataset", ["Quantity", "Value"], rows,
              "Dataset summary.", "dataset")
    if not (lab or leg or fl):
        say("  (no dataset CSVs found)")


# ================================================= 2. event accuracy
def events():
    say("\n" + "=" * 70)
    say("2. EVENT DETECTION ACCURACY")
    say("=" * 70)

    files = sorted(glob.glob("events_*.csv"))
    if not files and os.path.exists("events.csv"):
        files = ["events.csv"]
    if not files:
        say("  (no events CSVs -- run extract_events.py --out events_<model>.csv)")
        return {}

    tols = list(range(0, 11))
    per_model = {}
    for f in files:
        m = re.sub(r"^events_|\.csv$", "", os.path.basename(f)) or "model"
        rows = read(f)
        if not rows:
            continue
        d = {"_n_clips": len(rows)}
        for k, _ in EVENTS:
            e = [abs(num(r.get(f"err_{k}"))) for r in rows
                 if num(r.get(f"err_{k}")) is not None]
            d[k] = e
        per_model[m] = d

    # ---- figure 2: tolerance curves for the primary model
    primary = PRIMARY if PRIMARY in per_model else sorted(per_model)[0]
    d = per_model[primary]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ends = []
    for (k, nice), col in zip(EVENTS, (S1, S2, S3)):
        e = d[k]
        if not e:
            continue
        N = d.get("_n_clips", len(e))
        ys = [100 * sum(1 for v in e if v <= t) / N for t in tols]
        ax.plot(tols, ys, color=col, lw=2, marker="o", ms=4.5,
                markeredgecolor="white", markeredgewidth=0.8, label=nice)
        ends.append([ys[-1], nice, col])
    # Curves converge at the right edge, so direct labels collide. Push them
    # apart vertically, keeping their order, before drawing.
    ends.sort(key=lambda t: t[0])
    GAP = 6.0
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i-1][0] < GAP:
            ends[i][0] = ends[i-1][0] + GAP
    for y, nice, col in ends:
        ax.text(tols[-1] + 0.2, y, nice, color=col, fontsize=8.5,
                va="center", fontweight="bold")
    ax.axvline(3, color=INK2, lw=1, ls=":", zorder=0)
    ax.text(3.1, 4, "3 frames = 25 ms", fontsize=8, color=INK2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 103)
    ax.set_xticks(tols)
    style(ax, xlabel="Tolerance (frames at 120 fps)",
          ylabel="Deliveries correct (% of all)",
          title="Event detection accuracy by tolerance")
    fig.subplots_adjust(right=0.74)
    save(fig, "fig2_tolerance")

    # ---- table + console
    hdr = ["Event", "Detected", "Error (fr)", "Error (ms)",
           "Within 1", "Within 3", "Within 5"]
    trows = []
    N = d.get("_n_clips", 0)
    say(f"\n  all rates below are out of {N} deliveries, including the ones")
    say("  where no block was found -- a model that gives up is not rewarded")
    for k, nice in EVENTS:
        e = d[k]
        if not e:
            continue
        say(f"\n  {nice}  ({primary})")
        say(f"    detected          {len(e)}/{N}  ({len(e)/N:.1%})")
        say(f"    mean |error|      {np.mean(e):.2f} frames "
            f"({np.mean(e)*1000/120:.1f} ms)   [of those detected]")
        for t in (1, 2, 3, 5):
            say(f"    within {t:>2}         {sum(1 for v in e if v<=t)/N:>6.1%}")
        trows.append([nice, f"{len(e)}/{N}", f"{np.mean(e):.2f}",
                      f"{np.mean(e)*1000/120:.1f}"] +
                     [f"{sum(1 for v in e if v<=t)/N:.1%}"
                      for t in (1, 3, 5)])
    if trows:
        table("events", hdr, trows,
              f"Event detection accuracy ({primary}), leave-one-bowler-out. "
              f"Rates are out of all {N} deliveries; 'Detected' is how many "
              f"yielded a phase block at all. 'Error' is the mean absolute "
              f"timing error, in frames and in milliseconds, over the "
              f"deliveries detected.",
              "events", size="small")
    return per_model


# ============================================== 3. model comparison
def models(per_model):
    say("\n" + "=" * 70)
    say("3. MODEL COMPARISON")
    say("=" * 70)

    pred = read("predictions_all.csv") or read("predictions.csv")
    names = []
    if pred:
        names = sorted(c[5:] for c in pred[0] if c.startswith("pred_"))
        y = np.array([r["phase"] for r in pred])

        say(f"\n  per-frame, {len(pred)} frames, out-of-fold\n")
        say(f"  {'model':<12}{'accuracy':>10}{'macro F1':>10}")
        prows, accs, f1s = [], [], []
        for n in names:
            p = np.array([r[f"pred_{n}"] for r in pred])
            acc = float((p == y).mean())
            f1 = float(np.mean([
                (2 * ((p == c) & (y == c)).sum() /
                 max(((p == c).sum() + (y == c).sum()), 1))
                for c in PHASES]))
            accs.append(acc)
            f1s.append(f1)
            say(f"  {n:<12}{acc:>10.3f}{f1:>10.3f}")
            prows.append([n, f"{acc:.3f}", f"{f1:.3f}"])
        table("models_perframe", ["Model", "Accuracy", "Macro F1"], prows,
              "Per-frame phase classification, leave-one-bowler-out.",
              "models_perframe")

        # figure 3 -- per-frame
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        x = np.arange(len(names))
        w = 0.38
        ax.bar(x - w/2, accs, w * 0.92, color=S1, label="Accuracy")
        ax.bar(x + w/2, f1s, w * 0.92, color=S2, label="Macro F1")
        # No value labels: 14 bars at these widths collide, and Table 3 above
        # carries the exact numbers. The figure's job is the comparison.
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, max(accs + f1s) * 1.18)
        style(ax, ylabel="Score", title="Per-frame phase classification")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
        save(fig, "fig3_models_perframe")

        # figure 5 -- confusion matrix for the PRIMARY model, not the best
        # per-frame scorer: section 5.3 argues the per-frame leader is not the
        # system we report, so showing its confusion matrix would contradict
        # the surrounding text.
        best = PRIMARY if PRIMARY in names else names[int(np.argmax(accs))]
        p = np.array([r[f"pred_{best}"] for r in pred])
        M = np.zeros((len(PHASES), len(PHASES)))
        for t, q in zip(y, p):
            if t in PHASES and q in PHASES:
                M[PHASES.index(t), PHASES.index(q)] += 1
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(PHASES)):
            for j in range(len(PHASES)):
                ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if Mn[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(PHASES)))
        ax.set_yticks(range(len(PHASES)))
        ax.set_xticklabels([NICE[p_] for p_ in PHASES], rotation=35,
                           ha="right", fontsize=8)
        ax.set_yticklabels([NICE[p_] for p_ in PHASES], fontsize=8)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, colors=INK2)
        style(ax, xlabel="Predicted", ylabel="True",
              title=f"Confusion matrix, row-normalised  ({best})", grid=None)
        save(fig, "fig5_confusion")

        # per-bowler spread
        if "bowler" in pred[0]:
            g = np.array([r["bowler"] for r in pred])
            accs_b = sorted(((b, float((p[g == b] == y[g == b]).mean()))
                             for b in set(g)), key=lambda t: t[1])
            say(f"\n  per-bowler accuracy ({best}): "
                f"mean {np.mean([a for _, a in accs_b]):.3f}  "
                f"worst {accs_b[0][1]:.3f} ({accs_b[0][0]})")
            fig, ax = plt.subplots(figsize=(6.2, max(3, 0.24*len(accs_b))))
            bs = ax.barh([b for b, _ in accs_b], [a for _, a in accs_b],
                         color=S1, height=0.66)
            m = float(np.mean([a for _, a in accs_b]))
            ax.axvline(m, color=S2, lw=1.6, zorder=3)
            ax.text(m + 0.006, len(accs_b) - 0.4, f"mean {m:.3f}",
                    color=S2, fontsize=8.5, fontweight="bold")
            ax.set_xlim(0, 1.0)
            style(ax, xlabel="Out-of-fold accuracy",
                  title=f"Accuracy per held-out bowler  ({best})", grid="x")
            save(fig, "fig7_per_bowler")

    # figure 4 -- event accuracy across models
    if per_model and len(per_model) > 1:
        ms = sorted(per_model)
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        x = np.arange(len(ms))
        w = 0.26
        erows = []
        for i, ((k, nice), col) in enumerate(zip(EVENTS, (S1, S2, S3))):
            vals = []
            for m in ms:
                e = per_model[m][k]
                N_ = per_model[m].get("_n_clips", len(e))
                vals.append(100 * sum(1 for v in e if v <= 3) / N_ if e else 0)
            bb = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=col, label=nice)
            for r_ in bb:
                ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 1.6,
                        f"{r_.get_height():.0f}", ha="center", fontsize=7.5,
                        color=INK2)
            erows.append([nice] + [f"{v:.1f}%" for v in vals])
        ax.set_xticks(x)
        ax.set_xticklabels(ms)
        ax.set_ylim(0, 108)
        style(ax, ylabel="Within 3 frames (%)",
              title="Event detection accuracy by model")
        ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16))
        save(fig, "fig4_models_events")
        table("models_events", ["Event"] + ms, erows,
              "Event detection within 3 frames, by model, as a share of all "
              "deliveries (clips where no block was found count as failures).",
              "models_events")

        say("\n  within 3 frames, by model")
        w_ = max(11, max(len(m) for m in ms) + 2)
        say("  " + " " * 22 + "".join(f"{m:>{w_}}" for m in ms))
        for r in erows:
            say(f"  {r[0]:<22}" +
                "".join(f"{v.replace(chr(92)+'%','%'):>{w_}}" for v in r[1:]))

    if not pred and not per_model:
        say("  (run compare_models.py first)")


# ============================================ 4. measurements
def measurements():
    say("\n" + "=" * 70)
    say("4. BIOMECHANICAL MEASUREMENTS vs WORTHINGTON (2010)")
    say("=" * 70)

    rows = read("measurements.csv")
    if not rows:
        say("  (no measurements.csv)")
        return
    legal = [r for r in rows if r.get("legal", "1") != "0"]
    use = legal or rows
    say(f"  {len(use)} deliveries, {len(set(r['bowler'] for r in use))} bowlers\n")

    say(f"  {'measurement':<24}{'ours (mean+-sd)':>18}{'elite':>16}"
        f"{'z':>7}{'in +-1sd':>9}")
    trows, plot, byb_z = [], [], defaultdict(dict)
    for k in REF:
        pairs = [(r.get("bowler"), num(r.get(k))) for r in use]
        pairs = [(b, v) for b, v in pairs if v is not None]
        vals = [v for _, v in pairs]
        if not vals:
            continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rm, rs = REF[k]
        z = (mu - rm) / rs if rs else 0.0
        # Share of OUR individual deliveries inside the elite 1-sd band. A mean
        # comparison cannot tell a whole-group shift from a few outliers.
        inside = sum(1 for v in vals if abs(v - rm) <= rs) / len(vals)
        say(f"  {REF_NICE[k]:<24}{mu:>11.1f} ±{sd:<5.1f}{rm:>9.1f} ±{rs:<5.1f}"
            f"{z:>+7.2f}{inside:>9.0%}")
        trows.append([REF_NICE[k], len(vals), f"{mu:.1f} $\\pm$ {sd:.1f}",
                      f"{rm:.1f} $\\pm$ {rs:.1f}", f"{z:+.2f}",
                      f"{inside:.0%}"])
        plot.append((REF_NICE[k], mu, rm, rs, z, (sd / rs) if rs else 0.0))
        for b, v in pairs:
            byb_z[b].setdefault(k, []).append((v - rm) / rs if rs else 0.0)

    if trows:
        table("measurements",
              ["Measurement", "n", "Ours", "Elite", "$z$",
               "In $\\pm$1\\,sd"],
              trows,
              "Measured kinematics against Worthington (2010) elite reference "
              "values. Ours and Elite are mean $\\pm$ sd. $z$ is the shift of "
              "our mean in elite standard deviations; the final column is the "
              "share of our individual deliveries falling inside the elite "
              "one-sd band.",
              "measurements", size="footnotesize")

    if plot:
        # z-scores: a signed deviation from a reference -> bars about zero
        plot.sort(key=lambda t: t[4])
        fig, ax = plt.subplots(figsize=(6.4, 0.46 * len(plot) + 1.9))
        ys = np.arange(len(plot))
        zs = [p[4] for p in plot]
        sp = [p[5] for p in plot]        # our own sd, in elite-sd units
        # Dot = where our mean sits. Bar = how widely OUR bowlers scatter. A
        # big shift with a narrow bar is a group difference; a big shift with
        # a wide bar may be a handful of bowlers.
        for yy, z, w in zip(ys, zs, sp):
            ax.plot([z - w, z + w], [yy, yy], color=GRID, lw=3.2,
                    solid_capstyle="round", zorder=1)
        # Same encoding as the fig9 heatmap: blue below the elite mean,
        # orange above it. Inconsistent colour across two figures of the same
        # quantity is worse than no colour at all.
        ax.scatter(zs, ys, s=60, c=[S2 if z >= 0 else S1 for z in zs],
                   zorder=3, edgecolor="white", linewidth=1.1)
        for yy, z, w in zip(ys, zs, sp):
            ax.text(z + w + 0.12, yy, f"{z:+.2f}", va="center", ha="left",
                    fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, lw=1.2)
        for b in (-1, 1):
            ax.axvline(b, color=GRID, lw=1, ls="--", zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([p[0] for p in plot], fontsize=8.5)
        lim = max(2.0, max(abs(z) + w for z, w in zip(zs, sp)) * 1.08)
        ax.set_xlim(-lim, lim)
        style(ax, xlabel="Deviations from the elite mean ($z$)",
              title="Measured kinematics vs elite reference", grid="x")
        # explanation lives in the LaTeX caption, not jammed under the title
        save(fig, "fig6_measurements")

    # ---- per bowler: is the pattern universal, or a few people?
    if byb_z:
        keys = [k for k in REF if any(k in v for v in byb_z.values())]
        bs = sorted(byb_z, key=lambda b: np.mean(
            [np.mean(byb_z[b][k]) for k in keys if k in byb_z[b]] or [0]))
        M = np.full((len(bs), len(keys)), np.nan)
        for i, b in enumerate(bs):
            for j, k in enumerate(keys):
                if k in byb_z[b]:
                    M[i, j] = float(np.mean(byb_z[b][k]))
        fig, ax = plt.subplots(figsize=(0.78 * len(keys) + 2.6,
                                        0.3 * len(bs) + 2.2))
        v = 2.5
        im = ax.imshow(M, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([REF_NICE[k] for k in keys], rotation=38,
                           ha="right", fontsize=8)
        ax.set_yticks(range(len(bs)))
        ax.set_yticklabels(bs, fontsize=7.5)
        ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.tick_params(length=0, colors=INK2)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                          ticks=[-v, -1, 0, 1, v])
        cb.set_label("deviations from the elite mean ($z$)", fontsize=8,
                     color=INK2)
        cb.ax.tick_params(labelsize=7.5, colors=INK2, length=0)
        cb.outline.set_visible(False)
        style(ax, title="Each bowler against the elite reference", grid=None)
        save(fig, "fig9_per_bowler_z")

        say("\n  per-bowler mean z (sorted; a column of one colour means the")
        say("  difference is the whole group, not a few individuals)")
        hdr2 = "  " + " " * 14 + "".join(
            f"{REF_NICE[k][:9]:>10}" for k in keys)
        say(hdr2)
        prows = []
        for i, b in enumerate(bs):
            say(f"  {b:<14}" + "".join(
                ("    --    " if np.isnan(M[i, j]) else f"{M[i,j]:>+10.2f}")
                for j in range(len(keys))))
            prows.append([b] + [("--" if np.isnan(M[i, j])
                                 else f"{M[i,j]:+.2f}")
                                for j in range(len(keys))])
        key = "; ".join(f"{REF_SHORT.get(k, k)} = {REF_NICE[k]}"
                        for k in keys)
        table("per_bowler_z",
              ["Bowler"] + [REF_SHORT.get(k, k) for k in keys], prows,
              "Each bowler's mean deviation from the elite reference, in "
              "elite standard deviations, sorted by overall deviation. "
              "Columns: " + key + ".",
              "perbowlerz",
              align="l" + "r" * len(keys), size="footnotesize")


# ============================================ 5. technique
def technique():
    say("\n" + "=" * 70)
    say("5. FRONT-LEG TECHNIQUE vs PORTUS et al. (2004)")
    say("=" * 70)

    rows = read("technique.csv")
    if not rows:
        say("  (no technique.csv)")
        return
    byb = defaultdict(list)
    for r in rows:
        byb[r["bowler"]].append(r["front_leg_class"])
    modal = {b: Counter(v).most_common(1)[0][0] for b, v in byb.items()}
    c = Counter(modal.values())
    nb = len(modal)
    order = ["flexor", "flexor-extender", "extender", "constant brace"]

    say(f"  {nb} bowlers, modal class each\n")
    say(f"  {'class':<20}{'ours':>14}{'Portus 42':>14}")
    trows = []
    for k in order:
        ours, them = c[k] / nb, PORTUS[k] / 42
        say(f"  {k:<20}{c[k]:>4} ({ours:>5.1%}){PORTUS[k]:>6} ({them:>5.1%})")
        trows.append([k.capitalize(), c[k], f"{ours:.1%}",
                      PORTUS[k], f"{them:.1%}"])
    table("technique", ["Class", "Ours (n)", "Ours", "Portus (n)", "Portus"],
          trows, "Front-leg technique classification against Portus et al. "
          "(2004), 42 elite bowlers.", "technique")

    # per bowler, with how consistent each one is across their own deliveries
    say("\n  per bowler  (modal class, and how consistent across deliveries)")
    brows = []
    for b in sorted(byb):
        c2 = Counter(byb[b])
        top, n2 = c2.most_common(1)[0]
        share = n2 / len(byb[b])
        mark = "" if share >= 0.7 else "   <- varies"
        say(f"    {b:<14}{top:<18}{n2}/{len(byb[b])}{mark}")
        brows.append([b, top.capitalize(), f"{n2}/{len(byb[b])}",
                      f"{share:.0%}"])
    table("technique_per_bowler",
          ["Bowler", "Modal class", "Deliveries", "Consistency"], brows,
          "Front-leg class per bowler. Consistency is the share of that "
          "bowler's own deliveries falling in their modal class.",
          "techniqueperbowler")

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(order))
    w = 0.38
    a = ax.bar(x - w/2, [100*c[k]/nb for k in order], w, color=S1,
               label=f"This study (n={nb})")
    b = ax.bar(x + w/2, [100*PORTUS[k]/42 for k in order], w, color=S2,
               label="Portus et al. 2004 (n=42)")
    for bars in (a, b):
        for bb in bars:
            if bb.get_height() > 0.5:
                ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+1.2,
                        f"{bb.get_height():.0f}%", ha="center", fontsize=8,
                        color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" ", "\n").replace("-", "-\n")
                        for k in order], fontsize=8.5)
    ax.set_ylim(0, 100)
    style(ax, ylabel="Bowlers (%)", title="Front-leg technique classification")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig8_technique")


# ==================================================================== main
def main():
    global PRIMARY
    if len(sys.argv) > 1:
        PRIMARY = sys.argv[1]
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    print("building report assets from the CSVs in", os.getcwd(), "\n")

    dataset()
    per_model = events()
    models(per_model)
    measurements()
    technique()

    say("\n" + "=" * 70)
    say("All figures are 300 dpi PNG in report/figures/.")
    say("All tables are booktabs LaTeX in report/tables/ -- \\usepackage{booktabs}")
    say("and \\input{report/tables/events} to include one.")
    say("=" * 70)

    open("report/summary.txt", "w", encoding="utf-8").write(
        "\n".join(summary) + "\n")
    print("\n    summary: report/summary.txt")


if __name__ == "__main__":
    main()
