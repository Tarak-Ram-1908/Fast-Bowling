"""
Train the per-frame phase classifier and report how well it separates the phases.

    pip install scikit-learn
    python train_classifier.py features.csv

Splits by BOWLER, never by frame. Adjacent frames are 8 ms apart and nearly
identical, so a random split would put near-duplicates in both train and test
and report an accuracy that means nothing. GroupKFold holds out whole bowlers,
which is the question that matters: does this work on someone it has never seen?

Two models, deliberately:
  logistic regression   linear, interpretable, tells you which features carry
                        the decision through its coefficients
  random forest         non-linear, handles the run-up class having two shapes
                        (running strides and an airborne bound)

Writes predictions.csv with out-of-fold predictions for every frame -- each row
predicted by a model that never saw that bowler. That file is the input to the
event extraction step.
"""

import argparse
import csv
import sys

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("pip install scikit-learn")

FEATURES = [
    "knee_lead", "knee_trail", "elbow_high", "elbow_low",
    "ankle_sep_x", "ankle_diff_y", "low_ankle_y", "high_ankle_y",
    "trunk_lean", "arm_high_elev", "arm_low_elev", "wrist_high_vs_head",
    "shoulder_width", "hip_width", "shoulder_hip_offset", "lead_ankle_fwd",
]
PHASES = ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([r["phase"] for r in rows])
    g = np.array([r["bowler"] for r in rows])
    return rows, X, y, g


def show_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=PHASES)
    print(f"\n{title}")
    print("  rows = truth, columns = predicted\n")
    head = "  " + " " * 17 + "".join(f"{p[:9]:>11s}" for p in PHASES) + f"{'recall':>10s}"
    print(head)
    for i, p in enumerate(PHASES):
        row = cm[i]
        tot = row.sum()
        rec = row[i] / tot if tot else 0
        line = f"  {p:17s}" + "".join(f"{v:11d}" for v in row) + f"{rec:9.1%}"
        print(line)
    prec = []
    for j in range(len(PHASES)):
        col = cm[:, j].sum()
        prec.append(cm[j, j] / col if col else 0)
    print("  " + "precision".ljust(17) + "".join(f"{v:10.1%} " for v in prec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--auto-features", action="store_true",
                    help="derive the feature list from the CSV header instead of\nthe hardcoded FEATURES, for window-stacked files")
    args = ap.parse_args()

    if args.auto_features:
        import csv as _csv
        hdr = next(_csv.reader(open(args.features, newline="")))
        meta = {"video", "bowler", "frame", "phase", "direction", "torso_px",
                "is_bfc_event", "is_ffc_event", "is_release_event",
                "width", "height", "legal", "usable"}
        global FEATURES
        FEATURES = [c for c in hdr if c not in meta]
        print(f"auto-features: {len(FEATURES)} columns from the header")
    rows, X, y, g = load(args.features)
    bowlers = sorted(set(g))
    print(f"{len(rows)} frames, {len(bowlers)} bowlers, {len(FEATURES)} features")
    print("class balance:", {p: int((y == p).sum()) for p in PHASES})
    print(f"\n{args.folds}-fold GroupKFold, holding out whole bowlers\n")

    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced")),
        "forest": RandomForestClassifier(
            n_estimators=args.trees, min_samples_leaf=3,
            class_weight="balanced", random_state=0, n_jobs=-1),
    }

    oof = {name: np.empty(len(y), dtype=object) for name in models}
    oof_prob = {name: np.zeros((len(y), len(PHASES))) for name in models}
    per_fold = {name: [] for name in models}

    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=g), 1):
        held = sorted(set(g[te]))
        line = f"  fold {k}  held out: {', '.join(held)}"
        for name, m in models.items():
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            oof[name][te] = pred
            classes = list(m.classes_)
            prob = m.predict_proba(X[te])
            for j, p in enumerate(PHASES):
                if p in classes:
                    oof_prob[name][te, j] = prob[:, classes.index(p)]
            acc = (pred == y[te]).mean()
            per_fold[name].append(acc)
            line += f"   {name} {acc:.1%}"
        print(line, flush=True)

    print()
    for name in models:
        a = np.array(per_fold[name])
        print(f"  {name:8s} accuracy {a.mean():.1%} +- {a.std():.1%} across folds")

    for name in models:
        show_confusion(y, oof[name], f"=== {name} ===")

    print("\n\nper-class detail (random forest)\n")
    print(classification_report(y, oof["forest"], labels=PHASES, digits=3,
                                zero_division=0))

    # ---------------------------------------------------- what drives it
    rf = models["forest"]
    imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda z: -z[1])
    print("feature importance (random forest)\n")
    for f, v in imp:
        bar = "#" * int(round(v * 120))
        print(f"  {f:22s} {v:.3f}  {bar}")

    # ---------------------------------------------------- save
    # Both models' probabilities are saved. The forest is better overall, but
    # the logistic regression has higher ffc_phase RECALL, and recall is what
    # decides whether an ffc block exists at all for the event extraction to
    # read an instant off. Which one wins is an empirical question, so keep
    # both and let extract_events.py --model settle it.
    cols = ["video", "bowler", "frame", "phase",
            "is_bfc_event", "is_ffc_event", "is_release_event",
            "pred_logreg", "pred_forest"] \
        + [f"p_forest_{p}" for p in PHASES] \
        + [f"p_logreg_{p}" for p in PHASES] \
        + [f"p_{p}" for p in PHASES]          # legacy alias = forest
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(rows):
            pf = [round(float(v), 4) for v in oof_prob["forest"][i]]
            pl = [round(float(v), 4) for v in oof_prob["logreg"][i]]
            w.writerow([r["video"], r["bowler"], r["frame"], r["phase"],
                        r["is_bfc_event"], r["is_ffc_event"], r["is_release_event"],
                        oof["logreg"][i], oof["forest"][i]]
                       + pf + pl + pf)
    print(f"\nout-of-fold predictions written to {args.out}")

    print("""
Read the confusion matrix, not the accuracy. What matters is the two contact
phases: how often bfc_phase is called ffc_phase and the reverse, because that
boundary is where the event instants come from. Confusion between run_up and
none costs you nothing -- both mean 'not yet bowling'.""")


if __name__ == "__main__":
    main()
