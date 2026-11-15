"""
Fit the final phase classifier on every bowler and save it for inference.

    py -3.13 train_production.py feat_noab_win.csv

train_classifier.py answers "does this generalise to a bowler it has never
seen" -- it holds each bowler out in turn and never keeps a model. That is the
right design for reporting accuracy and the wrong one for serving predictions.
This script fits one forest on ALL the data and persists it, together with the
exact feature list and window size, so inference cannot silently disagree with
training about column order.

Saves models/phase_model.joblib holding:
    model       the fitted RandomForestClassifier
    features    ordered feature column names
    phases      class labels
    k           window half-width the features were built with
    n_bowlers   how many bowlers it was fitted on
    trained     ISO timestamp
"""
import argparse, csv, json, os, sys, datetime
import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    sys.exit("pip install scikit-learn joblib")

META = {"video","bowler","frame","phase","direction","torso_px",
        "is_bfc_event","is_ffc_event","is_release_event",
        "width","height","legal","usable"}
PHASES = ["run_up","bfc_phase","ffc_phase","follow_through","none"]

ap = argparse.ArgumentParser()
ap.add_argument("features", nargs="?", default="feat_noab_win.csv")
ap.add_argument("--out", default="models/phase_model.joblib")
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--k", type=int, default=5, help="window half-width used to build the file")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.features, newline="")))
if not rows:
    sys.exit("empty feature file")
feats = [c for c in rows[0] if c not in META]
X = np.array([[float(r[c]) for c in feats] for r in rows], dtype=np.float32)
y = np.array([r["phase"] for r in rows])
bowlers = sorted({r["bowler"] for r in rows})

print(f"{len(rows)} frames, {len(bowlers)} bowlers, {len(feats)} features")
print("fitting on ALL of it (no hold-out -- accuracy comes from train_classifier.py)")

m = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=3,
                           class_weight="balanced", random_state=0, n_jobs=-1)
m.fit(X, y)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
joblib.dump({"model": m, "features": feats, "phases": list(m.classes_),
             "k": a.k, "n_bowlers": len(bowlers), "bowlers": bowlers,
             "trained": datetime.datetime.now().isoformat(timespec="seconds")},
            a.out, compress=3)
size = os.path.getsize(a.out) / 1e6
print(f"\nsaved {a.out}  ({size:.1f} MB)")
print(f"classes: {list(m.classes_)}")
print("\nin-sample accuracy %.1f%% -- this is NOT a performance figure, the model"
      "\nhas seen every row. The honest number is the 24-fold leave-one-bowler-out"
      "\nresult from train_classifier.py." % (100 * m.score(X, y)))
