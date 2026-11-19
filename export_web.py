"""
Package the model for the browser version of the site in docs/.

    py -3.13 export_web.py

Writes:
    docs/model/forest.bin      the 300-tree random forest as flat arrays
    docs/model/meta.json       feature order, phases, window size, and every
                               reference value the report uses
    docs/model/pose_landmarker_heavy.task   the same MediaPipe model Python uses
    webtest/forest_check.json  feature rows with scikit-learn's probabilities,
                               so the JavaScript forest can be checked exactly

forest.bin layout, little-endian, every array starting on its natural
alignment:

    int32[4]   nTrees, nNodes, nLeaves, nClasses
    float64[nNodes]           split threshold (unused at leaves)
    float32[nLeaves*nClasses] class fractions at each leaf
    int32[nTrees]             root node of each tree
    int32[nNodes]             left child, or leaf index at a leaf
    int32[nNodes]             right child, or -1 at a leaf
    int16[nNodes]             split feature, or -1 at a leaf

Thresholds stay float64 because scikit-learn compares float32 features to
float64 thresholds; the browser rounds features to float32 the same way, so
every split goes the same direction.
"""
import json, os, shutil
import numpy as np
import joblib

import analyze

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "model")


def export_forest(model, path):
    trees = [e.tree_ for e in model.estimators_]
    n_cls = model.n_classes_
    thr, leafv, roots, left, right, feat = [], [], [], [], [], []
    base = 0
    n_leaves = 0
    for t in trees:
        roots.append(base)
        is_leaf = t.children_left == -1
        val = t.value[:, 0, :].astype(np.float64)
        val = val / val.sum(axis=1, keepdims=True)       # fractions, as predict_proba
        for i in range(t.node_count):
            thr.append(t.threshold[i])
            if is_leaf[i]:
                left.append(n_leaves); right.append(-1); feat.append(-1)
                leafv.extend(val[i]); n_leaves += 1
            else:
                left.append(base + t.children_left[i])
                right.append(base + t.children_right[i])
                feat.append(t.feature[i])
        base += t.node_count
    with open(path, "wb") as fh:
        np.array([len(trees), base, n_leaves, n_cls], "<i4").tofile(fh)
        np.array(thr, "<f8").tofile(fh)
        np.array(leafv, "<f4").tofile(fh)
        np.array(roots, "<i4").tofile(fh)
        np.array(left, "<i4").tofile(fh)
        np.array(right, "<i4").tofile(fh)
        np.array(feat, "<i2").tofile(fh)
    return len(trees), base, n_leaves


def main():
    os.makedirs(OUT, exist_ok=True)
    bundle = joblib.load(os.path.join(ROOT, "models", "phase_model.joblib"))
    model = bundle["model"]
    nt, nn, nl = export_forest(model, os.path.join(OUT, "forest.bin"))
    print(f"forest.bin: {nt} trees, {nn} nodes, {nl} leaves, "
          f"{os.path.getsize(os.path.join(OUT, 'forest.bin')) / 1e6:.1f} MB")

    meta = {
        "features": list(bundle["features"]),
        "base_features": analyze.FEATURE_NAMES,
        "phases": [str(p) for p in bundle["phases"]],
        "model_classes": [str(c) for c in model.classes_],
        "k": int(bundle["k"]),
        "smooth": 9,
        "ref": {k: list(v) for k, v in analyze.REF.items()},
        "not_comparable": analyze.NOT_COMPARABLE,
        "cohort": analyze.COHORT,
        "trunk_ratio": analyze.TRUNK_RATIO,
        "measures": [list(m) for m in analyze.MEASURES],
        "event_deps": {k: list(v) for k, v in analyze.EVENT_DEPS.items()},
        "event_name": analyze.EVENT_NAME,
        "event_hit_rate": analyze.EVENT_HIT_RATE,
        "event_base": analyze.EVENT_BASE,
        "assoc": analyze._ASSOC,
        "describe": analyze._DESCRIBE,
        "focus_at": analyze.FOCUS_AT,
        "focus_text": analyze.FOCUS_TEXT,
        "implausible_sd": analyze.IMPLAUSIBLE_SD,
        "unusual_sd": analyze.UNUSUAL_SD,
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("meta.json written")

    shutil.copyfile(os.path.join(ROOT, "models", "pose_landmarker_heavy.task"),
                    os.path.join(OUT, "pose_landmarker_heavy.task"))
    print("pose model copied")

    # the page shares the server version's stylesheet and definitions text
    docs = os.path.dirname(OUT)
    shutil.copyfile(os.path.join(ROOT, "web", "static", "style.css"),
                    os.path.join(docs, "style.css"))
    tpl = open(os.path.join(ROOT, "web", "templates", "report.html"),
               encoding="utf-8").read()
    a = tpl.index('<section class="defs">')
    defs = tpl[a:tpl.index("</section>", a) + len("</section>")]
    page_path = os.path.join(docs, "index.html")
    page = open(page_path, encoding="utf-8").read()
    s, e = page.index("<!-- DEFS:START"), page.index("<!-- DEFS:END -->")
    head = page[s:page.index("-->", s) + 3]
    page = page[:s] + head + "\n" + defs + "\n" + page[e:]
    open(page_path, "w", encoding="utf-8").write(page)
    open(os.path.join(docs, ".nojekyll"), "w").close()
    print("style.css and definitions copied into docs/")

    # rows to check the browser forest against scikit-learn, from real clips
    src = os.path.join(ROOT, "feat_noab_win.csv")
    if os.path.exists(src):
        import csv
        rng = np.random.default_rng(0)
        with open(src, newline="") as fh:
            rows = list(csv.DictReader(fh))
        pick = rng.choice(len(rows), size=min(3000, len(rows)), replace=False)
        X = np.array([[float(rows[i][f]) for f in meta["features"]] for i in pick],
                     dtype=np.float32)
        P = model.predict_proba(X)
        os.makedirs(os.path.join(ROOT, "webtest"), exist_ok=True)
        with open(os.path.join(ROOT, "webtest", "forest_check.json"), "w") as fh:
            json.dump({"X": X.astype(float).tolist(), "P": P.tolist()}, fh)
        print(f"forest_check.json: {len(pick)} rows")


if __name__ == "__main__":
    main()
