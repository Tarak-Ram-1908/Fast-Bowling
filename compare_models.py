"""
Benchmark several classifiers on the same leave-one-bowler-out folds.

    py -3.13 compare_models.py feat_noab_win.csv --folds 24
    py -3.13 compare_models.py feat_noab_win.csv --folds 24 --only logreg,forest,hgb
    py -3.13 compare_models.py feat_noab_win.csv --folds 24 --no-deep

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
This is not a leaderboard. Every model here answers a specific question, and a
model that cannot be given a reason is not in the list:

    logreg      linear baseline. If the forest cannot beat it, the problem was
                linear all along and the extra complexity is unearned.
    forest      the incumbent. Everything else is measured against it.
    extratrees  more randomised splits. Tests whether the forest is overfitting
                bowler identity through its split thresholds.
    hgb         gradient boosting -- the strongest classical competitor on
                tabular data. If anything classical beats the forest, this.
    mlp         a dense network on the stacked features. The honest test of
                whether "add a neural network" alone buys anything.
    tcn         1D convolution over the BASE features. Window stacking IS a
                hand-built single-layer convolution with fixed weights; this
                learns the kernel instead, over several layers.
    bilstm      recurrent, both directions, over the BASE features. The only
                model that can learn "back foot contact precedes front foot
                contact" rather than having it imposed afterwards by
                extract_events.py.

THE SAMPLE SIZE, STATED PLAINLY
-------------------------------
There are ~30,000 frames but only ~24 INDEPENDENT subjects. Frames inside one
bowler are near-duplicates of each other and share one body, one action and one
camera position. The effective n is the bowler count. A model with hundreds of
thousands of parameters has 24 examples to generalise from, and GroupKFold will
expose that. Expect the deep models to struggle. That result is worth
reporting: it is evidence about the dataset, not a failure of the method.

THE METRIC THAT ACTUALLY MATTERS
--------------------------------
Per-frame accuracy is NOT the headline. The system is judged on whether it
finds three frames per delivery, and a model can score higher per-frame while
being worse at that -- if its errors sit at the block EDGES instead of the
middles. So this script writes every model's probabilities into one
predictions file and then you score each on events with the existing,
already-validated code:

    py -3.13 extract_events.py predictions_all.csv --model forest
    py -3.13 extract_events.py predictions_all.csv --model hgb

If the per-frame ranking and the event ranking disagree, say so in the report.
That disagreement is a finding.
"""

import argparse, csv, os, sys, time
from collections import defaultdict

import numpy as np

try:
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

PHASES = ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]
META = {"video", "bowler", "frame", "phase", "direction", "torso_px",
        "is_bfc_event", "is_ffc_event", "is_release_event"}


# ------------------------------------------------------------------ loading

def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit(f"{path} is empty")
    hdr = list(rows[0].keys())
    feats = [c for c in hdr if c not in META]
    base = [c for c in feats if "__" not in c]
    if not feats:
        sys.exit("no feature columns found")

    def mat(cols):
        X = np.empty((len(rows), len(cols)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                try:
                    X[i, j] = float(r[c])
                except (TypeError, ValueError):
                    X[i, j] = 0.0
        return X

    y = np.array([r["phase"] for r in rows])
    g = np.array([r["bowler"] for r in rows])
    return rows, mat(feats), mat(base), y, g, feats, base


def reindex(prob, classes):
    """Model class order -> PHASES order. Never assume they match.

    Models are fitted on INTEGER labels that already index PHASES, so each
    entry of classes_ is a position in PHASES. Integer labels are not a style
    choice: some scikit-learn builds call np.isnan() on the predictions inside
    MLPClassifier's early-stopping check, which raises TypeError on a string
    array. Fitting on integers avoids that path entirely."""
    out = np.zeros((prob.shape[0], len(PHASES)), dtype=np.float64)
    for j, c in enumerate(classes):
        out[:, int(c)] = prob[:, j]
    return out


# ------------------------------------------------------------ deep models

if HAVE_TORCH:
    class TCN(nn.Module):
        """Three dilated 1D conv layers. Deliberately small: ~15k parameters.

        Window stacking takes offsets -5..+5 with fixed weights. One conv layer
        of kernel 3 at dilations 1, 2, 4 reaches +-7 frames and learns the
        weights. That is the entire idea."""
        def __init__(self, n_in, n_out, ch=48):
            super().__init__()
            def blk(i, o, d):
                return nn.Sequential(
                    nn.Conv1d(i, o, 3, padding=d, dilation=d),
                    nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(0.3))
            self.net = nn.Sequential(blk(n_in, ch, 1), blk(ch, ch, 2),
                                     blk(ch, ch, 4), nn.Conv1d(ch, n_out, 1))

        def forward(self, x):            # (B, C, T) -> (B, n_out, T)
            return self.net(x)

    class BiLSTM(nn.Module):
        def __init__(self, n_in, n_out, hid=48):
            super().__init__()
            self.rnn = nn.LSTM(n_in, hid, num_layers=1, batch_first=True,
                               bidirectional=True, dropout=0.0)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(hid * 2, n_out)

        def forward(self, x):            # (B, C, T) -> (B, n_out, T)
            h, _ = self.rnn(x.transpose(1, 2))
            return self.out(self.drop(h)).transpose(1, 2)


def clip_sequences(rows, Xb, idx):
    """Group a fold's rows into per-clip sequences. A sequence model must never
    see two deliveries as one series."""
    byv = defaultdict(list)
    for i in idx:
        byv[rows[i]["video"]].append(i)
    for v in byv:
        byv[v].sort(key=lambda i: int(rows[i]["frame"]))
    return list(byv.values())


def fit_deep(kind, rows, Xb, y_i, tr, te, epochs, seed, device):
    torch.manual_seed(seed)
    n_in, n_out = Xb.shape[1], len(PHASES)
    model = (TCN(n_in, n_out) if kind == "tcn" else BiLSTM(n_in, n_out)).to(device)

    mu, sd = Xb[tr].mean(0), Xb[tr].std(0) + 1e-6
    tr_seqs, te_seqs = clip_sequences(rows, Xb, tr), clip_sequences(rows, Xb, te)

    cnt = np.bincount(y_i[tr], minlength=n_out).astype(np.float64)
    w = torch.tensor((cnt.sum() / (n_out * np.maximum(cnt, 1))),
                     dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def batch(seq):
        x = torch.tensor(((Xb[seq] - mu) / sd).T[None], device=device)
        t = torch.tensor(y_i[seq][None], dtype=torch.long, device=device)
        return x, t

    model.train()
    for _ in range(epochs):
        np.random.shuffle(tr_seqs)
        for seq in tr_seqs:
            if len(seq) < 8:
                continue
            x, t = batch(seq)
            opt.zero_grad()
            loss = lossf(model(x), t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    prob = np.zeros((len(y_i), n_out))
    with torch.no_grad():
        for seq in te_seqs:
            x, _ = batch(seq)
            p = torch.softmax(model(x), dim=1)[0].T.cpu().numpy()
            prob[seq] = p
    return prob


# ------------------------------------------------------------------- write

def dump(path, rows, names, oof_lbl, oof_prob):
    """Write every finished model to disk. Called after EACH model, so a crash
    in model five never costs the four before it."""
    primary = "forest" if "forest" in names else names[0]
    cols = ["video", "bowler", "frame", "phase",
            "is_bfc_event", "is_ffc_event", "is_release_event"]
    cols += [f"pred_{n}" for n in names]
    for n in names:
        cols += [f"p_{n}_{p}" for p in PHASES]
    cols += [f"p_{p}" for p in PHASES]       # legacy alias, for older scripts
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(rows):
            out = [r["video"], r["bowler"], r["frame"], r["phase"],
                   r.get("is_bfc_event", ""), r.get("is_ffc_event", ""),
                   r.get("is_release_event", "")]
            out += [oof_lbl[n][i] for n in names]
            for n in names:
                out += [round(float(v), 4) for v in oof_prob[n][i]]
            out += [round(float(v), 4) for v in oof_prob[primary][i]]
            w.writerow(out)
    os.replace(tmp, path)          # atomic: never a half-written file


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--folds", type=int, default=24)
    ap.add_argument("--out", default="predictions_all.csv")
    ap.add_argument("--only", default="", help="comma list of model names")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep models already in --out and only fit the rest")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows, X, Xb, y, g, feats, base = load(a.features)
    bowlers = sorted(set(g))
    folds = min(a.folds, len(bowlers))
    print(f"{len(rows)} frames | {len(bowlers)} bowlers | {len(feats)} stacked "
          f"features | {len(base)} base features")
    print(f"class balance: " +
          "  ".join(f"{p}={int((y == p).sum())}" for p in PHASES))
    print(f"\n{folds}-fold GroupKFold, holding out whole bowlers")
    print(f"effective n for generalisation = {len(bowlers)} bowlers, "
          f"not {len(rows)} frames\n")

    classical = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000,
                                                   class_weight="balanced")),
        "forest": RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=3,
                                         class_weight="balanced",
                                         random_state=a.seed, n_jobs=-1),
        "extratrees": ExtraTreesClassifier(n_estimators=a.trees, min_samples_leaf=3,
                                           class_weight="balanced",
                                           random_state=a.seed, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              random_state=a.seed),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64),
                                           max_iter=400, early_stopping=True,
                                           random_state=a.seed)),
    }
    deep = [] if (a.no_deep or not HAVE_TORCH) else ["tcn", "bilstm"]
    if not HAVE_TORCH and not a.no_deep:
        print("  note: torch not installed, skipping tcn and bilstm")
        print("  pip install torch   (CPU build is enough at this size)\n")

    names = list(classical) + deep
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        names = [n for n in names if n in want]
        if not names:
            sys.exit(f"--only matched nothing. available: "
                     f"{','.join(list(classical) + deep)}")

    y_i = np.array([PHASES.index(v) for v in y])

    device = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"
    gkf = GroupKFold(n_splits=folds)
    splits = list(gkf.split(X, y, groups=g))

    oof_lbl, oof_prob, timing = {}, {}, {}

    # Resume: anything already in the output file is kept, so a crash or a
    # Ctrl-C never costs more than the model that was running.
    if a.resume and os.path.exists(a.out):
        prev = list(csv.DictReader(open(a.out, newline="")))
        if len(prev) == len(rows):
            for n in [c[5:] for c in prev[0] if c.startswith("pred_")]:
                if all(f"p_{n}_{p}" in prev[0] for p in PHASES):
                    oof_prob[n] = np.array([[float(r[f"p_{n}_{p}"])
                                             for p in PHASES] for r in prev])
                    oof_lbl[n] = np.array([r[f"pred_{n}"] for r in prev])
                    timing[n] = float("nan")
            if oof_prob:
                print(f"  resuming -- already done: {', '.join(sorted(oof_prob))}\n")
        else:
            print(f"  {a.out} has {len(prev)} rows, this run has {len(rows)}"
                  " -- ignoring it\n")

    failed = []
    for name in names:
        if name in oof_prob:
            print(f"  {name:11s} skipped (already in {a.out})")
            continue
        t0 = time.time()
        prob = np.zeros((len(rows), len(PHASES)))
        try:
            for k, (tr, te) in enumerate(splits, 1):
                print(f"\r  {name:11s} fold {k}/{folds}", end="", flush=True)
                if name in deep:
                    prob[te] = fit_deep(name, rows, Xb, y_i, tr, te,
                                        a.epochs, a.seed, device)[te]
                else:
                    m = classical[name]
                    m.fit(X[tr], y_i[tr])
                    prob[te] = reindex(m.predict_proba(X[te]), list(m.classes_))
        except KeyboardInterrupt:
            print(f"\r  {name:11s} interrupted" + " " * 20)
            break
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"\r  {name:11s} FAILED -- {type(e).__name__}: "
                  f"{str(e)[:60]}" + " " * 6)
            print(f"  {'':11s} other models are unaffected and already saved")
            continue
        oof_prob[name] = prob
        oof_lbl[name] = np.array([PHASES[i] for i in prob.argmax(1)])
        timing[name] = time.time() - t0
        print(f"\r  {name:11s} done in {timing[name]:6.1f}s" + " " * 12)
        # Save NOW. A model that took ten minutes is never lost to a later
        # model's crash.
        dump(a.out, rows, sorted(oof_prob), oof_lbl, oof_prob)

    names = [n for n in names if n in oof_prob]
    if not names:
        sys.exit("no model completed")

    # ---------------------------------------------------------- per-frame
    print("\n" + "=" * 78)
    print("PER-FRAME METRICS  (not the headline -- see event scores below)")
    print("per-class columns are F1")
    print("=" * 78)
    SHORT = {"run_up": "runup", "bfc_phase": "bfc", "ffc_phase": "ffc",
             "follow_through": "follow", "none": "none"}
    hdr = f"{'model':<12}{'acc':>7}{'bal acc':>9}{'macro F1':>10}"
    hdr += "".join(f"{SHORT[p]:>9}" for p in PHASES)
    hdr += f"{'secs':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        p = oof_lbl[name]
        per = f1_score(y, p, labels=PHASES, average=None, zero_division=0)
        line = (f"{name:<12}{(p == y).mean():>7.3f}"
                f"{balanced_accuracy_score(y, p):>9.3f}"
                f"{f1_score(y, p, average='macro', zero_division=0):>10.3f}")
        line += "".join(f"{v:>9.3f}" for v in per)
        line += f"{timing[name]:>8.1f}"
        print(line)

    # per-bowler spread -- a single mean hides a model that fails on one person
    print(f"\n{'model':<12}{'per-bowler acc: mean':>22}{'sd':>8}{'worst':>8}")
    print("-" * 50)
    for name in names:
        accs = [float((oof_lbl[name][g == b] == y[g == b]).mean())
                for b in bowlers]
        print(f"{name:<12}{np.mean(accs):>22.3f}{np.std(accs):>8.3f}"
              f"{min(accs):>8.3f}")

    print(f"\nwritten: {a.out}")
    if failed:
        print("\nmodels that failed (everything else is in the file):")
        for n, why in failed:
            print(f"  {n:11s} {why[:90]}")

    print("""
NOW SCORE THEM ON EVENTS -- this is the ranking that counts.
Per-frame accuracy rewards getting block MIDDLES right; your measurements only
need the block EDGES. Run each model through the event extractor:""")
    for n in names:
        print(f"    py -3.13 extract_events.py {a.out} --model {n}")
    print("""
Then build the report table from the 'within 3 frames' rates for release, FFC
and BFC. If a model wins on per-frame accuracy but loses on events, say so --
that is a result about the metric, and it is more interesting than the ranking.""")


if __name__ == "__main__":
    main()
