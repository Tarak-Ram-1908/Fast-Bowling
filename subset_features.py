"""
Filter features.csv by bowler, so an exclusion is a reported choice rather than
a deleted file.

    py -3.13 subset_features.py features.csv --exclude Abhiram --out feat_noab.csv
    py -3.13 subset_features.py features.csv --only Charan,Jagan --out feat_oct.csv

Nothing is destroyed: features.csv is untouched and you can always rebuild the
full set. Whatever you exclude, say so in the report and give the number both
ways -- an excluded participant is a scope statement, not a secret.
"""
import argparse, csv, collections, sys

ap = argparse.ArgumentParser()
ap.add_argument("features")
ap.add_argument("--out", required=True)
ap.add_argument("--exclude", default="")
ap.add_argument("--only", default="")
a = ap.parse_args()

if a.exclude and a.only:
    sys.exit("use --exclude or --only, not both")

rows = list(csv.DictReader(open(a.features, newline="")))
if not rows:
    sys.exit("empty input")
allb = sorted({r["bowler"] for r in rows})

ex = {s.strip() for s in a.exclude.split(",") if s.strip()}
on = {s.strip() for s in a.only.split(",") if s.strip()}
unknown = (ex | on) - set(allb)
if unknown:
    sys.exit(f"not in {a.features}: {sorted(unknown)}\navailable: {allb}")

keep = [r for r in rows if (r["bowler"] in on) if on] if on else \
       [r for r in rows if r["bowler"] not in ex]

with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(keep)

kb = sorted({r["bowler"] for r in keep})
print(f"in : {len(rows):6d} frames, {len(allb)} bowlers")
print(f"out: {len(keep):6d} frames, {len(kb)} bowlers -> {a.out}")
print(f"dropped: {sorted(set(allb) - set(kb)) or 'none'}")
c = collections.Counter(r["phase"] for r in keep)
print("class balance:", dict(c))
print(f"\nnext:  py -3.13 train_classifier.py {a.out} --folds {len(kb)} --out predictions.csv")
