"""Python's own results on a few clips, for checking the browser port.

    py -3.13 webtest/dump_reference.py

Writes webtest/ref/<clip>.json: the raw MediaPipe landmarks, the video size
and fps, the phase probabilities, the detected events and every measurement.
The JavaScript port is fed the same landmarks and must reproduce the rest.
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import analyze

CLIPS = ["videos/Asish/VID20260919081610.mp4", "videos/Asish/VID20260919081519.mp4",
         "videos/Hemanth/*", "videos/Junior4/*", "videos/Anirudh/*", "videos/Srijay/*"]
os.makedirs(os.path.join(HERE, "ref"), exist_ok=True)
for pat in CLIPS:
    path = sorted(glob.glob(pat))[0]
    st, err = analyze.prepare(path, "models/phase_model.joblib")
    name = os.path.splitext(os.path.basename(path))[0]
    out = {"video": path, "error": err}
    if st is not None:
        res = analyze.measure_at(st, st["auto_events"])
        out.update(
            W=st["W"], H=st["H"], fps=st["fps"],
            poses={str(f): P.tolist() for f, P in st["poses"].items()},
            frames=st["frames"], P=np.asarray(st["P"]).tolist(),
            events=st["auto_events"], result=res)
    with open(os.path.join(HERE, "ref", name + ".json"), "w") as fh:
        json.dump(out, fh)
    print(name, "ok" if st is not None else err, flush=True)
