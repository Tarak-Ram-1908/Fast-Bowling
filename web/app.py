"""
Delivery Stride -- upload side-on videos, get the front-leg numbers back.

    pip install flask
    py -3.13 web/app.py
    open http://127.0.0.1:5000

Analysis runs in a worker thread so the page can show progress; MediaPipe takes
roughly 20-40 s on a 3-second 4K clip. Jobs live in memory and vanish on
restart, which is the right scope for a demo.

A job is one or more deliveries. Each keeps the pose state and a strip of
frames around the detected events, so the report can show the frames the
system chose and the user can nudge them and re-measure without re-running
MediaPipe. With several deliveries the report leads with their mean.
"""
import os, sys, threading, uuid, traceback
from flask import (Flask, request, jsonify, render_template, abort, Response)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                 # analyze.py and measure_deliveries.py
import analyze                           # noqa: E402

UPLOADS = os.path.join(HERE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)
MODEL = os.path.join(ROOT, "models", "phase_model.joblib")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1200 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True          # pick up template edits
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0          # never cache static files
ASSET_V = str(int(os.path.getmtime(os.path.join(HERE, "static", "style.css"))))
JOBS = {}
OK_EXT = {".mp4", ".mov", ".m4v", ".avi"}
MAX_FILES = 12


def work(job_id, items):
    job = JOBS[job_id]
    n = len(items)
    for k, (name, path) in enumerate(items):
        d = job["deliveries"][k]
        try:
            def prog(frac, k=k):
                job["progress"] = 3 + int((k + frac * 0.9) / n * 94)
            job["stage"] = (f"Reading delivery {k + 1} of {n}" if n > 1
                            else "Reading the delivery")
            st, err = analyze.prepare(path, MODEL, progress=prog)
            if st is None:
                d.update(state="failed", error=err)
                continue
            job["stage"] = "Measuring"
            d["st"] = st
            d["thumbs"] = analyze.grab_frames(path, st)
            d["result"] = analyze.measure_at(st, st["auto_events"])
            d["state"] = "done" if d["result"].get("ok") else "failed"
            if not d["result"].get("ok"):
                d["error"] = d["result"].get("error", "analysis failed")
        except Exception:
            d.update(state="failed", error="Something broke while analysing "
                     "this clip.", trace=traceback.format_exc())
        finally:
            try: os.remove(path)
            except OSError: pass
    ok = [d for d in job["deliveries"] if d["state"] == "done"]
    job["state"] = "done" if ok else "failed"
    if not ok:
        job["error"] = job["deliveries"][0].get("error", "analysis failed")
    job["progress"] = 100


@app.route("/")
def index():
    return render_template("index.html", model_ready=os.path.exists(MODEL),
                           v=ASSET_V, max_files=MAX_FILES)


@app.post("/analyse")
def start():
    files = [f for f in request.files.getlist("video") if f and f.filename]
    if not files:
        return jsonify(error="Choose a video first."), 400
    if len(files) > MAX_FILES:
        return jsonify(error=f"Up to {MAX_FILES} deliveries at a time."), 400
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in OK_EXT:
            return jsonify(error=f"{f.filename}: {ext or 'that file type'} "
                                 "will not open. Use MP4 or MOV."), 400
    if not os.path.exists(MODEL):
        return jsonify(error="The model is missing. Run train_production.py."), 503

    jid = uuid.uuid4().hex[:12]
    items = []
    for k, f in enumerate(files):
        path = os.path.join(UPLOADS, f"{jid}_{k}"
                            + os.path.splitext(f.filename)[1].lower())
        f.save(path)
        items.append((f.filename, path))
    JOBS[jid] = {"state": "running", "progress": 2, "stage": "Opening the files",
                 "deliveries": [{"name": nm, "state": "queued"}
                                for nm, _ in items]}
    threading.Thread(target=work, args=(jid, items), daemon=True).start()
    return jsonify(job=jid)


@app.get("/progress/<jid>")
def progress(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify(error="unknown job"), 404
    return jsonify(state=j["state"], progress=j["progress"],
                   stage=j.get("stage", ""), error=j.get("error"))


def delivery(jid, k):
    j = JOBS.get(jid)
    if not j or j["state"] != "done" or not 0 <= k < len(j["deliveries"]):
        abort(404)
    d = j["deliveries"][k]
    if d["state"] != "done":
        abort(404)
    return j, d


@app.get("/frame/<jid>/<int:k>/<int:n>.jpg")
def frame(jid, k, n):
    _, d = delivery(jid, k)
    b = d["thumbs"].get(n)
    if b is None:
        abort(404)
    return Response(b, mimetype="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.post("/events/<jid>/<int:k>")
def set_events(jid, k):
    """Re-measure one delivery at frames the user picked or confirmed."""
    _, d = delivery(jid, k)
    body = request.get_json(silent=True) or {}
    try:
        ev = {e: int(body[e]) for e in ("bfc", "ffc", "release")}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Send bfc, ffc and release frame numbers."), 400
    if body.get("reset"):
        ev = dict(d["st"]["auto_events"])
    lo, hi = min(d["thumbs"]), max(d["thumbs"])
    if not all(lo <= v <= hi for v in ev.values()):
        return jsonify(error=f"Frames must lie between {lo} and {hi}."), 400
    res = analyze.measure_at(d["st"], ev, checked=bool(body.get("checked"))
                             and not body.get("reset"))
    if not res.get("ok"):
        return jsonify(error=res.get("error", "could not measure")), 400
    d["result"] = res
    return jsonify(ok=True)


@app.get("/report/<jid>")
def report(jid):
    j = JOBS.get(jid)
    if not j or j["state"] != "done":
        abort(404)
    done = [(k, d) for k, d in enumerate(j["deliveries"]) if d["state"] == "done"]
    failed = [d for d in j["deliveries"] if d["state"] != "done"]
    summary = (analyze.summarise([d["result"] for _, d in done])
               if len(done) > 1 else None)
    strips = {k: {"lo": min(d["thumbs"]), "hi": max(d["thumbs"])}
              for k, d in done}
    return render_template("report.html", jid=jid, done=done, failed=failed,
                           summary=summary, strips=strips,
                           event_name=analyze.EVENT_NAME,
                           hit_rate=analyze.EVENT_HIT_RATE, v=ASSET_V)


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        print("WARNING: models/phase_model.joblib not found.")
        print("Run:  py -3.13 train_production.py feat_noab_win.csv\n")
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(debug=False, host=host, port=port)
