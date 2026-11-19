"""
Delivery Stride -- upload side-on videos, get the front-leg numbers back.

    pip install flask
    py -3.13 web/app.py
    open http://127.0.0.1:5000

Analysis runs on ONE worker thread fed by a queue, so many visitors at once
wait their turn instead of each starting a MediaPipe graph and starving the
CPU; the page shows their place in line. MediaPipe takes roughly 20-40 s on a
3-second 4K clip. Jobs live in memory, are only reachable through their
unguessable link, and are deleted JOB_TTL seconds after they finish.

A job is one or more deliveries. Each keeps the pose state and a strip of
frames around the detected events, so the report can show the frames the
system chose and the user can nudge them and re-measure without re-running
MediaPipe. With several deliveries the report leads with their mean.
"""
import os, sys, threading, uuid, traceback, time, queue
from flask import (Flask, request, jsonify, render_template, abort, Response)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                 # analyze.py and measure_deliveries.py
import analyze                           # noqa: E402

UPLOADS = os.path.join(HERE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)
MODEL = os.path.join(ROOT, "models", "phase_model.joblib")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True          # pick up template edits
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0          # never cache static files
ASSET_V = str(int(os.path.getmtime(os.path.join(HERE, "static", "style.css"))))
JOBS = {}
OK_EXT = {".mp4", ".mov", ".m4v", ".avi"}
MAX_FILES = 6           # deliveries per upload
MAX_SECONDS = 12        # per clip; one delivery needs 3-5 s
MAX_WAITING = 15        # uploads queued before new ones are turned away
JOB_TTL = 60 * 60       # results kept this long after they finish
MAX_JOBS = 40           # and never more than this many in memory
WORK = queue.Queue()
WAITING = []            # job ids in queue order, for "N ahead of you"
LOCK = threading.Lock()


def purge():
    """Drop finished jobs past their lifetime, and the oldest beyond MAX_JOBS."""
    now = time.time()
    with LOCK:
        done = sorted((j.get("finished", now), jid) for jid, j in JOBS.items()
                      if j["state"] in ("done", "failed"))
        for fin, jid in done:
            if now - fin > JOB_TTL:
                JOBS.pop(jid, None)
        extra = len(JOBS) - MAX_JOBS
        for fin, jid in done:
            if extra <= 0:
                break
            if jid in JOBS:
                JOBS.pop(jid); extra -= 1


def worker():
    while True:
        jid, items = WORK.get()
        with LOCK:
            if jid in WAITING:
                WAITING.remove(jid)
        if jid in JOBS:
            JOBS[jid]["state"] = "running"
            work(jid, items)
            JOBS[jid]["finished"] = time.time()
        else:                                   # expired while queued
            for _, path in items:
                try: os.remove(path)
                except OSError: pass
        WORK.task_done()


def clip_seconds(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return n / fps if fps > 0 else None


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
    purge()
    if len(WAITING) >= MAX_WAITING:
        return jsonify(error="The site is busy right now. Try again in a few "
                             "minutes."), 503

    jid = uuid.uuid4().hex                # the link is the only key to it
    items = []

    def discard():
        for _, pth in items:
            try: os.remove(pth)
            except OSError: pass

    for k, f in enumerate(files):
        path = os.path.join(UPLOADS, f"{jid}_{k}"
                            + os.path.splitext(f.filename)[1].lower())
        f.save(path)
        items.append((f.filename, path))
        secs = clip_seconds(path)
        if secs is None:
            discard()
            return jsonify(error=f"{f.filename} could not be read as a video."), 400
        if secs > MAX_SECONDS:
            discard()
            return jsonify(error=f"{f.filename} is {secs:.0f} s long. Trim it to "
                                 f"one delivery, under {MAX_SECONDS} s."), 400
    JOBS[jid] = {"state": "queued", "progress": 1, "stage": "Waiting",
                 "created": time.time(),
                 "deliveries": [{"name": nm, "state": "queued"}
                                for nm, _ in items]}
    with LOCK:
        WAITING.append(jid)
    WORK.put((jid, items))
    return jsonify(job=jid)


@app.get("/progress/<jid>")
def progress(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify(error="unknown job"), 404
    stage = j.get("stage", "")
    if j["state"] == "queued":
        with LOCK:
            ahead = WAITING.index(jid) if jid in WAITING else 0
        if any(x["state"] == "running" for x in list(JOBS.values())):
            ahead += 1                   # the one being analysed now
        stage = ("Starting" if ahead == 0 else
                 f"Waiting for {ahead} analysis ahead of yours" if ahead == 1
                 else f"Waiting for {ahead} analyses ahead of yours")
    return jsonify(state="running" if j["state"] == "queued" else j["state"],
                   progress=j["progress"], stage=stage, error=j.get("error"))


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
    return Response(analyze.render_frame(d["st"], b, n), mimetype="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.get("/focus/<jid>/<int:k>/<key>.jpg")
def focus(jid, k, key):
    """The frame one measurement was taken at, drawn against the elite."""
    _, d = delivery(jid, k)
    row = d["result"]["rows"].get(key)
    if not row or row.get("focus_frame") is None:
        abort(404)
    f = row["focus_frame"]
    b = d["thumbs"].get(f)
    if b is None:
        abort(404)
    return Response(analyze.render_focus(d["st"], b, f, key, row),
                    mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


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
    # for the averaged view, show each value far from elite on the delivery
    # whose own value is closest to the bowler's mean -- a typical example
    sfocus = []
    if summary:
        for key, srow in sorted(summary.items(),
                                key=lambda kv: -abs(kv[1].get("z") or 0)):
            if srow.get("z") is None or abs(srow["z"]) <= 1:
                continue
            cands = [(abs(d["result"]["rows"][key]["value"] - srow["value"]), k)
                     for k, d in done if key in d["result"]["rows"]
                     and d["result"]["rows"][key]["trusted"]
                     and d["result"]["rows"][key].get("focus_frame") is not None]
            if cands:
                sfocus.append((key, min(cands)[1]))
    return render_template("report.html", jid=jid, done=done, failed=failed,
                           summary=summary, strips=strips, sfocus=sfocus,
                           focus_text=analyze.FOCUS_TEXT,
                           done_map={k: d for k, d in done},
                           event_name=analyze.EVENT_NAME,
                           hit_rate=analyze.EVENT_HIT_RATE, v=ASSET_V)


threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        print("WARNING: models/phase_model.joblib not found.")
        print("Run:  py -3.13 train_production.py feat_noab_win.csv\n")
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(debug=False, host=host, port=port)
