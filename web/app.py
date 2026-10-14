"""
Delivery Stride -- upload one side-on video, get the front-leg numbers back.

    pip install flask
    py -3.13 web/app.py
    open http://127.0.0.1:5000

Analysis runs in a worker thread so the page can show progress; MediaPipe takes
roughly 20-40 s on a 3-second 4K clip. Jobs live in memory and vanish on
restart, which is the right scope for a demo.
"""
import os, sys, threading, uuid, traceback
from flask import Flask, request, jsonify, render_template, abort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                 # analyze.py and measure_deliveries.py
import analyze                           # noqa: E402

UPLOADS = os.path.join(HERE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True          # pick up template edits
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0          # never cache static files
ASSET_V = str(int(os.path.getmtime(os.path.join(HERE, "static", "style.css"))))
JOBS = {}
OK_EXT = {".mp4", ".mov", ".m4v", ".avi"}


def work(job_id, path):
    job = JOBS[job_id]
    try:
        def prog(frac):
            job["progress"] = 5 + int(frac * 80)
        job["stage"] = "Reading the delivery"
        res = analyze.analyse(path, os.path.join(ROOT, "models", "phase_model.joblib"),
                              progress=prog)
        job["progress"] = 95
        job["stage"] = "Measuring"
        job["result"] = res
        job["state"] = "done" if res.get("ok") else "failed"
        if not res.get("ok"):
            job["error"] = res.get("error", "analysis failed")
        job["progress"] = 100
    except Exception:
        job["state"] = "failed"
        job["error"] = "Something broke while analysing the clip."
        job["trace"] = traceback.format_exc()
        job["progress"] = 100
    finally:
        try: os.remove(path)
        except OSError: pass


@app.route("/")
def index():
    ready = os.path.exists(os.path.join(ROOT, "models", "phase_model.joblib"))
    return render_template("index.html", model_ready=ready, v=ASSET_V)


@app.post("/analyse")
def start():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="Choose a video first."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in OK_EXT:
        return jsonify(error=f"{ext or 'That file type'} will not open. "
                             "Use MP4 or MOV."), 400
    if not os.path.exists(os.path.join(ROOT, "models", "phase_model.joblib")):
        return jsonify(error="The model is missing. Run train_production.py."), 503

    jid = uuid.uuid4().hex[:12]
    path = os.path.join(UPLOADS, jid + ext)
    f.save(path)
    JOBS[jid] = {"state": "running", "progress": 2,
                 "stage": "Opening the file", "name": f.filename}
    threading.Thread(target=work, args=(jid, path), daemon=True).start()
    return jsonify(job=jid)


@app.get("/progress/<jid>")
def progress(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify(error="unknown job"), 404
    return jsonify(state=j["state"], progress=j["progress"],
                   stage=j.get("stage", ""), error=j.get("error"))


@app.get("/report/<jid>")
def report(jid):
    j = JOBS.get(jid)
    if not j or j["state"] != "done":
        abort(404)
    return render_template("report.html", r=j["result"], name=j["name"], v=ASSET_V)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(ROOT, "models", "phase_model.joblib")):
        print("WARNING: models/phase_model.joblib not found.")
        print("Run:  py -3.13 train_production.py feat_noab_win.csv\n")
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(debug=False, host=host, port=port)
