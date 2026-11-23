"""
Run MediaPipe pose estimation on the frames listed in frame_labels.csv.

Reads the videos directly -- no intermediate JPEGs. For each video it decodes
once, runs pose on the frames you labelled, and writes one row per frame with
33 landmarks (x, y, visibility) in image-normalised coordinates.

    pip install mediapipe opencv-python

    python run_pose.py frame_labels.csv . --out keypoints.csv
    python run_pose.py frame_labels.csv . --out keypoints.csv --limit 3 --draw 6
    python run_pose.py frame_labels.csv . --out keypoints.csv --mode image

Resumable: if --out already exists, videos already in it are skipped. Ctrl-C is
safe; rerun the same command to continue.

--mode video (default) tracks the pose across frames within a clip: faster and
steadier, but it can lock onto the wrong person if someone else is in shot.
--mode image detects independently on every frame: slower, no drift.

Coordinates are normalised to the IMAGE (0-1 across width/height), not to the
bowler. Normalising to the bowler happens in the next script. width/height are
stored so pixel coordinates can be recovered.
"""

import argparse
import csv
import os
import time
from collections import defaultdict, Counter

import cv2

N_LM = 33
MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


# ---------------------------------------------------------------- pose backend

class LegacyPose:
    """mediapipe <= 0.10.x solutions API."""

    def __init__(self, complexity, mode):
        import mediapipe as mp
        self.mp = mp
        self.complexity = complexity
        self.static = (mode == "image")
        self.pose = None
        self.restart()

    def restart(self):
        if self.pose is not None:
            self.pose.close()
        self.pose = self.mp.solutions.pose.Pose(
            static_image_mode=self.static,
            model_complexity=self.complexity,
            smooth_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, rgb):
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        return [(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark]

    def close(self):
        self.pose.close()


class TasksPose:
    """mediapipe tasks API (1.x and newer 0.10.x builds without .solutions)."""

    def __init__(self, complexity, mode, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.vision = vision
        self.mp_python = mp_python
        self.model_path = model_path
        self.image_mode = (mode == "image")
        self.lm = None
        self.ts = 0
        self.restart()

    def restart(self):
        if self.lm is not None:
            self.lm.close()
        rm = self.vision.RunningMode.IMAGE if self.image_mode else self.vision.RunningMode.VIDEO
        opts = self.vision.PoseLandmarkerOptions(
            base_options=self.mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=rm,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = self.vision.PoseLandmarker.create_from_options(opts)
        self.ts = 0

    def __call__(self, rgb):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.image_mode:
            res = self.lm.detect(img)
        else:
            self.ts += 33          # must be strictly increasing within a clip
            res = self.lm.detect_for_video(img, self.ts)
        if not res.pose_landmarks:
            return None
        return [(l.x, l.y, getattr(l, "visibility", 1.0)) for l in res.pose_landmarks[0]]

    def close(self):
        self.lm.close()


def ensure_model(complexity, cache_dir="models"):
    os.makedirs(cache_dir, exist_ok=True)
    url = MODEL_URLS[complexity]
    path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print("downloading pose model (one time) ...", flush=True)
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path


def make_pose(complexity, mode):
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        print(f"mediapipe solutions API, complexity {complexity}, mode {mode}")
        return LegacyPose(complexity, mode)
    print(f"mediapipe tasks API, complexity {complexity}, mode {mode}")
    return TasksPose(complexity, mode, ensure_model(complexity))


# ---------------------------------------------------------------- skeleton draw

EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
         (23,25),(25,27),(24,26),(26,28),(27,31),(28,32),(0,11),(0,12)]


def draw_skeleton(img, lms):
    h, w = img.shape[:2]
    pts = [(int(x*w), int(y*h)) for x, y, _ in lms]
    for a, b in EDGES:
        cv2.line(img, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y, v) in enumerate(lms):
        col = (0, 255, 0) if v > 0.5 else (0, 0, 255)
        cv2.circle(img, pts[i], 4, col, -1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------- main

def header():
    cols = ["video", "bowler", "frame", "phase",
            "is_bfc_event", "is_ffc_event", "is_release_event",
            "width", "height", "detected"]
    for i in range(N_LM):
        cols += [f"x{i}", f"y{i}", f"v{i}"]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_labels")
    ap.add_argument("video_root")
    ap.add_argument("--out", default="keypoints.csv")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--mode", default="video", choices=["video", "image"])
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--draw", type=int, default=0,
                    help="annotated frames per bowler saved into pose_check/")
    args = ap.parse_args()

    wanted = defaultdict(dict)
    with open(args.frame_labels, newline="") as fh:
        for r in csv.DictReader(fh):
            wanted[r["video"]][int(r["frame"])] = r

    index = {}
    for root, _, files in os.walk(args.video_root):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".m4v")):
                index.setdefault(f, os.path.join(root, f))

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add(r["video"])
        print(f"{len(done)} videos already done, skipping")

    todo = [v for v in sorted(wanted) if v not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return

    pose = make_pose(args.complexity, args.mode)
    drawn = defaultdict(int)
    if args.draw:
        os.makedirs("pose_check", exist_ok=True)

    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(header())

    t0 = time.time()
    tot = 0
    miss_by_phase = Counter()
    seen_by_phase = Counter()
    worst = []

    for vi, name in enumerate(todo, 1):
        path = index.get(name)
        if not path:
            print(f"  [{vi}/{len(todo)}] {name}: VIDEO NOT FOUND")
            continue

        keep = wanted[name]
        last = max(keep)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  [{vi}/{len(todo)}] {name}: could not open")
            continue

        pose.restart()          # fresh tracker + timestamp clock per clip

        i = n = missed = 0
        rows = []
        while i <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if i in keep:
                H, W = fr.shape[:2]
                if W > args.max_width:
                    s = args.max_width / W
                    small = cv2.resize(fr, (args.max_width, int(H * s)))
                else:
                    small = fr
                lms = pose(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

                r = keep[i]
                seen_by_phase[r["phase"]] += 1
                row = [r["video"], r["bowler"], i, r["phase"],
                       r["is_bfc_event"], r["is_ffc_event"], r["is_release_event"],
                       W, H, 1 if lms else 0]
                if lms:
                    for x, y, v in lms:
                        row += [round(x, 5), round(y, 5), round(v, 3)]
                else:
                    row += [""] * (N_LM * 3)
                    missed += 1
                    miss_by_phase[r["phase"]] += 1
                rows.append(row)
                n += 1

                if lms and args.draw and drawn[r["bowler"]] < args.draw \
                        and r["phase"] in ("bfc_phase", "ffc_phase"):
                    cv2.imwrite(
                        os.path.join("pose_check",
                                     f"{r['bowler']}_{os.path.splitext(name)[0]}_{i:05d}.jpg"),
                        draw_skeleton(small.copy(), lms),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
                    drawn[r["bowler"]] += 1
            i += 1
        cap.release()

        w.writerows(rows)
        fh.flush()
        tot += n
        if missed:
            worst.append((missed / max(n, 1), name, missed, n))
        el = time.time() - t0
        rate = tot / el if el else 0
        remain = sum(len(wanted[v]) for v in todo[vi:])
        eta = remain / rate / 60 if rate else 0
        flag = f"  missed {missed}" if missed else ""
        print(f"  [{vi}/{len(todo)}] {name}: {n}{flag}   "
              f"{rate:.1f} fps, ~{eta:.0f} min left", flush=True)

    fh.close()
    pose.close()

    print(f"\n{tot} frames written to {args.out}")
    tm = sum(miss_by_phase.values())
    print(f"no pose detected in {tm} frames ({100*tm/max(tot,1):.1f}%)")
    if tm:
        print("\n  phase            seen   missed    %")
        for ph in ["run_up", "bfc_phase", "ffc_phase", "follow_through", "none"]:
            s, m = seen_by_phase[ph], miss_by_phase[ph]
            if s:
                print(f"  {ph:16s} {s:5d} {m:7d}  {100*m/s:5.1f}")
        worst.sort(reverse=True)
        print("\n  worst clips:")
        for frac, nm, m, n in worst[:8]:
            print(f"    {nm}: {m}/{n} ({100*frac:.0f}%)")
    if args.draw:
        print("\nskeleton images in pose_check/ -- look at these before going further")


if __name__ == "__main__":
    main()
