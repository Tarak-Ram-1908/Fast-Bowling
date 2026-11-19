# Delivery Stride

Fast bowling technique measured from a single side-on phone video. Upload one
delivery and get twelve kinematic measurements back, each compared against the
elite reference values in Worthington (2010).

Built as a data science course project at IIITDM Kancheepuram.

**Try it: https://tarak-ram-1908.github.io/Fast-Bowling/** — the whole
analysis runs in your browser, so your video never leaves your device.

## How it works

MediaPipe gives 33 body landmarks per frame. Sixteen per-frame features are
computed from those, then stacked across an eleven-frame window so the model can
see motion rather than only position. A random forest labels each frame with
the phase of the delivery it belongs to, the phase blocks give the three event
instants — back foot contact, front foot contact, ball release — and the twelve
measurements are geometry at those frames.

## Accuracy

Validated leave-one-bowler-out across 24 bowlers, so every number below comes
from a model that never saw that bowler during training. Rates are shares of
all 268 deliveries: a clip that yields no phase block counts as a failure, not
as a missing value.

| | within 3 frames of a hand mark |
|---|---|
| ball release | 98.5% |
| front foot contact | 91.8% |
| back foot contact | 59.0% |

Mean absolute timing error is 0.76 frames (6.3 ms) for ball release and 1.53
frames (12.7 ms) for front foot contact.

Plant angle, measured to Worthington's own definition, reproduces the
laboratory value almost exactly: 36.6° against the elite 36.3 ± 3.9.

## Reading a single delivery

The numbers above are validated for the cohort. One clip is a much weaker
measurement, mostly because back foot contact is right only 59% of the time,
so the report guards each delivery:

- **Differences are shown on the body.** Every value more than 1 sd from the
  elite mean is drawn on the frame it was measured at: the measured segments
  in red, and where the elite average would put them dashed in green.
- **The chosen frames are shown.** Each event frame is drawn with its skeleton
  and can be stepped ±1 or ±5 frames and re-measured, or confirmed as right.
- **Plausibility check.** Every value is also scored against our own 262
  deliveries (`models/cohort_stats.json`, rebuilt with
  `py -3.13 analyze.py --build-cohort measurements.csv`). Past 3 cohort sd it
  is reported as a probable detection error, not a finding; two or more values
  past 2 sd together raise a warning naming the likely mis-detected event.
- **Confidence per measurement**, inherited from the event frames it depends
  on: back foot contact starts low, and phase fragmentation, an unsure
  classifier edge, pose drop-out or hidden legs lower it further.
- **Several deliveries are averaged.** Upload up to twelve clips at once; the
  report leads with their mean and spread and leaves implausible values out.
- **Explanations, not prescriptions.** Values more than 1 sd from elite get a
  rule-based sentence quoting the published association, stated as an
  association across bowlers, never as advice on what to change.

## Running it

### The website (GitHub Pages)

`docs/` is a static site that runs the whole pipeline in the browser:
MediaPipe's browser build (the same pose model, `@mediapipe/tasks-vision`
1.0.1 to match Python's `mediapipe` 1.0.1) finds the body in every frame,
and `docs/js/core.js` is a line-for-line JavaScript port of `analyze.py` and
`measure_deliveries.py` — features, the 300-tree forest, event extraction,
measurements, plausibility, confidence and explanations. GitHub Pages serves
it from the `docs/` folder of `main`; there is no server, and videos stay on
the visitor's device.

After changing the model, the analysis or the report text, rebuild and check:

    py -3.13 export_web.py               # forest, reference values, styles -> docs/
    py -3.13 webtest/dump_reference.py   # Python's results on six clips
    node webtest/check_port.mjs          # the port must reproduce them

**How closely the website matches Python.** Fed the same landmarks, the port
reproduces Python exactly: the forest's probabilities agree to 3e-9 over
3,000 rows, and on six clips from five bowlers every event, measurement
(within Python's 0.1 rounding), confidence level, warning and explanation is
identical. Run end to end in Chrome on three of those clips, with the
browser decoding the video and running MediaPipe itself, all nine events
matched Python's frame for frame and landmarks differed by 1–2 pixels on
average. Most measurements agreed within 1–2°; the exceptions are values at
blurred or tiny body parts — wrist height at release (up to 6.5% of height)
and foot strike angle (up to 10.5°) — where small differences in image decoding
move MediaPipe's point. Treat those two as the least repeatable, in either
version.

A clip took 40–105 seconds in testing on a laptop (MediaPipe runs on the GPU
through WebGL where available, otherwise on the CPU) and longer on a phone.

### The Flask version (local, or any Python host)

    pip install -r requirements.txt
    py -3.13 web/app.py          # http://127.0.0.1:5000

Analyses run one at a time through a queue, so visitors see their place in
line. Clips are limited to 12 seconds and six per upload. Uploaded videos are
deleted once analysed, and reports an hour after they finish. The
`Dockerfile` runs it on any container host, and `deploy_space.sh` publishes it
to a Hugging Face Space, though CPU Spaces now need a paid account.

## What it does not do

It does not assess whether the bowling action is legal — elbow extension past
the permitted 15° cannot be judged from one side-on view. It says nothing about
injury risk: shoulder counter-rotation, the strongest published link between
technique and lower-back injury, is a transverse-plane measurement needing a
second camera behind the bowler.

## Data

25 bowlers, 304 deliveries, filmed at 120 fps over two sessions in September and
October 2026. Every frame-level phase label, event instant and legality
judgement was annotated by hand.
