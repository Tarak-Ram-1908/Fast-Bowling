# Delivery Stride

Fast bowling technique measured from a single side-on phone video. Upload one
delivery and get twelve kinematic measurements back, each compared against the
elite reference values in Worthington (2010).

Built as a data science course project at IIITDM Kancheepuram.

## How it works

MediaPipe gives 33 body landmarks per frame. Sixteen per-frame features are
computed from those, then stacked across an eleven-frame window so the model can
see motion rather than only position. A random forest labels each frame with
the phase of the delivery it belongs to, the phase blocks give the three event
instants â€” back foot contact, front foot contact, ball release â€” and the twelve
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

Two measurements reproduce laboratory values almost exactly: plant angle
36.6Â° against the elite 36.3 Â± 3.9, and front knee at foot contact 167.8Â°
against 164.1 Â± 6.1.

## Reading a single delivery

The numbers above are validated for the cohort. One clip is a much weaker
measurement, mostly because back foot contact is right only 59% of the time,
so the report guards each delivery:

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

## What it does not do

It does not assess whether the bowling action is legal â€” elbow extension past
the permitted 15Â° cannot be judged from one side-on view. It says nothing about
injury risk: shoulder counter-rotation, the strongest published link between
technique and lower-back injury, is a transverse-plane measurement needing a
second camera behind the bowler.

## Data

25 bowlers, 304 deliveries, filmed at 120 fps over two sessions in September and
October 2026. Every frame-level phase label, event instant and legality
judgement was annotated by hand.
