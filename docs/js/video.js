// Reading every frame of a phone video in the browser, in order, the way
// OpenCV reads it in the Python version.
//
// A <video> element can show any frame, but it only takes times, not frame
// numbers, and it does not report the frame rate. mp4box.js reads the file's
// own sample table, which gives each frame's presentation time; the page then
// seeks to the middle of each frame in turn. That works for any codec the
// browser can play (H.264 and, on most devices, HEVC).
import { createFile } from "https://cdn.jsdelivr.net/npm/mp4box@2.4.1/+esm";

// { times: [seconds at the start of each frame, presentation order],
//   frameDur: seconds, fps, n }
export async function frameTiming(file) {
  try {
    const buf = await file.arrayBuffer();
    buf.fileStart = 0;
    const mp = createFile();
    const info = await new Promise((resolve, reject) => {
      mp.onReady = resolve;
      mp.onError = e => reject(new Error(String(e)));
      mp.appendBuffer(buf);
      mp.flush();
    });
    const vt = info.videoTracks && info.videoTracks[0];
    if (!vt) throw new Error("no video track");
    const trak = mp.getTrackById(vt.id);
    const ts = vt.timescale;
    const cts = (trak.samples || []).map(s => s.cts).sort((a, b) => a - b);
    if (!cts.length) throw new Error("no samples");
    // an edit list can start playback part-way into the media timeline
    let offset = 0;
    const entries = trak.edts && trak.edts.elst && trak.edts.elst.entries;
    if (entries) for (const e of entries) if (e.media_time >= 0) { offset = e.media_time; break; }
    const times = cts.map(c => (c - offset) / ts);
    const n = times.length;
    const span = n > 1 ? (times[n - 1] - times[0]) / (n - 1) : 1 / 30;
    return { times, frameDur: span, fps: snapFps(1 / span), n };
  } catch (err) {
    console.warn("mp4box could not read the file, falling back to its duration", err);
    return null;          // caller falls back to duration and an assumed rate
  }
}

// Phone frame rates are nominal (120, 240, 60, 30); the measured average is a
// hair off. OpenCV reports the nominal one, and MediaPipe's timestamps are
// built from it, so snap when within 0.5%.
function snapFps(f) {
  for (const nominal of [24, 25, 30, 48, 50, 60, 90, 100, 120, 240, 480])
    if (Math.abs(f - nominal) / nominal < 0.005) return nominal;
  return f;
}

export function fallbackTiming(duration, fps = 120) {
  const n = Math.floor(duration * fps);
  return { times: Array.from({ length: n }, (_, i) => i / fps), frameDur: 1 / fps, fps, n };
}

// Where, in the <video> element's own timeline, each frame actually starts.
// Browsers apply a file's edit list differently (this project's Android clips
// open with a 47 ms empty edit, which Chrome turns into a 5.3 ms shift), so
// rather than assume, step through the first dozen frames in small
// increments, note the times the picture changes, and take the median phase
// of those changes relative to the nominal frame grid. The caller then seeks
// to times[i] + phase + half a frame: the middle of frame i as displayed.
export async function displayPhase(video, T, frames = 12) {
  const c = new OffscreenCanvas(160, 90), x = c.getContext("2d", { willReadFrequently: true });
  const sig = () => {
    x.drawImage(video, 0, 0, 160, 90);
    const d = x.getImageData(0, 0, 160, 90).data;
    let h = 0;
    for (let i = 0; i < d.length; i += 7) h = (h * 31 + d[i]) >>> 0;
    return h;
  };
  const dur = T.frameDur, step = dur / 10, end = T.times[0] + dur * frames;
  const phases = [];
  let prev = null;
  for (let t = T.times[0]; t <= end; t += step) {
    await seek(video, t);
    const s = sig();
    if (prev !== null && s !== prev) {
      let r = (t - T.times[0]) % dur;
      if (r > dur - step / 2) r -= dur;            // a change right on the grid
      phases.push(r);
    }
    prev = s;
  }
  if (phases.length < 3) return 0;                 // a still clip: assume none
  phases.sort((a, b) => a - b);
  return phases[Math.floor(phases.length / 2)];
}

export function once(el, ev) {
  return new Promise(res => el.addEventListener(ev, res, { once: true }));
}

// Seek and wait until that frame can be drawn.
export function seek(video, t) {
  return new Promise(resolve => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    video.addEventListener("seeked", finish, { once: true });
    setTimeout(finish, 3000);                 // never hang on a stuck seek
    video.currentTime = Math.max(0, t);
  });
}
