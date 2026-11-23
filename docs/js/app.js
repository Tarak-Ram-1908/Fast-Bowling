// Delivery Stride in the browser: the page, the video reading and the report.
// The analysis itself lives in core.js. Nothing is uploaded anywhere: the
// video, the pose and the report all stay on the visitor's device.
import * as core from "./core.js";
import { frameTiming, fallbackTiming, displayPhase, once, seek } from "./video.js";

const TASKS = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
const MAX_FILES = 6, MAX_SECONDS = 12, MAX_WIDTH = 1280;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f1 = v => (Math.round(v * 10) / 10).toFixed(1);
const sgn1 = v => (v >= 0 ? "+" : "") + f1(v);

let M = null, F = null, vision = null, poseModel = null;
// GPU is about 2.5x faster than CPU and lands within ~1.5 px of Python's
// landmarks (CPU: ~1 px); CPU is the fallback where WebGL is unavailable.
let delegate = "GPU";
let deliveries = [];          // { name, st, video, T, crop, result, error }

// ------------------------------------------------------------------ assets
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onBytes(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

let assetsPromise = null;
function loadAssets() {
  if (assetsPromise) return assetsPromise;
  const status = $("#loadstate");
  const parts = { forest: [0, 18.6e6], pose: [0, 30.7e6] };
  const show = () => {
    const got = parts.forest[0] + parts.pose[0], tot = parts.forest[1] + parts.pose[1];
    status.textContent = `Loading the models: ${(got / 1e6).toFixed(0)} of ${(tot / 1e6).toFixed(0)} MB (first visit only)`;
  };
  assetsPromise = (async () => {
    show();
    M = core.prepareMeta(await (await fetch("model/meta.json")).json());
    const [forestBytes, poseBytes, mod] = await Promise.all([
      fetchWithProgress("model/forest.bin", (g, t) => { parts.forest = [g, t || parts.forest[1]]; show(); }),
      fetchWithProgress("model/pose_landmarker_heavy.task", (g, t) => { parts.pose = [g, t || parts.pose[1]]; show(); }),
      import(`${TASKS}/vision_bundle.mjs`),
    ]);
    F = core.loadForest(forestBytes.buffer);
    poseModel = poseBytes;
    vision = { mod, files: await mod.FilesetResolver.forVisionTasks(`${TASKS}/wasm`) };
    status.textContent = "Models ready.";
    $("#go").disabled = false;
  })().catch(err => {
    status.textContent = "The models could not be loaded. Check your connection and reload the page.";
    console.error(err);
    throw err;
  });
  return assetsPromise;
}

// a fresh tracker per clip, as Python creates one per run
async function newLandmarker() {
  const make = dlg => vision.mod.PoseLandmarker.createFromOptions(vision.files, {
    baseOptions: { modelAssetBuffer: poseModel, delegate: dlg },
    runningMode: "VIDEO", numPoses: 1,
  });
  try { return await make(delegate); }
  catch (e) {
    if (delegate === "CPU") throw e;
    console.warn("GPU pose detection unavailable, using CPU", e);
    delegate = "CPU";
    return make("CPU");
  }
}

// ------------------------------------------------------------ reading clips
async function openVideo(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true; video.playsInline = true; video.preload = "auto"; video.src = url;
  await Promise.race([once(video, "loadeddata"), once(video, "error").then(() => { throw new Error("unreadable"); })]);
  return { video, url };
}

function procSize(video) {
  let w = video.videoWidth, h = video.videoHeight;
  if (w > MAX_WIDTH) { h = Math.floor(h * MAX_WIDTH / w); w = MAX_WIDTH; }
  return [w, h];
}

async function poseClip(file, onFrac) {
  const timing = (await frameTiming(file));
  const { video, url } = await openVideo(file);
  const T = timing || fallbackTiming(video.duration);
  T.phase = await displayPhase(video, T);
  T.at = i => T.times[i] + T.phase + T.frameDur / 2;   // middle of frame i as shown
  const [w, h] = procSize(video);
  const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext("2d");
  const lm = await newLandmarker();
  const poses = [];
  try {
    for (let i = 0; i < T.n; i++) {
      await seek(video, T.at(i));
      ctx.drawImage(video, 0, 0, w, h);
      const r = lm.detectForVideo(canvas, Math.floor(i * 1000 / T.fps));
      if (r.landmarks && r.landmarks.length)
        poses.push([i, r.landmarks[0].map(p => [p.x, p.y, p.visibility ?? 1])]);
      if (i % 5 === 0) onFrac(i / T.n);
    }
  } finally { lm.close(); }
  return { poses, W: w, H: h, T, video, url };
}

// The crop around the bowler, as grab_frames in analyze.py: one box for
// everywhere the bowler goes from pad before back foot contact to pad after
// release, so frames stay comparable. Frames themselves are not stored --
// encoding them is slow in some browsers -- but fetched from the still-open
// video when the report needs one (frameAt).
function setupFrames(d, clip, st, pad = 25, height = 420) {
  const { T, W, H } = clip, ev = st.autoEvents;
  const lo = Math.max(0, ev.bfc - pad), hi = Math.min(T.n - 1, ev.release + pad);
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const [f, P] of st.poses) if (f >= lo && f <= hi)
    for (const p of P) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  let box = [0, 0, 1, 1];
  if (x1 > x0) {
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, aspect = W / H;
    let hh = (y1 - y0) * 1.25;
    const ww = Math.max((x1 - x0) * 1.15, hh * 16 / 9 / aspect);
    hh = Math.max(hh, ww * 9 / 16 * aspect);
    box = [Math.max(0, cx - ww / 2), Math.max(0, cy - hh / 2), Math.min(1, cx + ww / 2), Math.min(1, cy + hh / 2)];
  }
  const cx0 = Math.floor(box[0] * W), cy0 = Math.floor(box[1] * H);
  const cw = Math.floor(box[2] * W) - cx0, ch = Math.floor(box[3] * H) - cy0, s = height / ch;
  Object.assign(d, {
    video: clip.video, url: clip.url, T, lo: 0, hi: T.n - 1,
    crop: { w: W, h: H, x0: cx0, y0: cy0, s, cw, ch, ow: Math.round(cw * s), oh: height },
    full: new OffscreenCanvas(W, H), lock: Promise.resolve(), cache: new Map(),
  });
}

// One frame's crop as an ImageBitmap. Seeks are queued per video, and the
// last 40 frames are kept so stepping back and forth is instant.
function frameAt(d, f) {
  if (d.cache.has(f)) return d.cache.get(f);
  const job = d.lock.then(async () => {
    const c = d.crop;
    await seek(d.video, d.T.at(f));
    d.full.getContext("2d").drawImage(d.video, 0, 0, c.w, c.h);
    const out = new OffscreenCanvas(c.ow, c.oh);
    out.getContext("2d").drawImage(d.full, c.x0, c.y0, c.cw, c.ch, 0, 0, c.ow, c.oh);
    return createImageBitmap(out);
  });
  d.lock = job.catch(() => {});
  d.cache.set(f, job);
  if (d.cache.size > 40) d.cache.delete(d.cache.keys().next().value);
  return job;
}

// ----------------------------------------------------------------- drawing
const EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32], [0, 11], [0, 12]];
const YOU = "rgb(255,70,60)", ELITE = "rgb(60,160,255)", GUIDE = "rgb(200,200,200)";

function pixels(d, f) {
  const P = d.st.poses.get(f), c = d.crop;
  if (!P || !c) return null;
  return P.map(([x, y]) => [(x * c.w - c.x0) * c.s, (y * c.h - c.y0) * c.s]);
}

function skeleton(ctx, d, f, faint) {
  const X = pixels(d, f);
  if (!X) return;
  ctx.strokeStyle = faint ? "rgb(170,170,170)" : "rgb(255,255,0)";
  ctx.lineWidth = faint ? 1 : 2;
  for (const [a, b] of EDGES) { ctx.beginPath(); ctx.moveTo(...X[a]); ctx.lineTo(...X[b]); ctx.stroke(); }
  if (faint) return;
  const vis = d.st.poses.get(f);
  X.forEach((p, i) => {
    ctx.fillStyle = vis[i][2] > 0.5 ? "rgb(0,255,0)" : "rgb(255,0,0)";
    ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI); ctx.fill();
  });
}

async function drawFrame(canvas, d, f) {
  if (!d.video) return;
  const img = await frameAt(d, f);
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  skeleton(ctx, d, f, false);
}

function rot(v, deg) {
  const r = deg * Math.PI / 180;
  return [v[0] * Math.cos(r) - v[1] * Math.sin(r), v[0] * Math.sin(r) + v[1] * Math.cos(r)];
}

// render_focus in analyze.py: the measured segments in red, the elite
// average dashed in blue, cropped to the bowler and the overlays
async function drawFocus(canvas, d, key, row) {
  const f = row.focus_frame, X = pixels(d, f);
  if (!d.video || !X) return;
  const img = await frameAt(d, f);
  const src = new OffscreenCanvas(img.width, img.height), ctx = src.getContext("2d");
  ctx.drawImage(img, 0, 0);
  skeleton(ctx, d, f, true);
  const drawn = [];
  const line = (p, q, col, w = 3, dash = false) => {
    drawn.push(p, q);
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash ? [9, 9] : []);
    ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke(); ctx.setLineDash([]);
  };
  const dir = d.st.flip ? -1 : 1, B = p => [p[0] * dir, p[1]];
  const leadL = X[core.L_AN][0] * dir >= X[core.R_AN][0] * dir;
  const [hip, kn, an, toe] = leadL ? [core.L_HIP, core.L_KN, core.L_AN, core.L_FOOT] : [core.R_HIP, core.R_KN, core.R_AN, core.R_FOOT];
  const backAn = leadL ? core.R_AN : core.L_AN;
  const ref = row.ref_mean, sub = (a, b) => [a[0] - b[0], a[1] - b[1]], plus = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const len = v => Math.hypot(v[0], v[1]), mul = (v, s) => [v[0] * s, v[1] * s];

  if (["front_knee_ffc", "front_knee_release", "knee_extension"].includes(key)) {
    const H = X[hip], K = X[kn], A = X[an];
    line(H, K, YOU); line(K, A, YOU);
    if (key !== "knee_extension") {
      const cur = core.kneeAngle(B(H), B(K), B(A));
      let best = null;
      for (const s of [1, -1]) {
        const cand = plus(K, rot(sub(A, K), s * (ref - cur)));
        const err = Math.abs(core.kneeAngle(B(H), B(K), B(cand)) - ref);
        if (!best || err < best[0]) best = [err, cand];
      }
      line(K, best[1], ELITE, 2, true);
    }
    ctx.strokeStyle = YOU; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(K[0], K[1], 8, 0, 2 * Math.PI); ctx.stroke();
  } else if (key === "plant_angle") {
    const H = X[hip], A = X[an], L = len(sub(A, H)), r = ref * Math.PI / 180;
    line(H, [H[0], H[1] + L], GUIDE, 1, true); line(H, A, YOU);
    line(H, plus(H, B(mul([Math.sin(r), Math.cos(r)], L))), ELITE, 2, true);
  } else if (key === "foot_strike") {
    const A = X[an], T = X[toe], v = sub(T, A), L = Math.max(len(v) * 1.8, 30), r = ref * Math.PI / 180;
    line(A, plus(A, B([L, 0])), GUIDE, 1, true);
    line(A, plus(A, mul(v, L / Math.max(len(v), 1e-6))), YOU);
    line(A, plus(A, B(mul([Math.cos(r), -Math.sin(r)], L))), ELITE, 2, true);
  } else if (key === "lean_back_bfc") {
    const A = X[core.L_AN][1] >= X[core.R_AN][1] ? X[core.L_AN] : X[core.R_AN];
    const S = [(X[core.L_SH][0] + X[core.R_SH][0]) / 2, (X[core.L_SH][1] + X[core.R_SH][1]) / 2];
    const L = len(sub(S, A)), r = ref * Math.PI / 180;
    line(A, [A[0], A[1] - L], GUIDE, 1, true); line(A, S, YOU);
    line(A, plus(A, B(mul([-Math.sin(r), -Math.cos(r)], L))), ELITE, 2, true);
  } else if (key === "stride_pct_height") {
    const Fp = X[an], K = X[backAn], y = Math.max(Fp[1], K[1]) + 14;
    line([K[0], y], [Fp[0], y], YOU);
    line(K, [K[0], y], YOU, 1, true); line(Fp, [Fp[0], y], YOU, 1, true);
    const span = (Fp[0] - K[0]) * ref / row.value;
    line([K[0], y + 12], [K[0] + span, y + 12], ELITE, 3, true);
  } else if (key === "release_pct_height") {
    const Wr = X[core.L_WR][1] <= X[core.R_WR][1] ? X[core.L_WR] : X[core.R_WR];
    const g = Math.max(X[core.L_AN][1], X[core.R_AN][1]);
    line([Wr[0], g], Wr, YOU);
    const top = g - (g - Wr[1]) * ref / row.value;
    line([Wr[0] + 12, g], [Wr[0] + 12, top], ELITE, 2, true);
    line([Wr[0] + 4, top], [Wr[0] + 20, top], ELITE, 2);
  }

  // crop to the bowler and the overlays, 4:3
  const pts = X.concat(drawn);
  let x0 = Math.min(...pts.map(p => p[0])), y0 = Math.min(...pts.map(p => p[1]));
  let x1 = Math.max(...pts.map(p => p[0])), y1 = Math.max(...pts.map(p => p[1]));
  const m = 0.12 * Math.max(x1 - x0, y1 - y0);
  x0 -= m; y0 -= m; x1 += m; y1 += m;
  let cw = x1 - x0, ch = y1 - y0;
  if (cw / ch < 4 / 3) { x0 -= (ch * 4 / 3 - cw) / 2; cw = ch * 4 / 3; } else { y0 -= (cw * 3 / 4 - ch) / 2; ch = cw * 3 / 4; }
  const xa = Math.max(0, x0), ya = Math.max(0, y0), xb = Math.min(src.width, x0 + cw), yb = Math.min(src.height, y0 + ch);
  const sc = 480 / (yb - ya);
  canvas.width = Math.round((xb - xa) * sc); canvas.height = 480;
  const c = canvas.getContext("2d");
  c.drawImage(src, xa, ya, xb - xa, yb - ya, 0, 0, canvas.width, canvas.height);
  c.fillStyle = "#000"; c.fillRect(0, canvas.height - 34, canvas.width, 34);
  c.font = "14px system-ui, sans-serif"; c.fillStyle = "#e6e6e6"; c.fillText(`frame ${f}`, 10, canvas.height - 12);
  c.font = "bold 15px system-ui, sans-serif";
  c.fillStyle = YOU; c.fillText("you", canvas.width - 112, canvas.height - 12);
  if (key !== "knee_extension") { c.fillStyle = ELITE; c.fillText("elite", canvas.width - 64, canvas.height - 12); }
}

// ------------------------------------------------------------------ report
const chip = (level, why = "") => `<span class="chip c-${level}"${why ? ` title="${esc(why)}"` : ""}>${level}</span>`;
const unitOf = row => (row.unit === "°" ? "" : row.unit);
const eliteCell = row => (row.ref_mean !== undefined ? `${row.ref_mean} ± ${row.ref_sd}` : `<span class="nc">${esc(row.ref_note)}</span>`);

function focusCard(k, key, row, suffix = "") {
  return `<figure class="focus"><canvas data-focus="${k}" data-key="${key}" aria-label="${esc(row.label)} drawn on the frame it was measured at"></canvas>
    <figcaption><b>${esc(row.label)}</b>
      <span class="fz">${f1(row.value)}${row.unit === "°" ? "°" : "%"} against elite ${row.ref_mean} ± ${row.ref_sd} <em>(${sgn1(row.z)} sd)</em></span>
      <small>${esc(M.focus_text[key])}${esc(suffix)}</small></figcaption></figure>`;
}

const WHO = `<p class="who">The system measures; a coach decides whether anything should change.
  Altering a bowling action on the strength of a cross-sectional correlation is a recognised route to injury.</p>`;

function renderReport() {
  const done = deliveries.map((d, k) => [k, d]).filter(([, d]) => d.result && d.result.ok);
  const failed = deliveries.filter(d => !(d.result && d.result.ok));
  let h = `<section class="rhead"><div class="wrap"><h2>Delivery report</h2>`;
  if (done.length > 1) h += `<p>${done.length} deliveries analysed. The table below is their average; each delivery, with the frames the system chose, follows it.</p>`;
  else if (done.length === 1) h += `<p>${esc(done[0][1].name)}. One delivery carries event-detection noise; analyse several of the same bowler and the report averages them.</p>`;
  for (const d of failed) h += `<p class="err">${esc(d.name)} could not be analysed: ${esc(d.error || (d.result && d.result.error) || "unknown error")}</p>`;
  h += `</div></section>`;

  if (done.length > 1) {
    const S = core.summarise(done.map(([, d]) => d.result), M);
    h += `<section class="wrap block"><div class="table-scroll"><table>
      <caption>Mean over ${done.length} deliveries, against Worthington (2010), twenty elite male fast bowlers. ± is the spread between your deliveries. Values that failed the plausibility check on unchecked frames are left out of the mean.</caption>
      <thead><tr><th>Measurement</th><th>You</th><th>Elite</th><th>Difference</th><th>Deliveries</th></tr></thead><tbody>`;
    for (const [key, s] of Object.entries(S)) {
      h += `<tr${s.z !== undefined && Math.abs(s.z) > 1 ? ' class="flag"' : ""}><td>${esc(s.label)}</td>
        <td class="v">${f1(s.value)}${unitOf(s)}${s.spread !== null ? ` <small class="pm">± ${f1(s.spread)}</small>` : ""}</td>
        <td>${eliteCell(s)}</td><td class="z">${s.z !== undefined ? sgn1(s.z) + " sd" : "—"}</td>
        <td class="n">${s.n}${s.dropped ? ` <small>(${s.dropped} left out)</small>` : ""}</td></tr>`;
    }
    h += `</tbody></table></div>`;
    const cards = [];
    for (const [key, s] of Object.entries(S).sort((a, b) => Math.abs(b[1].z || 0) - Math.abs(a[1].z || 0))) {
      if (s.z === undefined || Math.abs(s.z) <= 1) continue;
      let best = null;
      for (const [k, d] of done) {
        const r = d.result.rows[key];
        if (r && r.trusted && r.focus_frame !== null && r.focus_frame !== undefined) {
          const diff = Math.abs(r.value - s.value);
          if (!best || diff < best[0]) best = [diff, k, r];
        }
      }
      if (best) cards.push(focusCard(best[1], key, s, ` Drawn on ${deliveries[best[1]].name}, which measured ${f1(best[2].value)}.`));
    }
    if (cards.length) h += `<div class="focusbox"><h3>Where your action differs most from elite</h3>
      <p class="hint">Each value more than one elite standard deviation from the average, drawn on the frame it was measured at. The example is the delivery closest to your own average.</p>
      <div class="focusgrid">${cards.join("")}</div></div>`;
    const ex = Object.values(S).filter(s => s.explanation);
    if (ex.length) h += `<div class="explain"><h3>What the differences mean</h3>${ex.map(s => `<p>${esc(s.explanation)}</p>`).join("")}${WHO}</div>`;
    h += `</section>`;
  }

  for (const [k, d] of done) {
    const r = d.result;
    h += `<section class="wrap block delivery" id="d${k}"><details${done.length === 1 || r.warning || d.open ? " open" : ""}>
      <summary><h3>${esc(d.name)}</h3>${r.warning && !r.checked ? '<span class="chip c-low">check frames</span>' : ""}${r.checked ? '<span class="chip c-checked">frames checked</span>' : ""}</summary>`;
    if (r.warning) h += `<div class="warnbox${r.checked ? " soft" : ""}">${esc(r.warning.text)}</div>`;
    if (r.focus.length) h += `<div class="focusbox"><h3>Where ${done.length > 1 ? "this delivery" : "your action"} differs most from elite</h3>
      <p class="hint">Each value more than one elite standard deviation from the average, drawn on the frame it was measured at, furthest first.${r.warning && !r.checked ? " Check the frames below first: some values on this delivery look like detection errors." : ""}</p>
      <div class="focusgrid">${r.focus.map(key => focusCard(k, key, r.rows[key])).join("")}</div></div>`;
    h += `<div class="frames" data-k="${k}">`;
    for (const e of ["bfc", "ffc", "release"]) {
      const q = r.quality[e];
      const sus = r.warning && r.warning.suspects.includes(e) && !r.checked;
      h += `<figure class="ev${sus ? " suspect" : ""}${r.events[e] !== r.autoEvents[e] ? " moved" : ""}" data-e="${e}" data-f="${r.events[e]}">
        <canvas data-frame="${k}"></canvas><figcaption>
        <b>${esc(M.event_name[e][0].toUpperCase() + M.event_name[e].slice(1))}</b> ${chip(q.level)}
        <span class="fno">frame <output>${r.events[e]}</output></span>
        <span class="nudge">${[-5, -1, 1, 5].map(x => `<button type="button" data-d="${x}" aria-label="${Math.abs(x)} frame${Math.abs(x) > 1 ? "s" : ""} ${x < 0 ? "earlier" : "later"}">${x < 0 ? "−" : "+"}${Math.abs(x)}</button>`).join("")}</span>
        ${q.reasons.length ? `<small class="why">${esc(q.reasons.join("; "))}</small>` : ""}</figcaption></figure>`;
    }
    h += `</div><div class="actions" data-k="${k}">
      <button type="button" class="act" data-act="measure">Re-measure at these frames</button>
      <button type="button" class="act ghost" data-act="confirm">These frames are right</button>
      <button type="button" class="act ghost" data-act="reset">Back to detected</button>
      <span class="msg" role="status"></span></div>
      <p class="hint">Step each instant until the picture matches: back foot contact is the first frame the back foot is planted, front foot contact the first frame the front foot is planted, release the last frame the ball is in the hand. Confirming marks every measurement as checked.</p>
      <div class="table-scroll"><table><caption>Measured against Worthington (2010), twenty elite male fast bowlers. Confidence is how far to trust the event frames each value is measured at.</caption>
      <thead><tr><th>Measurement</th><th>You</th><th>Elite</th><th>Difference</th><th>Confidence</th></tr></thead><tbody>`;
    for (const row of Object.values(r.rows)) {
      const cls = !row.trusted ? "bad" : row.z !== undefined && Math.abs(row.z) > 1 ? "flag" : "";
      h += `<tr class="${cls}"><td>${esc(row.label)}${row.plausibility === "unusual" && row.trusted ? `<small class="odd">unusual even for our cohort (${f1(row.cohort_mean)})</small>` : ""}</td>
        <td class="v">${f1(row.value)}${unitOf(row)}</td><td>${eliteCell(row)}</td>
        <td class="z">${!row.trusted ? '<span class="det">looks like a detection error — check the frames</span>' : row.z !== undefined ? sgn1(row.z) + " sd" : "—"}</td>
        <td>${chip(row.confidence, row.confidence_reason)}</td></tr>`;
    }
    h += `</tbody></table></div>`;
    if (done.length === 1) {
      const ex = Object.values(r.rows).filter(x => x.explanation);
      if (ex.length) h += `<div class="explain"><h3>What the differences mean</h3>${ex.map(x => `<p>${x.confidence === "low" ? "<em>Low confidence:</em> " : ""}${esc(x.explanation)}</p>`).join("")}${WHO}</div>`;
    }
    h += `</details></section>`;
  }
  $("#report").innerHTML = h;
  $("#reportwrap").hidden = false;
  wireReport();
}

function wireReport() {
  document.querySelectorAll("canvas[data-focus]").forEach(cv => {
    const k = +cv.dataset.focus, d = deliveries[k];
    drawFocus(cv, d, cv.dataset.key, d.result.rows[cv.dataset.key]);
  });
  document.querySelectorAll(".frames").forEach(strip => {
    const k = +strip.dataset.k, d = deliveries[k];
    strip.querySelectorAll("figure.ev").forEach(fig => {
      const cv = fig.querySelector("canvas"), out = fig.querySelector("output");
      drawFrame(cv, d, +fig.dataset.f);
      fig.querySelectorAll(".nudge button").forEach(b => b.addEventListener("click", () => {
        const f = Math.min(d.hi, Math.max(d.lo, +fig.dataset.f + +b.dataset.d));
        fig.dataset.f = f; out.textContent = f;
        drawFrame(cv, d, f);
        fig.classList.toggle("moved", f !== d.st.autoEvents[fig.dataset.e]);
      }));
    });
  });
  document.querySelectorAll(".actions").forEach(bar => {
    const k = +bar.dataset.k, d = deliveries[k], msg = bar.querySelector(".msg");
    bar.querySelectorAll("button.act").forEach(b => b.addEventListener("click", () => {
      const ev = {};
      document.querySelectorAll(`.frames[data-k="${k}"] figure.ev`).forEach(f => { ev[f.dataset.e] = +f.dataset.f; });
      let res;
      if (b.dataset.act === "reset") res = core.measureAt(d.st, { ...d.st.autoEvents }, M, false);
      else res = core.measureAt(d.st, ev, M, b.dataset.act === "confirm");
      if (!res.ok) { msg.textContent = res.error; return; }
      d.result = res; d.open = true;
      renderReport();
      document.getElementById(`d${k}`).scrollIntoView({ block: "start" });
    }));
  });
}

// -------------------------------------------------------------------- flow
function setProgress(frac, text) {
  $("#bar").classList.add("on");
  $("#fill").style.width = Math.max(2, Math.round(frac * 100)) + "%";
  $("#stage").textContent = text;
}

async function run(files) {
  $("#err").textContent = "";
  $("#go").disabled = true; $("#go").textContent = "Working";
  for (const d of deliveries) if (d.url) URL.revokeObjectURL(d.url);
  deliveries = files.map(f => ({ name: f.name }));
  $("#reportwrap").hidden = true;
  try {
    setProgress(0.01, "Loading the models");
    await loadAssets();
    for (let k = 0; k < files.length; k++) {
      const d = deliveries[k], n = files.length;
      const label = n > 1 ? `Delivery ${k + 1} of ${n}` : "Reading the delivery";
      try {
        const probe = await openVideo(files[k]);
        const secs = probe.video.duration; URL.revokeObjectURL(probe.url);
        if (secs > MAX_SECONDS) { d.error = `it is ${Math.round(secs)} s long. Trim it to one delivery, under ${MAX_SECONDS} s.`; continue; }
        const clip = await poseClip(files[k], frac => setProgress((k + frac * 0.9) / n, `${label}: finding the body, ${Math.round(frac * 100)}%`));
        d.url = clip.url; d.video = clip.video;
        setProgress((k + 0.9) / n, `${label}: measuring`);
        const { st, err } = core.analysePoses(clip.poses, clip.W, clip.H, F, M);
        if (!st) { d.error = err; continue; }
        d.st = st;
        setupFrames(d, clip, st);
        d.result = core.measureAt(st, { ...st.autoEvents }, M);
      } catch (e) {
        console.error(e);
        d.error = e.message === "unreadable"
          ? "this browser cannot play that video. Try MP4 (H.264), or another browser."
          : "something broke while analysing this clip.";
      }
    }
    setProgress(1, "");
    $("#bar").classList.remove("on");
    renderReport();
    $("#reportwrap").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    $("#err").textContent = "Analysis failed: " + (e.message || e);
  } finally {
    $("#go").disabled = false; $("#go").textContent = "Measure it";
  }
}

function init() {
  const form = $("#f"), input = $("#video");
  form.addEventListener("dragover", e => { e.preventDefault(); form.classList.add("hot"); });
  form.addEventListener("dragleave", () => form.classList.remove("hot"));
  form.addEventListener("drop", e => { e.preventDefault(); form.classList.remove("hot"); input.files = e.dataTransfer.files; });
  form.addEventListener("submit", e => {
    e.preventDefault();
    const files = [...input.files];
    if (!files.length) { $("#err").textContent = "Choose a video first."; return; }
    if (files.length > MAX_FILES) { $("#err").textContent = `Up to ${MAX_FILES} deliveries at a time.`; return; }
    run(files);
  });
  $("#again").addEventListener("click", () => { window.scrollTo({ top: $("#analyse").offsetTop, behavior: "smooth" }); });
  loadAssets().catch(() => {});
}

// exposed for automated checks (webtest/browser_check.html)
window.deliveryStride = { loadAssets, poseClip, core, get M() { return M; }, get F() { return F; },
  setDelegate(d) { delegate = d; } };
init();
