// Delivery Stride -- the analysis, ported from analyze.py and
// measure_deliveries.py so it runs in the visitor's browser.
//
// Nothing here touches the page or the video: it takes MediaPipe landmarks
// and returns events, measurements and the report's safeguards. That keeps it
// testable in Node against the Python results (webtest/check_port.mjs).
// Function names follow the Python ones; when changing a definition there,
// change it here too.

// MediaPipe landmark indices
export const NOSE = 0, L_SH = 11, R_SH = 12, L_EL = 13, R_EL = 14,
  L_WR = 15, R_WR = 16, L_HIP = 23, R_HIP = 24, L_KN = 25, R_KN = 26,
  L_AN = 27, R_AN = 28, L_HEEL = 29, R_HEEL = 30, L_FOOT = 31, R_FOOT = 32;

const DEG = 180 / Math.PI;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const scale = (a, s) => [a[0] * s, a[1] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const norm = a => Math.hypot(a[0], a[1]);
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

// ---------------------------------------------------------------- geometry
// analyze.joint_angle: the feature version
function jointAngleF(a, b, c) {
  const v1 = sub(a, b), v2 = sub(c, b);
  const n = norm(v1) * norm(v2);
  if (n === 0) return NaN;
  return Math.acos(Math.min(1, Math.max(-1, dot(v1, v2) / n))) * DEG;
}

// measure_deliveries.joint_angle: the measurement version
export function jointAngle(a, b, c) {
  const v1 = sub(a, b), v2 = sub(c, b);
  const n1 = norm(v1), n2 = norm(v2);
  if (n1 < 1e-9 || n2 < 1e-9) return NaN;
  return Math.acos(Math.max(-1, Math.min(1, dot(v1, v2) / (n1 * n2)))) * DEG;
}

// 180 = straight, above 180 = past straight (knee behind the hip-ankle line)
export function kneeAngle(hip, knee, ankle) {
  const inner = jointAngle(hip, knee, ankle);
  const d = sub(ankle, hip);
  const forward = dot(sub(knee, hip), [d[1], -d[0]]);
  return forward >= 0 ? inner : 360 - inner;
}

export function fromVertical(v) {          // from straight up, + toward +x
  if (Math.abs(v[0]) < 1e-12 && Math.abs(v[1]) < 1e-12) return NaN;
  return Math.atan2(v[0], -v[1]) * DEG;
}
function signedFromVertical(v) { return Math.atan2(v[0], -v[1]) * DEG; }
export function fromHorizontal(v) {        // from +x, + upward
  if (Math.abs(v[0]) < 1e-12 && Math.abs(v[1]) < 1e-12) return NaN;
  return Math.atan2(-v[1], v[0]) * DEG;
}
export function fromDown(v) {              // from straight down, + toward +x
  if (Math.abs(v[0]) < 1e-12 && Math.abs(v[1]) < 1e-12) return NaN;
  return Math.atan2(v[0], v[1]) * DEG;
}

// ---------------------------------------------------------------- features
// poses: [[frame, P]] with P = 33 x [x, y, visibility], normalised to the image
export function featuresFor(poses, W, H, minTorso = 0.02) {
  const ar = W / H;
  const fr = poses.map(p => p[0]);
  const hx = poses.map(([, P]) => (P[L_HIP][0] + P[R_HIP][0]) / 2 * ar);
  let slope = 1.0;
  if (fr.length > 2) {                       // least-squares slope, as polyfit
    const n = fr.length;
    const mx = fr.reduce((s, v) => s + v, 0) / n, my = hx.reduce((s, v) => s + v, 0) / n;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (fr[i] - mx) * (hx[i] - my); sxx += (fr[i] - mx) ** 2; }
    slope = sxy / sxx;
  }
  const flip = slope < 0;

  const rows = [];
  for (const [f, Praw] of poses) {
    const P = Praw.map(p => { const x = p[0] * ar; return [flip ? -x : x, p[1]]; });
    const hipC = mid(P[L_HIP], P[R_HIP]), shC = mid(P[L_SH], P[R_SH]);
    const torso = norm(sub(shC, hipC));
    if (torso < minTorso) continue;
    const Q = P.map(p => scale(sub(p, hipC), 1 / torso));

    const [lead, trail] = Q[L_AN][0] >= Q[R_AN][0]
      ? [[L_HIP, L_KN, L_AN], [R_HIP, R_KN, R_AN]]
      : [[R_HIP, R_KN, R_AN], [L_HIP, L_KN, L_AN]];
    const [high, low] = Q[L_WR][1] <= Q[R_WR][1]
      ? [[L_SH, L_EL, L_WR], [R_SH, R_EL, R_WR]]
      : [[R_SH, R_EL, R_WR], [L_SH, L_EL, L_WR]];
    const aLead = Q[lead[2]], aTrail = Q[trail[2]];
    const lower = aLead[1] >= aTrail[1] ? aLead : aTrail;
    const upper = aLead[1] >= aTrail[1] ? aTrail : aLead;

    const d = {
      knee_lead: jointAngleF(Q[lead[0]], Q[lead[1]], Q[lead[2]]),
      knee_trail: jointAngleF(Q[trail[0]], Q[trail[1]], Q[trail[2]]),
      elbow_high: jointAngleF(Q[high[0]], Q[high[1]], Q[high[2]]),
      elbow_low: jointAngleF(Q[low[0]], Q[low[1]], Q[low[2]]),
      ankle_sep_x: aLead[0] - aTrail[0],
      ankle_diff_y: aLead[1] - aTrail[1],
      low_ankle_y: lower[1],
      high_ankle_y: upper[1],
      trunk_lean: signedFromVertical(add(scale(Q[L_SH], 0.5), scale(Q[R_SH], 0.5))),
      arm_high_elev: signedFromVertical(sub(Q[high[2]], Q[high[0]])),
      arm_low_elev: signedFromVertical(sub(Q[low[2]], Q[low[0]])),
      wrist_high_vs_head: Q[high[2]][1] - Q[NOSE][1],
      shoulder_width: Math.abs(Q[L_SH][0] - Q[R_SH][0]),
      hip_width: Math.abs(Q[L_HIP][0] - Q[R_HIP][0]),
      shoulder_hip_offset: (Q[L_SH][0] + Q[R_SH][0]) / 2,
      lead_ankle_fwd: aLead[0],
    };
    if (Object.values(d).some(v => Number.isNaN(v))) continue;
    rows.push({ f, d, P });                  // P: scaled and flipped, for measuring
  }
  return { rows, flip };
}

// window stacking, as stack_windows.py
export function stack(rows, k, baseFeatures) {
  const L = rows.length, out = [];
  for (let i = 0; i < L; i++) {
    const d = rows[i].d, v = { ...d };
    for (const f of baseFeatures) {
      for (let off = -k; off <= k; off++) {
        if (off === 0) continue;
        const j = Math.min(Math.max(i + off, 0), L - 1);
        v[`${f}__${off < 0 ? "m" : "p"}${Math.abs(off)}`] = rows[j].d[f];
      }
      const pm1 = rows[Math.max(i - 1, 0)].d[f], pp1 = rows[Math.min(i + 1, L - 1)].d[f];
      const pm3 = rows[Math.max(i - 3, 0)].d[f];
      v[`${f}__d1`] = d[f] - pm1;
      v[`${f}__d3`] = d[f] - pm3;
      v[`${f}__v`] = (pp1 - pm1) / 2;
    }
    out.push(v);
  }
  return out;
}

// ------------------------------------------------------------------ forest
export function loadForest(buf) {
  const h = new Int32Array(buf, 0, 4);
  const [nTrees, nNodes, nLeaves, nClasses] = h;
  let o = 16;
  const thr = new Float64Array(buf, o, nNodes); o += nNodes * 8;
  const leaf = new Float32Array(buf, o, nLeaves * nClasses); o += nLeaves * nClasses * 4;
  const roots = new Int32Array(buf, o, nTrees); o += nTrees * 4;
  const left = new Int32Array(buf, o, nNodes); o += nNodes * 4;
  const right = new Int32Array(buf, o, nNodes); o += nNodes * 4;
  const feat = new Int16Array(buf, o, nNodes);
  return { nTrees, nClasses, thr, leaf, roots, left, right, feat };
}

// x: Float32Array of features (already float32, as scikit-learn casts them)
export function predictProba(F, x) {
  const p = new Float64Array(F.nClasses);
  for (let t = 0; t < F.nTrees; t++) {
    let n = F.roots[t];
    while (F.feat[n] !== -1) n = x[F.feat[n]] <= F.thr[n] ? F.left[n] : F.right[n];
    const base = F.left[n] * F.nClasses;
    for (let c = 0; c < F.nClasses; c++) p[c] += F.leaf[base + c];
  }
  for (let c = 0; c < F.nClasses; c++) p[c] /= F.nTrees;
  return p;
}

// ------------------------------------------------------------------ events
function median(a) {
  const s = [...a].sort((x, y) => x - y), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

export function medianFilter(P, k) {
  if (k < 3 || k % 2 === 0) return P;
  const h = Math.floor(k / 2), n = P.length, C = P[0].length, out = [];
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - h), hi = Math.min(n, i + h + 1), row = [];
    for (let c = 0; c < C; c++) {
      const col = [];
      for (let j = lo; j < hi; j++) col.push(P[j][c]);
      row.push(median(col));
    }
    out.push(row);
  }
  return out;
}

function blocksOf(labels, want) {
  const out = []; let s = null;
  labels.forEach((v, i) => {
    if (v === want && s === null) s = i;
    else if (v !== want && s !== null) { out.push([s, i - 1]); s = null; }
  });
  if (s !== null) out.push([s, labels.length - 1]);
  return out;
}

function argmax(r) { let b = 0; for (let i = 1; i < r.length; i++) if (r[i] > r[b]) b = i; return b; }

export function extractEvents(frames, prob, phases, smooth = 9) {
  const P = medianFilter(prob, smooth);
  const lab = P.map(r => phases[argmax(r)]);
  const bi = phases.indexOf("bfc_phase"), fi = phases.indexOf("ffc_phase");
  const bb = blocksOf(lab, "bfc_phase"), fb = blocksOf(lab, "ffc_phase");
  const info = { P, nBlocks: { bfc: bb.length, ffc: fb.length, release: fb.length } };
  if (!bb.length || !fb.length) return { ev: null, err: "could not find both foot contacts", info };
  let best = null, score = -1;
  for (const b of bb) for (const f of fb) {
    if (b[0] >= f[0]) continue;
    let s = 0;
    for (let i = b[0]; i <= b[1]; i++) s += P[i][bi];
    for (let i = f[0]; i <= f[1]; i++) s += P[i][fi];
    if (s > score) { score = s; best = [b, f]; }
  }
  if (!best) return { ev: null, err: "back foot contact was not found before front foot contact", info };
  const [b, f] = best;
  return { ev: { bfc: frames[b[0]], ffc: frames[f[0]], release: frames[f[1]] }, err: "", info };
}

// --------------------------------------------------------------- measuring
function nearest(frames, want, tol) {
  if (want === null || want === undefined || !frames.length) return null;
  let f = frames[0];
  for (const x of frames) if (Math.abs(x - want) < Math.abs(f - want)) f = x;
  return Math.abs(f - want) <= tol ? f : null;
}

export function measure(P, flip) {
  const Q = P.map(p => [p[0] * flip, p[1]]);
  const hipC = mid(Q[L_HIP], Q[R_HIP]), shC = mid(Q[L_SH], Q[R_SH]);
  const torso = norm(sub(shC, hipC));
  if (torso < 0.02) return null;
  const [lead, trail] = Q[L_AN][0] >= Q[R_AN][0]
    ? [[L_HIP, L_KN, L_AN, L_FOOT], [R_HIP, R_KN, R_AN, R_FOOT]]
    : [[R_HIP, R_KN, R_AN, R_FOOT], [L_HIP, L_KN, L_AN, L_FOOT]];
  const high = Q[L_WR][1] <= Q[R_WR][1] ? [L_SH, L_EL, L_WR] : [R_SH, R_EL, R_WR];
  const trunk = fromVertical(sub(shC, hipC));
  const ground = Math.max(Q[L_AN][1], Q[R_AN][1]);
  const planted = Q[L_AN][1] >= Q[R_AN][1] ? Q[L_AN] : Q[R_AN];
  return {
    torso,
    knee: kneeAngle(Q[lead[0]], Q[lead[1]], Q[lead[2]]),
    plant: fromDown(sub(Q[lead[2]], Q[lead[0]])),
    foot_strike: fromHorizontal(sub(Q[lead[3]], Q[lead[2]])),
    trunk,
    lean_back: -fromVertical(sub(shC, planted)),
    arm_vs_trunk: fromVertical(sub(Q[high[1]], Q[high[0]])) - trunk,
    stride: Math.abs(Q[lead[2]][0] - Q[trail[2]][0]) / torso,
    lead_ankle_x: Q[lead[2]][0],
    trail_ankle_x: Q[trail[2]][0],
    release_hgt: (ground - Q[high[2]][1]) / torso,
  };
}

// byf: Map frame -> P (33 x [x, y]) already scaled and flipped
export function measureClip(byf, ev, flip, tol) {
  const idx = [...byf.keys()].sort((a, b) => a - b);
  const fB = nearest(idx, ev.bfc, tol), fF = nearest(idx, ev.ffc, tol), fR = nearest(idx, ev.release, tol);
  if (fF === null || fR === null) return { m: null, note: "no pose at front foot contact or release" };
  const mF = measure(byf.get(fF), flip), mR = measure(byf.get(fR), flip);
  if (!mF || !mR) return { m: null, note: "degenerate pose at an event frame" };
  const mB = fB !== null ? measure(byf.get(fB), flip) : null;
  let kMin = NaN, fMin = null;
  for (const f of idx) {
    if (f < fF || f > fR) continue;
    const mm = measure(byf.get(f), flip);
    if (mm && !Number.isNaN(mm.knee) && (fMin === null || mm.knee < kMin)) { kMin = mm.knee; fMin = f; }
  }
  const m = {
    front_knee_ffc: mF.knee,
    front_knee_release: mR.knee,
    knee_flexion: mF.knee - kMin,
    knee_extension: mR.knee - kMin,
    plant_angle: mF.plant,
    foot_strike: mF.foot_strike,
    trunk_lean_ffc: mF.trunk,
    trunk_lean_release: mR.trunk,
    lean_back_bfc: mB ? mB.lean_back : NaN,
    arm_angle_release: mR.arm_vs_trunk,
    stride_torso: mF.stride,
    stride_contacts_torso: mB ? Math.abs(mF.lead_ankle_x - mB.trail_ankle_x) / mF.torso : NaN,
    release_hgt_torso: mR.release_hgt,
    _f: { bfc: fB, ffc: fF, rel: fR, min: fMin },
  };
  return { m, note: mB ? "" : "no pose at back foot contact" };
}

// ------------------------------------------------------- report safeguards
const LEVELS = ["low", "moderate", "high"];
const LEG = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

export function band(z) {
  const a = Math.abs(z);
  return a <= 1 ? "typical" : a <= 2 ? "outside the usual range" : "well outside the usual range";
}

// st: { frames, P (smoothed), phases, nBlocks, vis: Map frame -> visibility[33], autoEvents }
export function eventQuality(st, ev, M, checked = false) {
  const out = {};
  for (const e of ["bfc", "ffc", "release"]) {
    const f = ev[e];
    if (checked) { out[e] = { level: "checked", reasons: ["frame checked by you"] }; continue; }
    if (f === null || f === undefined || !st.frames.length) { out[e] = { level: "low", reasons: ["not detected"] }; continue; }
    let i = 0;
    for (let j = 1; j < st.frames.length; j++) if (Math.abs(st.frames[j] - f) < Math.abs(st.frames[i] - f)) i = j;
    let lvl = M.event_base[e];
    const reasons = [];
    if (e === "bfc") reasons.push(`back foot contact is found within 3 frames on only ${Math.round(M.event_hit_rate.bfc)}% of clips`);
    if (Math.abs(st.frames[i] - f) > 0) { lvl--; reasons.push("no pose on that exact frame"); }
    const lo = Math.max(0, i - 3), hi = Math.min(st.frames.length - 1, i + 3);
    let gap = false;
    for (let j = lo; j < hi; j++) if (st.frames[j + 1] - st.frames[j] > 1) gap = true;
    if (gap) { lvl--; reasons.push("pose dropped out around this instant"); }
    const vis = st.vis.get(st.frames[i]);
    if (vis) {
      const mv = LEG.reduce((s, k) => s + vis[k], 0) / LEG.length;
      if (mv < 0.6) { lvl--; reasons.push("legs poorly visible"); }
    }
    if (st.autoEvents[e] === f) {
      const p = st.phases.indexOf(e === "bfc" ? "bfc_phase" : "ffc_phase");
      if (st.P[i][p] < 0.6) { lvl--; reasons.push("the classifier was unsure where the phase " + (e === "release" ? "ends" : "begins")); }
      const nb = st.nBlocks[e] ?? 1;
      if (nb > 1) { lvl--; reasons.push(`the phase was split into ${nb} pieces`); }
    }
    out[e] = { level: LEVELS[Math.max(0, lvl)], reasons };
  }
  return out;
}

function measureConfidence(key, q, M) {
  const deps = M.event_deps[key] || [];
  if (!deps.length) return ["moderate", ""];
  if (deps.every(e => q[e].level === "checked")) return ["checked", "event frames checked by you"];
  const rank = e => { const i = LEVELS.indexOf(q[e].level); return i < 0 ? 3 : i; };
  const worst = deps.reduce((a, b) => (rank(b) < rank(a) ? b : a));
  const lvl = q[worst].level;
  return [lvl, lvl !== "high" ? `${M.event_name[worst]} uncertain: ${q[worst].reasons.join("; ")}` : ""];
}

export function explain(key, z, M) {
  if (z === null || z === undefined || Math.abs(z) <= 1) return null;
  const label = M.label[key];
  const head = `Your ${label.toLowerCase()} is ${Math.abs(z).toFixed(1)} sd ${z < 0 ? "below" : "above"} the elite mean.`;
  if (M.assoc[key]) return `${head} ${M.assoc[key]} This is an association across bowlers, not a prediction about you.`;
  const tail = M.describe[key] || "";
  return `${head} ${tail} Worthington reports this as a descriptive reference; being different from it is not by itself a fault.`.replaceAll("  ", " ");
}

export function measureAt(st, ev, M, checked = false) {
  if (!(ev.bfc < ev.ffc && ev.ffc < ev.release))
    return { ok: false, error: "back foot contact must come before front foot contact, and that before release" };
  const { m, note } = measureClip(st.byf, ev, 1.0, 3);
  if (!m) return { ok: false, error: note || "could not measure the delivery" };
  const vals = { ...m };
  for (const [src, dst] of [["stride_torso", "stride_pct_height"], ["release_hgt_torso", "release_pct_height"]]) {
    const v = vals[src];
    vals[dst] = v === null || Number.isNaN(v) ? null : v * M.trunk_ratio * 100;
  }
  const quality = eventQuality(st, ev, M, checked);
  const rows = {};
  for (const [key, label, unit] of M.measures) {
    const v = vals[key];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    const row = { label, unit, value: v, ref_note: M.not_comparable[key] || "not published" };
    let z = null;
    if (M.ref[key]) {
      const [mu, sd] = M.ref[key];
      z = (v - mu) / sd;
      Object.assign(row, { ref_mean: mu, ref_sd: sd, z, band: band(z) });
    }
    const coh = M.cohort[key];
    let plaus = "ok";
    if (coh && coh.sd > 0) {
      const cz = (v - coh.mean) / coh.sd;
      Object.assign(row, { cohort_mean: coh.mean, cohort_z: cz });
      plaus = Math.abs(cz) > M.implausible_sd ? "implausible" : Math.abs(cz) > M.unusual_sd ? "unusual" : "ok";
    }
    row.plausibility = plaus;
    [row.confidence, row.confidence_reason] = measureConfidence(key, quality, M);
    const trusted = checked || plaus !== "implausible";
    row.trusted = trusted;
    row.explanation = trusted ? explain(key, z, M) : null;
    const fk = M.focus_at[key];
    row.focus_frame = fk ? m._f[fk] : null;
    rows[key] = row;
  }
  return {
    ok: true, events: { ...ev }, checked, autoEvents: { ...st.autoEvents },
    quality, rows, warning: patternWarning(rows, checked, M), focus: focusList(rows), note,
  };
}

export function focusList(rows) {
  return Object.keys(rows)
    .filter(k => rows[k].z !== undefined && Math.abs(rows[k].z) > 1 && rows[k].trusted && rows[k].focus_frame !== null && rows[k].focus_frame !== undefined)
    .sort((a, b) => Math.abs(rows[b].z) - Math.abs(rows[a].z));
}

export function patternWarning(rows, checked, M) {
  const bad = Object.keys(rows).filter(k => rows[k].plausibility === "implausible");
  const odd = Object.keys(rows).filter(k => rows[k].plausibility !== "ok");
  if (!bad.length && odd.length < 2) return null;
  const hits = {};
  for (const k of odd) for (const e of M.event_deps[k] || []) hits[e] = (hits[e] || 0) + 1;
  const top = Math.max(...Object.values(hits));
  const suspects = ["bfc", "ffc", "release"].filter(e => hits[e] === top);
  const names = suspects.map(e => M.event_name[e]).join(" and ");
  const listed = odd.map(k => M.label[k].toLowerCase()).join(", ");
  const lead = odd.length > 1
    ? `${odd.length} values sit far outside our own 262-delivery cohort (${listed}).`
    : `${M.label[odd[0]]} sits far outside our own 262-delivery cohort.`;
  const text = checked
    ? `${lead} You have checked the frames, so ${odd.length > 1 ? "these are" : "it is"} reported, but MediaPipe's joint-angle error on fast bowling has not been validated -- treat with care.`
    : `${lead} ${odd.length > 1 ? "Extremes that move together are the signature" : "A value this extreme is usually the sign"} of a wrongly detected event, most likely ${names}. Check the frames below before reading anything into these numbers.`;
  return { flagged: odd, implausible: bad, suspects, text };
}

export function summarise(results, M) {
  const good = results.filter(r => r.ok), out = {};
  for (const [key, label, unit] of M.measures) {
    const use = []; let dropped = 0;
    for (const r of good) {
      const row = r.rows[key];
      if (!row) continue;
      if (row.trusted) use.push(row.value); else dropped++;
    }
    if (!use.length) continue;
    const mean = use.reduce((s, v) => s + v, 0) / use.length;
    const sd = use.length > 1 ? Math.sqrt(use.reduce((s, v) => s + (v - mean) ** 2, 0) / (use.length - 1)) : null;
    const row = { label, unit, value: mean, spread: sd, n: use.length, dropped, ref_note: M.not_comparable[key] || "not published" };
    let z = null;
    if (M.ref[key]) {
      const [mu, rsd] = M.ref[key];
      z = (mean - mu) / rsd;
      Object.assign(row, { ref_mean: mu, ref_sd: rsd, z, band: band(z) });
    }
    row.explanation = explain(key, z, M);
    out[key] = row;
  }
  return out;
}

// ------------------------------------------------------------ whole chain
// From landmarks to the first report, as analyze.prepare + measure_at.
// poses: [[frame, P(33 x [x, y, vis])]], F: loaded forest, M: meta.json
export function analysePoses(poses, W, H, F, M) {
  if (poses.length < 30) return { st: null, err: "no person detected for long enough -- check the bowler is fully in frame" };
  const { rows, flip } = featuresFor(poses, W, H);
  if (rows.length < 30) return { st: null, err: "pose was too unstable to measure" };
  const stacked = stack(rows, M.k, M.base_features);
  const prob = stacked.map(v => predictProba(F, Float32Array.from(M.features, f => v[f])));
  const frames = rows.map(r => r.f);
  const { ev, err, info } = extractEvents(frames, prob, M.phases, M.smooth);
  if (!ev) return { st: null, err };
  const st = {
    frames, P: info.P, phases: M.phases, nBlocks: info.nBlocks, flip, W, H,
    autoEvents: { ...ev },
    byf: new Map(rows.map(r => [r.f, r.P])),
    poses: new Map(poses),
    vis: new Map(poses.map(([f, P]) => [f, P.map(p => p[2])])),
  };
  return { st, err: "" };
}

// label lookup, built once from meta.measures
export function prepareMeta(M) {
  M.label = Object.fromEntries(M.measures.map(([k, l]) => [k, l]));
  return M;
}
