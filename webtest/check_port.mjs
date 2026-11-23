// Check the browser port (docs/js/core.js) against Python's own results.
//
//   node webtest/check_port.mjs
//
// 1. The forest: 3000 real feature rows, JavaScript probabilities against
//    scikit-learn's predict_proba (webtest/forest_check.json).
// 2. The whole chain: for each clip in webtest/ref/, the SAME MediaPipe
//    landmarks Python used go through the JavaScript features, forest,
//    smoothing, events, measurements and safeguards, and every output is
//    compared with Python's. Any difference is printed; exit code 1 if so.
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as core from "../docs/js/core.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(HERE);
const buf = readFileSync(join(ROOT, "docs/model/forest.bin"));
const F = core.loadForest(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const M = core.prepareMeta(JSON.parse(readFileSync(join(ROOT, "docs/model/meta.json"), "utf8")));
let failures = 0;
const fail = msg => { failures++; console.log("  FAIL " + msg); };

// ---- 1. forest
const fc = JSON.parse(readFileSync(join(HERE, "forest_check.json"), "utf8"));
let maxd = 0, argDiff = 0;
fc.X.forEach((row, i) => {
  const p = core.predictProba(F, Float32Array.from(row));
  const q = fc.P[i];
  p.forEach((v, c) => { maxd = Math.max(maxd, Math.abs(v - q[c])); });
  const am = a => a.indexOf(Math.max(...a));
  if (am([...p]) !== am(q)) argDiff++;
});
console.log(`forest: ${fc.X.length} rows, largest probability difference ${maxd.toExponential(2)}, predicted phase differs on ${argDiff}`);
if (maxd > 1e-5 || argDiff) fail("forest disagrees with scikit-learn");

// ---- 2. whole chain
const refDir = join(HERE, "ref");
const files = existsSync(refDir) ? readdirSync(refDir).filter(f => f.endsWith(".json")) : [];
for (const file of files) {
  const R = JSON.parse(readFileSync(join(refDir, file), "utf8"));
  console.log(`\n${file}  (${R.video})`);
  if (!R.result) { console.log("  python could not analyse it: " + R.error); continue; }
  const poses = Object.entries(R.poses).map(([f, P]) => [Number(f), P]).sort((a, b) => a[0] - b[0]);
  const { st, err } = core.analysePoses(poses, R.W, R.H, F, M);
  if (!st) { fail("javascript could not analyse it: " + err); continue; }

  if (JSON.stringify(st.frames) !== JSON.stringify(R.frames)) fail(`frames kept differ (${st.frames.length} vs ${R.frames.length})`);
  let pd = 0;
  st.P.forEach((r, i) => r.forEach((v, c) => { pd = Math.max(pd, Math.abs(v - R.P[i][c])); }));
  console.log(`  smoothed phase probabilities: largest difference ${pd.toExponential(2)}`);
  if (pd > 1e-5) fail("phase probabilities differ");

  const ev = st.autoEvents;
  console.log(`  events  js ${ev.bfc}/${ev.ffc}/${ev.release}   python ${R.events.bfc}/${R.events.ffc}/${R.events.release}`);
  for (const e of ["bfc", "ffc", "release"]) if (ev[e] !== R.events[e]) fail(`${e} event differs`);

  const res = core.measureAt(st, ev, M);
  const py = R.result;
  let worst = 0;
  for (const [key, pr] of Object.entries(py.rows)) {
    const jr = res.rows[key];
    if (!jr) { fail(`${key} missing in javascript`); continue; }
    const d = Math.abs(jr.value - pr.value);          // python rounds to 0.1
    worst = Math.max(worst, d);
    if (d > 0.051) fail(`${key}: js ${jr.value.toFixed(3)} python ${pr.value}`);
    if ("z" in pr && Math.abs(jr.z - pr.z) > 0.006) fail(`${key} z: js ${jr.z.toFixed(3)} python ${pr.z}`);
    for (const f of ["plausibility", "confidence", "trusted", "focus_frame"])
      if (jr[f] !== pr[f]) fail(`${key} ${f}: js ${jr[f]} python ${pr[f]}`);
    if ((jr.explanation || null) !== (pr.explanation || null)) fail(`${key} explanation text differs`);
  }
  for (const key of Object.keys(res.rows)) if (!(key in py.rows)) fail(`${key} only in javascript`);
  for (const e of ["bfc", "ffc", "release"])
    if (JSON.stringify(res.quality[e]) !== JSON.stringify(py.quality[e])) fail(`${e} confidence differs: ${JSON.stringify(res.quality[e])} vs ${JSON.stringify(py.quality[e])}`);
  if (JSON.stringify(res.focus) !== JSON.stringify(py.focus)) fail(`focus list differs: ${res.focus} vs ${py.focus}`);
  if ((res.warning?.text || null) !== (py.warning?.text || null)) fail("warning differs");
  console.log(`  ${Object.keys(py.rows).length} measurements, largest difference ${worst.toFixed(3)} (python rounds to 0.1)`);
}

console.log(failures ? `\n${failures} differences` : "\nALL MATCH");
process.exit(failures ? 1 : 0);
