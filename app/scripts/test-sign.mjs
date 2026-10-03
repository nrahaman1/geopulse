// Check of worker.js::sign(): read tokens for the Planetary Computer are requested once per storage container, even
// when every scene asks at the same moment (a burst gets rate-limited, see worker.js). Runs the real code against a
// stand-in token service that counts requests.   node scripts/test-sign.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("../src/worker.js", import.meta.url), "utf8");
const code = src.slice(src.indexOf("const tokens = {};"), src.indexOf("\n}\n", src.indexOf("async function sign")) + 2);
let calls = [], failNext = 0, expiry = 3600e3;
async function fetchRetry(url) {
  const n = calls.push(url);
  await new Promise((r) => setTimeout(r, 50));
  if (failNext-- > 0) return { ok: false, status: 504 };
  return { ok: true, json: async () => ({ token: `sig=${n}`, "msft:expiry": new Date(Date.now() + expiry).toISOString() }) };
}
const fresh = () => new Function("fetchRetry", "PC", `${code}; return sign;`)(fetchRetry, "https://pc/api");
const s1 = (i) => `https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/scene${i}/vv.tif`;
const s2 = "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif";

// 8 files at once (4 scenes x VV/VH) plus one from another container: one request per container, then cached.
let sign = fresh();
const out = await Promise.all([...Array(8).keys()].map((i) => sign(s1(i))).concat(sign(s2)));
assert.equal(calls.length, 2);
assert.ok(out.slice(0, 8).every((u) => u.endsWith("?sig=1")) && out[8].endsWith("?sig=2"));
await sign(s1(9));
assert.equal(calls.length, 2);

// A failed request is not cached: the next caller asks again.
calls = []; failNext = 1; sign = fresh();
await assert.rejects(sign(s1(1)), /could not get a read token/);
assert.ok((await sign(s1(2))).endsWith("?sig=2"));

// A token close to expiry is renewed once, however many callers are waiting.
calls = []; expiry = 60e3; sign = fresh();
await sign(s1(1));
expiry = 3600e3;
await Promise.all([...Array(6).keys()].map((i) => sign(s1(i))));
assert.equal(calls.length, 2);
console.log("✓ sign(): one token request per container; failures retried; stale tokens renewed once");
