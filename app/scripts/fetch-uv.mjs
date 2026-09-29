// Bundles uv, which the desktop app uses to install its Python engine, for one target triple:
// src-tauri/binaries/uv-<target>[.exe] (Tauri `externalBin`). Verified against the release's published SHA-256.
//   node scripts/fetch-uv.mjs [target]      (default: this machine, from `rustc -vV`)
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

const UV = "0.11.26";
const target = process.argv[2] ?? execFileSync("rustc", ["-vV"], { encoding: "utf8" }).match(/host: (\S+)/)[1];
const win = target.includes("windows");
const asset = `uv-${target}.${win ? "zip" : "tar.gz"}`;
const url = `https://github.com/astral-sh/uv/releases/download/${UV}/${asset}`;

async function get(u) {
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

const archive = await get(url);
const want = (await get(`${url}.sha256`)).toString().split(/\s+/)[0];
if (createHash("sha256").update(archive).digest("hex") !== want) throw new Error(`checksum mismatch for ${asset}`);

const tmp = mkdtempSync(join(tmpdir(), "uv-"));
writeFileSync(join(tmp, asset), archive);
// Windows' own tar (bsdtar) also reads zip archives; a Git-for-Windows tar earlier on PATH would not.
const tar = win && process.env.SystemRoot ? join(process.env.SystemRoot, "System32", "tar.exe") : "tar";
execFileSync(tar, ["-xf", asset], { cwd: tmp });
const exe = `uv${win ? ".exe" : ""}`;
const found = readdirSync(tmp, { recursive: true }).find((p) => basename(p) === exe);
if (!found) throw new Error(`${exe} not found in ${asset}`);

const out = resolve(import.meta.dirname, "../src-tauri/binaries", `uv-${target}${win ? ".exe" : ""}`);
mkdirSync(dirname(out), { recursive: true });
copyFileSync(join(tmp, found), out);
chmodSync(out, 0o755);
rmSync(tmp, { recursive: true, force: true });
console.log(`✓ uv ${UV} for ${target} -> ${out}`);
