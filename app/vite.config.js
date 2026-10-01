// One build for three hosts: GitHub Pages (PWA), the desktop app (Tauri) and `geopulse serve` (FastAPI).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import { parse } from "yaml";

const ROOT = resolve(import.meta.dirname, "..");
const DESKTOP = !!process.env.TAURI_ENV_PLATFORM; // set by the Tauri CLI; the app needs no service worker
const VERSION = readFileSync(resolve(ROOT, "geopulse/__init__.py"), "utf8").match(/__version__ = "(.+)"/)[1];

// The example events, from the same examples/*/request.yaml the Python CLI and API use.
function examples() {
  const dir = resolve(ROOT, "examples");
  return readdirSync(dir).filter((d) => existsSync(resolve(dir, d, "request.yaml"))).sort().map((id) => {
    const { aoi_file = "aoi.geojson", ...req } = parse(readFileSync(resolve(dir, id, "request.yaml"), "utf8"));
    return { id, ...req, aoi: JSON.parse(readFileSync(resolve(dir, id, aoi_file), "utf8")) };
  });
}

// ONNX models for the in-browser engine, from models/ (`geopulse models pull`, or the models release in CI).
function weights() {
  const dir = process.env.GEOPULSE_MODELS ?? resolve(ROOT, "models");
  if (!existsSync(dir)) return { cards: [], files: [] };
  const cards = existsSync(resolve(dir, "index.json"))
    ? JSON.parse(readFileSync(resolve(dir, "index.json"), "utf8"))
    : readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(resolve(dir, f), "utf8")));
  const usable = cards.filter((c) => c.onnx && existsSync(resolve(dir, c.onnx.file)));
  usable.sort((a, b) => (b.created ?? "").localeCompare(a.created ?? "")); // newest first, like `geopulse models`
  return { cards: usable, files: usable.map((c) => [c.onnx.file, resolve(dir, c.onnx.file)]) };
}

// The live Scout's cases and health, when GEOPULSE_SCOUT points at its state (the scout-data branch in CI).
function scout() {
  const dir = process.env.GEOPULSE_SCOUT;
  const files = { "cases.json": "cases.json", "scout-status.json": "status.json" };
  return Object.fromEntries(Object.entries(files).filter(([, f]) => dir && existsSync(resolve(dir, f)))
    .map(([name, f]) => [name, readFileSync(resolve(dir, f))]));
}

function geopulseData() {
  const assets = () => {
    const w = weights();
    return {
      "examples.json": JSON.stringify(examples()),
      "site.json": JSON.stringify({ version: VERSION }),
      ...scout(),
      "weights/index.json": JSON.stringify(w.cards),
      ...Object.fromEntries(w.files.map(([name, path]) => [`weights/${name}`, readFileSync(path)])),
    };
  };
  return {
    name: "geopulse-data",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const body = assets()[req.url.split("?")[0].slice(1)];
        if (body === undefined) return next();
        res.setHeader("Content-Type", req.url.includes(".json") ? "application/json" : "application/octet-stream");
        res.end(body);
      });
    },
    generateBundle() {
      for (const [fileName, source] of Object.entries(assets())) this.emitFile({ type: "asset", fileName, source });
    },
  };
}

export default defineConfig({
  base: "./", // relative URLs: works at a GitHub Pages sub-path, inside the app and behind `geopulse serve`
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  worker: { format: "es" },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        // MapLibre changes rarely: its own long-cached chunk, so app updates download only the small app chunk.
        manualChunks: (id) => (id.includes("node_modules/maplibre-gl") ? "maplibre" : undefined),
      },
    },
  },
  plugins: [
    geopulseData(),
    VitePWA({
      disable: DESKTOP,
      registerType: "autoUpdate",
      includeAssets: ["icon.svg"],
      manifest: {
        name: "GeoPulse",
        short_name: "GeoPulse",
        description: "Flood, wildfire and forest-loss maps from Sentinel-1/2, computed on your own machine",
        theme_color: "#0e1116",
        background_color: "#0e1116",
        display: "standalone",
        icons: [
          { src: "icons/192x192.png", sizes: "192x192", type: "image/png" },
          { src: "icons/512x512.png", sizes: "512x512", type: "image/png" },
          { src: "icon.svg", sizes: "any", type: "image/svg+xml" },
        ],
      },
      workbox: {
        // The app shell is precached (instant, offline-capable start). ONNX Runtime's WebAssembly and the models are
        // large, so they are cached the first time a job needs them; their URLs change whenever their content does.
        globPatterns: ["**/*.{js,css,html,svg,png,json}"],
        globIgnores: ["weights/**"],
        maximumFileSizeToCacheInBytes: 4 * 2 ** 20,
        runtimeCaching: [
          { urlPattern: /\.wasm$/, handler: "CacheFirst", options: { cacheName: "geopulse-wasm", expiration: { maxEntries: 4 } } },
          { urlPattern: /\/weights\/.+\.onnx/, handler: "CacheFirst", options: { cacheName: "geopulse-weights", expiration: { maxEntries: 16 } } },
          { urlPattern: /\/weights\/index\.json$/, handler: "NetworkFirst", options: { cacheName: "geopulse-weights-index" } },
        ],
      },
    }),
  ],
});
