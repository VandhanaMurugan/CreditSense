// Copies the repo's own evidence into ui/public/repo so the UI never hard-codes a number.
// Source of truth stays in data/demo, phase2/outputs and results; public/repo is git-ignored.
import { cpSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ui = join(dirname(fileURLToPath(import.meta.url)), "..");
const repo = join(ui, "..");
const out = join(ui, "public", "repo");
const sets = [
  ["data/demo", "demo", /\.csv$/],
  ["phase2/outputs", "phase2", /\.(json|png)$/],
  ["results", "phase1", /\.(json|csv)$/],
];
rmSync(out, { recursive: true, force: true });
const manifest = {};
for (const [src, dst, pattern] of sets) {
  mkdirSync(join(out, dst), { recursive: true });
  manifest[dst] = [];
  for (const f of readdirSync(join(repo, src)).filter((f) => pattern.test(f)).sort()) {
    cpSync(join(repo, src, f), join(out, dst, f));
    manifest[dst].push(f);
  }
}
writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log("synced repo evidence:", Object.entries(manifest).map(([k, v]) => `${k}=${v.length}`).join(" "));
