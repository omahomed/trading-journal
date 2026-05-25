/**
 * Integration test: every (app)-group authenticated route must render
 * as ƒ (dynamic), not ○ (static).
 *
 * Why: pre-fix, /position-sizer + /trade-journal + 22 sibling auth-
 * gated client-component routes were prerendered as static at build
 * time, with chunk hashes baked into the served HTML. On rebuild,
 * Vercel emitted new chunk hashes; the cached prerender pointed at
 * 404'd chunks; auth-gated users saw broken/inert pages until hard
 * refresh. The fix wraps each route's page.tsx in a server component
 * that calls `await connection()` from "next/server", forcing dynamic
 * rendering per request. See cd5e402 + 1e6f14f for the canonical
 * /dashboard + /login fixes.
 *
 * This test enumerates (app)-group routes from the filesystem (so it
 * auto-tracks new routes) and asserts each one shows up as ƒ in the
 * `next build` route table. Gated by RUN_BUILD_INTEGRATION=1 because
 * `next build` adds ~10s to the test run, which doubles the default
 * vitest suite duration. Run via `npm run test:routes`.
 */
import { describe, test, expect } from "vitest";
import { execSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const RUN = process.env.RUN_BUILD_INTEGRATION === "1";

describe.skipIf(!RUN)("Build output — (app)-group routes are dynamic", () => {
  test("every (app) page.tsx renders as ƒ in `next build` output", () => {
    const appDir = join(process.cwd(), "src/app/(app)");
    const routes = readdirSync(appDir)
      .filter((name) => {
        const full = join(appDir, name);
        return statSync(full).isDirectory() && existsSync(join(full, "page.tsx"));
      })
      .map((name) => `/${name}`)
      .sort();

    // Sanity check: the project has ~25 (app) routes. If this fails, the
    // filesystem walk is broken, not the build.
    expect(routes.length).toBeGreaterThan(20);

    const output = execSync("node_modules/.bin/next build", {
      encoding: "utf8",
      env: { ...process.env, FORCE_COLOR: "0", NO_COLOR: "1" },
    });

    const offenders: string[] = [];
    for (const route of routes) {
      // Build output lines look like "├ ƒ /dashboard" or "├ ○ /admin".
      // Use a non-greedy match for the route to avoid /more matching /more-*.
      const re = new RegExp(`[├└┌]\\s+([ƒ○])\\s+${route.replace(/\//g, "\\/")}(?:\\s|$)`, "m");
      const match = output.match(re);
      if (!match) {
        offenders.push(`${route}: route not present in build output`);
      } else if (match[1] !== "ƒ") {
        offenders.push(`${route}: rendered as ${match[1]} (static), expected ƒ (dynamic)`);
      }
    }

    expect(offenders, offenders.join("\n")).toEqual([]);
  }, 180_000);
});
