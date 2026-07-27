import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";
import fs from "fs";
import path from "path";

// Build identifier — used by the "new version available" detection in
// update-banner.tsx. The client bundle bakes this in at build time
// (LOADED_BUILD_ID); the server returns it at runtime via /api/version.
// A mismatch surfaces the upgrade prompt to the user.
//
// CI env-var cascade (in order of preference):
//   - VERCEL_GIT_COMMIT_SHA  — set by Vercel
//   - RAILWAY_GIT_COMMIT_SHA — set by Railway
//   - RAILWAY_DEPLOYMENT_ID  — Railway fallback if SHA isn't propagated
//   - GIT_COMMIT_SHA         — generic CI convention
//   - GITHUB_SHA             — GitHub Actions
//   - local-<timestamp>      — local-dev only; the
//                              `runAfterProductionCompile` hook below
//                              writes the file exactly once per build,
//                              so the timestamp doesn't drift at
//                              runtime cold start.
const BUILD_ID =
  process.env.VERCEL_GIT_COMMIT_SHA ||
  process.env.RAILWAY_GIT_COMMIT_SHA ||
  process.env.RAILWAY_DEPLOYMENT_ID ||
  process.env.GIT_COMMIT_SHA ||
  process.env.GITHUB_SHA ||
  `local-${Date.now()}`;

const nextConfig: NextConfig = {
  devIndicators: false,
  generateBuildId: async () => BUILD_ID,
  env: {
    NEXT_PUBLIC_BUILD_ID: BUILD_ID,
  },
  compiler: {
    // Persist BUILD_ID to a static file that ships with the deploy
    // artifact. `/api/version/route.ts` reads from this file at runtime,
    // guaranteeing the server returns the SAME value the client bundle
    // was built with — regardless of whether the runtime env has
    // NEXT_PUBLIC_BUILD_ID set.
    //
    // Next 16's native `runAfterProductionCompile` hook fires exactly
    // once after production compile finishes and is never invoked at
    // runtime cold start, so we don't need the brittle
    // `process.env.NEXT_PHASE === "phase-production-build"` gate the
    // prior implementation depended on. That env var stopped matching
    // somewhere in Next 16's lifecycle, the file silently wasn't
    // written, `/api/version` returned a stale or default BUILD_ID, and
    // UpdateBanner's auto-reload poll became inert across deploys.
    runAfterProductionCompile: async ({ projectDir }) => {
      const publicDir = path.join(projectDir, "public");
      await fs.promises.mkdir(publicDir, { recursive: true });
      await fs.promises.writeFile(
        path.join(publicDir, "build-info.json"),
        JSON.stringify({ buildId: BUILD_ID, builtAt: new Date().toISOString() }),
      );
    },
  },
  // Permanent redirects for renamed routes. Next.js's `permanent: true`
  // emits HTTP 308, functionally interchangeable with 301 for browsers
  // and SEO. Query strings are preserved (Next.js default behavior),
  // so /daily-report?date=2026-07-24 lands on /daily-journal?date=2026-07-24.
  async redirects() {
    return [
      // Market Cycle Tracker → M Factor
      { source: "/market-cycle",      destination: "/m-factor",     permanent: true },
      // Phase 2 Daily Routine merger (2026-07-26): Daily Report absorbed
      // Trading Checklist and took over the /daily-routine URL; the
      // former Daily Routine page (NLV entry only) moved to /nlv-entry.
      //
      // Renamed again 2026-07-26 evening: "Daily Routine" (shell) →
      // "Daily Journal"; previous "Daily Journal" (historical browse) →
      // "Journal Log". The /daily-routine URL redirects to the new
      // /daily-journal shell; historical /daily-report chain now lands
      // there too. No redirect from the old /daily-journal browse — the
      // URL is being reused for the shell, and user is aware of the swap.
      { source: "/daily-report",      destination: "/daily-journal", permanent: true },
      { source: "/trading-checklist", destination: "/daily-journal", permanent: true },
      { source: "/daily-routine",     destination: "/daily-journal", permanent: true },
    ];
  },
};

export default withSentryConfig(nextConfig, {
  // Source-map upload is skipped until we add SENTRY_AUTH_TOKEN to Vercel;
  // without it stack traces land in Sentry minified. Acceptable for v1.
  //
  // No tunnelRoute: Sentry events POST directly to ingest.sentry.io from the
  // browser. Tunneling through our own domain (via /monitoring) would dodge
  // ad-blockers, but our auth proxy gates every non-excluded path and was
  // swallowing the POSTs. If we want to enable tunneling later, add the tunnel
  // path to proxy.ts's matcher exclusions.
  silent: true,
});
