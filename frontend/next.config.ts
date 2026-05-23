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
//   - local-<timestamp>      — local-dev only; the file-persistence
//                              path below makes this safe across the
//                              build/runtime split that would otherwise
//                              cause Date.now() to evaluate twice and
//                              produce the false-positive banner the
//                              old config exhibited.
const BUILD_ID =
  process.env.VERCEL_GIT_COMMIT_SHA ||
  process.env.RAILWAY_GIT_COMMIT_SHA ||
  process.env.RAILWAY_DEPLOYMENT_ID ||
  process.env.GIT_COMMIT_SHA ||
  process.env.GITHUB_SHA ||
  `local-${Date.now()}`;

// Persist BUILD_ID to a static file that ships with the deploy artifact.
// /api/version/route.ts reads from this file at runtime, guaranteeing
// the server returns the SAME value the client bundle was built with —
// regardless of whether the runtime env has NEXT_PUBLIC_BUILD_ID set.
// Without this, next.config.ts re-evaluates Date.now() on every server
// cold-start (every Vercel serverless instance, every Railway restart),
// producing a divergent server buildId that the dismiss-by-buildId
// logic in update-banner.tsx can't suppress.
try {
  const publicDir = path.join(process.cwd(), "public");
  if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });
  fs.writeFileSync(
    path.join(publicDir, "build-info.json"),
    JSON.stringify({ buildId: BUILD_ID, builtAt: new Date().toISOString() }),
  );
} catch (e) {
  console.warn(`[next.config] Failed to write build-info.json: ${e}`);
}

const nextConfig: NextConfig = {
  devIndicators: false,
  generateBuildId: async () => BUILD_ID,
  env: {
    NEXT_PUBLIC_BUILD_ID: BUILD_ID,
  },
  // /market-cycle → /m-factor permanent redirect. The page was renamed
  // from "Market Cycle Tracker" to "M Factor"; Next.js's `permanent: true`
  // emits HTTP 308, the modern permanent-redirect status that's
  // functionally interchangeable with 301 for browsers and SEO. The
  // redirect preserves query strings (Next.js default behavior).
  async redirects() {
    return [
      { source: "/market-cycle", destination: "/m-factor", permanent: true },
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
