import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

// Prefer the Vercel-provided commit SHA when available so the build ID
// matches what's in git; fall back to a timestamp for local/preview
// builds. Captured in a const so it's identical across generateBuildId
// and the inlined NEXT_PUBLIC_BUILD_ID below.
const BUILD_ID = process.env.VERCEL_GIT_COMMIT_SHA || `build-${Date.now()}`;

const nextConfig: NextConfig = {
  devIndicators: false,
  generateBuildId: async () => BUILD_ID,
  env: {
    NEXT_PUBLIC_BUILD_ID: BUILD_ID,
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
