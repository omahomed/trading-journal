import type { NextConfig } from "next";

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

export default nextConfig;
