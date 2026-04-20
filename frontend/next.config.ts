import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  // Force clean build — cache bust 2026-04-19
  generateBuildId: async () => `build-${Date.now()}`,
};

export default nextConfig;
