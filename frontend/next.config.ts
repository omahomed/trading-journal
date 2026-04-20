import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  // Force clean build
  generateBuildId: async () => `build-${Date.now()}-v2`,
};

export default nextConfig;
