import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // ── MO Money Design Tokens ──
      colors: {
        // Canvas
        bg: { DEFAULT: "#f6f7fb", 2: "#eef0f6" },
        surface: { DEFAULT: "#ffffff", 2: "#fafbfd" },
        border: { DEFAULT: "#e6e8ef", 2: "#d8dbe5" },

        // Ink (text hierarchy)
        ink: {
          DEFAULT: "#0f1524",
          2: "#2c3243",
          3: "#5a6175",
          4: "#8a90a2",
          5: "#b6bac7",
        },

        // Semantic
        up: { DEFAULT: "#08a86b", soft: "#e5f7ee" },
        down: { DEFAULT: "#e5484d", soft: "#fdecec" },
        warn: { DEFAULT: "#f59f00", soft: "#fff4dd" },

        // Group accents
        "g-dash": { DEFAULT: "#6366f1", soft: "#eef0ff" },
        "g-ops": { DEFAULT: "#08a86b", soft: "#e6f8ef" },
        "g-risk": { DEFAULT: "#e5484d", soft: "#fdecec" },
        "g-daily": { DEFAULT: "#f59f00", soft: "#fff4dd" },
        "g-mkt": { DEFAULT: "#8b5cf6", soft: "#f1ecfe" },
        "g-ai": { DEFAULT: "#0ea5a4", soft: "#e0f5f4" },
        "g-deep": { DEFAULT: "#0d6efd", soft: "#e7f0ff" },
        "g-legacy": { DEFAULT: "#64748b", soft: "#eef1f5" },
        "g-admin": { DEFAULT: "#0f1524", soft: "#eceef3" },
      },

      fontFamily: {
        ui: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        num: ["JetBrains Mono", "ui-monospace", "Menlo", "monospace"],
        display: ["Fraunces", "Instrument Serif", "Georgia", "serif"],
      },

      borderRadius: {
        "r-1": "6px",
        "r-2": "10px",
        "r-3": "14px",
        "r-4": "20px",
      },

      boxShadow: {
        "sh-1": "0 1px 2px rgba(14,20,38,0.04), 0 0 0 1px rgba(14,20,38,0.04)",
        "sh-2": "0 4px 14px rgba(14,20,38,0.06), 0 0 0 1px rgba(14,20,38,0.04)",
        "sh-3": "0 20px 48px rgba(14,20,38,0.14), 0 0 0 1px rgba(14,20,38,0.06)",
      },

      spacing: {
        "sidebar": "260px",
        "sidebar-rail": "64px",
        "header": "56px",
      },

      animation: {
        "pulse-dot": "pulse-dot 2s ease-in-out infinite",
        "fade-in": "fade-in 0.2s ease-out",
        "slide-up": "slide-up 0.18s ease-out",
      },

      keyframes: {
        "pulse-dot": {
          "0%, 100%": { boxShadow: "0 0 0 3px var(--tw-shadow-color)" },
          "50%": { boxShadow: "0 0 0 6px var(--tw-shadow-color)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
