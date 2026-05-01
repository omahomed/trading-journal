# Frontend Architecture Snapshot

Snapshot taken: 2026-04-30. All paths relative to [frontend/](../../frontend/).

## 1. Routing

**App Router** (Next.js 16.2.4, React 19.2.4). No `pages/` directory exists.

### Top-level routing files
- [src/app/layout.tsx](../../frontend/src/app/layout.tsx) — Root HTML/body layout, font loading, theme bootstrap, `UpdateBanner`.
- [src/app/page.tsx](../../frontend/src/app/page.tsx) — Root redirect to `/dashboard`.
- [src/app/error.tsx](../../frontend/src/app/error.tsx) — Route-level error boundary.
- [src/app/global-error.tsx](../../frontend/src/app/global-error.tsx) — App-level error boundary.
- [src/app/globals.css](../../frontend/src/app/globals.css) — Tailwind import + CSS custom properties (light/dark) + recharts/markdown overrides.
- [src/app/(app)/layout.tsx](../../frontend/src/app/(app)/layout.tsx) — Authenticated app shell (sidebar + header + tape pill + command palette).
- [src/proxy.ts](../../frontend/src/proxy.ts) — Edge proxy/middleware (auth gate; mentioned in `next.config.ts` comments).

### Page routes
| Route | File |
|---|---|
| `/` | [src/app/page.tsx](../../frontend/src/app/page.tsx) (redirects to `/dashboard`) |
| `/login` | [src/app/login/page.tsx](../../frontend/src/app/login/page.tsx) |
| `/login/check-email` | [src/app/login/check-email/page.tsx](../../frontend/src/app/login/check-email/page.tsx) |
| `/dashboard` | [src/app/(app)/dashboard/page.tsx](../../frontend/src/app/(app)/dashboard/page.tsx) |
| `/overview` | [src/app/(app)/overview/page.tsx](../../frontend/src/app/(app)/overview/page.tsx) |
| `/active-campaign` | [src/app/(app)/active-campaign/page.tsx](../../frontend/src/app/(app)/active-campaign/page.tsx) |
| `/import-trades` | [src/app/(app)/import-trades/page.tsx](../../frontend/src/app/(app)/import-trades/page.tsx) |
| `/log-buy` | [src/app/(app)/log-buy/page.tsx](../../frontend/src/app/(app)/log-buy/page.tsx) |
| `/log-sell` | [src/app/(app)/log-sell/page.tsx](../../frontend/src/app/(app)/log-sell/page.tsx) |
| `/position-sizer` | [src/app/(app)/position-sizer/page.tsx](../../frontend/src/app/(app)/position-sizer/page.tsx) |
| `/trade-journal` | [src/app/(app)/trade-journal/page.tsx](../../frontend/src/app/(app)/trade-journal/page.tsx) |
| `/trade-manager` | [src/app/(app)/trade-manager/page.tsx](../../frontend/src/app/(app)/trade-manager/page.tsx) |
| `/earnings` | [src/app/(app)/earnings/page.tsx](../../frontend/src/app/(app)/earnings/page.tsx) |
| `/portfolio-heat` | [src/app/(app)/portfolio-heat/page.tsx](../../frontend/src/app/(app)/portfolio-heat/page.tsx) |
| `/risk-manager` | [src/app/(app)/risk-manager/page.tsx](../../frontend/src/app/(app)/risk-manager/page.tsx) |
| `/daily-journal` | [src/app/(app)/daily-journal/page.tsx](../../frontend/src/app/(app)/daily-journal/page.tsx) |
| `/daily-report` | [src/app/(app)/daily-report/page.tsx](../../frontend/src/app/(app)/daily-report/page.tsx) |
| `/daily-routine` | [src/app/(app)/daily-routine/page.tsx](../../frontend/src/app/(app)/daily-routine/page.tsx) |
| `/weekly-retro` | [src/app/(app)/weekly-retro/page.tsx](../../frontend/src/app/(app)/weekly-retro/page.tsx) |
| `/period-review` | [src/app/(app)/period-review/page.tsx](../../frontend/src/app/(app)/period-review/page.tsx) |
| `/m-factor` | [src/app/(app)/m-factor/page.tsx](../../frontend/src/app/(app)/m-factor/page.tsx) |
| `/rally-context` | [src/app/(app)/rally-context/page.tsx](../../frontend/src/app/(app)/rally-context/page.tsx) |
| `/analytics` | [src/app/(app)/analytics/page.tsx](../../frontend/src/app/(app)/analytics/page.tsx) |
| `/performance-heatmap` | [src/app/(app)/performance-heatmap/page.tsx](../../frontend/src/app/(app)/performance-heatmap/page.tsx) |
| `/ai-coach` | [src/app/(app)/ai-coach/page.tsx](../../frontend/src/app/(app)/ai-coach/page.tsx) |
| `/settings` | [src/app/(app)/settings/page.tsx](../../frontend/src/app/(app)/settings/page.tsx) |
| `/admin` | [src/app/(app)/admin/page.tsx](../../frontend/src/app/(app)/admin/page.tsx) |
| `/api/auth/[...nextauth]` | [src/app/api/auth/[...nextauth]/route.ts](../../frontend/src/app/api/auth/[...nextauth]/route.ts) |
| `/api/version` | [src/app/api/version/route.ts](../../frontend/src/app/api/version/route.ts) |
| `/market-cycle` → `/m-factor` (308 permanent redirect, configured in [next.config.ts](../../frontend/next.config.ts)) |

## 2. Layout shell

- **Root layout / `_app` equivalent**: [src/app/layout.tsx](../../frontend/src/app/layout.tsx) (Server Component; imports fonts, sets `<html>`/`<body>`, mounts `UpdateBanner`).
- **App shell (sidebar + header + content) layout**: [src/app/(app)/layout.tsx](../../frontend/src/app/(app)/layout.tsx) (Client Component, route-group scoped to `(app)`).
- **Sidebar component**: [src/components/sidebar.tsx](../../frontend/src/components/sidebar.tsx).
- **Tape pill / cycle indicator**: [src/components/tape-status-pill.tsx](../../frontend/src/components/tape-status-pill.tsx).

### Composition (structural skeleton)

`src/app/layout.tsx`:
```tsx
<html lang="en" className="… h-full antialiased">
  <head>
    <script /* light/dark theme bootstrap */ />
  </head>
  <body className="min-h-full">
    {children}
    <UpdateBanner />
  </body>
</html>
```

`src/app/(app)/layout.tsx` (`AppShell`):
```tsx
<PortfolioProvider>
  <AppGate>
    {/* loading / error / Onboarding states first */}
    <div className={`flex h-screen ${privacy ? "privacy" : ""}`}>
      <Sidebar privacy dark rail … />
      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-[56px] sticky top-0 z-30">
          {/* breadcrumb (group/page) */}
          <div className="flex-1" />
          <TapeStatusPill />
          <button>Quick jump (⌘K)</button>
        </header>
        <div className="flex-1 overflow-auto px-7 py-6">
          {children}
        </div>
      </main>
      <CommandPalette />
    </div>
  </AppGate>
</PortfolioProvider>
```

The shell is a horizontal flexbox: fixed-width `<aside>` (260px expanded / 64px rail) + `<main>` with sticky header.

## 3. Styling

- **Tailwind**: yes, **v4** (`tailwindcss: ^4`, `@tailwindcss/postcss: ^4`). Loaded via PostCSS ([postcss.config.mjs](../../frontend/postcss.config.mjs)) and imported with `@import "tailwindcss"` at the top of [src/app/globals.css](../../frontend/src/app/globals.css). A v3-style [tailwind.config.ts](../../frontend/tailwind.config.ts) is also present and defines tokens (see below); under v4 the config file is non-canonical (theme is normally declared via CSS `@theme`), so the actual runtime token resolution may not pull from this file — flagged for verification.
- **`tailwind.config.ts` extends**:
  - Colors: `bg`, `surface`, `border`, `ink` (1–5), `up`/`down`/`warn` (semantic), 9 group-accent palettes (`g-dash`, `g-ops`, `g-risk`, `g-daily`, `g-mkt`, `g-ai`, `g-deep`, `g-legacy`, `g-admin`), each with a `soft` variant.
  - Font families: `ui` (Inter), `num` (JetBrains Mono), `display` (Fraunces).
  - Border radius: `r-1`–`r-4` (6/10/14/20px).
  - Box shadows: `sh-1`/`sh-2`/`sh-3`.
  - Spacing: `sidebar` 260px, `sidebar-rail` 64px, `header` 56px.
  - Animations: `pulse-dot`, `fade-in`, `slide-up`.
- **Other styling approaches**:
  - Heavy use of inline `style={{ background: "var(--bg)", color: "var(--ink)", … }}` driven by CSS custom properties.
  - One `<style jsx global>` block in `(app)/layout.tsx` for two keyframes (duplicates ones in `tailwind.config.ts`).
  - Custom prose CSS classes (`.prose-custom*`, `.callout-*`) defined in `globals.css`.
  - No CSS Modules, no styled-components.
- **Custom fonts**: loaded via `next/font/google` in [src/app/layout.tsx](../../frontend/src/app/layout.tsx) — Inter, JetBrains Mono, Fraunces — exposed as CSS variables `--font-inter`, `--font-jetbrains`, `--font-fraunces` on `<html>`.
- **Custom CSS variables**: defined in `:root` and `.dark` in [src/app/globals.css](../../frontend/src/app/globals.css) — `--bg`, `--bg-2`, `--surface`, `--surface-2`, `--border`, `--border-2`, `--ink` (1–5), `--sidebar-bg`, `--header-bg`, `--card-shadow`, `--kpi-shadow`. Dark mode is class-based (`.dark` on `<html>`).

## 4. Component library

- **No shadcn/ui, Radix, Headless UI, Mantine, MUI, or Chakra are installed or imported.** No `components.json` exists.
- **`lucide-react` is in `package.json`** (`^1.8.0`) but **not imported anywhere** in `src/`.
- All UI primitives are bespoke. Icons live in [src/components/icons.tsx](../../frontend/src/components/icons.tsx) as inline SVG functions (`Icons.search()`, `NAV_ICONS[…]`, etc.).
- Charts: [`recharts`](https://recharts.org) (`^3.8.1`) and [`lightweight-charts`](https://www.tradingview.com/lightweight-charts/) (`^5.1.0`).
- Markdown: `react-markdown` + `remark-gfm` + `rehype-raw`.
- Image export: `html-to-image`.

## 5. State & data

- **State management**: React Context only. The single global is `PortfolioProvider` in [src/lib/portfolio-context.tsx](../../frontend/src/lib/portfolio-context.tsx) (active portfolio + portfolio list). No Zustand/Redux/Jotai/Recoil. UI state (`privacy`, `dark`, `rail`) lives in `AppShell` `useState`.
- **API calls**: bespoke `fetchWithAuth` wrapper in [src/lib/api.ts](../../frontend/src/lib/api.ts) calling a FastAPI backend at `NEXT_PUBLIC_API_URL` (Railway in prod, `localhost:8000` in dev). Each component does its own `useEffect` + `useState` fetch — no React Query, no SWR, no server components for data. `cache: "no-store"` is the default.
- **Auth**: **NextAuth v5 beta** (`next-auth: ^5.0.0-beta.31`) with Google + Resend (magic link) + Credentials providers, `@auth/pg-adapter` against Postgres. Mints a short-lived JWT (`apiToken`) attached to outbound API calls. Defined in [src/auth.ts](../../frontend/src/auth.ts); route handler at [src/app/api/auth/[...nextauth]/route.ts](../../frontend/src/app/api/auth/[...nextauth]/route.ts); login UI at [src/app/login/page.tsx](../../frontend/src/app/login/page.tsx).

## 6. Key files for the redesign

| Surface | Page route | Main component |
|---|---|---|
| Dashboard | [src/app/(app)/dashboard/page.tsx](../../frontend/src/app/(app)/dashboard/page.tsx) | [src/components/dashboard.tsx](../../frontend/src/components/dashboard.tsx) |
| Position Sizer | [src/app/(app)/position-sizer/page.tsx](../../frontend/src/app/(app)/position-sizer/page.tsx) | [src/components/position-sizer.tsx](../../frontend/src/components/position-sizer.tsx) |
| Trade Journal | [src/app/(app)/trade-journal/page.tsx](../../frontend/src/app/(app)/trade-journal/page.tsx) | [src/components/trade-journal.tsx](../../frontend/src/components/trade-journal.tsx) |
| Active Campaign Summary | [src/app/(app)/active-campaign/page.tsx](../../frontend/src/app/(app)/active-campaign/page.tsx) | [src/components/active-campaign.tsx](../../frontend/src/components/active-campaign.tsx) |
| M Factor | [src/app/(app)/m-factor/page.tsx](../../frontend/src/app/(app)/m-factor/page.tsx) | [src/components/m-factor.tsx](../../frontend/src/components/m-factor.tsx) |
| Daily Routine | [src/app/(app)/daily-routine/page.tsx](../../frontend/src/app/(app)/daily-routine/page.tsx) | [src/components/daily-routine.tsx](../../frontend/src/components/daily-routine.tsx) |
| Daily Journal | [src/app/(app)/daily-journal/page.tsx](../../frontend/src/app/(app)/daily-journal/page.tsx) | [src/components/daily-journal.tsx](../../frontend/src/components/daily-journal.tsx) |
| Daily Report Card | [src/app/(app)/daily-report/page.tsx](../../frontend/src/app/(app)/daily-report/page.tsx) | [src/components/daily-report-card.tsx](../../frontend/src/components/daily-report-card.tsx) |
| Rally Context | [src/app/(app)/rally-context/page.tsx](../../frontend/src/app/(app)/rally-context/page.tsx) | [src/components/rally-context.tsx](../../frontend/src/components/rally-context.tsx) |
| Settings | [src/app/(app)/settings/page.tsx](../../frontend/src/app/(app)/settings/page.tsx) | [src/components/settings.tsx](../../frontend/src/components/settings.tsx) |
| Admin | [src/app/(app)/admin/page.tsx](../../frontend/src/app/(app)/admin/page.tsx) | [src/components/admin.tsx](../../frontend/src/components/admin.tsx) |
| Log Buy | [src/app/(app)/log-buy/page.tsx](../../frontend/src/app/(app)/log-buy/page.tsx) | [src/components/log-buy.tsx](../../frontend/src/components/log-buy.tsx) |

## 7. Build & config

### `frontend/package.json` — dependencies / devDependencies

```json
"dependencies": {
  "@auth/pg-adapter": "^1.11.2",
  "@sentry/nextjs": "^10.49.0",
  "clsx": "^2.1.1",
  "html-to-image": "^1.11.13",
  "lightweight-charts": "^5.1.0",
  "lucide-react": "^1.8.0",
  "next": "16.2.4",
  "next-auth": "^5.0.0-beta.31",
  "pg": "^8.20.0",
  "react": "19.2.4",
  "react-dom": "19.2.4",
  "react-markdown": "^10.1.0",
  "recharts": "^3.8.1",
  "rehype-raw": "^7.0.0",
  "remark-gfm": "^4.0.1",
  "tailwind-merge": "^3.5.0"
},
"devDependencies": {
  "@tailwindcss/postcss": "^4",
  "@testing-library/dom": "^10.4.1",
  "@testing-library/jest-dom": "^6.9.1",
  "@testing-library/react": "^16.3.2",
  "@types/node": "^20",
  "@types/pg": "^8.20.0",
  "@types/react": "^19",
  "@types/react-dom": "^19",
  "@vitejs/plugin-react": "^6.0.1",
  "eslint": "^9",
  "eslint-config-next": "16.2.4",
  "jsdom": "^29.0.2",
  "tailwindcss": "^4",
  "typescript": "^5",
  "vite-tsconfig-paths": "^6.1.1",
  "vitest": "^4.1.5"
}
```

### `frontend/next.config.ts`

```ts
import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

const BUILD_ID = process.env.VERCEL_GIT_COMMIT_SHA || `build-${Date.now()}`;

const nextConfig: NextConfig = {
  devIndicators: false,
  generateBuildId: async () => BUILD_ID,
  env: { NEXT_PUBLIC_BUILD_ID: BUILD_ID },
  async redirects() {
    return [
      { source: "/market-cycle", destination: "/m-factor", permanent: true },
    ];
  },
};

export default withSentryConfig(nextConfig, { silent: true });
```

(Sentry is wired but source-map upload is intentionally skipped pending `SENTRY_AUTH_TOKEN`.)

### TypeScript ([tsconfig.json](../../frontend/tsconfig.json))

- `"strict": true` (full strict mode on).
- Path alias: `@/*` → `./src/*`.
- `target: ES2017`, `module: esnext`, `moduleResolution: bundler`, `jsx: react-jsx`.

## 8. Notable observations for a mobile-first effort

- **Effectively zero responsive Tailwind breakpoints in use.** A repo-wide grep for `sm:`, `md:`, `lg:`, `xl:` returned a single hit, which was a string match inside a `preprocessCallouts` helper — not a class. The current UI is laid out for desktop only.
- **126 `grid-cols-N` declarations**, many at fixed wide counts (`grid-cols-5`, `grid-cols-6`, `grid-cols-7` in dashboard / active-campaign / trade-journal). None are gated behind responsive prefixes, so they will overflow on narrow viewports.
- **Sidebar width (260 / 64 px) and header (56 px) are baked into the shell as `flex` siblings with a sticky header.** There is no `<700px` collapse path; mobile will need a separate top-level layout (drawer/bottom-nav) rather than an extension of `(app)/layout.tsx`.
- **`min-h-screen` / `h-screen` are used at the shell root**, plus `overflow-auto` on the content column. iOS Safari will need `100dvh` handling.
- **Inline `style={{}}` is pervasive**, often computing `color-mix(in oklab, …)` per element. Token-driven mobile theming will work but bulk find/replace won't — many values reference CSS vars rather than Tailwind utility classes.
- **Hover-dependent UX**: the sidebar nav items, pill buttons, and dashboard cards drive their hover state via inline `onMouseEnter` / `onMouseLeave` plus `hover:` classes (`hover:scale-[1.01]`, `hover:bg-…`). On touch these will fire briefly on tap and need an alternative active-state pattern.
- **Command Palette (⌘K)** is the de-facto navigation shortcut; on touch there is currently no equivalent.
- **PWA**: no `manifest.json`, no service worker, no app icons in [public/](../../frontend/public/) (only the default Next.js demo SVGs). Phase 1 will add these from scratch.
- **Tailwind v4 with a v3-style config file**: `tailwindcss@^4` is installed and `@import "tailwindcss"` is used in `globals.css`, but [tailwind.config.ts](../../frontend/tailwind.config.ts) is the v3 JS-config format. Whether v4 reads it depends on the `@tailwindcss/postcss` setup; design tokens may need to be migrated to a `@theme` block in CSS to be authoritative.
- **Data fetching is per-component `useEffect`**. Mobile network conditions favor cached/stale-while-revalidate; introducing React Query/SWR is out of scope for Phase 1 but worth noting as a downstream concern.
- **`(app)` route group is a Client Component**, so every authenticated page ships the shell as client JS. Bottom-nav addition will land in the same client tree.
- **Two `lucide-react` is installed but unused** (`^1.8.0` is also an unusual pin — current upstream is sub-1.0). Phase 1 may want to either adopt it for the bottom-nav icons or remove it.
- **Stray duplicated files exist at the repo root** from iCloud sync conflicts (`layout 2.tsx`, `log-buy.test 2.tsx`, `position-sizer.test 2.tsx`, `sizing-mode 2.ts`, etc., plus matching ones in the parent project). They are tracked as untracked in `git status`. None should be edited as part of the redesign.
