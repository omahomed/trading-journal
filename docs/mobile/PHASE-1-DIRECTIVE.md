# Phase 1 Directive — Mobile Foundation

**Prepared:** 2026-04-30
**Branch:** `mobile/phase-1-foundation`
**Scope:** Frontend only. No backend changes.
**Visible impact for desktop users:** None. Desktop UX is unchanged.
**Estimated tasks:** 8 sequenced steps with stop-and-report gates between each.

---

## Mission

Lay the foundation for a mobile-first redesign of the trading journal app **without changing the desktop experience**. This phase delivers the design system tokens, mobile shell (bottom nav + sticky tape pill), PWA install capability, and stubbed anchor screens (Position Sizer, Trade Journal) that prove the design system works in real React.

This phase does **NOT**:

- Wire mobile screens to real data (Phase 2+)
- Modify any existing desktop component
- Touch the backend
- Add push notifications
- Migrate existing screens to SWR (that's Phase 1.5)

---

## Pre-flight: Do not touch

The following must be left **byte-for-byte unchanged** in this phase:

- `src/app/(app)/layout.tsx` — the desktop shell. We wrap it, we do not modify it.
- `src/components/sidebar.tsx`
- `src/components/tape-status-pill.tsx` — we will reuse this component on mobile, but we do not edit it. If a mobile-specific variant is needed, create a new component.
- All files under `src/components/` that match the surface list (`dashboard.tsx`, `position-sizer.tsx`, `trade-journal.tsx`, `active-campaign.tsx`, `m-factor.tsx`, `daily-routine.tsx`, `daily-journal.tsx`, `daily-report-card.tsx`, `rally-context.tsx`, `settings.tsx`, `admin.tsx`, `log-buy.tsx`).
- `src/app/(app)/*/page.tsx` page files — these will be modified in Phase 2+, not here.
- `src/lib/portfolio-context.tsx`
- `src/lib/api.ts`
- `src/auth.ts` and the NextAuth route handler.
- `next.config.ts` except where this directive explicitly authorizes a change (PWA service worker registration may require an addition; if so, it is the only change permitted).
- The duplicated iCloud-sync files (`layout 2.tsx`, etc.) — leave them. A separate cleanup task at the end of this directive handles them.

> If a step in this directive seems to require touching one of these files, **stop and report** before proceeding.

---

## Step 1 — Resolve Tailwind v4 / v3 config ambiguity

The repo currently has both Tailwind v4 (`@tailwindcss/postcss@^4`, `@import "tailwindcss"` in `globals.css`) and a v3-style `tailwind.config.ts`. Under v4, the JS config may or may not be authoritative depending on postcss setup. We must resolve this before adding any new tokens.

**Tasks:**

1. Determine empirically whether `tailwind.config.ts` is being read by the v4 build:
   - Add a temporary, obviously-unique color (e.g. `colors: { "phase1-probe": "#ff00ff" }`) to the existing config.
   - Use it in a temporary div in `src/app/(app)/dashboard/page.tsx` (`className="bg-phase1-probe"`).
   - Run `npm run build` or `npm run dev` and inspect the rendered output / generated CSS.
   - **Stop and report** the finding: is the v3 config live, ignored, or partially read?

2. Based on the finding, choose one of:
   - **If v3 config is fully read by v4:** keep it as-is for now, document the behavior in `docs/mobile/00b-tailwind-config-decision.md`, and proceed to Step 2.
   - **If v3 config is ignored or partial:** migrate all theme tokens from `tailwind.config.ts` into a `@theme` block in `src/app/globals.css`, delete the JS config, verify nothing visually regressed on desktop dashboard, active-campaign, and trade-journal pages.

3. Remove the temporary `phase1-probe` color and the test div before moving on.

**Stop-and-report gate:** Report which path was taken and any visual regressions caught. Do not proceed to Step 2 until the human confirms.

---

## Step 2 — Mobile design tokens

Create the mobile design system as tokens added to the resolved Tailwind theme (CSS `@theme` if v4-native, JS config if v3 is live). Tokens must be **additive** — they extend, never replace, existing tokens.

**Tasks:**

1. Create `src/styles/mobile-tokens.css` (or equivalent CSS layer) defining:

   - **Mobile color palette** as CSS custom properties scoped under `:root`:
     - `--m-bg: #100D0B` (warm near-black background)
     - `--m-surface: #1A1612` (elevated surface / cards)
     - `--m-surface-2: #2A211A` (selected segmented control state)
     - `--m-border: rgba(255, 235, 210, 0.06)` (subtle warm border)
     - `--m-border-strong: rgba(255, 235, 210, 0.22)`
     - `--m-text: #F5EFE5` (off-white, warm)
     - `--m-text-muted: #B0A89E`
     - `--m-text-dim: #7A7268`
     - `--m-text-faint: #5A5249`
     - `--m-accent: #4ADE80` (primary action / gains)
     - `--m-accent-text-on: #052e16` (text on green pill)
     - `--m-down: #F87171` (losses / stop)
     - `--m-warn: #FBBF24` (at risk / amber callouts)
     - `--m-purple: #AFA9EC` (POWERTREND / M Factor inheritance)
     - `--m-purple-text: #C9C5F0`
     - Tinted variants for Ready-state card border, at-risk card background, etc., as `rgba()` derived values.

   - **Mobile font stack mapping:** `--m-font-ui` should resolve to the existing `--font-inter` chain; `--m-font-num` to `--font-jetbrains`; `--m-font-display-italic` to a serif italic stack (`'Iowan Old Style', 'Palatino', Georgia, serif`) for the wordmark.

   - **Spacing/radius tokens:**
     - `--m-radius-sm: 10px`, `--m-radius-md: 14px`, `--m-radius-lg: 18px`, `--m-radius-xl: 22px`, `--m-radius-pill: 999px`
     - `--m-space-1: 4px` through `--m-space-6: 24px` on a 4-px scale

   - **Motion tokens:**
     - `--m-ease-spring: cubic-bezier(0.32, 0.72, 0, 1)` (iOS-feeling spring)
     - `--m-duration-tap: 120ms`, `--m-duration-sheet: 320ms`

2. Import this file in `src/app/globals.css` (at the bottom, so it can override if needed but does not pollute the desktop cascade).

3. Add a Tailwind utility extension (in v4 `@theme` or v3 config, whichever resolved) that exposes these as Tailwind-friendly classes prefixed with `m-` (e.g. `bg-m-surface`, `text-m-text-muted`, `rounded-m-lg`). This means mobile components write `className="bg-m-surface rounded-m-lg"` instead of inline `style={{}}`.

**Stop-and-report gate:** Report the new token file location, the integration approach taken, and confirm `npm run build` succeeds with no warnings. Show one example of a class like `bg-m-surface` resolving correctly in the browser inspector.

---

## Step 3 — Viewport detection utility

Create the primitive that drives all adaptive rendering.

**Tasks:**

1. Create `src/lib/use-viewport.ts` exporting:
   - `useIsMobile(): boolean` — returns `true` when viewport width is `< 1024px`. Uses `window.matchMedia('(max-width: 1023px)')`. Must SSR-safely return `false` until hydration to avoid layout flash on desktop. Document this behavior in a JSDoc block on the hook.
   - `MOBILE_BREAKPOINT_PX = 1024` exported constant for use elsewhere.

2. Add a corresponding **CSS-only** utility class system. In `mobile-tokens.css`:
   - `.m-only { display: none; }` `@media (max-width: 1023px) { .m-only { display: block; } }`
   - `.d-only { display: block; }` `@media (max-width: 1023px) { .d-only { display: none; } }`
   - These let us hide/show shells without JS, which is the visibility mechanism used by the AdaptiveShell in Step 4.

**Stop-and-report gate:** Show the file contents and confirm SSR safety (no hydration warnings).

---

## Step 4 — Mobile shell components

Build the mobile chrome that wraps mobile pages.

**Tasks:**

1. Create `src/components/mobile/mobile-shell.tsx` — the top-level mobile chrome:
   - Sticky tape pill at top (reuses existing `TapeStatusPill` component, restyled in a mobile-specific wrapper that applies the warm-dark palette).
   - Page header slot (passed via prop: title with optional italic-green word, back button, right-side icon slot).
   - Children slot for page content with appropriate padding.
   - Sticky bottom nav at bottom (separate component, see below).
   - Background `bg-m-bg`, full-height with iOS-safe `100dvh`.

2. Create `src/components/mobile/mobile-bottom-nav.tsx`:
   - Five destinations: **Dashboard, Sizer, Journal, Cycle, More.**
   - Uses `lucide-react` icons (your call from decision D — adopt lucide for this nav). Map: `LineChart` (Dashboard), `Calculator` or `ListOrdered` (Sizer), `Menu` or `BookOpen` (Journal), `Clock` or `CircleDot` (Cycle), `MoreHorizontal` (More).
   - Active state: green icon + green label using `--m-accent`.
   - Inactive: `--m-text-faint`.
   - Uses Next.js `<Link>` for navigation. Routes: `/dashboard`, `/position-sizer`, `/trade-journal`, `/m-factor`, `/more` (the More page is a Phase 1 stub — see Step 6).
   - Touch targets minimum 44×44px.
   - No hover effects. Active state on tap only.

3. Create `src/components/mobile/mobile-page-header.tsx`:
   - Renders centered title with the italic-green wordmark pattern (e.g. `Position <em>Sizer</em>`).
   - Left slot for back button (optional), right slot for action icon (optional, e.g. settings, search, more menu).
   - Reusable across all mobile pages.

4. Create `src/components/mobile/mobile-tape-pill.tsx`:
   - Visual variant of the existing tape pill, restyled with `--m-purple-text`, `--m-border-strong`, mobile padding/sizing.
   - Reads from the same data source as the existing `TapeStatusPill` (do not duplicate data fetching). The simplest path: import the existing one and wrap it with mobile styling, OR factor out the data hook from the existing component into a shared hook that both pills consume.
   - **Stop and report** if the existing pill cannot be reused without modification — that decision affects the do-not-touch list.

**Stop-and-report gate:** Render mobile shell standalone in a test page (`/dev/mobile-shell-preview`, mark with `// PHASE 1 PREVIEW — REMOVE BEFORE PHASE 2 SHIP` comment). Confirm: looks like the anchor designs, no console errors, bottom nav routes work, tape pill displays.

---

## Step 5 — Adaptive shell wrapper

Compose desktop and mobile shells under one root so the same routes serve both.

**Tasks:**

1. Create `src/components/mobile/adaptive-shell.tsx`:
   - Renders both `<DesktopShell>` (the existing `(app)/layout.tsx` content) and `<MobileShell>` simultaneously.
   - Uses CSS `.d-only` and `.m-only` classes (from Step 3) to show only the active one — no JavaScript viewport detection needed for the shell itself.
   - Children render inside both shells. Pages that have a mobile variant will internally branch on `useIsMobile()` to render the mobile component.

2. Refactor `src/app/(app)/layout.tsx`:
   - Extract the current shell JSX into a new `<DesktopShell>` component (`src/components/desktop-shell.tsx`).
   - Replace the body of `(app)/layout.tsx` with `<AdaptiveShell>{children}</AdaptiveShell>`.
   - **This is the only modification to `(app)/layout.tsx` permitted in Phase 1.** The structural composition stays identical from the desktop user's perspective.

3. Verify: load any existing page (e.g. `/dashboard`) on a desktop viewport — it must look pixel-identical to before. Resize to mobile width — desktop shell hides, mobile shell appears, page content still shows but unstyled (because no mobile component exists yet for these screens, which is correct for Phase 1).

**Stop-and-report gate:** Side-by-side screenshots of `/dashboard` before and after at desktop width. They must match exactly.

---

## Step 6 — Anchor screens stubbed

Build the two anchor screens (Position Sizer and Trade Journal) as mobile-only components with mock data. They prove the design system works in real React; Phase 2 wires them to live data.

**Tasks:**

1. Create `src/components/mobile/mobile-position-sizer.tsx`:
   - Translates the locked v6 anchor design to React.
   - Uses `bg-m-surface`, `rounded-m-lg`, etc. — no inline `style={{}}` for color or spacing.
   - Mock data hardcoded inside the component (NVDA, the values from the anchor mockup).
   - Mode picker chip is a tappable button that does nothing yet (`onClick={() => {}}` with `// TODO Phase 2: open mode sheet` comment).
   - Three picker tiles (Mode/Profile/Size) similarly stubbed.
   - "Log buy" button does nothing yet (`// TODO Phase 3: hand off to log-buy with prefilled values`).

2. Create `src/components/mobile/mobile-trade-journal.tsx`:
   - Translates the locked v3 anchor design to React.
   - Mock data: 6 positions matching the anchor (NVDA, ARM with READY, HOOD, ASML at-risk, ANET, LUMN calls).
   - Search field is a non-functional input with placeholder.
   - Filter chips are tappable but don't filter (mock data is fixed).
   - Position cards are tappable but go nowhere yet (`// TODO Phase 4: navigate to position detail`).

3. Wire the page-level branching for these two surfaces only:
   - Modify `src/app/(app)/position-sizer/page.tsx` to read `useIsMobile()` and render `<MobilePositionSizer />` on mobile, existing `<PositionSizer />` on desktop. **This is one of the few page-file modifications permitted in Phase 1, and is scoped strictly to these two pages.**
   - Modify `src/app/(app)/trade-journal/page.tsx` identically.

4. Create a stub mobile page for the "More" bottom-nav destination:
   - `src/app/(app)/more/page.tsx` rendering a mobile-only screen with a list of links to Dashboard, M Factor, Daily Routine, Daily Journal, Daily Report, Rally Context, Settings, Admin (anything not on the bottom nav).
   - On desktop, this page should redirect to `/dashboard` (since desktop has the sidebar for nav).

**Stop-and-report gate:** Walk through `/position-sizer` and `/trade-journal` on a phone-width viewport. Both must look like the locked anchor designs. Walk through `/more` and confirm the link list works. Walk through both at desktop width and confirm desktop UX is unchanged.

---

## Step 7 — PWA: install + offline shell

Make the app installable to home screen with a proper icon and offline-capable shell.

**Tasks:**

1. Create `public/manifest.json`:
   - `name`: full app name (read from `package.json` or hardcode "Trading Journal")
   - `short_name`: short version that fits under an icon (~12 chars max)
   - `start_url`: `"/dashboard"`
   - `display`: `"standalone"`
   - `background_color`: `"#100D0B"`, `theme_color`: `"#100D0B"`
   - `icons` array referencing `/icon-192.png`, `/icon-512.png`, `/icon-maskable-512.png`
   - `orientation`: `"portrait"`

2. Generate placeholder app icons in `public/`:
   - `icon-192.png` (192×192), `icon-512.png` (512×512), `icon-maskable-512.png` (with safe-area padding for adaptive icons), `apple-touch-icon.png` (180×180).
   - Placeholder design: warm-black background `#100D0B`, single italic green serif character (the app's first letter) in `#4ADE80`. Final icons can be designed later — these unblock the manifest.

3. Add `<link rel="manifest" href="/manifest.json">` and the apple-touch-icon link in `src/app/layout.tsx`'s `<head>`. Add `<meta name="apple-mobile-web-app-capable" content="yes">` and `<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">`.

4. Create a minimal service worker at `public/sw.js`:
   - Caches the app shell (`/`, `/dashboard`, `/manifest.json`, the icon files, the main JS bundle if statically known).
   - `fetch` event: network-first for API calls (`/api/*` and the `NEXT_PUBLIC_API_URL` host), cache-first for static assets.
   - **Does not cache authenticated API responses.**

5. Register the service worker from a new `src/components/mobile/pwa-register.tsx` client component, mounted from `src/app/layout.tsx`. Use feature detection (`if ('serviceWorker' in navigator)`).

6. Test: open the app on a phone (or Chrome devtools mobile emulation), open the browser menu, confirm "Add to Home Screen" appears. Install. Open from home screen. Verify it launches in standalone mode without browser chrome.

**Stop-and-report gate:** Screenshots of (a) the install prompt, (b) the app running standalone after install, (c) the offline state when network is killed (the cached dashboard route should still render the shell, even if API data is empty).

---

## Step 8 — Biometric unlock plumbing (scaffolded, not enforced)

We scaffold biometric unlock so it's ready for Phase 2+ to enable. We do not enforce it on every visit yet — that's a UX decision for later.

**Tasks:**

1. Create `src/lib/biometric.ts`:
   - `isBiometricAvailable(): Promise<boolean>` — checks for `window.PublicKeyCredential` and `isUserVerifyingPlatformAuthenticatorAvailable()`.
   - `registerBiometric(userId: string): Promise<void>` — stub that calls `navigator.credentials.create(...)` with placeholder challenge. Real challenge generation requires a backend endpoint (out of scope for Phase 1).
   - `verifyBiometric(): Promise<boolean>` — stub similarly.
   - Each function logs `[Phase 1 stub]` and returns a sensible default. Phase 2+ wires the real backend challenge endpoint.

2. Create `src/components/mobile/biometric-prompt.tsx` — unused in Phase 1, just exists as the future mount point.

3. Document the integration shape in `docs/mobile/01-biometric-plan.md` so the Phase 2+ author has a clear plan.

**No stop-and-report gate.** Just confirm files exist and TypeScript compiles.

---

## Step 9 — Cleanup duplicated iCloud-sync files

Separate from the above. Mark this as a stop-and-report gate before action.

**Tasks:**

1. List all files matching the pattern ` 2.tsx`, ` 2.ts`, ` 2.test.tsx`, etc. in `src/`.
2. Confirm they are not referenced by any other file (`grep -r` for the file basenames).
3. **Stop and report** the list. Wait for explicit human approval before deletion.
4. On approval, delete and commit in a separate commit (`"chore: remove iCloud-sync duplicate files"`).

---

## Final checks

Before merging this branch:

- [ ] `npm run build` succeeds with zero new warnings vs. baseline.
- [ ] `npm test` (Vitest) all green; no existing tests broken.
- [ ] **Visual regression:** open `/dashboard`, `/position-sizer`, `/trade-journal`, `/active-campaign`, `/m-factor` at desktop width. All must look pixel-identical to pre-branch state. (Take screenshots before starting Phase 1 for comparison.)
- [ ] At mobile width (375px wide, e.g. iPhone 14 viewport):
  - `/position-sizer` renders the locked anchor design with mock data.
  - `/trade-journal` renders the locked anchor design with mock data.
  - `/more` shows the link list.
  - All other routes show the mobile shell chrome with content area unstyled (expected — those screens get mobile components in later phases).
- [ ] PWA install works on real iPhone Safari and Chrome desktop devtools.
- [ ] No `console.error` or `console.warn` from any of the new mobile components in normal use.
- [ ] `docs/mobile/00-architecture-snapshot.md`, `00b-tailwind-config-decision.md`, `01-biometric-plan.md` all committed.

---

## What ships at the end of Phase 1

A user opening the app on their phone for the first time will see:

- The app installable to home screen.
- After install, opens in standalone mode with the warm-dark mobile chrome.
- Bottom nav with five destinations.
- `/position-sizer` and `/trade-journal` look like real mobile apps (with mock data — they don't actually work yet, but they look beautiful).
- All other routes load the chrome but the content area is the existing desktop component squeezed into a narrow viewport (intentionally ugly — those are Phase 2+).

A user on desktop sees: **exactly what they saw before Phase 1 started.** No changes. No regressions.

---

## Working agreement

- Each numbered step ends with a commit to `mobile/phase-1-foundation`. Commit messages: `phase-1: step N — <short description>`.
- **Stop-and-report gates are mandatory.** Do not proceed past a gate without explicit human confirmation in chat.
- If a step requires touching a do-not-touch file, stop and report. Do not work around it silently.
- If a step reveals an architectural assumption that's wrong (e.g. the tape pill cannot be reused without modification), stop and report. Do not invent a workaround.
- Final commit on this branch opens a PR to `main` with this directive linked in the description.
