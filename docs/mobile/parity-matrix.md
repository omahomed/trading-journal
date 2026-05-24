# Mobile Parity Matrix

Source of truth for what motrading.net workflows live on mobile, what's read-only
on mobile, and what stays desktop-only by design.

Every feature PR going forward must include a **Mobile** line in its scope section
choosing one of:

- "Parity in this PR" — Tier 1, ships on both surfaces
- "Parity in follow-up PR within N days, tracked as issue #N" — Tier 1, deferred
- "Read-only display in follow-up PR" — Tier 2
- "Out of scope, desktop-only by design" — Tier 3

## Mental model

Mobile is a **read + plan + glance companion**, not an action surface.

The only things you *do* on mobile are: Daily Routine EOD save, Position Sizer
planning (no execution), and portfolio switching. Everything else is viewing.

Trade logging (Log Buy, Log Sell) happens on desktop. Mobile observes.

## Tier 1 — Daily companion (mobile parity required)

| Workflow | Area | Status | Notes |
|---|---|---|---|
| Login (NextAuth) | Auth | ❌ untested on mobile dev | tested against deployed staging instead |
| Main Dashboard | Dashboard | ✅ shipped | Phase 2 Step 2 — Featured NLV, 2×2 KPI grid, equity curve, last-10 strip |
| EOD / Live Exposure cards | Dashboard | ✅ shipped | EOD Exposure tile in 2×2 KPI grid (Phase 2 Step 2) |
| Tape pill (rally state) | Market | ✅ live | Phase 1 Step 4 + Phase 2 Step 2 (format → "Since {MMM D}"); cycle state carried globally in the shell, no separate M Factor banner needed |
| Trade Journal (campaign cards: open + closed) | Trades | 🟡 mock (v3 anchor) | real data, strategy pill, status + strategy filter, notes preview, P&L |
| Position Sizer (planning only) | Trades | 🟡 mock (v6 anchor) | real backend, portfolio context, no Log Buy handoff |
| Portfolio switcher | Multi-portfolio | ✅ shipped | Phase 2 Step 1 — bottom-sheet picker in header right slot |
| Daily Routine EOD save | Daily | ❌ missing | multi-portfolio batch save (CanSlim + 457B + LTG) |

## Tier 2 — Read-mostly (mobile gets viewing, edits stay desktop)

| Workflow | Area | Status | Notes |
|---|---|---|---|
| Trade detail view (notes, trade reflection, full history) | Trades | ❌ missing | tap-into from journal card; Trade Reflection design conversation pending |
| Tag pills on trade cards | Tags | ❌ missing | display only; arrives when Phase 8 of tag arc ships tags on trade summaries |
| Weekly retro view (read-only) | Reviews | ❌ missing | drafting/editing stays desktop |
| Daily journal history scroll | Daily | ❌ missing | look back at prior entries |
| Discipline Pulse (Last 10 stats) | Dashboard | ❌ missing | win rate already on dashboard last-10 strip; full pulse cards deferred |

## Tier 3 — Desktop-only by design

| Workflow | Why |
|---|---|
| Log Buy | User doesn't trade from phone; logs on desktop |
| Log Sell | same |
| Weekly retro drafting / editing | too dense for mobile |
| Trade Manager (edit / delete trades) | LIFO-consequential edits need careful review |
| Edit transaction modal | same |
| Database Health / recompute LIFO | admin action |
| Drift-scan admin (10 checks) | power-user, low frequency, dense |
| Strategy admin UI | low frequency, dense form |
| Tag create / rename / recolor UI | low frequency |
| All Campaigns 4-axis filter row | dense; mobile gets simplified strategy + status filter |
| Robinhood importer | bulk, manual |
| Migration runner / SQL fixes | manual deploy step |
| Audit trail viewer | dense, low frequency |
| Cash transactions ledger / mirrors | dense, admin-flavored |
| Heat Tape chart | not a priority on mobile |
| Capture EOD Snapshot (Downloads + Sentry) | desktop screenshot flow |

## Bottom-nav structure

Phase 1 shipped a 5-destination bottom nav: Dashboard / Sizer / Journal / Cycle / More.

With the M Factor cycle state now carried in the global tape pill (Phase 2 Step 2),
the Cycle destination no longer has unique content.

**Phase 2 collapses to 4 destinations: Dashboard / Sizer / Journal / More.** Tier 2
read-only surfaces (weekly retro view, daily journal history) live behind More.

## Auth on mobile dev

Mobile work is verified against **deployed staging on the phone**, not local dev.
This matches the pattern used during Phase 1 PWA install verification.

Configuring Resend + Google OAuth for localhost is deferred housekeeping for a
later session if dev iteration becomes painful.

## Phase plan

- **Phase 1**: Mobile Foundation — shipped, merged at `ed3e070`
- **Phase 1.5**: Originally scoped for SWR adoption; cancelled (tag-system Phase 1
  chose plain fetch + useState pattern, so SWR migration is a non-goal)
- **Phase 2: Mobile Catch-Up** — current sprint; bring every Tier 1 row to ✅
  - Step 1: Portfolio context plumbing — shipped
  - Step 2: Mobile Dashboard surface — shipped
  - Remaining: Trade Journal real data, Position Sizer real backend, Daily Routine
- **Phase 3+**: TBD — likely T2 surfaces (trade detail view, retro/journal read)
- **Phase 7**: Biometric enforcement — deferred per Phase 1 biometric plan
