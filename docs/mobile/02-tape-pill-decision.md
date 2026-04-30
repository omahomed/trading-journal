# Tape Pill — Carve-out Decision

**Date:** 2026-04-30
**Phase 1 step:** 4
**Branch:** `mobile/phase-1-foundation`

## Background

Phase 1 mounts a sticky cycle indicator (the "tape pill") at the top of every mobile page, visually equivalent to the existing desktop `TapeStatusPill` but restyled to the warm-dark mobile palette and the locked anchor's split-element layout. The Phase 1 directive lists `frontend/src/components/tape-status-pill.tsx` as do-not-touch with one explicit Step-4 escape hatch:

> Reads from the same data source as the existing `TapeStatusPill` (do not duplicate data fetching). The simplest path: import the existing one and wrap it with mobile styling, OR factor out the data hook from the existing component into a shared hook that both pills consume.
> **Stop and report** if the existing pill cannot be reused without modification — that decision affects the do-not-touch list.

## Empirical finding

Three concrete blockers prevented a true zero-modification reuse:

1. **No exported data hook.** The `api.rallyPrefix()` fetch lived inside `TapeStatusPill`'s own `useEffect`. Nothing was importable.
2. **Hard-coded desktop palette baked into the inner element.** The rendered `<Link>` carried `bg-[var(--surface)] border-[var(--border)] text-[var(--ink-2)] hover:bg-[#eef0f6]`. A wrapping element cannot override these because Tailwind utilities target the inner element directly with class-level specificity.
3. **DOM structure differs from the mobile anchor.** The existing pill concatenates label + detail into a single string (`"Power-Trend · since Apr 22"`); the anchor splits the pill into discrete spans (`POWERTREND` · `Day 5` · right-aligned `Cap …`). No CSS-only restyling can reshape DOM.

Duplicating the fetch in a self-contained `MobileTapePill` was forbidden by the directive and would have caused two parallel `api.rallyPrefix()` requests once the AdaptiveShell mounts both shells on the same route in Step 5.

## Decision

Approved by the human in chat: **Path A — Carve `tape-status-pill.tsx` out of the do-not-touch list with a strictly-additive scope, for Step 4 only.**

Strict scope:

- Replace the inline `useEffect` / `useState` fetch logic with a single call to a new `useRallyState()` hook in `frontend/src/lib/use-rally-state.ts`.
- No other changes to `tape-status-pill.tsx`. No surrounding cleanup.
- The user-visible behavior of `TapeStatusPill` (markup, classes, formatting, link target, lock-icon rendering) is byte-identical to before. **Hard requirement:** every pre-existing test in `tape-status-pill.test.tsx` must pass byte-identically. Any test failure is evidence of behavioral drift and would be a stop-and-report — not a "fix the test" event.

Outside Step 4, `tape-status-pill.tsx` returns to fully do-not-touch for the rest of Phase 1.

## Implementation summary

- **New file:** `frontend/src/lib/use-rally-state.ts` — exports `useRallyState()` plus the `RallyV11State` and `RallyState` types. The hook subscribes once on mount, returns `null` until the first valid response (or permanently on error), and never polls. Behavior contract preserved verbatim from the inline `useEffect` it replaces.
- **Refactor:** `frontend/src/components/tape-status-pill.tsx` — diff is `+3 / −38` lines, all of which are the swap. The component imports `useRallyState` and the shared types in place of `useEffect` / `useState` / `api`, and the component body shrinks from `const [data, setData] = useState(...); useEffect(...)` to a single `const data = useRallyState();`.
- **New file:** `frontend/src/components/mobile/mobile-tape-pill.tsx` — consumes the same `useRallyState()` and renders the anchor design with `bg-m-purple-tint`, `border-m-purple-border`, all-caps state label, `Day N` detail, and an optional right-aligned cap indicator. Single shared fetch across both pills.

## Tests

All 11 pre-existing `tape-status-pill.test.tsx` tests pass byte-identically after the carve-out. No tests were modified, added to, or removed from that file as part of Step 4.

## Future work

A post-Phase-1 cleanup can simplify further by inlining the desktop pill's `STATE_STYLE` lookup and `formatDetail` into the hook (so both pills present identical data shapes). That is intentionally out of scope here — the carve-out is meant to enable mobile, not refactor desktop.
