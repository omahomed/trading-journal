"use client";

// SR8 Cascade Monitor — daily position-management screen for positions
// tagged sr8. Build sequence:
//   Commit 1: backend endpoint + route stub + nav rename (merged).
//   Commit 2: page scaffold — control strip + summary chips + weekly
//     snapshot + data fetch (merged).
//   Commit 3 (this commit): Action / Hold sections + Mark-done state.
//   Commit 4: All-clear / Loading / Empty / Retry states.
//
// The cascade math lives in mors/monitor.py (Python). The
// /api/sr8/monitor endpoint wraps it; this page renders the response.

import { useState, useEffect, useCallback, useMemo, type ReactNode } from "react";
import { api, getActivePortfolio, type SR8MonitorResponse, type SR8AnalyzedPosition } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

const mono = "var(--font-jetbrains), monospace";

// Default NLV seed used while we wait for journalLatest to resolve.
// Picked to match the design's prototype constant so the loading state
// renders something plausible until real NLV arrives.
const DEFAULT_NLV = 500_000;

// Friendly compact-dollar formatter for the "TO TRIM" summary chip
// (e.g. $161K). Matches the design's k-suffix style.
function compactDollars(n: number): string {
  const sign = n < 0 ? "-" : "";
  const a = Math.abs(n);
  if (a >= 1000) {
    return `${sign}$${(a / 1000).toFixed(a >= 100_000 ? 0 : 1)}K`;
  }
  return `${sign}$${a.toFixed(0)}`;
}

// Days since `iso` (ISO datetime string). Returns null for unparseable.
function daysSince(iso: string): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  const diff = Date.now() - t;
  return Math.floor(diff / 86_400_000);
}

function shortMonthDay(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// Signal display — collapses MORS's verbose "QUICKSAND" into the design's
// short "QS" badge, and aliases the GREEN sub-entry variant. Driven by
// the live cascade tier (current_tier on the position) — the 4 visible
// tier classes (GREEN/QUICK/QS/GD) each map to a (bg, text) color pair.
// ENTRY remains tolerated as a legacy fallback (defensive — current_tier
// always returns a real tier name).
//
// Tokens follow the campaign's SellRuleBadge convention — a `color-mix`
// soft tint over `var(--surface)` for the bg, raw hex for the text —
// because the unprefixed runtime vars (`--up`, `--sig-quick-text`, …)
// the badges used to reference aren't defined in `:root` (Tailwind v4's
// `@theme` exposes them as `--color-*`, not `--*`), so on the live light
// theme every pill was falling back to inherited grey. The three tiers
// that overlap with the campaign's sell-rule palette intentionally use
// the same hex pairs as `SellRuleBadge` so SR8/SR11/SR1 → GREEN/QUICK/GD
// read identically; QS uses the orange-600 family for the orange step
// between QUICK (amber) and GD (rose).
//
// Palette:
//   GREEN  → emerald  bg #08a86b @ 12% / fg #16a34a  (= SellRuleBadge.sr8)
//   QUICK  → amber    bg #f59f00 @ 12% / fg #d97706  (= SellRuleBadge.sr11)
//   QS     → orange   bg #f97316 @ 12% / fg #ea580c
//   GD     → rose     bg #e5484d @ 14% / fg #dc2626  (= SellRuleBadge.sr1)
const TIER_PALETTE = {
  GREEN: { bg: "color-mix(in oklab, #08a86b 12%, var(--surface))", text: "#16a34a" },
  QUICK: { bg: "color-mix(in oklab, #f59f00 12%, var(--surface))", text: "#d97706" },
  QS:    { bg: "color-mix(in oklab, #f97316 12%, var(--surface))", text: "#ea580c" },
  GD:    { bg: "color-mix(in oklab, #e5484d 14%, var(--surface))", text: "#dc2626" },
  ENTRY: { bg: "color-mix(in oklab, #0d6efd 12%, var(--surface))", text: "#1d4ed8" },
} as const;

function signalDisplay(signal: string): { label: string; bg: string; text: string; severity: number } {
  const s = (signal || "").toUpperCase();
  if (s === "GREEN" || s === "GREEN(SUB-ENTRY)") return { label: "GREEN", ...TIER_PALETTE.GREEN, severity: 1 };
  if (s === "QUICK") return { label: "QUICK", ...TIER_PALETTE.QUICK, severity: 2 };
  if (s === "QUICKSAND" || s === "QS") return { label: "QS", ...TIER_PALETTE.QS, severity: 3 };
  if (s === "GD" || s === "TERMINATED") return { label: "GD", ...TIER_PALETTE.GD, severity: 4 };
  if (s === "ENTRY") return { label: "ENTRY", ...TIER_PALETTE.ENTRY, severity: 0 };
  return { label: s || "—", bg: "var(--surface-2)", text: "var(--ink-3)", severity: -1 };
}

// Per-tier text color for the Tiers KPI counts — same hex as the row
// badges' text color so "Q" in the chip reads the same amber as the
// QUICK pill below it. Pinned to the table above so a palette change
// only needs one edit.
const TIER_KPI_COLOR = {
  green: TIER_PALETTE.GREEN.text,
  quick: TIER_PALETTE.QUICK.text,
  quicksand: TIER_PALETTE.QS.text,
  gd: TIER_PALETTE.GD.text,
} as const;

// NLV currency formatter — display "792,792" in the input while keeping
// the underlying numeric value clean for the API. en-US locale matches
// the rest of the app's $ rendering (formatCurrency / formatMoney).
function formatNlvDisplay(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "";
  return Math.round(n).toLocaleString("en-US");
}

// localStorage key for the per-snapshot Mark-done list. We namespace by
// the snapshot's fetched_at so a new weekly refresh implicitly resets
// the done set (new bar = fresh decisions) without an explicit clear.
const SR8_DONE_KEY = "sr8_monitor_done_v1";

interface DoneCache {
  fetched_at: string;
  tickers: string[];
}

function loadDoneCache(): DoneCache | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(SR8_DONE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.fetched_at === "string" && Array.isArray(parsed.tickers)) {
      return { fetched_at: parsed.fetched_at, tickers: parsed.tickers.filter((t: unknown) => typeof t === "string") };
    }
  } catch { /* fall through */ }
  return null;
}

function saveDoneCache(fetched_at: string, tickers: string[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(SR8_DONE_KEY, JSON.stringify({ fetched_at, tickers }));
  } catch { /* localStorage may be full or disabled — non-fatal */ }
}

// Mark-done animation: ms before we commit to state. Matches the
// 320ms collapse the design specifies.
const MARK_DONE_ANIMATION_MS = 320;

// Numeric vs text columns for Hold-table sort. Comparing signal uses
// severity order (ENTRY < GREEN < QUICK < QS < GD); failed rows always
// pinned to the bottom regardless of direction.
const HOLD_NUMERIC_KEYS = new Set<keyof SR8AnalyzedPosition>([
  "current_pct_nlv", "shares_held", "avg_price", "current_price",
  "unreal_dollars", "unreal_pct",
]);
type HoldSortKey =
  | "ticker" | "last_signal" | "current_pct_nlv" | "shares_held"
  | "avg_price" | "current_price" | "unreal_dollars" | "unreal_pct"
  // "anchor" is a display-only column (activation date + NLV chip);
  // not sortable — clicking the header no-ops (guarded in onSortHold).
  | "anchor";

export function Sr8Monitor({ navColor }: { navColor: string }) {
  const [nlv, setNlv] = useState<number>(DEFAULT_NLV);
  // Draft holds the formatted display string (e.g. "792,792"). The
  // applied value lives in `nlv`. Commas are stripped on parse so the
  // user can type them or not.
  const [nlvDraft, setNlvDraft] = useState<string>(formatNlvDisplay(DEFAULT_NLV));
  const [data, setData] = useState<SR8MonitorResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [nlvSeeded, setNlvSeeded] = useState<boolean>(false);

  // Mark-done state. `done` is the set of tickers the user has
  // dismissed from the Action list this snapshot week; `exiting` is
  // mid-animation (320ms collapse). Both persist by namespacing on the
  // snapshot's fetched_at so a Refresh implicitly resets them.
  const [done, setDone] = useState<string[]>([]);
  const [exiting, setExiting] = useState<Set<string>>(() => new Set());

  // Hold-table sort. Default: % NLV desc — the biggest positions
  // floating to the top reads as "what's most concentrated."
  const [sort, setSort] = useState<{ key: HoldSortKey; dir: "asc" | "desc" }>({
    key: "current_pct_nlv",
    dir: "desc",
  });

  // Per-row Retry state. While `rowLoading === ticker`, that row's
  // Retry button shows a spinner instead of the ⟳ glyph. Reused
  // across Retry attempts (a click on a second failed row replaces
  // the in-flight one — design says clicks during retry are ignored).
  const [rowLoading, setRowLoading] = useState<string | null>(null);

  // Seed NLV from the active portfolio's latest journal entry once on
  // mount — same anchor active-campaign uses. User edits override.
  useEffect(() => {
    let cancelled = false;
    api.journalLatest(getActivePortfolio())
      .then(j => {
        if (cancelled) return;
        const v = parseFloat(String((j as { end_nlv?: number | string })?.end_nlv ?? 0));
        if (v > 0) {
          setNlv(v);
          setNlvDraft(formatNlvDisplay(v));
        }
      })
      .catch(err => log.debug.devOnly("sr8-monitor", "journalLatest seed failed (expected on first run)", err))
      .finally(() => { if (!cancelled) setNlvSeeded(true); });
    return () => { cancelled = true; };
  }, []);

  // Data fetch — fires when NLV changes (after seed) or on explicit
  // refresh. The endpoint is fast enough that refetching on every NLV
  // edit is fine; backend rate-limited to 30/min as a safety net.
  const fetchData = useCallback(async (nlvForFetch: number) => {
    setError(null);
    try {
      const r = await api.sr8Monitor(nlvForFetch, getActivePortfolio());
      if (r && "error" in r) {
        setError(r.error);
        setData(null);
      } else {
        setData(r as SR8MonitorResponse);
      }
    } catch (e) {
      log.error("sr8-monitor", "fetch failed", e);
      setError(e instanceof Error ? e.message : String(e));
      setData(null);
    }
  }, []);

  useEffect(() => {
    if (!nlvSeeded) return;
    let cancelled = false;
    setLoading(true);
    fetchData(nlv).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [nlv, nlvSeeded, fetchData]);

  // NLV input — apply on blur or Enter so we don't fire a fetch on
  // every keystroke. Edits invalid numbers fall back to the last good
  // value silently.
  const applyNlv = useCallback(() => {
    // Strip commas/$ before parsing so the user can paste "$792,792".
    const parsed = parseFloat(nlvDraft.replace(/[^0-9.]/g, ""));
    if (Number.isFinite(parsed) && parsed > 0) {
      setNlv(parsed);
      setNlvDraft(formatNlvDisplay(parsed));
    } else {
      setNlvDraft(formatNlvDisplay(nlv));
    }
  }, [nlvDraft, nlv]);

  const onRefresh = useCallback(async () => {
    if (refreshing) return;
    setRefreshing(true);
    setError(null);
    try {
      const r = await api.sr8Refresh(nlv, getActivePortfolio());
      if (r && "error" in r) {
        setError(r.error);
      } else {
        setData(r as SR8MonitorResponse);
      }
    } catch (e) {
      log.error("sr8-monitor", "refresh failed", e);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }, [nlv, refreshing]);

  // Snapshot freshness — drives the green/amber dot + cadence line.
  const fetchedAt = data?.meta?.fetched_at || "";
  const ageDays = useMemo(() => daysSince(fetchedAt), [fetchedAt]);
  const isFresh = ageDays != null && ageDays < 7;
  const dotColor = isFresh ? "#08a86b" : "#f59f00";
  const fetchedLabel = useMemo(() => {
    if (!fetchedAt) return "—";
    const short = shortMonthDay(fetchedAt);
    if (ageDays == null) return short;
    return `${short} · ${ageDays === 0 ? "today" : `${ageDays}d ago`}`;
  }, [fetchedAt, ageDays]);
  // Next pull date = fetchedAt + 7 days (also drives "Pull due Nd ago"
  // when ageDays >= 7).
  const cadenceLabel = useMemo(() => {
    if (!fetchedAt || ageDays == null) return "";
    if (ageDays >= 7) return `Pull due · ${ageDays - 7 === 0 ? "today" : `${ageDays - 7}d ago`}`;
    const next = new Date(fetchedAt);
    next.setDate(next.getDate() + 7);
    return `Next pull · ${next.toLocaleDateString("en-US", { month: "short", day: "numeric" })}`;
  }, [fetchedAt, ageDays]);

  const summary = data?.summary;

  // Hydrate done from localStorage whenever the snapshot's fetched_at
  // changes. The cache is namespaced by fetched_at — if the cached
  // payload's fetched_at no longer matches (i.e., a refresh produced a
  // new snapshot), the done list implicitly resets. This is why we
  // don't need an explicit "clear on refresh" branch.
  useEffect(() => {
    if (!fetchedAt) return;
    const cached = loadDoneCache();
    if (cached && cached.fetched_at === fetchedAt) {
      setDone(cached.tickers);
    } else {
      setDone([]);
    }
  }, [fetchedAt]);

  const onMarkDone = useCallback((ticker: string) => {
    if (!ticker) return;
    setExiting(prev => {
      if (prev.has(ticker)) return prev;
      const next = new Set(prev);
      next.add(ticker);
      return next;
    });
    // Commit after the collapse animation finishes so the row doesn't
    // snap out — visual continuity with the design's 320ms fade.
    setTimeout(() => {
      setDone(prevDone => {
        if (prevDone.includes(ticker)) return prevDone;
        const nextDone = [...prevDone, ticker];
        saveDoneCache(fetchedAt, nextDone);
        return nextDone;
      });
      setExiting(prev => {
        if (!prev.has(ticker)) return prev;
        const next = new Set(prev);
        next.delete(ticker);
        return next;
      });
    }, MARK_DONE_ANIMATION_MS);
  }, [fetchedAt]);

  // Split derived rows: actions (flagged + not failed + not done) vs
  // holds (everything else). Failed rows go in the Hold table per spec.
  const positions = data?.positions ?? [];
  const doneSet = useMemo(() => new Set(done), [done]);

  const actions = useMemo(
    () => positions.filter(p => p.is_action && !p.fetch_failed && !doneSet.has(p.ticker)),
    [positions, doneSet],
  );
  const holds = useMemo(
    () => positions.filter(p => p.fetch_failed || !p.is_action || doneSet.has(p.ticker)),
    [positions, doneSet],
  );

  // Hold-table sort comparator. Failed rows ALWAYS pin to the bottom
  // (regardless of sort key/direction) — they don't have meaningful
  // numeric values, so sorting them mixed in with priced rows reads
  // as confusing.
  const sortedHolds = useMemo(() => {
    const arr = [...holds];
    arr.sort((a, b) => {
      if (a.fetch_failed !== b.fetch_failed) return a.fetch_failed ? 1 : -1;
      let va: number | string;
      let vb: number | string;
      if (sort.key === "last_signal") {
        // Sort by live cascade tier severity, not the last log emission.
        // (Column key kept as "last_signal" for sort-state stability; the
        // user-facing column label is "Signal" — see HoldSection header.)
        va = signalDisplay(a.current_tier).severity;
        vb = signalDisplay(b.current_tier).severity;
      } else if (sort.key === "anchor") {
        // "anchor" is display-only (onSortHold guards this key), but keep
        // a stable comparator here in case a caller ever routes past
        // that guard.
        va = String(a.activation_nlv ?? 0);
        vb = String(b.activation_nlv ?? 0);
      } else if (HOLD_NUMERIC_KEYS.has(sort.key as keyof SR8AnalyzedPosition)) {
        va = Number((a as unknown as Record<string, unknown>)[sort.key] ?? 0);
        vb = Number((b as unknown as Record<string, unknown>)[sort.key] ?? 0);
      } else {
        va = String((a as unknown as Record<string, unknown>)[sort.key] ?? "");
        vb = String((b as unknown as Record<string, unknown>)[sort.key] ?? "");
      }
      const cmp = typeof va === "number" && typeof vb === "number"
        ? va - vb
        : String(va).localeCompare(String(vb));
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [holds, sort]);

  const onSortHold = (key: HoldSortKey) => {
    // "anchor" is display-only (activation-date chip). No stable sort
    // semantic when half the rows are on live_fallback (no anchor).
    if (key === "anchor") return;
    setSort(s => s.key === key
      ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
      : { key, dir: HOLD_NUMERIC_KEYS.has(key as keyof SR8AnalyzedPosition) || key === "last_signal" ? "desc" : "asc" });
  };

  // Per-row Retry — triggers a backend refresh which re-runs the engine
  // for ALL positions (the existing /api/sr8/refresh endpoint). The
  // failed ticker will resolve to a normal row in the next payload if
  // yfinance succeeds; if not, it stays muted. UX is per-row so only
  // this row shows a spinner. Ignores re-clicks while in flight.
  const onRowRetry = useCallback(async (ticker: string) => {
    if (rowLoading) return;
    setRowLoading(ticker);
    try {
      const r = await api.sr8Refresh(nlv, getActivePortfolio());
      if (r && !("error" in r)) {
        setData(r as SR8MonitorResponse);
      }
    } catch (e) {
      log.error("sr8-monitor", "row retry failed", e);
    } finally {
      setRowLoading(null);
    }
  }, [nlv, rowLoading]);

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="sr8-root">
      {/* Page header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0"
            style={{ fontFamily: "var(--font-fraunces), Georgia, serif", letterSpacing: "-0.02em" }}>
          SR8 Cascade <em className="italic" style={{ color: navColor }}>Monitor</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Weekly-chart signals across positions tagged sr8 · prices cached vs SPY, pulled weekly
        </div>
      </div>

      {error && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             data-testid="sr8-error"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {error}
        </div>
      )}

      {/* Control strip */}
      <div className="flex flex-wrap gap-[14px] mb-5" data-testid="sr8-control-strip">
        {/* NLV input */}
        <div className="rounded-[14px] p-[14px_16px]"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 1px 2px rgba(14,20,38,0.04)", minWidth: 230 }}>
          <div className="text-[9.5px] font-semibold uppercase mb-1.5" style={{ color: "var(--ink-4)", letterSpacing: "0.10em" }}>
            Net Liq Value
          </div>
          <div className="flex items-center gap-1.5 h-[38px] px-3 rounded-[10px]"
               style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            <span className="text-[14px]" style={{ color: "var(--ink-4)", fontFamily: mono }}>$</span>
            <input type="text" inputMode="numeric"
                   value={nlvDraft}
                   onChange={e => setNlvDraft(e.target.value)}
                   onBlur={applyNlv}
                   onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                   data-testid="sr8-nlv-input"
                   className="flex-1 bg-transparent text-[18px] font-semibold outline-none border-none"
                   style={{ color: "var(--ink)", fontFamily: mono }} />
          </div>
        </div>

        {/* Summary chips */}
        <div className="flex rounded-[14px] overflow-hidden"
             data-testid="sr8-summary"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 1px 2px rgba(14,20,38,0.04)", flex: "1 1 320px", minWidth: 320 }}>
          <SummaryChip
            label="Positions"
            value={summary ? String(summary.total_positions) : "—"}
            sub={summary ? "tagged sr8" : ""}
            testId="sr8-chip-positions"
          />
          <SummaryChip
            label="At Risk"
            value={summary ? `${summary.at_risk_pct.toFixed(1)}%` : "—"}
            sub={summary ? `${summary.flagged_count} flagged today` : ""}
            valueColor="var(--g-risk)"
            testId="sr8-chip-at-risk"
            divider
          />
          <SummaryChip
            label="To Trim"
            value={summary ? compactDollars(summary.to_trim_dollars) : "—"}
            sub={summary && summary.to_trim_dollars > 0 && nlv > 0
              ? `${((summary.to_trim_dollars / nlv) * 100).toFixed(1)}% of NLV`
              : ""}
            testId="sr8-chip-to-trim"
            divider
          />
          <SummaryChip
            label="Tiers"
            value={summary ? (
              <span data-testid="sr8-chip-tiers-counts">
                <span style={{ color: TIER_KPI_COLOR.green }}
                      data-testid="sr8-chip-tier-green">
                  {summary.tier_breakdown.green}G
                </span>
                <span style={{ color: "var(--ink-5)" }}> · </span>
                <span style={{ color: TIER_KPI_COLOR.quick }}
                      data-testid="sr8-chip-tier-quick">
                  {summary.tier_breakdown.quick}Q
                </span>
                <span style={{ color: "var(--ink-5)" }}> · </span>
                <span style={{ color: TIER_KPI_COLOR.quicksand }}
                      data-testid="sr8-chip-tier-quicksand">
                  {summary.tier_breakdown.quicksand}QS
                </span>
                <span style={{ color: "var(--ink-5)" }}> · </span>
                <span style={{ color: TIER_KPI_COLOR.gd }}
                      data-testid="sr8-chip-tier-gd">
                  {summary.tier_breakdown.gd}GD
                </span>
              </span>
            ) : "—"}
            sub=""
            testId="sr8-chip-tiers"
            divider
            small
          />
        </div>

        {/* Weekly snapshot + refresh */}
        <div className="rounded-[14px] p-[14px_16px] flex flex-col gap-2"
             data-testid="sr8-snapshot"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 1px 2px rgba(14,20,38,0.04)", minWidth: 212 }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: dotColor }} />
            <div className="text-[9.5px] font-semibold uppercase" style={{ color: "var(--ink-4)", letterSpacing: "0.10em" }}>
              Weekly Snapshot
            </div>
          </div>
          <div className="text-[12px] font-semibold" style={{ fontFamily: mono }}>{fetchedLabel}</div>
          <button type="button" onClick={onRefresh} disabled={refreshing}
                  data-testid="sr8-refresh-btn"
                  className="h-[32px] px-3 rounded-[8px] text-[12px] flex items-center justify-center gap-1.5 transition-colors disabled:opacity-60 disabled:cursor-wait"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            <span className={refreshing ? "inline-block animate-spin" : "inline-block"}>⟳</span>
            <span>{refreshing ? "Fetching…" : "Refresh weekly data"}</span>
          </button>
          {cadenceLabel && (
            <div className="text-[10px]" style={{ color: ageDays != null && ageDays >= 7 ? "#f59f00" : "var(--ink-4)" }}>
              {cadenceLabel}
            </div>
          )}
        </div>
      </div>

      {/* Action / Hold sections — or one of three polished alt-states. */}
      {loading && !data ? (
        <LoadingState />
      ) : data && data.positions.length === 0 ? (
        <EmptyState />
      ) : (
        <>
          {/* Action needed */}
          <ActionSection
            actions={actions}
            doneCount={done.length}
            activeHoldCount={holds.filter(h => !h.fetch_failed).length}
            exiting={exiting}
            onMarkDone={onMarkDone}
            navColor={navColor}
          />

          {/* Hold table */}
          <HoldSection
            holds={sortedHolds}
            sort={sort}
            onSort={onSortHold}
            navColor={navColor}
            rowLoading={rowLoading}
            onRetry={onRowRetry}
          />
        </>
      )}
    </div>
  );
}

// ─── Action section ──────────────────────────────────────────────────

function ActionSection({ actions, doneCount, activeHoldCount, exiting, onMarkDone, navColor }: {
  actions: SR8AnalyzedPosition[];
  doneCount: number;
  activeHoldCount: number;
  exiting: Set<string>;
  onMarkDone: (ticker: string) => void;
  navColor: string;
}) {
  const isAllClear = actions.length === 0;
  return (
    <section className="mb-6" data-testid="sr8-action-section">
      <div className="flex items-center gap-2 mb-3">
        <h2 className="text-[19px] font-normal m-0"
            style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Action needed
        </h2>
        {!isAllClear && (
          <span className="px-2 py-0.5 rounded-md text-[11px] font-semibold"
                style={{ background: "color-mix(in oklab, var(--down) 14%, var(--surface))", color: "var(--down)", fontFamily: mono }}>
            {actions.length}
          </span>
        )}
        {doneCount > 0 && (
          <span className="ml-auto text-[11px] font-medium" style={{ color: "var(--up)" }}>
            ✓ {doneCount} done today
          </span>
        )}
      </div>

      {isAllClear ? (
        <AllClearPanel activeHoldCount={activeHoldCount} doneCount={doneCount} />
      ) : (
        <div className="flex flex-col gap-2.5" data-testid="sr8-action-rows">
          {actions.map(p => (
            <ActionRow key={p.ticker}
                       p={p}
                       exiting={exiting.has(p.ticker)}
                       onMarkDone={onMarkDone}
                       navColor={navColor} />
          ))}
        </div>
      )}
    </section>
  );
}

// "Pilot's panel — calm when nothing's wrong." Renders when no
// actions are currently flagged but positions exist. Per the
// handoff spec: --up at 9% bg over surface, --up at 32% border,
// 48px rounded check-circle tile, Fraunces italic 22px heading.
function AllClearPanel({ activeHoldCount, doneCount }: { activeHoldCount: number; doneCount: number }) {
  return (
    <div className="px-6 py-7 rounded-[14px] flex items-center gap-4"
         data-testid="sr8-all-clear"
         style={{
           background: "color-mix(in oklab, var(--up) 9%, var(--surface))",
           border: "1px solid color-mix(in oklab, var(--up) 32%, var(--border))",
         }}>
      <div className="w-12 h-12 rounded-full flex items-center justify-center text-[22px] shrink-0"
           style={{ background: "color-mix(in oklab, var(--up) 16%, var(--surface))", color: "var(--up)" }}>
        ✓
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[22px] italic font-normal mb-0.5"
             style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: "var(--ink)" }}>
          No actions today
        </div>
        <div className="text-[12px]" style={{ color: "var(--ink-3)" }}>
          All <b style={{ color: "var(--ink)" }}>{activeHoldCount}</b> active sr8 position{activeHoldCount === 1 ? "" : "s"} are holding to plan
          {doneCount > 0 ? <>. <b style={{ color: "var(--up)" }}>{doneCount}</b> trim{doneCount === 1 ? "" : "s"} marked done.</> : "."}
        </div>
      </div>
    </div>
  );
}

function ActionRow({ p, exiting, onMarkDone, navColor: _navColor }: {
  p: SR8AnalyzedPosition;
  exiting: boolean;
  onMarkDone: (ticker: string) => void;
  navColor: string;
}) {
  const sig = signalDisplay(p.current_tier);
  const isExit = p.terminated;
  return (
    <div className="rounded-[14px] transition-all duration-200 hover:shadow-md"
         data-testid={`sr8-action-${p.ticker}`}
         style={{
           background: "var(--surface)",
           border: "1px solid var(--border)",
           borderLeft: `4px solid ${sig.text}`,
           boxShadow: "0 1px 2px rgba(14,20,38,0.04)",
           padding: "14px 16px",
           // Collapse animation: when exiting, snap maxHeight + opacity to 0
           // over MARK_DONE_ANIMATION_MS so the row visibly slides away.
           maxHeight: exiting ? 0 : 240,
           opacity: exiting ? 0 : 1,
           overflow: "hidden",
           marginBottom: exiting ? -10 : 0,  // collapse the gap too
           transition: `max-height ${MARK_DONE_ANIMATION_MS}ms ease, opacity ${MARK_DONE_ANIMATION_MS}ms ease, margin-bottom ${MARK_DONE_ANIMATION_MS}ms ease`,
         }}>
      <div className="grid items-center gap-[14px]"
           style={{ gridTemplateColumns: "72px minmax(116px,0.9fr) minmax(196px,1.5fr) 116px auto" }}>
        {/* Signal badge — bound to live cascade tier, not the last
            emission (the emission can be ENTRY for newly-entered SR8
            positions; the tier is GREEN/QUICK/QS/GD from the ratchet). */}
        <SignalBadge signal={p.current_tier} />

        {/* Ticker block */}
        <div className="flex flex-col gap-0.5 min-w-0">
          <div className="text-[19px] font-bold leading-none" style={{ fontFamily: mono, color: "var(--ink)" }}>
            {p.ticker}
          </div>
          <div className="text-[11px]" style={{ color: "var(--ink-4)", fontFamily: mono }}>
            {Math.round(p.shares_held).toLocaleString()} sh{" "}
            <span className="ml-1 inline-block px-1.5 py-0.5 rounded-md text-[10px] font-bold"
                  style={{ background: "color-mix(in oklab, var(--g-mkt) 12%, var(--surface))", color: "var(--g-mkt)" }}>
              Phase {p.phase}
            </span>
          </div>
        </div>

        {/* Recommended action */}
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-[30px] h-[30px] rounded-[8px] flex items-center justify-center text-[15px] shrink-0"
               style={{ background: `color-mix(in oklab, ${sig.text} 14%, var(--surface))`, color: sig.text }}>
            {isExit ? "↗" : "✂"}
          </div>
          <div className="flex flex-col gap-0.5 min-w-0">
            <div className="text-[15px] leading-tight" style={{ color: "var(--ink)" }}>
              {isExit ? (
                <>
                  <span className="font-bold">EXIT</span>{" "}
                  <span style={{ color: sig.text }}>all {Math.round(p.shares_held).toLocaleString()} sh</span>
                </>
              ) : (
                <>
                  <span className="font-bold">TRIM</span>{" "}
                  <span style={{ color: sig.text, fontFamily: mono }}>{Math.round(p.delta_shares).toLocaleString()} sh</span>
                  <span className="mx-1.5" style={{ color: "var(--ink-5)" }}>→</span>
                  <span style={{ fontFamily: mono }}>{p.tier_pct_nlv.toFixed(p.tier_pct_nlv >= 10 ? 0 : 2)}% NLV target</span>
                </>
              )}
            </div>
            <div className="text-[11px] truncate" style={{ color: "var(--ink-4)", fontFamily: mono }}>
              {isExit
                ? "Weekly GD · full exit ends campaign"
                : `${p.current_pct_nlv.toFixed(1)}% → ${p.tier_pct_nlv.toFixed(p.tier_pct_nlv >= 10 ? 0 : 2)}% · ${formatCurrency(p.delta_dollars, { decimals: 0 })}`}
            </div>
          </div>
        </div>

        {/* Price / NLV */}
        <div className="text-right" style={{ fontFamily: mono }}>
          <div className="text-[16px] font-semibold leading-none privacy-mask" style={{ color: "var(--ink)" }}>
            {p.current_price != null && p.current_price > 0 ? `$${p.current_price.toFixed(2)}` : "—"}
          </div>
          <div className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }}>
            <span className="font-bold" style={{ color: sig.text }}>{p.current_pct_nlv.toFixed(1)}%</span> NLV now
          </div>
        </div>

        {/* Mark done */}
        <button type="button"
                onClick={() => onMarkDone(p.ticker)}
                disabled={exiting}
                data-testid={`sr8-mark-done-${p.ticker}`}
                className="h-[38px] px-4 rounded-[10px] text-[12px] font-semibold flex items-center gap-1.5 transition-all hover:brightness-110 disabled:opacity-60"
                style={{ background: "var(--ink)", color: "#ffffff" }}>
          <span>✓</span>
          <span>Mark done</span>
        </button>
      </div>
    </div>
  );
}

function SignalBadge({ signal }: { signal: string }) {
  const sig = signalDisplay(signal);
  return (
    <span className="inline-flex items-center justify-center px-2.5 h-[26px] rounded-full text-[10.5px] font-bold uppercase tracking-[0.06em]"
          data-testid={`sr8-signal-${sig.label}`}
          style={{ background: sig.bg, color: sig.text, fontFamily: mono }}>
      {sig.label}
    </span>
  );
}

// ─── Hold section ────────────────────────────────────────────────────

function HoldSection({ holds, sort, onSort, navColor, rowLoading, onRetry }: {
  holds: SR8AnalyzedPosition[];
  sort: { key: HoldSortKey; dir: "asc" | "desc" };
  onSort: (key: HoldSortKey) => void;
  navColor: string;
  rowLoading: string | null;
  onRetry: (ticker: string) => void;
}) {
  if (holds.length === 0) return null;
  return (
    <section data-testid="sr8-hold-section">
      <div className="rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 1px 2px rgba(14,20,38,0.04)" }}>
        <div className="px-[18px] py-[14px] flex items-center gap-2"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Hold</span>
          <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            {holds.length} position{holds.length === 1 ? "" : "s"} · within plan
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" data-testid="sr8-hold-table"
                 style={{ borderCollapse: "collapse", whiteSpace: "nowrap" }}>
            <thead>
              <tr>
                {([
                  ["ticker", "Ticker", "left"],
                  ["last_signal", "Signal", "left"],
                  ["current_pct_nlv", "% NLV / target", "right"],
                  ["anchor", "Anchor", "right"],
                  ["shares_held", "Shares", "right"],
                  ["avg_price", "Avg", "right"],
                  ["current_price", "Price", "right"],
                  ["unreal_dollars", "P&L $", "right"],
                  ["unreal_pct", "P&L %", "right"],
                ] as const).map(([key, label, align]) => {
                  const active = sort.key === key;
                  const caret = active ? (sort.dir === "asc" ? "▲" : "▼") : "";
                  return (
                    <th key={key as string}
                        onClick={() => onSort(key as HoldSortKey)}
                        data-testid={`sr8-hold-th-${key}`}
                        className="px-3 py-2.5 text-[9.5px] font-bold uppercase tracking-[0.08em] cursor-pointer select-none"
                        style={{
                          background: "var(--surface-2)",
                          color: active ? "var(--g-risk)" : "var(--ink-4)",
                          textAlign: align,
                          borderBottom: "1px solid var(--border)",
                        }}>
                      {label}{caret ? ` ${caret}` : ""}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {holds.map(p => (
                <HoldRow key={p.ticker}
                         p={p}
                         retrying={rowLoading === p.ticker}
                         onRetry={onRetry} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

// AnchorCell — surfaces the migration-048 activation anchor on each
// hold row. When activation_nlv is set (backfill or live capture),
// the cell shows a compact date chip with a tooltip carrying the
// full NLV + derived Quick/QS target shares. When null (position
// pre-dates the backfill, hasn't crossed +50% cushion yet, or the
// row is a legacy shape), we render a muted "live" chip so the
// operator sees at a glance which rows are still on the fallback.
function AnchorCell({ p }: { p: SR8AnalyzedPosition }) {
  const anchored = p.anchor_source === "activation" && p.activation_nlv != null;
  if (!anchored) {
    return (
      <span
        data-testid={`sr8-anchor-fallback-${p.ticker}`}
        className="inline-block px-1.5 py-0.5 rounded text-[9.5px] font-semibold"
        style={{ background: "var(--warn-soft)", color: "#a87108" }}
        title="Live-NAV fallback: this position pre-dates the backfill or hasn't crossed +50% cushion yet. Quick/QS targets fall back to live NAV — subject to the pre-2026-07-18 inflation bug. Run scripts/sr8_activation_backfill.py to anchor."
      >
        live
      </span>
    );
  }
  // Compact activation date chip. Tooltip carries the full NLV and the
  // derived Quick / Quicksand share targets so the trader sees "the
  // ladder still has teeth" without opening the Trim Calculator.
  const anchorNlv = p.activation_nlv as number;
  const price = p.current_price ?? 0;
  const qShs = price > 0 ? Math.round((anchorNlv * 0.10) / price) : 0;
  const qsShs = price > 0 ? Math.round((anchorNlv * 0.05) / price) : 0;
  const b1Iso = p.b1_date || "";
  // b1_date is YYYY-MM-DD; render MM/DD for column density.
  const shortDate = b1Iso.length >= 10 ? `${b1Iso.slice(5, 7)}/${b1Iso.slice(8, 10)}` : "—";
  const nlvStr = `$${Math.round(anchorNlv).toLocaleString()}`;
  return (
    <span
      data-testid={`sr8-anchor-${p.ticker}`}
      className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold"
      style={{ background: "color-mix(in oklab, var(--up) 10%, var(--surface-2))", color: "var(--up)", fontFamily: mono }}
      title={`Anchored to ${nlvStr} on B1 date ${b1Iso}. Derived Quick target ${qShs} shs; Quicksand target ${qsShs} shs. See migration 048 / mors/monitor.py docstring.`}
    >
      {shortDate}
    </span>
  );
}

function HoldRow({ p, retrying, onRetry }: {
  p: SR8AnalyzedPosition;
  retrying: boolean;
  onRetry: (ticker: string) => void;
}) {
  // Signal badge + early-warn coloring reads live tier, not last log
  // emission. isEntry stays as a defensive fallback for the legacy
  // ENTRY display path; current_tier shouldn't produce it.
  const sig = signalDisplay(p.current_tier);
  const isEntry = sig.label === "ENTRY";
  const plUp = p.unreal_dollars > 0;
  const plColor = plUp ? "var(--up)" : p.unreal_dollars < 0 ? "var(--down)" : "var(--ink-3)";

  if (p.fetch_failed) {
    // Fetch-failed row: muted with a left-edge --down insert, ⚠ pill,
    // and a per-row Retry button that triggers the full /api/sr8/refresh.
    // The spinner during retry is the design's "~0.8s per-row" UX.
    return (
      <tr data-testid={`sr8-hold-row-${p.ticker}`}
          style={{ borderBottom: "1px solid var(--border)", background: "color-mix(in oklab, var(--down) 4%, var(--surface))" }}>
        <td className="px-3 py-2.5 font-semibold" style={{ fontFamily: mono, color: "var(--ink-3)", borderLeft: "3px solid var(--down)" }}>
          {p.ticker}
        </td>
        <td className="px-3 py-2.5">
          <span className="inline-flex items-center gap-1 text-[10.5px] font-semibold px-2 py-0.5 rounded-md"
                style={{ background: "var(--down-soft)", color: "var(--down)" }}>
            ⚠ fetch failed
          </span>
        </td>
        <td className="px-3 py-2.5 text-right" style={{ color: "var(--ink-4)", fontFamily: mono }}>—</td>
        <td className="px-3 py-2.5 text-right"><AnchorCell p={p} /></td>
        <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
          {Math.round(p.shares_held).toLocaleString()}
        </td>
        <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
          ${p.avg_price.toFixed(2)}
        </td>
        <td className="px-3 py-2.5 text-right" colSpan={3} style={{ color: "var(--ink-4)", fontStyle: "italic" }}>
          price unavailable
          <button type="button"
                  onClick={() => onRetry(p.ticker)}
                  disabled={retrying}
                  data-testid={`sr8-retry-${p.ticker}`}
                  title="Re-fetch the weekly snapshot from yfinance"
                  className="ml-3 px-2 py-0.5 rounded-md text-[10.5px] font-semibold transition-colors hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                  style={{ background: "var(--surface)", border: "1px solid var(--down)", color: "var(--down)" }}>
            <span className={retrying ? "inline-block animate-spin mr-1" : "inline-block mr-1"}>⟳</span>
            {retrying ? "Retrying…" : "Retry"}
          </button>
        </td>
      </tr>
    );
  }

  return (
    <tr data-testid={`sr8-hold-row-${p.ticker}`}
        style={{
          borderBottom: "1px solid var(--border)",
          background: p.early_warn ? "color-mix(in oklab, var(--warn) 7%, transparent)" : undefined,
        }}>
      <td className="px-3 py-2.5 font-bold" style={{
            fontFamily: mono,
            color: "var(--ink)",
            borderLeft: p.early_warn ? "3px solid var(--warn)" : "3px solid transparent",
          }}>
        {p.ticker}
      </td>
      <td className="px-3 py-2.5">
        <span className="inline-flex items-center gap-2">
          <SignalBadge signal={p.current_tier} />
          {p.early_warn && (
            <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md"
                  data-testid={`sr8-near-${p.ticker}`}
                  style={{ background: "var(--warn-soft)", color: "#a87108" }}>
              ⚠ NEAR
            </span>
          )}
        </span>
      </td>
      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono }}>
        <span className="font-bold" style={{ color: "var(--ink)" }}>{p.current_pct_nlv.toFixed(1)}%</span>{" "}
        <span style={{ color: isEntry ? "var(--ink-5)" : "var(--ink-4)" }}>
          / {isEntry ? "building" : `${p.tier_pct_nlv.toFixed(p.tier_pct_nlv >= 10 ? 0 : 2)}% tgt`}
        </span>
      </td>
      <td className="px-3 py-2.5 text-right"><AnchorCell p={p} /></td>
      <td className="px-3 py-2.5 text-right" style={{ fontFamily: mono }}>
        {Math.round(p.shares_held).toLocaleString()}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
        ${p.avg_price.toFixed(2)}
      </td>
      <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
        {p.current_price != null && p.current_price > 0 ? `$${p.current_price.toFixed(2)}` : "—"}
      </td>
      <td className="px-3 py-2.5 text-right font-semibold privacy-mask"
          style={{ fontFamily: mono, color: plColor }}>
        {formatCurrency(p.unreal_dollars, { showSign: true, decimals: 0 })}
      </td>
      <td className="px-3 py-2.5 text-right font-semibold"
          style={{ fontFamily: mono, color: plColor }}>
        {(p.unreal_pct >= 0 ? "+" : "")}{p.unreal_pct.toFixed(1)}%
      </td>
    </tr>
  );
}

function SummaryChip({ label, value, sub, valueColor, testId, divider, small }: {
  label: string;
  value: ReactNode;
  sub: string;
  valueColor?: string;
  testId?: string;
  divider?: boolean;
  small?: boolean;
}) {
  return (
    <div className="flex-1 p-[14px_16px]"
         data-testid={testId}
         style={{ borderLeft: divider ? "1px solid var(--border)" : undefined, minWidth: 0 }}>
      <div className="text-[9px] font-semibold uppercase tracking-[0.10em] mb-1"
           style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className={`${small ? "text-[14px]" : "text-[20px]"} font-semibold tracking-tight whitespace-nowrap`}
           style={{ color: valueColor ?? "var(--ink)", fontFamily: mono }}>
        {value}
      </div>
      {sub && (
        <div className="text-[10px] mt-1" style={{ color: "var(--ink-4)" }}>{sub}</div>
      )}
    </div>
  );
}

// Loading state — pulsing rose dot + "Fetching latest prices…" line
// + three 70px skeleton rows with a left-to-right gradient sweep. Lands
// when initial mount fires before the first sr8Monitor response.
function LoadingState() {
  return (
    <section data-testid="sr8-loading">
      <div className="flex items-center gap-2 mb-3 text-[12px] font-medium" style={{ color: "var(--ink-3)" }}>
        <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: "var(--down)" }} />
        Fetching latest prices from yfinance…
      </div>
      <div className="flex flex-col gap-2.5">
        {[0, 1, 2].map(i => (
          <div key={i}
               data-testid={`sr8-skeleton-${i}`}
               className="rounded-[14px] overflow-hidden relative"
               style={{
                 height: 70,
                 background: "var(--surface-2)",
                 border: "1px solid var(--border)",
               }}>
            <div className="absolute inset-0"
                 style={{
                   background: "linear-gradient(90deg, transparent 0%, color-mix(in oklab, var(--ink) 6%, transparent) 50%, transparent 100%)",
                   animation: "sr8-shimmer 1.3s linear infinite",
                 }} />
          </div>
        ))}
      </div>
      <style jsx global>{`
        @keyframes sr8-shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @media (prefers-reduced-motion: reduce) {
          [data-testid^="sr8-skeleton-"] > div { animation: none !important; }
        }
      `}</style>
    </section>
  );
}

// Empty state — dashed panel with the design's calm "no positions
// tagged sr8 yet" copy. `sr8` rendered as a mono rose chip inline.
function EmptyState() {
  return (
    <div className="px-8 py-10 rounded-[14px] flex items-center gap-5"
         data-testid="sr8-empty-state"
         style={{
           background: "var(--surface)",
           border: "1.5px dashed color-mix(in oklab, var(--g-risk) 30%, var(--border))",
         }}>
      <div className="w-14 h-14 rounded-full flex items-center justify-center text-[26px] shrink-0"
           style={{ background: "color-mix(in oklab, var(--g-risk) 14%, var(--surface))", color: "var(--g-risk)" }}>
        ⚡
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[24px] italic font-normal mb-1"
             style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: "var(--ink)" }}>
          No positions tagged sr8
        </div>
        <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>
          Tag an open campaign with{" "}
          <code className="px-1.5 py-0.5 rounded-md text-[11px] font-semibold"
                style={{ background: "var(--g-risk-soft)", color: "var(--g-risk)", fontFamily: mono }}>
            sr8
          </code>{" "}
          to start monitoring its cascade. Tagged positions surface here with a daily hold / trim / exit signal computed against SPY.
        </div>
      </div>
    </div>
  );
}
