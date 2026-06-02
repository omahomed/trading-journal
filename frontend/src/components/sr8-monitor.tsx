"use client";

// SR8 Cascade Monitor — daily position-management screen for positions
// tagged sr8. Build sequence:
//   Commit 1: backend endpoint + route stub + nav rename (merged).
//   Commit 2 (this commit): page scaffold — control strip + summary
//     chips + weekly snapshot card + data fetch. Action/Hold sections
//     still a placeholder.
//   Commit 3: Action / Hold sections + Mark-done state.
//   Commit 4: All-clear / Loading / Empty / Retry states.
//
// The cascade math lives in mors/monitor.py (Python). The
// /api/sr8/monitor endpoint wraps it; this page renders the response.

import { useState, useEffect, useCallback, useMemo } from "react";
import { api, getActivePortfolio, type SR8MonitorResponse } from "@/lib/api";
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

export function Sr8Monitor({ navColor }: { navColor: string }) {
  const [nlv, setNlv] = useState<number>(DEFAULT_NLV);
  const [nlvDraft, setNlvDraft] = useState<string>(String(DEFAULT_NLV));
  const [data, setData] = useState<SR8MonitorResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [nlvSeeded, setNlvSeeded] = useState<boolean>(false);

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
          setNlvDraft(String(Math.round(v)));
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
      const r = await api.sr8Monitor(nlvForFetch);
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
    const parsed = parseFloat(nlvDraft.replace(/[^0-9.]/g, ""));
    if (Number.isFinite(parsed) && parsed > 0) {
      setNlv(parsed);
      setNlvDraft(String(Math.round(parsed)));
    } else {
      setNlvDraft(String(Math.round(nlv)));
    }
  }, [nlvDraft, nlv]);

  const onRefresh = useCallback(async () => {
    if (refreshing) return;
    setRefreshing(true);
    setError(null);
    try {
      const r = await api.sr8Refresh(nlv);
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
            label="Cascades"
            value={summary
              ? `${summary.cascade_breakdown.cascade_20} 20-cas / ${summary.cascade_breakdown.cascade_15} 15-cas`
              : "—"}
            sub=""
            testId="sr8-chip-cascades"
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

      {/* Action / Hold sections placeholder for Commit 3. Loading skeleton
          renders here for now so the page doesn't look empty during fetch. */}
      {loading && !data ? (
        <div data-testid="sr8-loading" className="animate-pulse">
          <div className="h-[100px] rounded-[14px] mb-3" style={{ background: "var(--bg-2)" }} />
          <div className="h-[100px] rounded-[14px] mb-3" style={{ background: "var(--bg-2)" }} />
          <div className="h-[100px] rounded-[14px]" style={{ background: "var(--bg-2)" }} />
        </div>
      ) : (
        <div className="px-4 py-8 text-center text-[12px] rounded-[14px]"
             data-testid="sr8-body-placeholder"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
          {data && data.positions.length === 0
            ? "No positions tagged sr8. (Empty-state polish lands in Commit 4.)"
            : "Action / Hold sections coming in Commit 3 — for each position, a hold/trim/exit recommendation rendered from the cascade engine."}
        </div>
      )}
    </div>
  );
}

function SummaryChip({ label, value, sub, valueColor, testId, divider, small }: {
  label: string;
  value: string;
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
