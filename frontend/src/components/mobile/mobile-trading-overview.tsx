"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Line,
} from "recharts";
import { api, getActivePortfolio, type JournalHistoryPoint, type TradePosition } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

/**
 * Mobile-native Trading Overview. Same "portfolio vs SPY vs NDX equity
 * curve" template Mobile Dashboard uses, plus a compact closed-trades
 * summary (count / win-rate / total P&L / avg / best / worst) and a
 * short list of the most recent closed campaigns.
 *
 * Kept read-only: the desktop Trading Overview has a richer table +
 * recent-activity ledger; on mobile you scan the curve, sanity-check
 * the summary, and jump to desktop for detail work.
 */
type ECRange = "All" | "1Y" | "6M" | "3M";

export function MobileTradingOverview() {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [closed, setClosed] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<ECRange>("6M");

  useEffect(() => {
    Promise.all([
      api.journalHistory(getActivePortfolio(), 0).catch((err) => {
        log.error("mobile-trading-overview", "journalHistory failed", err);
        return [] as JournalHistoryPoint[];
      }),
      api.tradesClosed(getActivePortfolio(), 200).catch((err) => {
        log.error("mobile-trading-overview", "tradesClosed failed", err);
        return [] as TradePosition[];
      }),
    ])
      .then(([hist, cls]) => {
        setHistory(hist as JournalHistoryPoint[]);
        setClosed(cls as TradePosition[]);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const summary = useMemo(() => buildClosedSummary(closed), [closed]);
  const ecData = useMemo(() => buildEcData(history, range), [history, range]);
  const recent = useMemo(() => sortByCloseDateDesc(closed).slice(0, 5), [closed]);

  return (
    <div className="pb-4 flex flex-col gap-3" data-testid="mobile-trading-overview-root">
      {error && (
        <div className="px-4 py-3 rounded-m-sm text-[12px]"
             style={{
               background: "color-mix(in oklab, var(--m-down) 12%, var(--m-surface))",
               border: "1px solid var(--m-warn-border-soft)",
               color: "var(--m-down)",
             }}>
          Failed to load: {error}
        </div>
      )}

      {/* Summary tile — total closed trades, win rate, P&L */}
      <div className="rounded-m-md p-4"
           style={{
             background: "var(--m-surface)",
             border: "0.5px solid var(--m-border)",
           }}>
        <div className="grid grid-cols-3 gap-3">
          <SummaryStat label="Closed" main={String(summary.count)}
                       sub={summary.count === 1 ? "trade" : "trades"} />
          <SummaryStat label="Win rate"
                       main={summary.count > 0 ? `${(summary.winRate * 100).toFixed(0)}%` : "—"}
                       sub={`${summary.wins}W · ${summary.losses}L`}
                       color={summary.winRate >= 0.5 ? "var(--m-accent)" : "var(--m-warn)"} />
          <SummaryStat label="Total P&L"
                       main={formatCurrency(summary.totalPl, { decimals: 0, showSign: true })}
                       sub={summary.count > 0
                         ? `avg ${formatCurrency(summary.avgPl, { decimals: 0, showSign: true })}`
                         : "—"}
                       color={summary.totalPl >= 0 ? "var(--m-accent)" : "var(--m-down)"} />
        </div>
      </div>

      {/* Equity curve — same template as Mobile Dashboard EquityCurveCard */}
      <div className="rounded-m-md px-[14px] py-3"
           style={{
             background: "var(--m-surface)",
             border: "0.5px solid var(--m-border)",
           }}>
        <div className="flex items-center justify-between">
          <div className="text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
            Equity vs. Benchmarks
          </div>
          <div className="flex gap-1">
            {(["All", "1Y", "6M", "3M"] as const).map((r) => {
              const active = r === range;
              return (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRange(r)}
                  aria-pressed={active}
                  className={
                    "rounded-m-pill px-2.5 py-0.5 text-[11px] " +
                    (active
                      ? "bg-m-accent-tint font-medium text-m-accent"
                      : "bg-transparent text-m-text-dim")
                  }
                >
                  {r}
                </button>
              );
            })}
          </div>
        </div>
        <div className="mt-2 h-[140px] w-full">
          {loading ? (
            <div className="h-full rounded-m-sm animate-pulse"
                 style={{ background: "var(--m-surface-2)" }} />
          ) : ecData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={ecData} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="0" vertical={false} />
                <XAxis dataKey="date" hide />
                <YAxis hide domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    background: "var(--m-surface-2)",
                    border: "1px solid var(--m-border-strong)",
                    borderRadius: 8,
                    fontSize: 11,
                    color: "var(--m-text)",
                  }}
                  labelStyle={{ color: "var(--m-text-dim)" }}
                  formatter={(v) => {
                    const n = typeof v === "number" ? v : Number(v);
                    return Number.isFinite(n) ? `${n >= 0 ? "+" : ""}${n.toFixed(2)}%` : "—";
                  }}
                />
                <Line type="monotone" dataKey="spy" stroke="#B0A89E" strokeWidth={1}
                      strokeDasharray="3 3" strokeOpacity={0.4} dot={false}
                      isAnimationActive={false} name="SPY" />
                <Line type="monotone" dataKey="ndx" stroke="#AFA9EC" strokeWidth={1}
                      strokeDasharray="3 3" strokeOpacity={0.55} dot={false}
                      isAnimationActive={false} name="NDX" />
                <Line type="monotone" dataKey="portfolio" stroke="#4ADE80" strokeWidth={2}
                      dot={false} isAnimationActive={false} name="Portfolio" />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center text-[11px] text-m-text-dim">
              No data in this range
            </div>
          )}
        </div>
        <div className="mt-2 flex items-center gap-3 text-[10px] text-m-text-dim">
          <LegendSwatch color="#4ADE80" thickness={2} label="Portfolio" />
          <LegendSwatch color="#AFA9EC" thickness={1} opacity={0.55} label="NDX" />
          <LegendSwatch color="#B0A89E" thickness={1} opacity={0.4} label="SPY" />
        </div>
      </div>

      {/* Best / worst tile */}
      {summary.count > 0 && (
        <div className="rounded-m-md p-4 grid grid-cols-2 gap-4"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
             }}>
          <div>
            <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
              Best
            </div>
            {summary.best ? (
              <>
                <div className="mt-0.5 text-[13px] font-semibold text-m-text"
                     style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                  {summary.best.ticker}
                </div>
                <div className="mt-0.5 text-[13px] font-semibold privacy-mask"
                     style={{ color: "var(--m-accent)", fontFamily: "var(--font-jetbrains), monospace" }}>
                  {formatCurrency(summary.best.pl, { decimals: 0, showSign: true })}
                </div>
              </>
            ) : (
              <div className="mt-0.5 text-[13px] text-m-text-faint">—</div>
            )}
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
              Worst
            </div>
            {summary.worst ? (
              <>
                <div className="mt-0.5 text-[13px] font-semibold text-m-text"
                     style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                  {summary.worst.ticker}
                </div>
                <div className="mt-0.5 text-[13px] font-semibold privacy-mask"
                     style={{ color: "var(--m-down)", fontFamily: "var(--font-jetbrains), monospace" }}>
                  {formatCurrency(summary.worst.pl, { decimals: 0, showSign: true })}
                </div>
              </>
            ) : (
              <div className="mt-0.5 text-[13px] text-m-text-faint">—</div>
            )}
          </div>
        </div>
      )}

      {/* Recent closed trades — top 5 */}
      {recent.length > 0 && (
        <div className="rounded-m-md overflow-hidden"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
             }}>
          <div className="px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.06em] text-m-text-dim"
               style={{ borderBottom: "0.5px solid var(--m-border)" }}>
            Recent Closed · {recent.length}
          </div>
          {recent.map((t, idx) => {
            const pl = plOf(t);
            const isLast = idx === recent.length - 1;
            return (
              <div key={t.trade_id ?? `${t.ticker}-${idx}`}
                   className="px-4 py-3 flex items-center justify-between gap-3"
                   style={{ borderBottom: isLast ? "none" : "0.5px solid var(--m-border)" }}>
                <div className="min-w-0 flex-1">
                  <div className="text-[14px] font-semibold text-m-text truncate"
                       style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                    {t.ticker}
                  </div>
                  <div className="mt-0.5 text-[11px] text-m-text-dim">
                    {t.closed_date ? String(t.closed_date).slice(0, 10) : "—"}
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-[13px] font-semibold privacy-mask"
                       style={{
                         color: pl >= 0 ? "var(--m-accent)" : "var(--m-down)",
                         fontFamily: "var(--font-jetbrains), monospace",
                       }}>
                    {formatCurrency(pl, { decimals: 0, showSign: true })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────

function SummaryStat({ label, main, sub, color }: {
  label: string;
  main: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
        {label}
      </div>
      <div className="mt-0.5 text-[16px] font-semibold privacy-mask"
           style={{ color: color ?? "var(--m-text)", fontFamily: "var(--font-jetbrains), monospace" }}>
        {main}
      </div>
      {sub && (
        <div className="mt-0.5 text-[10px] text-m-text-faint privacy-mask"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function LegendSwatch({ color, thickness, opacity = 1, label }: {
  color: string;
  thickness: number;
  opacity?: number;
  label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden="true"
            style={{
              display: "inline-block",
              width: 12,
              height: thickness,
              background: color,
              opacity,
              borderRadius: 1,
            }} />
      {label}
    </span>
  );
}

// ── Data shapers ────────────────────────────────────────────────────

interface EcDatum { date: string; portfolio: number; spy: number; ndx: number; }

function buildEcData(history: JournalHistoryPoint[], range: ECRange): EcDatum[] {
  if (history.length === 0) return [];
  const cutoffStr = range === "All"
    ? null
    : (() => {
        const now = new Date();
        const months = range === "1Y" ? 12 : range === "6M" ? 6 : 3;
        const cutoff = new Date(now.getFullYear(), now.getMonth() - months, now.getDate());
        return cutoff.toISOString().slice(0, 10);
      })();
  const filtered = cutoffStr == null
    ? history
    : history.filter((h) => String(h.day) >= cutoffStr);
  if (filtered.length === 0) return [];
  const base = {
    portfolio: filtered[0].portfolio_ltd || 0,
    spy: filtered[0].spy_ltd || 0,
    ndx: filtered[0].ndx_ltd || 0,
  };
  return filtered.map((h) => ({
    date: String(h.day).slice(5),
    portfolio: parseFloat((((h.portfolio_ltd || 0) - base.portfolio)).toFixed(2)),
    spy: parseFloat((((h.spy_ltd || 0) - base.spy)).toFixed(2)),
    ndx: parseFloat((((h.ndx_ltd || 0) - base.ndx)).toFixed(2)),
  }));
}

interface ClosedSummary {
  count: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPl: number;
  avgPl: number;
  best: { ticker: string; pl: number } | null;
  worst: { ticker: string; pl: number } | null;
}

function plOf(t: TradePosition): number {
  const realized = Number((t as { realized_pl?: unknown }).realized_pl ?? 0);
  return Number.isFinite(realized) ? realized : 0;
}

function buildClosedSummary(closed: TradePosition[]): ClosedSummary {
  if (closed.length === 0) {
    return {
      count: 0, wins: 0, losses: 0, winRate: 0,
      totalPl: 0, avgPl: 0, best: null, worst: null,
    };
  }
  let totalPl = 0;
  let wins = 0;
  let losses = 0;
  let best: { ticker: string; pl: number } | null = null;
  let worst: { ticker: string; pl: number } | null = null;
  for (const t of closed) {
    const pl = plOf(t);
    totalPl += pl;
    if (pl > 0) wins++;
    else if (pl < 0) losses++;
    const rec = { ticker: t.ticker, pl };
    if (!best || pl > best.pl) best = rec;
    if (!worst || pl < worst.pl) worst = rec;
  }
  const decided = wins + losses;
  return {
    count: closed.length,
    wins,
    losses,
    winRate: decided > 0 ? wins / decided : 0,
    totalPl,
    avgPl: closed.length > 0 ? totalPl / closed.length : 0,
    best,
    worst,
  };
}

function sortByCloseDateDesc(closed: TradePosition[]): TradePosition[] {
  return [...closed].sort((a, b) => {
    const da = String(a.closed_date ?? "");
    const db = String(b.closed_date ?? "");
    return db.localeCompare(da);
  });
}
