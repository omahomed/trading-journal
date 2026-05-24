"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  api,
  getActivePortfolio,
  type DashboardMetrics,
  type JournalEntry,
  type JournalHistoryPoint,
  type TradePosition,
} from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

/**
 * Phase 2 Step 2 — mobile Dashboard surface. Consumes the SAME data
 * hooks the desktop Dashboard uses (no parallel mobile-only context)
 * but renders a slim, glance-first layout per docs/mobile/anchors/
 * dashboard-v1.html.
 *
 * Sections, top to bottom:
 *   1. "As of {date}" caption
 *   2. Featured NLV card with day delta
 *   3. 2×2 KPI grid: LTD / YTD / EOD Exposure / Drawdown
 *   4. Equity Curve card with All/1Y/6M/3M toggle (default 6M)
 *   5. Last 10 Trades sequence strip (oldest → newest)
 *
 * Data: 5 endpoints (vs desktop's 8). dashboardMetrics + journalHistory
 * + journalLatest (for portfolio_heat) + tradesOpen (for the "N pos"
 * sub on Exposure, since DashboardMetrics.total_holdings is dollars
 * not count) + tradesClosed(limit 10) for the strip. Skipped from
 * desktop: events, tradesRecent, batchPrices.
 *
 * Active-portfolio switch triggers window.location.reload() via the
 * shared usePortfolio() context, so this component never needs to
 * teardown/rebuild on its own — the whole page remounts.
 */

// Match desktop's throttle so a focus-storm doesn't fan out into a
// 5-endpoint refetch on every visibility change.
const STALE_THROTTLE_MS = 5_000;

type ECRange = "All" | "1Y" | "6M" | "3M";

export function MobileDashboard() {
  const { activePortfolio } = usePortfolio();
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [latest, setLatest] = useState<JournalEntry | null>(null);
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [ecRange, setEcRange] = useState<ECRange>("6M");

  const lastFetchAtRef = useRef<number>(0);
  const inFlightRef = useRef<boolean>(false);

  const loadData = useCallback(async (opts?: { force?: boolean }) => {
    const force = !!opts?.force;
    if (!force && Date.now() - lastFetchAtRef.current < STALE_THROTTLE_MS) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const activeId = activePortfolio?.id;
      const [dash, hist, lat, open, closed] = await Promise.all([
        activeId != null
          ? api.dashboardMetrics(activeId).catch((err) => {
              log.error("mobile-dashboard", "metrics fetch failed", err);
              return null;
            })
          : Promise.resolve(null),
        api.journalHistory(getActivePortfolio(), 0).catch((err) => {
          log.error("mobile-dashboard", "journal history fetch failed", err);
          return [];
        }),
        api.journalLatest().catch((err) => {
          log.error("mobile-dashboard", "journal latest fetch failed", err);
          return null;
        }),
        api.tradesOpen().catch((err) => {
          log.error("mobile-dashboard", "open trades fetch failed", err);
          return [];
        }),
        api.tradesClosed(getActivePortfolio(), 10).catch((err) => {
          log.error("mobile-dashboard", "closed trades fetch failed", err);
          return [];
        }),
      ]);
      const safeMetrics =
        dash && typeof dash === "object" && !("error" in dash)
          ? (dash as DashboardMetrics)
          : null;
      setMetrics(safeMetrics);
      setHistory(hist as JournalHistoryPoint[]);
      setLatest(lat as JournalEntry | null);
      setOpenTrades(Array.isArray(open) ? (open as TradePosition[]) : []);
      setClosedTrades(Array.isArray(closed) ? (closed as TradePosition[]) : []);
      setLoading(false);
      lastFetchAtRef.current = Date.now();
    } finally {
      inFlightRef.current = false;
    }
  }, [activePortfolio?.id]);

  useEffect(() => {
    loadData({ force: true });
  }, [loadData]);

  useEffect(() => {
    const onFocus = () => {
      if (!document.hidden) loadData();
    };
    document.addEventListener("visibilitychange", onFocus);
    window.addEventListener("focus", onFocus);
    return () => {
      document.removeEventListener("visibilitychange", onFocus);
      window.removeEventListener("focus", onFocus);
    };
  }, [loadData]);

  if (loading) {
    return <DashboardSkeleton />;
  }

  return (
    <div className="flex flex-col gap-2.5 pt-1">
      <AsOfCaption asOf={metrics?.as_of_date ?? null} />
      <FeaturedNlv metrics={metrics} />
      <KpiGrid
        metrics={metrics}
        history={history}
        openCount={openTrades.length}
        portfolioHeat={latest?.portfolio_heat ?? 0}
      />
      <EquityCurveCard history={history} range={ecRange} setRange={setEcRange} />
      <Last10Strip trades={closedTrades} />
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────

function AsOfCaption({ asOf }: { asOf: string | null }) {
  if (!asOf) return null;
  return <div className="text-[11px] text-m-text-dim">As of {asOf}</div>;
}

function FeaturedNlv({ metrics }: { metrics: DashboardMetrics | null }) {
  const journalAvailable = metrics?.journal_available ?? false;
  const nlv = metrics?.nlv ?? 0;
  const dailyDol = metrics?.nlv_delta_dollar ?? null;
  const dailyPct = metrics?.nlv_delta_pct ?? null;

  return (
    <div className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-[18px] py-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-m-text-dim">
        Net Liq Value
      </div>
      <div className="mt-1.5 font-m-num text-[36px] font-medium tabular-nums tracking-[-0.02em] text-m-text">
        {journalAvailable ? formatCurrency(nlv, { decimals: 0 }) : "—"}
      </div>
      <DayDeltaRow dollar={dailyDol} pct={dailyPct} journalAvailable={journalAvailable} />
    </div>
  );
}

function DayDeltaRow({
  dollar,
  pct,
  journalAvailable,
}: {
  dollar: number | null;
  pct: number | null;
  journalAvailable: boolean;
}) {
  if (!journalAvailable) {
    return (
      <div className="mt-2 text-[11px] text-m-text-dim">
        Save your first daily routine
      </div>
    );
  }
  if (dollar == null || pct == null) {
    return (
      <div className="mt-2 text-[11px] text-m-text-dim">First entry — no prior day</div>
    );
  }
  const positive = dollar >= 0;
  const toneClass = positive ? "text-m-accent" : "text-m-down";
  return (
    <div className="mt-2 flex items-baseline gap-2">
      <span className={`font-m-num text-[13px] font-medium tabular-nums ${toneClass}`}>
        {formatCurrency(dollar, { showSign: true, signGlyph: "unicode", decimals: 0 })}
      </span>
      <span className={`font-m-num text-[13px] tabular-nums ${toneClass}`}>
        {pct >= 0 ? "+" : ""}
        {pct.toFixed(2)}%
      </span>
      <span className="ml-auto text-[11px] text-m-text-dim">today</span>
    </div>
  );
}

function KpiGrid({
  metrics,
  history,
  openCount,
  portfolioHeat,
}: {
  metrics: DashboardMetrics | null;
  history: JournalHistoryPoint[];
  openCount: number;
  portfolioHeat: number;
}) {
  const journalAvailable = metrics?.journal_available ?? false;
  const ltdPct = metrics?.ltd_pct ?? null;
  const ltdDol = metrics?.ltd_pl_dollar ?? null;
  const ytdPct = metrics?.ytd_pct ?? null;
  const ytdAvailable = metrics?.ytd_available ?? false;
  const exposure = metrics?.exposure_pct ?? null;
  const ddPct = metrics?.drawdown_current_pct ?? null;
  const peakNlv = metrics?.drawdown_peak_nlv ?? null;

  // SPY / NDX YTD — derived from the journal history tail, matching the
  // desktop pattern. Index fields land via the index signature on
  // JournalHistoryPoint; they're not surfaced on the typed shape.
  const { ytdSpy, ytdNdx } = useMemo(() => computeYtdBenchmarks(history), [history]);

  return (
    <div className="grid grid-cols-2 gap-2.5">
      <KpiCard
        label="LTD Return"
        value={
          journalAvailable && ltdPct != null
            ? `${ltdPct >= 0 ? "+" : ""}${ltdPct.toFixed(2)}%`
            : "—"
        }
        valueTone={ltdPct != null && ltdPct >= 0 ? "accent" : ltdPct != null ? "down" : "text"}
        sub={
          ltdDol != null
            ? formatCurrency(ltdDol, { showSign: true, decimals: 0 })
            : undefined
        }
      />
      <KpiCard
        label="YTD Return"
        value={
          ytdAvailable && ytdPct != null
            ? `${ytdPct >= 0 ? "+" : ""}${ytdPct.toFixed(2)}%`
            : "—"
        }
        valueTone={ytdPct != null && ytdPct >= 0 ? "accent" : ytdPct != null ? "down" : "text"}
        subLines={
          ytdAvailable
            ? [
                `SPY ${ytdSpy >= 0 ? "+" : ""}${ytdSpy.toFixed(2)}%`,
                `NDX ${ytdNdx >= 0 ? "+" : ""}${ytdNdx.toFixed(2)}%`,
              ]
            : undefined
        }
      />
      <KpiCard
        label="EOD Exposure"
        value={
          journalAvailable && exposure != null ? `${exposure.toFixed(1)}%` : "—"
        }
        valueTone={exposure != null && exposure > 100 ? "warn" : "text"}
        sub={
          journalAvailable
            ? `${openCount} pos · risk ${portfolioHeat.toFixed(1)}%`
            : undefined
        }
      />
      <KpiCard
        label="Drawdown"
        value={journalAvailable && ddPct != null ? `${ddPct.toFixed(2)}%` : "—"}
        valueTone="text"
        sub={
          journalAvailable && ddPct != null && Math.abs(ddPct) >= 0.01 && peakNlv != null
            ? `peak ${formatCurrency(peakNlv, { decimals: 0 })}`
            : undefined
        }
        tag={
          journalAvailable && ddPct != null && Math.abs(ddPct) < 0.01
            ? { label: "Clear", tone: "accent" }
            : undefined
        }
      />
    </div>
  );
}

function KpiCard({
  label,
  value,
  valueTone,
  sub,
  subLines,
  tag,
}: {
  label: string;
  value: string;
  valueTone: "text" | "accent" | "down" | "warn";
  sub?: string;
  subLines?: string[];
  tag?: { label: string; tone: "accent" };
}) {
  const valueClass =
    valueTone === "accent"
      ? "text-m-accent"
      : valueTone === "down"
        ? "text-m-down"
        : valueTone === "warn"
          ? "text-m-warn"
          : "text-m-text";
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-3.5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
        {label}
      </div>
      <div className={`mt-1 font-m-num text-[20px] font-medium tabular-nums ${valueClass}`}>
        {value}
      </div>
      {tag && (
        <div className="mt-1 inline-block rounded-m-pill bg-m-accent-tint px-2 py-0.5 text-[10px] font-semibold tracking-wide text-m-accent">
          {tag.label}
        </div>
      )}
      {sub && !tag && (
        <div className="mt-0.5 font-m-num text-[11px] tabular-nums text-m-text-dim">
          {sub}
        </div>
      )}
      {subLines && !tag && (
        <div className="mt-0.5 font-m-num text-[10px] leading-tight tabular-nums text-m-text-dim">
          {subLines.map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function EquityCurveCard({
  history,
  range,
  setRange,
}: {
  history: JournalHistoryPoint[];
  range: ECRange;
  setRange: (r: ECRange) => void;
}) {
  const data = useMemo(() => buildEcData(history, range), [history, range]);

  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-3">
      <div className="flex items-center justify-between">
        <div className="text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Equity Curve
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
      <div className="mt-2 h-[120px] w-full">
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
              <CartesianGrid
                stroke="rgba(255,255,255,0.04)"
                strokeDasharray="0"
                vertical={false}
              />
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
              <Line
                type="monotone"
                dataKey="spy"
                stroke="#B0A89E"
                strokeWidth={1}
                strokeDasharray="3 3"
                strokeOpacity={0.4}
                dot={false}
                isAnimationActive={false}
                name="SPY"
              />
              <Line
                type="monotone"
                dataKey="ndx"
                stroke="#AFA9EC"
                strokeWidth={1}
                strokeDasharray="3 3"
                strokeOpacity={0.55}
                dot={false}
                isAnimationActive={false}
                name="NDX"
              />
              <Line
                type="monotone"
                dataKey="portfolio"
                stroke="#4ADE80"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="Portfolio"
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-[11px] text-m-text-dim">
            No data in this range
          </div>
        )}
      </div>
      <div className="mt-2 flex gap-3.5 text-[11px] text-m-text-dim">
        <LegendSwatch color="#4ADE80" thickness={2} label="Portfolio" />
        <LegendSwatch color="#AFA9EC" thickness={1} opacity={0.55} label="NDX" />
        <LegendSwatch color="#B0A89E" thickness={1} opacity={0.4} label="SPY" />
      </div>
    </div>
  );
}

function LegendSwatch({
  color,
  thickness,
  opacity = 1,
  label,
}: {
  color: string;
  thickness: number;
  opacity?: number;
  label: string;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block w-3"
        style={{ height: thickness, background: color, opacity }}
      />
      {label}
    </span>
  );
}

function Last10Strip({ trades }: { trades: TradePosition[] }) {
  // Backend returns newest-first (closed_date DESC); reverse for OLDEST → NEWEST.
  const ordered = useMemo(() => [...trades].reverse(), [trades]);
  const wins = ordered.filter((t) => Number(t.realized_pl ?? 0) > 0).length;
  const winRate = ordered.length > 0 ? (wins / ordered.length) * 100 : 0;
  const slots = padToTen(ordered);

  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-3">
      <div className="flex items-baseline justify-between">
        <div className="text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Last 10 Trades
        </div>
        {ordered.length > 0 && (
          <div className="font-m-num text-xs font-medium tabular-nums text-m-accent">
            {winRate.toFixed(0)}% win rate
          </div>
        )}
      </div>
      <div className="mt-2.5 flex gap-1">
        {slots.map((slot, i) => (
          <div
            key={i}
            className={
              "h-7 flex-1 rounded-sm " +
              (slot === "win"
                ? "bg-m-accent opacity-85"
                : slot === "loss"
                  ? "bg-m-down opacity-85"
                  : "bg-m-border")
            }
            aria-label={
              slot === "win" ? "Winning trade" : slot === "loss" ? "Losing trade" : "Empty slot"
            }
          />
        ))}
      </div>
      <div className="mt-2 flex justify-between text-[10px] text-m-text-dim">
        <span>OLDEST</span>
        <span>NEWEST</span>
      </div>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-2.5 pt-1" aria-busy="true" aria-label="Loading dashboard">
      <div className="h-3.5 w-32 rounded bg-m-surface" />
      <div className="h-[112px] rounded-m-lg bg-m-surface" />
      <div className="grid grid-cols-2 gap-2.5">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-[82px] rounded-m-md bg-m-surface" />
        ))}
      </div>
      <div className="h-[176px] rounded-m-md bg-m-surface" />
      <div className="h-[80px] rounded-m-md bg-m-surface" />
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────

type Slot = "win" | "loss" | "empty";

function padToTen(trades: TradePosition[]): Slot[] {
  const out: Slot[] = trades
    .slice(-10)
    .map((t) => (Number(t.realized_pl ?? 0) > 0 ? "win" : "loss"));
  while (out.length < 10) out.unshift("empty");
  return out;
}

interface EcDatum {
  date: string;
  portfolio: number;
  spy: number;
  ndx: number;
}

function buildEcData(history: JournalHistoryPoint[], range: ECRange): EcDatum[] {
  if (history.length === 0) return [];

  const cutoffStr =
    range === "All"
      ? null
      : (() => {
          const now = new Date();
          const months = range === "1Y" ? 12 : range === "6M" ? 6 : 3;
          const cutoff = new Date(now.getFullYear(), now.getMonth() - months, now.getDate());
          return cutoff.toISOString().slice(0, 10);
        })();

  const filtered =
    cutoffStr == null
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

function computeYtdBenchmarks(history: JournalHistoryPoint[]): {
  ytdSpy: number;
  ytdNdx: number;
} {
  if (history.length === 0) return { ytdSpy: 0, ytdNdx: 0 };
  const currYearStr = String(new Date().getFullYear());
  const ytd = history.filter((h) => String(h.day).slice(0, 4) === currYearStr);
  if (ytd.length === 0) return { ytdSpy: 0, ytdNdx: 0 };
  const jan1 = ytd[0];
  const last = history[history.length - 1];
  // The wire response carries absolute index levels as `spy` / `nasdaq`
  // (separate from the rebased `spy_ltd` / `ndx_ltd` used by the chart).
  // Both reach us via JournalHistoryPoint's index signature.
  const spySt = Number((jan1 as any).spy) || 0;
  const spyCurr = Number((last as any).spy) || 0;
  const ndxSt = Number((jan1 as any).nasdaq) || 0;
  const ndxCurr = Number((last as any).nasdaq) || 0;
  return {
    ytdSpy: spySt > 0 ? (spyCurr / spySt - 1) * 100 : 0,
    ytdNdx: ndxSt > 0 ? (ndxCurr / ndxSt - 1) * 100 : 0,
  };
}
