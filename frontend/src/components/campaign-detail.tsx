"use client";

// Campaign Detail page — fill-by-fill ledger across all open stock campaigns.
// Build sequence:
//   Commit 1: route stub + nav rename (merged)
//   Commit 2 (this commit): page scaffold + data fetch + KPI strip
//   Commit 3: ledger table (17 cols, sort, filter, totals footer)
//   Commit 4: edit wiring (separate window — same flow as Trade Journal)
//
// Stocks only — option fills (instrument_type='OPTION' or option-shaped
// tickers) are filtered out at the campaign-scope step. Live prices only
// (no manual_price overlay), so batchPrices is called without portfolio.

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type LotClosure, type TradeDetailsBundle } from "@/lib/api";
import { walkLedger } from "@/lib/campaign-detail-walk";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

const mono = "var(--font-jetbrains), monospace";

// Per-tile gradient strings — matches the screenshot in the handoff.
// Negative variants for Realized/Unrealized swap to red.
const TILE_GRADIENTS = {
  indigo: "linear-gradient(135deg, #6366f1, #818cf8)",
  blue:   "linear-gradient(135deg, #2563eb, #60a5fa)",
  green:  "linear-gradient(135deg, #10b981, #34d399)",
  pink:   "linear-gradient(135deg, #ec4899, #f472b6)",
  orange: "linear-gradient(135deg, #f97316, #fb923c)",
  red:    "linear-gradient(135deg, #e5484d, #f87171)",
};

// KPI tile component — same shape as ACS's KPITile (active-campaign.tsx:34)
// but inlined here so the new page doesn't take a hard dependency on the
// neighbour. Uses the same radial-glow + gradient idiom.
function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub: string; gradient: string }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[18px] text-white flex flex-col justify-between min-h-[108px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "0 4px 14px rgba(0,0,0,0.08)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full"
           style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] opacity-85">{label}</div>
        <div className="text-[28px] font-semibold tracking-tight mt-1 privacy-mask" style={{ fontFamily: mono }}>{value}</div>
      </div>
      <div className="relative z-10 text-[11px] font-medium opacity-80 privacy-mask">{sub}</div>
    </div>
  );
}

// Same option-detection regex used elsewhere (parseOptionTicker etc).
// Kept local to avoid pulling in a module just for one regex.
function isOptionTicker(t: string): boolean {
  return /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(t || "").trim());
}

function isStockCampaign(t: TradePosition): boolean {
  const type = String((t as { instrument_type?: string }).instrument_type || "").toUpperCase();
  if (type === "STOCK") return true;
  if (type === "OPTION") return false;
  // Legacy fallback for rows without instrument_type set: regex on ticker.
  return !isOptionTicker(t.ticker || "");
}

export function CampaignDetail({ navColor }: { navColor: string }) {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closures, setClosures] = useState<LotClosure[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);

  const fetchAll = useCallback(async () => {
    setError(null);
    try {
      const [openRaw, bundleRaw] = await Promise.all([
        api.tradesOpen(getActivePortfolio()).catch(err => {
          log.error("campaign-detail", "tradesOpen failed", err);
          return [] as TradePosition[];
        }),
        api.tradesOpenDetails(getActivePortfolio()).catch(err => {
          log.error("campaign-detail", "tradesOpenDetails failed", err);
          return { details: [], lot_closures: [] } as TradeDetailsBundle;
        }),
      ]);

      // Stocks only. Scope details + closures to the surviving trade_ids.
      const allOpen = openRaw as TradePosition[];
      const stockOpen = allOpen.filter(isStockCampaign);
      const stockIds = new Set(stockOpen.map(t => String(t.trade_id || "")));
      const stockDetails = (bundleRaw.details || []).filter(d =>
        stockIds.has(String(d.trade_id || ""))
      );
      const stockClosures = (bundleRaw.lot_closures || []).filter(c =>
        stockIds.has(String(c.trade_id || ""))
      );

      // Live prices only — no portfolio arg means no manual_price overlay.
      const tickers = Array.from(new Set(stockOpen.map(t => t.ticker).filter(Boolean) as string[]));
      let prices: Record<string, number> = {};
      if (tickers.length > 0) {
        try {
          const r = await api.batchPrices(tickers);
          if (r && !("error" in r)) prices = r as Record<string, number>;
        } catch (e) {
          log.error("campaign-detail", "batchPrices failed", e);
        }
      }

      setOpenTrades(stockOpen);
      setDetails(stockDetails);
      setClosures(stockClosures);
      setLivePrices(prices);
      setLastUpdatedAt(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchAll().finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [fetchAll]);

  const onRefresh = useCallback(async () => {
    if (refreshing) return;
    setRefreshing(true);
    await fetchAll();
    setRefreshing(false);
  }, [fetchAll, refreshing]);

  // Walker output — per-detail Status + Remaining; openLotCount KPI input.
  const walked = useMemo(() => walkLedger(details), [details]);

  // KPI math (unfiltered — these tiles always reflect the WHOLE ledger
  // per the spec; the filter toolbar arriving in Commit 3 only narrows
  // the table body + footer totals).
  const kpis = useMemo(() => {
    const tradeMultiplier = new Map<string, number>();
    for (const t of openTrades) {
      const m = parseFloat(String((t as { multiplier?: string | number }).multiplier ?? 0));
      tradeMultiplier.set(String(t.trade_id), m > 0 ? m : 1);
    }

    let unrealized = 0;
    let marketValue = 0;
    for (const d of details) {
      if (String(d.action || "").toUpperCase() !== "BUY") continue;
      const detailId = (d as { detail_id?: number }).detail_id;
      const info = detailId != null ? walked.perDetail.get(detailId) : undefined;
      if (!info || info.remaining == null || info.remaining <= 0) continue;
      const remaining = info.remaining;
      const cost = parseFloat(String(d.amount || 0));
      const mark = livePrices[d.ticker || ""] ?? cost;
      const mult = tradeMultiplier.get(String(d.trade_id || "")) ?? 1;
      unrealized += (mark - cost) * remaining * mult;
      marketValue += mark * remaining * mult;
    }

    const realized = closures.reduce((sum, c) => sum + (parseFloat(String(c.realized_pl || 0)) || 0), 0);
    const transactionCount = details.length;
    const activeCampaignCount = openTrades.length;

    return {
      transactionCount,
      activeCampaignCount,
      openLotCount: walked.openLotCount,
      realized,
      unrealized,
      marketValue,
    };
  }, [openTrades, details, closures, livePrices, walked]);

  const lastUpdatedLabel = lastUpdatedAt
    ? lastUpdatedAt.toISOString().slice(0, 16).replace("T", " ")
    : "";

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="campaign-detail-root">
      {/* Page header — H1 + sub + Export / Refresh action buttons */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Active Campaign <em className="italic" style={{ color: navColor }}>Detail</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Every fill across all open stock campaigns
            {lastUpdatedLabel ? ` · as of ${lastUpdatedLabel}` : ""}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button type="button" disabled
                  data-testid="export-csv-btn"
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 cursor-not-allowed opacity-50"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}
                  title="CSV export lands with the ledger table in the next commit">
            ↓ Export CSV
          </button>
          <button type="button" onClick={onRefresh} disabled={refreshing}
                  data-testid="refresh-btn"
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: refreshing ? "var(--ink-4)" : "var(--ink-2)" }}>
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             data-testid="campaign-detail-error"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {error}
        </div>
      )}

      {/* KPI strip — 5 gradient tiles. These ALWAYS reflect the whole
          ledger (unfiltered). The Commit-3 filter toolbar will narrow
          the table body + footer totals only. */}
      {loading && !lastUpdatedAt ? (
        <div className="grid grid-cols-5 gap-[14px]" data-testid="kpi-strip-loading">
          {[0, 1, 2, 3, 4].map(i => (
            <div key={i} className="rounded-[14px] animate-pulse min-h-[108px]"
                 style={{ background: "var(--bg-2)" }} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-5 gap-[14px]" data-testid="kpi-strip">
          <KPITile
            label="Transactions"
            value={String(kpis.transactionCount)}
            sub={`${kpis.activeCampaignCount} active campaigns`}
            gradient={TILE_GRADIENTS.indigo}
          />
          <KPITile
            label="Open Lots"
            value={String(kpis.openLotCount)}
            sub="held, unrealized"
            gradient={TILE_GRADIENTS.blue}
          />
          <KPITile
            label="Realized P&L"
            value={formatCurrency(kpis.realized, { decimals: 0 })}
            sub="closed trims"
            gradient={kpis.realized >= 0 ? TILE_GRADIENTS.green : TILE_GRADIENTS.red}
          />
          <KPITile
            label="Unrealized P&L"
            value={formatCurrency(kpis.unrealized, { decimals: 0 })}
            sub="open at mark"
            gradient={kpis.unrealized >= 0 ? TILE_GRADIENTS.pink : TILE_GRADIENTS.red}
          />
          <KPITile
            label="Market Value"
            value={formatCurrency(kpis.marketValue, { decimals: 0 })}
            sub="open positions"
            gradient={TILE_GRADIENTS.orange}
          />
        </div>
      )}

      {/* Placeholder where the ledger table lands in Commit 3. */}
      <div className="mt-5 px-4 py-8 text-center text-[12px] rounded-[14px]"
           data-testid="ledger-placeholder"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
        Ledger table coming in Commit 3 — filter toolbar, 17 sortable columns, sticky totals footer, inline edit affordance.
      </div>
    </div>
  );
}
