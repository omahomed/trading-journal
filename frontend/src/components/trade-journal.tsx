"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type LotClosure, type Strategy } from "@/lib/api";
import { matchesAnyTradeQuery } from "@/lib/trade-search";
import { InteractiveChart } from "./interactive-chart";
import { StrategyChip } from "./strategy-chip";
import { StrategyFlyout, StrategyFlatList, useCoarsePointer } from "./strategy-flyout";

// Rule dropdowns for the closed-trade edit modal. Duplicated from
// trade-manager.tsx / log-buy.tsx / log-sell.tsx / import-trades.tsx —
// the codebase pattern is per-component copies rather than a shared
// module. Stable strings; keep in sync if those copies move.
const BUY_RULES = [
  "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
  "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot", "br1.8 High Tight Flag",
  "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
  "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", "br3.5 Reclaim 8e",
  "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", "br4.5 PB 8e", "br4.6 VWAP",
  "br5.1 Undercut & Rally", "br5.2 Upside Reversal",
  "br6.1 Gapper", "br6.2 Continuation Gap Up",
  "br7.1 TQQQ Strategy", "br7.2 New High after Gentle PB", "br7.3 JL Century Mark",
  "br8.1 Daily STL Break", "br8.2 Weekly STL Break", "br8.3 Monthly STL Break",
  "br9.1 21e Strategy",
  "br10.1 Hedging with leverage product",
  "br11.1 Shorting",
  "br12.1 Option Play",
];

const SELL_RULES = [
  "sr1 Capital Protection", "sr2 Trailing Stop", "sr3 Portfolio Management",
  "sr4 Time Stop", "sr5 Climax Top", "sr6 Exhaustion Gap",
  "sr7 200d Moving Avg Break", "sr8 Living Below 50d", "sr9 Failed Breakout",
  "sr10 Scale-Out T1 (-3%)", "sr11 Scale-Out T2 (-5%)", "sr12 Scale-Out T3 (-8%)",
  "sr13 Earnings Exit", "sr14 Market Correction Exit",
  "sr15 BE Stop Out (moved at +10%)",
  "sr16 Profit Taking",
];

type SortKey = "newest" | "oldest" | "best" | "worst" | "ticker";
type StatusFilter = "none" | "all" | "open" | "closed";
type DateRange = "all" | "7d" | "30d" | "90d" | "ytd";
type RecentActivity = "off" | "10" | "20" | "50";

function StatusBadge({ status }: { status: string }) {
  const isOpen = status.toUpperCase() === "OPEN";
  return (
    <span className="inline-block px-2.5 py-0.5 rounded-full text-[10px] font-semibold"
          style={{
            background: isOpen ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : "var(--bg-2)",
            color: isOpen ? "#16a34a" : "#6b7280",
          }}>
      {status}
    </span>
  );
}

function PLColor(val: number) {
  return val > 0 ? "#08a86b" : val < 0 ? "#e5484d" : "var(--ink-3)";
}

function GradeStars({ value, onChange }: { value: number | null | undefined; onChange: (v: number | null) => void }) {
  const [hover, setHover] = useState<number>(0);
  const current = typeof value === "number" && value >= 1 && value <= 5 ? value : 0;
  return (
    <div className="flex items-center gap-0.5" onMouseLeave={() => setHover(0)}
         onClick={e => e.stopPropagation()} title={current ? `${current}/5 — click again to clear` : "Grade this trade"}>
      {[1, 2, 3, 4, 5].map(n => {
        const filled = hover ? n <= hover : n <= current;
        return (
          <button key={n} type="button"
                  onMouseEnter={() => setHover(n)}
                  onClick={() => onChange(current === n ? null : n)}
                  className="p-0.5 bg-transparent border-0 cursor-pointer transition-transform hover:scale-110"
                  aria-label={`${n} star${n > 1 ? "s" : ""}`}>
            <svg width="14" height="14" viewBox="0 0 24 24"
                 fill={filled ? "#f59f00" : "none"}
                 stroke={filled ? "#f59f00" : "var(--ink-4)"}
                 strokeWidth="2" strokeLinejoin="round">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
          </button>
        );
      })}
    </div>
  );
}

function CardMetric({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[16px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: color || "var(--ink)" }}>{value}</div>
      {sub && <div className="text-[11px] mt-0.5 privacy-mask" style={{ color: "var(--ink-4)" }}>{sub}</div>}
    </div>
  );
}

const IMAGE_TYPE_MAP: Record<string, string> = {
  entry: "Entry Charts", weekly: "Entry Charts", daily: "Entry Charts",
  marketsurge: "Entry Charts",
  position_change: "Position Changes", position: "Position Changes", exit: "Position Changes",
};

const FUND_RATINGS = [
  { key: "composite_rating", label: "Composite", max: 99 },
  { key: "eps_rating", label: "EPS Rating", max: 99 },
  { key: "rs_rating", label: "RS Rating", max: 99 },
  { key: "group_rs_rating", label: "Group RS" },
  { key: "smr_rating", label: "SMR" },
  { key: "acc_dis_rating", label: "Acc/Dis" },
];

type LifoRow = {
  tx: TradeDetail; displayShares: number; remaining: number;
  exitPrice: number; realizedPl: number; returnPct: number;
  unrealizedPl: number; status: string; value: number; isSell: boolean;
};

// Walk the persisted lot_closures (migration 017) into the per-row P&L
// shape the trade-journal view consumes. Closures attribute to BUY rows
// (matching the LIFO semantics in trade_calc.compute_lifo_summary): each
// closure's realized P&L adds to its parent BUY's realizedPl, the BUY's
// remaining shares shrink by the closure's shares, and exitPrice tracks
// the most recent SELL price. SELL rows render with no P&L attribution —
// closures attribute to BUYs, never to SELLs.
//
// Open trades with no SELLs hit this with `closures = []` and produce
// only BUY rows in their initial state; the closure loop is a no-op.
function lotClosuresToLifoRows(
  txns: TradeDetail[],
  closures: LotClosure[],
  enrichedEntry: number,
  multiplier: number,
): { rowData: LifoRow[]; realizedBank: number } {
  // Chronological sort (BUY before SELL within the same date) so rowData
  // iteration order — and the table render order — matches the backend
  // LIFO walk that produced the closures.
  const sorted = [...txns].sort((a, b) => {
    const da = String(a.date || "");
    const db = String(b.date || "");
    if (da !== db) return da.localeCompare(db);
    return String(a.action).toUpperCase() === "BUY" ? -1 : 1;
  });

  // Initial rows. BUY rows start fully-open with zero closures attributed;
  // SELL rows are fully-closed with no P&L attribution (closures attribute
  // to BUY rows in the original walk; preserve that exactly).
  const rowData: LifoRow[] = sorted.map(tx => {
    const action = String(tx.action || "").toUpperCase();
    const txShares = Math.abs(parseFloat(String(tx.shares || 0)));
    const txAmount = parseFloat(String(tx.amount || 0));
    const txValue = txShares * txAmount * multiplier;
    if (action === "SELL") {
      return {
        tx, displayShares: -txShares, remaining: 0,
        exitPrice: 0, realizedPl: 0, returnPct: 0, unrealizedPl: 0,
        status: "Closed", value: -txValue, isSell: true,
      };
    }
    return {
      tx, displayShares: txShares, remaining: txShares,
      exitPrice: 0, realizedPl: 0, returnPct: 0, unrealizedPl: 0,
      status: "Open", value: txValue, isSell: false,
    };
  });

  // trx_id → BUY row lookup for O(1) closure attribution. Only BUY rows
  // are addressable; closures attribute to them, never to SELL rows.
  const buyRowByTrxId = new Map<string, LifoRow>();
  for (const row of rowData) {
    if (!row.isSell) {
      const trx = String((row.tx as any).trx_id || "");
      if (trx) buyRowByTrxId.set(trx, row);
    }
  }

  // Apply closures. The API returns them sorted by closed_at ASC within a
  // trade, matching the original walk's iteration order (chronological
  // SELLs). exitPrice on the BUY row is overwritten on each closure —
  // last write wins, so the chronologically-latest SELL's price ends up
  // displayed (same semantics as the original walk).
  let realizedBank = 0;
  for (const closure of closures) {
    const buyRow = buyRowByTrxId.get(closure.buy_trx_id);
    if (!buyRow) {
      // Stale or orphaned closure — shouldn't happen in well-formed data
      // (lot_closures rows are DELETE-then-INSERTed by the recompute path).
      // Logged so we notice if it ever does during the rewire's rollout.
      console.warn(
        "[lifo-rewire] closure references unknown buy_trx_id",
        { buy_trx_id: closure.buy_trx_id, sell_trx_id: closure.sell_trx_id, trade_id: closure.trade_id },
      );
      continue;
    }
    const closureShares = Number(closure.shares || 0);
    const sellPrice = Number(closure.sell_price || 0);
    const closureRpl = Number(closure.realized_pl || 0);

    buyRow.realizedPl += closureRpl;
    buyRow.remaining -= closureShares;
    buyRow.exitPrice = sellPrice;

    const buyRowPrice = parseFloat(String(buyRow.tx.amount || 0)) || enrichedEntry;
    buyRow.returnPct = buyRowPrice > 0 ? ((sellPrice - buyRowPrice) / buyRowPrice) * 100 : 0;

    if (buyRow.remaining < 0.00001) {
      buyRow.status = "Closed";
      buyRow.remaining = 0; // clamp tiny float residuals so "−0" doesn't render
    }

    realizedBank += closureRpl;
  }

  return { rowData, realizedBank };
}

function TradeCharts({ tradeId, ticker }: { tradeId: string; ticker: string }) {
  const [images, setImages] = useState<any[]>([]);
  const [fundamentals, setFundamentals] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.tradeImages(tradeId).catch(() => []),
      api.tradeFundamentals(tradeId).catch(() => []),
    ]).then(([imgs, funds]) => {
      setImages(Array.isArray(imgs) ? imgs : []);
      setFundamentals(Array.isArray(funds) ? funds : []);
      setLoading(false);
    });
  }, [tradeId]);

  // Deduplicate images by view_url (same chart saved as both entry + marketsurge)
  const seen = new Set<string>();
  const deduped = images.filter(img => {
    const url = img.view_url || img.image_url || "";
    if (!url || seen.has(url)) return false;
    seen.add(url);
    return true;
  });

  const groups: Record<string, any[]> = { "Entry Charts": [], "Position Changes": [] };
  deduped.forEach(img => {
    const group = IMAGE_TYPE_MAP[img.image_type] || "Entry Charts";
    if (groups[group]) groups[group].push(img);
  });

  const totalImages = deduped.length;
  const [uploading, setUploading] = useState(false);
  const [msUploading, setMsUploading] = useState(false);

  const handleUpload = async (files: FileList, imageType: string) => {
    setUploading(true);
    for (const file of Array.from(files)) {
      await api.uploadImage(file, getActivePortfolio(), tradeId, ticker, imageType);
    }
    // Reload images
    const imgs = await api.tradeImages(tradeId).catch(() => []);
    setImages(Array.isArray(imgs) ? imgs : []);
    setUploading(false);
  };

  const handleMsUpload = async (files: FileList) => {
    setMsUploading(true);
    for (const file of Array.from(files)) {
      await api.uploadImage(file, getActivePortfolio(), tradeId, ticker, "marketsurge");
    }
    // Reload both images and fundamentals (extraction happens server-side)
    const [imgs, funds] = await Promise.all([
      api.tradeImages(tradeId).catch(() => []),
      api.tradeFundamentals(tradeId).catch(() => []),
    ]);
    setImages(Array.isArray(imgs) ? imgs : []);
    setFundamentals(Array.isArray(funds) ? funds : []);
    setMsUploading(false);
  };

  const handleDelete = async (imageId: number) => {
    await api.deleteImage(imageId);
    setImages(prev => prev.filter(img => img.id !== imageId));
  };

  return (
    <>
      <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="text-[13px] font-semibold mb-3 flex items-center gap-2">
          <span>📊</span> Charts — {ticker}
          {!loading && <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>({totalImages} image{totalImages !== 1 ? "s" : ""})</span>}
        </div>

        {loading ? (
          <div className="text-[12px] py-3" style={{ color: "var(--ink-4)" }}>Loading charts...</div>
        ) : (
          <div className="flex flex-col gap-1.5">
            {Object.entries(groups).map(([label, imgs]) => {
              const isOpen = expandedGroup === label;
              const uploadType = label === "Entry Charts" ? "entry" : "position_change";
              return (
                <div key={label} className="rounded-[10px] overflow-hidden" style={{ border: "1px solid var(--border)" }}>
                  {/* Expander header */}
                  <div className="flex items-center px-3.5 py-2.5" style={{ background: "var(--surface-2)" }}>
                    <button onClick={() => setExpandedGroup(isOpen ? null : label)}
                            className="flex items-center gap-2 flex-1 text-left cursor-pointer">
                      <span className="text-[10px] transition-transform" style={{ transform: isOpen ? "rotate(90deg)" : "none", color: "var(--ink-4)" }}>▶</span>
                      <span className="text-[12px] font-semibold" style={{ color: "var(--ink)" }}>{label}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: "var(--bg)", color: "var(--ink-4)" }}>{imgs.length}</span>
                    </button>
                    {/* Upload button */}
                    <label className="text-[10px] px-2 py-1 rounded-[6px] cursor-pointer transition-colors hover:brightness-95"
                           style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
                      {uploading ? "..." : "+ Upload"}
                      <input type="file" accept="image/png,image/jpeg,application/pdf" multiple className="hidden"
                             onChange={e => e.target.files && handleUpload(e.target.files, uploadType)} />
                    </label>
                  </div>

                  {/* Expanded: show images */}
                  {isOpen && (
                    <div className="p-3 grid gap-3" style={{ gridTemplateColumns: imgs.length === 1 ? "1fr" : "1fr 1fr" }}>
                      {imgs.map((img, i) => {
                        const url = img.view_url || "";
                        const isPdf = /\.pdf($|\?)/i.test(String(img.file_name || "")) || /\.pdf($|\?)/i.test(url);
                        return (
                          <div key={img.id || i} className="rounded-[8px] overflow-hidden transition-all hover:shadow-md"
                               style={{ border: "1px solid var(--border)" }}>
                            {isPdf ? (
                              <a href={url} target="_blank" rel="noopener noreferrer"
                                 className="flex flex-col items-center justify-center p-8 gap-2 no-underline"
                                 style={{ background: "var(--bg)", color: "var(--ink)", minHeight: 140 }}>
                                <span className="text-[32px]">📄</span>
                                <span className="text-[12px] font-semibold">Open PDF</span>
                                <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>{img.file_name || "document.pdf"}</span>
                              </a>
                            ) : (
                              <div className="cursor-pointer" onClick={() => url && setLightbox(url)}>
                                {url ? (
                                  <img src={url} alt={`${label} ${i + 1}`} className="w-full h-auto" style={{ maxHeight: 300, objectFit: "contain", background: "var(--bg)" }} />
                                ) : (
                                  <div className="p-4 text-center text-[11px]" style={{ color: "var(--ink-4)" }}>No URL</div>
                                )}
                              </div>
                            )}
                            <div className="px-2.5 py-1.5 text-[10px] flex items-center justify-between" style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                              <span className="truncate flex-1">{img.file_name || `${label} ${i + 1}`}</span>
                              <button onClick={() => { if (confirm("Delete this image?")) handleDelete(img.id); }}
                                      className="ml-2 px-1.5 py-0.5 rounded text-[9px] cursor-pointer transition-colors hover:brightness-90"
                                      style={{ color: "#e5484d", background: "color-mix(in oklab, #e5484d 8%, var(--surface))" }}>
                                Delete
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Empty state with upload prompt */}
                  {isOpen && imgs.length === 0 && (
                    <div className="p-4 text-center text-[11px]" style={{ color: "var(--ink-4)" }}>
                      No images. Click "+ Upload" above to add charts.
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── MarketSurge Upload Zone ── */}
      <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="text-[13px] font-semibold mb-2 flex items-center gap-2">
          <span>🔬</span> MarketSurge Fundamentals
        </div>
        {fundamentals.length === 0 && !msUploading && (
          <p className="text-[11px] mb-2" style={{ color: "var(--ink-4)" }}>
            Upload a MarketSurge screenshot to auto-extract ratings via AI.
          </p>
        )}
        {msUploading ? (
          <div className="flex items-center gap-2 py-3 text-[12px]" style={{ color: "var(--ink-3)" }}>
            <span className="animate-spin inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
            Extracting fundamentals...
          </div>
        ) : (
          <label className="inline-flex items-center gap-1.5 text-[11px] px-3 py-1.5 rounded-[8px] cursor-pointer transition-colors hover:brightness-95"
                 style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
            {fundamentals.length > 0 ? "Re-upload Screenshot" : "+ Upload MarketSurge Screenshot"}
            <input type="file" accept="image/png,image/jpeg,application/pdf" className="hidden"
                   onChange={e => e.target.files && handleMsUpload(e.target.files)} />
          </label>
        )}
      </div>

      {/* ── MarketSurge Fundamentals (extracted data) ── */}
      {fundamentals.length > 0 && (
        <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-[13px] font-semibold flex items-center gap-2">
              <span>🔬</span> MarketSurge Fundamentals
            </div>
            <button
              onClick={async () => {
                if (!confirm("Delete extracted fundamentals for this trade?")) return;
                const r = await api.deleteFundamentals(tradeId);
                if (r.status === "ok") setFundamentals([]);
              }}
              className="text-[10px] px-2 py-1 rounded-[6px] cursor-pointer transition-colors hover:brightness-90"
              style={{ color: "#e5484d", background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid color-mix(in oklab, #e5484d 20%, var(--border))" }}>
              Delete
            </button>
          </div>
          {fundamentals.map((f, fi) => (
            <div key={fi} className="rounded-[10px] p-4" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              {/* Ratings row */}
              <div className="grid grid-cols-6 gap-2 mb-3">
                {FUND_RATINGS.map(r => {
                  const val = f[r.key];
                  if (val == null || val === "" || val === 0) return <div key={r.key} />;
                  const isNum = typeof val === "number";
                  const color = isNum ? (val >= 80 ? "#08a86b" : val >= 50 ? "#f59f00" : "#e5484d") : "var(--ink)";
                  return (
                    <div key={r.key} className="text-center p-2 rounded-[8px]" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>{r.label}</div>
                      <div className="text-[18px] font-bold" style={{ fontFamily: "var(--font-jetbrains), monospace", color }}>{val}</div>
                    </div>
                  );
                })}
              </div>
              {/* Extra details */}
              <div className="grid grid-cols-4 gap-2 text-[11px]">
                {f.eps_growth_rate != null && f.eps_growth_rate !== 0 && (
                  <div><span style={{ color: "var(--ink-4)" }}>EPS Growth:</span> <strong>{f.eps_growth_rate}%</strong></div>
                )}
                {f.ud_vol_ratio != null && f.ud_vol_ratio !== 0 && (
                  <div><span style={{ color: "var(--ink-4)" }}>U/D Vol:</span> <strong>{f.ud_vol_ratio}</strong></div>
                )}
                {f.num_funds != null && f.num_funds !== 0 && (
                  <div><span style={{ color: "var(--ink-4)" }}>Funds:</span> <strong>{f.num_funds}</strong></div>
                )}
                {f.industry_group && (
                  <div><span style={{ color: "var(--ink-4)" }}>Group:</span> <strong>{f.industry_group}</strong>{f.industry_group_rank ? ` (#${f.industry_group_rank})` : ""}</div>
                )}
                {f.funds_own_pct != null && f.funds_own_pct !== 0 && (
                  <div><span style={{ color: "var(--ink-4)" }}>Fund Own:</span> <strong>{f.funds_own_pct}%</strong></div>
                )}
                {f.mgmt_own_pct != null && f.mgmt_own_pct !== 0 && (
                  <div><span style={{ color: "var(--ink-4)" }}>Mgmt Own:</span> <strong>{f.mgmt_own_pct}%</strong></div>
                )}
                {f.market_cap && (
                  <div><span style={{ color: "var(--ink-4)" }}>Mkt Cap:</span> <strong>{f.market_cap}</strong></div>
                )}
              </div>
              {f.extracted_at && (
                <div className="text-[10px] mt-2" style={{ color: "var(--ink-5)" }}>Extracted: {String(f.extracted_at).slice(0, 10)}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Lightbox — full screen overlay */}
      {lightbox && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center cursor-pointer"
             style={{ background: "rgba(0,0,0,0.8)", backdropFilter: "blur(6px)" }}
             onClick={() => setLightbox(null)}>
          <div className="relative" onClick={e => e.stopPropagation()}>
            <img src={lightbox} alt="Chart" className="max-w-[92vw] max-h-[88vh] rounded-[10px]"
                 style={{ boxShadow: "0 20px 60px rgba(0,0,0,0.5)" }} />
            <button onClick={() => setLightbox(null)}
                    className="absolute -top-3 -right-3 w-8 h-8 rounded-full flex items-center justify-center text-white text-[14px] font-bold cursor-pointer"
                    style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}>
              x
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export function TradeJournal({ navColor }: { navColor: string }) {
  // Cohort-split storage so the page can fetch open trades on mount without
  // waiting on the closed-trade payload. tradesClosed(500) + tradesRecent(5000)
  // were the page's slowest fetches and most users only need them when they
  // explicitly switch the status filter or turn on Recent Activity.
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [openDetails, setOpenDetails] = useState<TradeDetail[]>([]);
  const [closedDetails, setClosedDetails] = useState<TradeDetail[]>([]);
  // Persisted lot_closures from the API — source of truth for per-row
  // realized P&L (migration 017, reconciled across all 478 trades on
  // 2026-05-04). Keeps cohort-split alongside details so each loader
  // populates its own slice. Open trades have empty closures (no SELLs
  // yet); lotClosuresToLifoRows handles that case correctly.
  const [openClosures, setOpenClosures] = useState<LotClosure[]>([]);
  const [closedClosures, setClosedClosures] = useState<LotClosure[]>([]);
  const [openLoaded, setOpenLoaded] = useState(false);
  const [closedLoaded, setClosedLoaded] = useState(false);
  const [filterLoading, setFilterLoading] = useState(false);
  // Phase 2 — strategies for the right-click flyout + card pill colors.
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [tjCtxMenu, setTjCtxMenu] = useState<{ x: number; y: number; trade: TradePosition } | null>(null);
  const coarsePointer = useCoarsePointer();
  // Lookup keyed by name so the card pill can grab a strategy's color
  // in O(1). When a trade references a strategy that's been deleted
  // from the table (shouldn't happen — FK is RESTRICT), we fall back
  // to grey so the chip still renders.
  const strategyByName = useMemo(() => {
    const m = new Map<string, Strategy>();
    for (const s of strategies) m.set(s.name, s);
    return m;
  }, [strategies]);

  // Derived combined views — keep the rest of the component code unchanged.
  const allTrades = useMemo(() => [...openTrades, ...closedTrades], [openTrades, closedTrades]);
  const allDetails = useMemo(() => [...openDetails, ...closedDetails], [openDetails, closedDetails]);
  // Lookup keyed by trade_id so the .map(trade => ...) below can grab a
  // trade's closures in O(1) without re-filtering allClosures per render.
  const closuresByTradeId = useMemo(() => {
    const m = new Map<string, LotClosure[]>();
    for (const c of openClosures) {
      const list = m.get(c.trade_id);
      if (list) list.push(c); else m.set(c.trade_id, [c]);
    }
    for (const c of closedClosures) {
      const list = m.get(c.trade_id);
      if (list) list.push(c); else m.set(c.trade_id, [c]);
    }
    return m;
  }, [openClosures, closedClosures]);

  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [equity, setEquity] = useState(0);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("none");
  const [sort, setSort] = useState<SortKey>("newest");
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tickerDropdownOpen, setTickerDropdownOpen] = useState(false);
  const [tickerQuery, setTickerQuery] = useState("");
  const [dateRange, setDateRange] = useState<DateRange>("all");
  // Default 'off' so a fresh page load doesn't have to fetch closed-trade
  // data just to render. Users who want the "show me what I just worked on"
  // mixed-status view click into 10/30/all explicitly, which lazy-fetches
  // closed cohort on demand.
  const [recentActivity, setRecentActivity] = useState<RecentActivity>("off");
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [scaleOutOpen, setScaleOutOpen] = useState<string | null>(null);
  const [txnFilter, setTxnFilter] = useState<"all" | "open" | "closed">("all");
  const [analysisOpen, setAnalysisOpen] = useState<string | null>(null);
  const [liveChartOpen, setLiveChartOpen] = useState<string | null>(null);

  // Closed-trade edit modal. Hoisted to page level so a single modal
  // instance handles every Edit button in every trade card's Transaction
  // History table. Form fields are controlled-input strings; the API call
  // parses them back to numbers at save time. trx_id / trade_id / ticker /
  // action are pass-through (read from editingTxn directly), so they
  // don't live in editForm.
  const [editingTxn, setEditingTxn] = useState<TradeDetail | null>(null);
  const [editForm, setEditForm] = useState<{
    date: string;
    shares: string;
    amount: string;
    stop_loss: string;
    rule: string;
    notes: string;
  }>({ date: "", shares: "", amount: "", stop_loss: "", rule: "", notes: "" });
  const [editError, setEditError] = useState<string | null>(null);
  const [editLoading, setEditLoading] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const openEditModal = useCallback((tx: TradeDetail) => {
    setEditingTxn(tx);
    setEditForm({
      date: String(tx.date || "").slice(0, 16),
      shares: String(tx.shares ?? ""),
      amount: String(tx.amount ?? ""),
      stop_loss: String((tx as any).stop_loss ?? ""),
      rule: tx.rule || "",
      notes: String((tx as any).notes ?? ""),
    });
    setEditError(null);
    setConfirmingDelete(false);
    setEditLoading(false);
  }, []);

  const closeEditModal = useCallback(() => {
    setEditingTxn(null);
    setEditError(null);
    setConfirmingDelete(false);
    setEditLoading(false);
  }, []);

  const loadOpen = useCallback(async () => {
    const [open, openDet, journal] = await Promise.all([
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      // catch fallback shape mirrors success shape so the destructuring
      // below doesn't crash on a fetch error.
      api.tradesOpenDetails(getActivePortfolio()).catch(() => ({ details: [], lot_closures: [] })),
      api.journalLatest(getActivePortfolio()).catch(() => ({ end_nlv: 100000 })),
    ]);
    const openArr = open as TradePosition[];
    setOpenTrades(openArr);
    setOpenDetails(openDet.details);
    setOpenClosures(openDet.lot_closures);
    setEquity(parseFloat(String((journal as any).end_nlv || 100000)));
    setOpenLoaded(true);

    // Live prices for open trades — fire-and-forget so it doesn't block the
    // page from rendering. The price provider's per-ticker cache means
    // subsequent navigations between Dashboard/ACS/Journal share the result.
    const tickers = openArr.map(t => t.ticker).filter(Boolean);
    if (tickers.length > 0) {
      api.batchPrices(tickers, getActivePortfolio()).then(prices => {
        if (prices && !("error" in prices)) setLivePrices(prices as Record<string, number>);
      }).catch(() => { /* fall back to entry prices */ });
    }
  }, []);

  const loadClosed = useCallback(async () => {
    const [closed, recent] = await Promise.all([
      api.tradesClosed(getActivePortfolio(), 500).catch(() => []),
      api.tradesRecent(getActivePortfolio(), 5000).catch(() => ({ details: [], lot_closures: [] })),
    ]);
    const closedArr = closed as TradePosition[];
    // tradesRecent returns details + lot_closures for ALL trades (open +
    // closed) sorted by recency. We already loaded open details via
    // tradesOpenDetails, so keep only the closed-trade rows here to avoid
    // duplicating when the two cohorts merge into allDetails / closuresByTradeId.
    const closedIds = new Set(closedArr.map(t => t.trade_id));
    const closedDet = recent.details.filter(d => closedIds.has(d.trade_id));
    const closedClos = recent.lot_closures.filter(c => closedIds.has(c.trade_id));
    setClosedTrades(closedArr);
    setClosedDetails(closedDet);
    setClosedClosures(closedClos);
    setClosedLoaded(true);
  }, []);

  const saveEdit = useCallback(async () => {
    if (!editingTxn) return;
    setEditLoading(true);
    setEditError(null);
    try {
      const shares = parseFloat(editForm.shares || "0") || 0;
      const amount = parseFloat(editForm.amount || "0") || 0;
      const res = await api.editTransaction({
        detail_id: (editingTxn as any).detail_id as number,
        trade_id: editingTxn.trade_id,
        ticker: editingTxn.ticker,
        action: editingTxn.action,
        date: editForm.date,
        shares,
        amount,
        // Backend recomputes value as shares × amount × multiplier; we
        // send shares × amount un-multiplied to match Trade Manager's
        // request shape. The field is required by the API client type.
        value: shares * amount,
        rule: editForm.rule,
        notes: editForm.notes,
        stop_loss: parseFloat(editForm.stop_loss || "0") || 0,
        trx_id: String((editingTxn as any).trx_id || ""),
      });
      if (res.error) {
        setEditError(res.error);
      } else {
        // Refresh BOTH cohorts — status may have flipped (open↔closed)
        // when the edit changed remaining shares, and we don't know which
        // side the trade ends up on without re-fetching.
        await Promise.all([loadOpen(), loadClosed()]);
        closeEditModal();
      }
    } catch (err: any) {
      setEditError(err?.message || "Failed to save");
    } finally {
      setEditLoading(false);
    }
  }, [editingTxn, editForm, loadOpen, loadClosed, closeEditModal]);

  const deleteTxn = useCallback(async () => {
    if (!editingTxn) return;
    // Two-click confirm: first click arms the destructive action, second
    // click executes. Avoids a modal-on-modal native confirm() dialog.
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      return;
    }
    setEditLoading(true);
    setEditError(null);
    try {
      const res = await api.deleteTransaction(
        (editingTxn as any).detail_id as number,
        editingTxn.trade_id,
        editingTxn.ticker,
      );
      if (res.error) {
        setEditError(res.error);
        setConfirmingDelete(false);
      } else {
        await Promise.all([loadOpen(), loadClosed()]);
        closeEditModal();
      }
    } catch (err: any) {
      setEditError(err?.message || "Failed to delete");
      setConfirmingDelete(false);
    } finally {
      setEditLoading(false);
    }
  }, [editingTxn, confirmingDelete, loadOpen, loadClosed, closeEditModal]);

  // ESC dismisses the edit modal (gated on !editLoading so an in-flight
  // save/delete can't be cancelled mid-request).
  useEffect(() => {
    if (!editingTxn) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !editLoading) closeEditModal();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editingTxn, editLoading, closeEditModal]);

  // Phase 2 — load active strategies once on mount. Used by both the
  // card pill (color lookup) and the right-click flyout (option list).
  useEffect(() => {
    api.listStrategies({ active: true }).then(setStrategies).catch(() => setStrategies([]));
  }, []);

  // Close the right-click context menu on outside click / Escape.
  useEffect(() => {
    if (!tjCtxMenu) return;
    const close = () => setTjCtxMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [tjCtxMenu]);

  // Phase 2 — single-trade retag from the right-click menu. Optimistic
  // update of both cohorts (the trade lives in exactly one), then fire
  // the PATCH. Reverts on error so the UI never lies about persisted
  // state. Mirrors how setTradeGrade is handled in the same component.
  const setTradeStrategyOptimistic = useCallback((trade_id: string, newStrategy: string) => {
    setTjCtxMenu(null);
    const patch = (s: string | undefined) => {
      setOpenTrades(prev => prev.map(t => t.trade_id === trade_id ? { ...t, strategy: s } as any : t));
      setClosedTrades(prev => prev.map(t => t.trade_id === trade_id ? { ...t, strategy: s } as any : t));
    };
    const prev = [...openTrades, ...closedTrades].find(t => t.trade_id === trade_id)?.strategy;
    patch(newStrategy);
    api.setTradeStrategy(trade_id, { strategy: newStrategy }).then(r => {
      if ("error" in r && r.error) patch(prev);
    }).catch(() => patch(prev));
  }, [openTrades, closedTrades]);

  // On-demand loader. Fires only when the user picks a status, turns on
  // Recent Activity, or has a ticker selected (via search, prefill, or
  // ?ticker= URL param). The default empty state ('none' status, no
  // ticker, recent activity off) does NOT fetch — clean visits don't
  // touch the backend until the user expresses intent.
  //
  // Ticker mode loads BOTH cohorts opportunistically so a ticker search
  // returns the full open + closed history. Each cohort renders as soon
  // as it resolves (the trade list reacts to openLoaded/closedLoaded
  // flipping), so open results show first and closed streams in behind.
  useEffect(() => {
    const tickerActive = selectedTickers.length > 0;
    const needsOpen =
      statusFilter === "open" || statusFilter === "all"
      || recentActivity !== "off" || tickerActive;
    const needsClosed =
      statusFilter === "closed" || statusFilter === "all"
      || recentActivity !== "off" || tickerActive;

    const fetches: Promise<void>[] = [];
    if (needsOpen && !openLoaded) fetches.push(loadOpen());
    if (needsClosed && !closedLoaded) fetches.push(loadClosed());

    if (fetches.length === 0) return;

    setFilterLoading(true);
    Promise.all(fetches).finally(() => setFilterLoading(false));
  }, [statusFilter, recentActivity, selectedTickers, openLoaded, closedLoaded, loadOpen, loadClosed]);

  // Mount-time prefill from explicit user intent: either a right-click
  // "View in Journal" from Active Campaign (localStorage) or a ?ticker=
  // URL param. Setting selectedTickers here is what triggers the
  // filter-driven loader above to fetch both cohorts. We do NOT set
  // statusFilter — the ticker is what drives the fetch in this mode.
  useEffect(() => {
    let pickedTicker: string | null = null;
    let pickedTradeId: string | null = null;
    try {
      const raw = localStorage.getItem("journal_prefill");
      if (raw) {
        localStorage.removeItem("journal_prefill");
        const data = JSON.parse(raw);
        if (data.ticker) pickedTicker = String(data.ticker).toUpperCase();
        if (data.trade_id) pickedTradeId = String(data.trade_id);
      }
    } catch { /* ignore */ }
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const urlTicker = params.get("ticker");
      if (urlTicker) pickedTicker = urlTicker.toUpperCase();
      const urlTradeId = params.get("trade_id");
      if (urlTradeId) pickedTradeId = urlTradeId;
    }
    if (pickedTicker) setSelectedTickers([pickedTicker]);
    if (pickedTradeId) setExpandedCard(pickedTradeId);
  }, []);

  const filtered = useMemo(() => {
    let result = [...allTrades];

    // Recent Activity: last N most-recently-touched trades across open AND
    // closed (ignores status filter / date range / sort so the list is
    // always "show me what I just worked on"). Activity = trades_summary
    // .last_updated, which bumps on any edit — scale-ins, stop changes,
    // grade updates, sells, etc.
    if (recentActivity !== "off") {
      const n = parseInt(recentActivity, 10);
      const activityTs = (t: TradePosition) => {
        const lu = new Date(String((t as any).last_updated || 0)).getTime() || 0;
        const o = new Date(String(t.open_date || 0)).getTime() || 0;
        const c = new Date(String(t.closed_date || 0)).getTime() || 0;
        return Math.max(lu, o, c);
      };
      // Still respect ticker multi-select; everything else is overridden.
      let base = [...allTrades];
      if (selectedTickers.length > 0) {
        base = base.filter(t => matchesAnyTradeQuery(t, selectedTickers));
      }
      base.sort((a, b) => activityTs(b) - activityTs(a));
      return base.slice(0, n);
    }

    // Status — 'none' and 'all' both mean "no status filter"; only 'open'
    // and 'closed' actually narrow.
    if (statusFilter === "open" || statusFilter === "closed") {
      result = result.filter(t => (t.status || "").toUpperCase() === statusFilter.toUpperCase());
    }

    // Ticker / trade-ID multi-select. Tokens may be equity tickers, option
    // underlyings (so "DOCN" matches DOCN equity AND every "DOCN ..." option),
    // exact trade IDs ("202605-013"), or month-prefix trade-ID queries
    // ("202605"). See lib/trade-search.ts for the full predicate.
    if (selectedTickers.length > 0) {
      result = result.filter(t => matchesAnyTradeQuery(t, selectedTickers));
    }

    // Date range
    if (dateRange !== "all") {
      const now = new Date();
      let cutoff: Date;
      if (dateRange === "7d") cutoff = new Date(now.getTime() - 7 * 86400000);
      else if (dateRange === "30d") cutoff = new Date(now.getTime() - 30 * 86400000);
      else if (dateRange === "90d") cutoff = new Date(now.getTime() - 90 * 86400000);
      else cutoff = new Date(now.getFullYear(), 0, 1);
      const cutoffStr = cutoff.toISOString().slice(0, 10);
      result = result.filter(t => String(t.open_date || "").slice(0, 10) >= cutoffStr);
    }

    // Sort
    switch (sort) {
      case "newest": result.sort((a, b) => String(b.open_date || "").localeCompare(String(a.open_date || "")) || String(b.trade_id || "").localeCompare(String(a.trade_id || ""))); break;
      case "oldest": result.sort((a, b) => String(a.open_date || "").localeCompare(String(b.open_date || "")) || String(a.trade_id || "").localeCompare(String(b.trade_id || ""))); break;
      case "best": result.sort((a, b) => parseFloat(String(b.realized_pl || 0)) - parseFloat(String(a.realized_pl || 0))); break;
      case "worst": result.sort((a, b) => parseFloat(String(a.realized_pl || 0)) - parseFloat(String(b.realized_pl || 0))); break;
      case "ticker": result.sort((a, b) => (a.ticker || "").localeCompare(b.ticker || "")); break;
    }

    return result;
  }, [allTrades, statusFilter, sort, selectedTickers, dateRange, recentActivity]);

  const openCount = allTrades.filter(t => (t.status || "").toUpperCase() === "OPEN").length;
  const closedCount = allTrades.filter(t => (t.status || "").toUpperCase() === "CLOSED").length;
  const hasLoadedAnyData = openLoaded || closedLoaded;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Trade <em className="italic" style={{ color: navColor }}>Journal</em>
        </h1>
        <div className="text-[13px] mt-1.5 flex items-center gap-2" style={{ color: "var(--ink-3)" }}>
          {hasLoadedAnyData ? (
            <span>{allTrades.length} campaigns ({openCount} open, {closedCount} closed)</span>
          ) : (
            <span>Pick a filter or search a ticker to begin.</span>
          )}
          {filterLoading && (
            <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>
              · loading…
            </span>
          )}
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-3 mb-5 flex-wrap">
        {/* Status tabs */}
        <div className="flex p-0.5 rounded-[10px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {(["all", "open", "closed"] as StatusFilter[]).map(s => (
            <button key={s} onClick={() => setStatusFilter(s)}
                    className="px-3 py-1.5 rounded-[8px] text-[12px] font-medium transition-all capitalize"
                    style={{
                      background: statusFilter === s ? "var(--surface)" : "transparent",
                      color: statusFilter === s ? "var(--ink)" : "var(--ink-4)",
                      boxShadow: statusFilter === s ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
                    }}>
              {s} {s === "all" ? `(${allTrades.length})` : s === "open" ? `(${openCount})` : `(${closedCount})`}
            </button>
          ))}
        </div>

        {/* Ticker multi-select: type + click to add */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {selectedTickers.map(t => (
            <span key={t} className="flex items-center gap-1 h-[28px] px-2.5 rounded-[8px] text-[11px] font-semibold"
                  style={{ background: `color-mix(in oklab, ${navColor} 10%, transparent)`, color: navColor, border: `1px solid ${navColor}30` }}>
              {t}
              <button onClick={() => setSelectedTickers(prev => prev.filter(x => x !== t))}
                      className="ml-0.5 opacity-60 hover:opacity-100" style={{ lineHeight: 1 }}>×</button>
            </span>
          ))}
          <div className="relative">
            <input type="text" value={tickerQuery}
                   placeholder={selectedTickers.length > 0 ? "Add ticker..." : "Search tickers..."}
                   onChange={e => {
                     setTickerQuery(e.target.value.toUpperCase());
                     setTickerDropdownOpen(true);
                   }}
                   onKeyDown={e => {
                     if (e.key === "Enter" && tickerQuery) {
                       const allTickerSet = [...new Set(allTrades.map(t => t.ticker).filter(Boolean))];
                       const match = allTickerSet.find(t => t.toUpperCase() === tickerQuery.trim());
                       // Free-form fallback when no autocomplete match — covers
                       // the empty-state case (allTrades=[] before any cohort
                       // load) where ticker entry is what should drive the
                       // initial fetch.
                       const ticker = match ?? tickerQuery.trim().toUpperCase();
                       if (ticker && !selectedTickers.includes(ticker)) {
                         setSelectedTickers(prev => [...prev, ticker]);
                       }
                       setTickerQuery("");
                       setTickerDropdownOpen(false);
                     }
                     if (e.key === "Backspace" && !tickerQuery && selectedTickers.length > 0) {
                       setSelectedTickers(prev => prev.slice(0, -1));
                     }
                   }}
                   onFocus={() => setTickerDropdownOpen(true)}
                   onBlur={() => setTimeout(() => setTickerDropdownOpen(false), 150)}
                   className="h-[34px] px-3 rounded-[10px] text-[12px] w-[140px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
            {tickerDropdownOpen && (() => {
              const available = [...new Set(allTrades.map(t => t.ticker).filter(Boolean))].sort()
                .filter(t => !selectedTickers.includes(t))
                .filter(t => !tickerQuery || t.toUpperCase().includes(tickerQuery.trim()));
              return available.length > 0 ? (
                <div className="absolute z-50 mt-1 w-[180px] rounded-[10px] overflow-hidden shadow-lg"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 200 }}>
                  <div className="overflow-y-auto" style={{ maxHeight: 200 }}>
                    {available.map(t => (
                      <button key={t} type="button"
                              onMouseDown={e => { e.preventDefault(); setSelectedTickers(prev => [...prev, t]); setTickerQuery(""); setTickerDropdownOpen(false); }}
                              className="w-full text-left px-3 py-1.5 text-[12px] transition-colors"
                              style={{ fontFamily: "var(--font-jetbrains), monospace" }}
                              onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                              onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null;
            })()}
          </div>
        </div>

        {/* Sort */}
        <select value={sort} onChange={e => setSort(e.target.value as SortKey)}
                className="h-[34px] px-3 rounded-[10px] text-[12px]"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
          <option value="newest">Newest First</option>
          <option value="oldest">Oldest First</option>
          <option value="best">Best P&L</option>
          <option value="worst">Worst P&L</option>
          <option value="ticker">Ticker A-Z</option>
        </select>

        {/* Date range */}
        <select value={dateRange} onChange={e => setDateRange(e.target.value as DateRange)}
                className="h-[34px] px-3 rounded-[10px] text-[12px]"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
          <option value="all">All Time</option>
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
          <option value="90d">Last 90 Days</option>
          <option value="ytd">This Year</option>
        </select>

        {/* Recent Activity — overrides status/date/sort when active. Shows
            the N most-recently-touched trades (added, edited, scaled-in,
            stop-changed, sold) across open + closed. */}
        <select value={recentActivity} onChange={e => setRecentActivity(e.target.value as RecentActivity)}
                className="h-[34px] px-3 rounded-[10px] text-[12px]"
                style={{
                  background: recentActivity !== "off" ? "color-mix(in oklab, #8b5cf6 12%, var(--surface))" : "var(--surface)",
                  border: `1px solid ${recentActivity !== "off" ? "color-mix(in oklab, #8b5cf6 40%, var(--border))" : "var(--border)"}`,
                  color: recentActivity !== "off" ? "#8b5cf6" : "var(--ink)",
                  appearance: "none" as any,
                  fontWeight: recentActivity !== "off" ? 600 : 400,
                }}>
          <option value="off">Recent Activity…</option>
          <option value="10">Last 10 Activity</option>
          <option value="20">Last 20 Activity</option>
          <option value="50">Last 50 Activity</option>
        </select>

        {hasLoadedAnyData && (
          <span className="text-[12px] ml-auto" style={{ color: "var(--ink-4)" }}>{filtered.length} results</span>
        )}
      </div>

      {/* Ticker-search banner: makes the active narrow obvious and gives a
          one-click way back to the empty state. */}
      {selectedTickers.length > 0 && (
        <div className="flex items-center gap-3 mb-4 px-3 py-2 rounded-[10px] text-[12px]"
             style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
          <span>
            Showing trades for <strong style={{ color: "var(--ink)" }}>{selectedTickers.join(", ")}</strong>
          </span>
          {filterLoading && selectedTickers.length > 0 && !closedLoaded && (
            <span style={{ color: "var(--ink-4)" }}>· loading closed trades…</span>
          )}
          <button onClick={() => { setSelectedTickers([]); setStatusFilter("none"); setRecentActivity("off"); }}
                  className="ml-auto px-2 py-0.5 rounded-[6px] text-[11px] font-medium cursor-pointer"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
            Clear filter
          </button>
        </div>
      )}

      {/* Trade list / loading skeleton / empty-state CTA. Empty state only
          renders when no fetch has completed AND no fetch is in flight, so
          the prefill/URL flow shows the skeleton (not the CTA) while data
          streams in. */}
      {!hasLoadedAnyData && !filterLoading ? (
        <div className="text-center py-20 px-6" style={{ color: "var(--ink-3)" }}>
          <div className="text-[15px] font-medium" style={{ color: "var(--ink-2)" }}>Pick a filter or search a ticker above</div>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-4)" }}>
            Choose <em>All</em>, <em>Open</em>, or <em>Closed</em> from the tabs, or type a ticker to see all activity for it.
          </div>
        </div>
      ) : !hasLoadedAnyData && filterLoading ? (
        <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>
      ) : (
      <>
      {/* Trade cards */}
      <div className="flex flex-col gap-4">
        {filtered.map(trade => {
          const isOpen = (trade.status || "").toUpperCase() === "OPEN";
          const shares = trade.shares || 0;
          const avgEntry = parseFloat(String(trade.avg_entry || 0));
          const avgExit = parseFloat(String(trade.avg_exit || 0));
          const totalCost = parseFloat(String(trade.total_cost || 0));
          // Migration 016 — when the row is an option, multiplier=100 turns
          // every per-contract dollar back into notional. Falls back to a
          // ticker-shape autodetect for any legacy row that pre-dates the
          // backfill (e.g. demo data, partial imports).
          const isOption = String((trade as any).instrument_type || "").toUpperCase() === "OPTION"
            || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(trade.ticker || ""));
          const multiplier = isOption
            ? Math.max(parseFloat(String((trade as any).multiplier || 0)) || 100, 1)
            : 1;
          const unitLabel = isOption ? "Contracts" : "Shares";

          // Transaction details for this trade
          const txns = allDetails.filter(d => d.trade_id === trade.trade_id);
          const buys = txns.filter(d => String(d.action).toUpperCase() === "BUY");
          const sells = txns.filter(d => String(d.action).toUpperCase() === "SELL");

          // Enrich avg_entry/avg_exit from details if summary has zeros
          let enrichedEntry = avgEntry;
          if (enrichedEntry === 0 && buys.length > 0) {
            const totalVal = buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0);
            const totalShs = buys.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
            enrichedEntry = totalShs > 0 ? totalVal / totalShs : 0;
          }
          let enrichedExit = avgExit;
          if (enrichedExit === 0 && sells.length > 0) {
            const totalVal = sells.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0);
            const totalShs = sells.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
            enrichedExit = totalShs > 0 ? totalVal / totalShs : 0;
          }

          // LIFO source for the Realized P&L tile and the Transaction History
          // table. The backend's lot_closures (migration 017) drive per-row
          // P&L; lotClosuresToLifoRows assembles them into the LifoRow shape
          // downstream consumers (the table render, the Realized P&L tile,
          // the live-exposure overwrite below) expect. Open trades hit this
          // with empty closures and produce BUY-only rows with realizedBank=0.
          const tradeClosures = closuresByTradeId.get(trade.trade_id) || [];
          const { rowData: lifoRowData, realizedBank } = lotClosuresToLifoRows(
            txns, tradeClosures, enrichedEntry, multiplier,
          );

          const livePrice = isOpen ? (livePrices[trade.ticker] || 0) : enrichedExit;
          const unrealizedPl = isOpen && livePrice > 0 ? (livePrice - enrichedEntry) * shares * multiplier : 0;
          const totalPl = isOpen ? unrealizedPl + realizedBank : realizedBank;

          // Return %: use summary if nonzero, else compute from enriched entry/exit
          let retPct = parseFloat(String(trade.return_pct || 0));
          if (isOpen) {
            retPct = livePrice > 0 && enrichedEntry > 0 ? ((livePrice - enrichedEntry) / enrichedEntry) * 100 : 0;
          } else if (retPct === 0 && enrichedEntry > 0 && enrichedExit > 0) {
            retPct = ((enrichedExit - enrichedEntry) / enrichedEntry) * 100;
          }

          const borderColor = totalPl > 0 ? "#08a86b" : totalPl < 0 ? "#e5484d" : "#f59f00";
          const bgTint = totalPl > 0 ? "rgba(8,168,107,0.03)" : totalPl < 0 ? "rgba(229,72,77,0.03)" : "rgba(245,159,0,0.03)";

          // Days held
          const openDate = new Date(trade.open_date || "");
          const closeDate = trade.closed_date ? new Date(trade.closed_date) : new Date();
          const daysHeld = Math.max(1, Math.floor((closeDate.getTime() - openDate.getTime()) / 86400000));

          // B1 entry (first buy)
          const b1 = buys.length > 0 ? buys[0] : null;
          const b1Price = b1 ? parseFloat(String(b1.amount || 0)) : 0;
          const bandLow = b1Price > 0 ? b1Price * 0.975 : 0;
          const bandHigh = b1Price > 0 ? b1Price * 1.025 : 0;

          // Core vs Add-on P&L
          const hasAddons = buys.length > 1;
          const b1Shares = b1 ? parseFloat(String(b1.shares || 0)) : 0;
          const corePl = b1Price > 0 && avgExit > 0 ? (avgExit - b1Price) * b1Shares * multiplier : 0;

          const isExpanded = expandedCard === trade.trade_id;

          return (
            <div key={trade.trade_id} className="rounded-[14px] overflow-hidden transition-all"
                 onContextMenu={e => { e.preventDefault(); setTjCtxMenu({ x: e.clientX, y: e.clientY, trade }); }}
                 style={{
                   background: bgTint,
                   border: "1px solid var(--border)",
                   borderLeft: `5px solid ${borderColor}`,
                   boxShadow: "var(--card-shadow)",
                 }}>
              {/* ── Card Header ── */}
              <div className="px-5 pt-4 pb-3">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-[20px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{trade.ticker}</span>
                    <StatusBadge status={trade.status || ""} />
                    {isOption && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold whitespace-nowrap"
                            title={`Equity option · ${multiplier}× contract multiplier`}
                            style={{ background: "color-mix(in oklab, #f59f00 14%, var(--surface))", color: "#b45309" }}>
                        OPTION ×{multiplier}
                      </span>
                    )}
                    <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{trade.trade_id}</span>
                    {(trade as any).be_stop_moved_at && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                            title={`Moved stop to BE on ${String((trade as any).be_stop_moved_at).slice(0, 10)} (+10% rule)`}
                            style={{ background: "color-mix(in oklab, #8b5cf6 12%, var(--surface))", color: "#8b5cf6" }}>
                        🎯 BE
                      </span>
                    )}
                    <GradeStars
                      value={typeof (trade as any).grade === "number" ? (trade as any).grade : null}
                      onChange={(v) => {
                        // Optimistic local update — patch whichever cohort the
                        // trade lives in. The .map is a no-op for the cohort
                        // that doesn't contain it.
                        const patch = (g: number | null) => {
                          setOpenTrades(prev => prev.map(t => t.trade_id === trade.trade_id ? { ...t, grade: g } as any : t));
                          setClosedTrades(prev => prev.map(t => t.trade_id === trade.trade_id ? { ...t, grade: g } as any : t));
                        };
                        patch(v);
                        api.setTradeGrade({ portfolio: getActivePortfolio(), trade_id: trade.trade_id, grade: v })
                          .catch(() => {
                            // revert on failure
                            patch((trade as any).grade ?? null);
                          });
                      }}
                    />
                  </div>
                  <div className="text-right">
                    <div className="text-[20px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: PLColor(totalPl) }}>
                      ${totalPl >= 0 ? "+" : ""}{totalPl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-[12px] font-medium" style={{ color: PLColor(retPct) }}>
                      {retPct >= 0 ? "+" : ""}{retPct.toFixed(2)}%
                    </div>
                  </div>
                </div>

                {/* Key metrics row */}
                <div className="grid grid-cols-4 gap-4 py-3" style={{ borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}>
                  <CardMetric label="Entry" value={`$${avgEntry.toFixed(2)}`} />
                  <CardMetric label={isOpen ? "Status" : "Exit"} value={isOpen ? "Active" : `$${avgExit.toFixed(2)}`} color={isOpen ? "#08a86b" : undefined} />
                  <CardMetric label={unitLabel} value={String(shares)} />
                  <CardMetric label="Days Held" value={String(daysHeld)} />
                </div>

                {/* Core/Add-on P&L (if multi-tranche) */}
                {hasAddons && b1Price > 0 && !isOpen && (
                  <div className="grid grid-cols-4 gap-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                    <CardMetric label="Core P&L" value={`$${corePl >= 0 ? "+" : ""}${corePl.toFixed(0)}`} color={PLColor(corePl)} />
                    <CardMetric label="Add P&L" value={`$${(totalPl - corePl) >= 0 ? "+" : ""}${(totalPl - corePl).toFixed(0)}`} color={PLColor(totalPl - corePl)} />
                    <CardMetric label="Core Band" value={`$${bandLow.toFixed(0)} – $${bandHigh.toFixed(0)}`} />
                    <CardMetric label="B1 Price" value={`$${b1Price.toFixed(2)}`} />
                  </div>
                )}

                {/* Scale-out plan (open trades only, collapsible) */}
                {isOpen && avgEntry > 0 && (
                  <div className="mt-3">
                    <button onClick={() => setScaleOutOpen(scaleOutOpen === trade.trade_id ? null : trade.trade_id)}
                            className="flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-[8px] transition-colors"
                            style={{ color: "#d97706", background: scaleOutOpen === trade.trade_id ? "rgba(245,159,0,0.08)" : "transparent" }}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                           style={{ transform: scaleOutOpen === trade.trade_id ? "rotate(90deg)" : "none", transition: "transform 0.15s" }}>
                        <path d="M9 18l6-6-6-6"/>
                      </svg>
                      Scale-Out Plan
                    </button>
                    {scaleOutOpen === trade.trade_id && (
                      <div className="mt-2 p-3 rounded-[10px]" style={{ background: "rgba(245,159,0,0.06)", border: "1px solid rgba(245,159,0,0.15)", animation: "slide-up 0.12s ease-out" }}>
                        <div className="grid grid-cols-3 gap-3 text-[11px]">
                          <div>
                            <span className="font-semibold">T1 (-3%)</span>
                            <span className="ml-1.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                              {Math.ceil(shares * 0.25)} shs @ ${(avgEntry * 0.97).toFixed(2)}
                            </span>
                          </div>
                          <div>
                            <span className="font-semibold">T2 (-5%)</span>
                            <span className="ml-1.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                              {Math.ceil(shares * 0.25)} shs @ ${(avgEntry * 0.95).toFixed(2)}
                            </span>
                          </div>
                          <div>
                            <span className="font-semibold">T3 (-7%)</span>
                            <span className="ml-1.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                              {shares - Math.ceil(shares * 0.25) * 2} shs @ ${(avgEntry * 0.93).toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Footer: dates + rule + strategy pill + expand toggle */}
                <div className="flex items-center justify-between mt-3 pt-2 gap-2 flex-wrap">
                  <div className="text-[11px] flex items-center gap-2 flex-wrap" style={{ color: "var(--ink-4)" }}>
                    <span>{trade.rule}</span>
                    {/* Phase 2 — strategy pill. Color from DB (strategies
                        table); chip continues to render even when the
                        strategy has been deactivated since this trade was
                        tagged. Falls back to a grey chip if the row's
                        strategy isn't in the loaded list (defensive — FK
                        is RESTRICT so this shouldn't happen). */}
                    {(trade as any).strategy && (
                      <>
                        <span>·</span>
                        <StrategyChip
                          name={(trade as any).strategy}
                          color={strategyByName.get((trade as any).strategy)?.color ?? "var(--ink-4)"}
                          size="lg"
                        />
                      </>
                    )}
                    <span>·</span>
                    <span>Opened {String(trade.open_date || "").slice(0, 10)}</span>
                    {trade.closed_date && <><span className="mx-2">·</span><span>Closed {String(trade.closed_date).slice(0, 10)}</span></>}
                  </div>
                  <button onClick={() => setExpandedCard(isExpanded ? null : trade.trade_id)}
                          className="flex items-center gap-1 text-[11px] font-medium px-2.5 py-1 rounded-[8px] transition-colors"
                          style={{ color: navColor, background: isExpanded ? `color-mix(in oklab, ${navColor} 8%, transparent)` : "transparent" }}>
                    {isExpanded ? "Collapse" : "Details"}
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                         style={{ transform: isExpanded ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}>
                      <path d="M6 9l6 6 6-6"/>
                    </svg>
                  </button>
                </div>
              </div>

              {/* ── Expanded Section ── */}
              {isExpanded && (
                <div style={{ borderTop: "1px solid var(--border)", animation: "slide-up 0.15s ease-out" }}>

                  {/* ── Section 3: Flight Deck (daily driver — always visible) ── */}
                  <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
                    <div className="text-[13px] font-semibold mb-3 flex items-center gap-2">
                      <span>🚁</span> Flight Deck: {trade.ticker}
                    </div>
                    <div className="grid grid-cols-8 gap-2">
                      {(() => {
                        const curPrice = isOpen ? (livePrices[trade.ticker] || avgEntry) : avgExit;
                        // Notional, not premium-per-contract: options need ×100 here
                        // so Total Equity, Unrealized P&L, and % Size all read in
                        // dollars at risk rather than per-share quotes.
                        const mktVal = isOpen ? shares * curPrice * multiplier : 0;
                        const unreal = isOpen ? (curPrice - avgEntry) * shares * multiplier : 0;
                        const unrealPct = avgEntry > 0 ? ((curPrice - avgEntry) / avgEntry) * 100 : 0;
                        const posSizePct = equity > 0 ? (mktVal / equity) * 100 : 0;
                        const riskBudget = parseFloat(String(trade.risk_budget || 0));
                        return [
                        { label: "Current Price", value: `$${curPrice.toFixed(2)}`, sub: undefined },
                        { label: "Orig Cost", value: b1Price > 0 ? `$${b1Price.toFixed(2)}` : `$${avgEntry.toFixed(2)}`, sub: undefined },
                        { label: "Avg Cost", value: `$${avgEntry.toFixed(2)}`, sub: isOption ? `×${multiplier}` : undefined },
                        { label: "Risk Budget", value: `$${riskBudget.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, sub: undefined },
                        { label: `${unitLabel} Held`, value: String(isOpen ? shares : 0), sub: undefined },
                        { label: "Unrealized P&L", value: `$${unreal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, sub: isOpen ? `${unrealPct.toFixed(2)}%` : undefined, color: PLColor(unreal) },
                        { label: "Realized P&L", value: `$${realizedBank.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, sub: undefined, color: PLColor(realizedBank) },
                        { label: isOpen ? "Total Equity" : "Final Cost Basis", value: isOpen
                            ? `$${mktVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                            : `$${totalCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                          sub: isOpen ? `${posSizePct.toFixed(1)}% Size` : "Closed" },
                        ];
                      })().map(m => (
                        <div key={m.label} className="p-2.5 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[9px] uppercase tracking-[0.06em] font-semibold truncate" style={{ color: "var(--ink-4)" }}>{m.label}</div>
                          <div className="text-[15px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: (m as any).color || "var(--ink)" }}>{m.value}</div>
                          {m.sub && <div className="text-[10px] mt-0.5" style={{ color: PLColor(parseFloat(m.sub) || 0) }}>{m.sub}</div>}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ── Transaction History (full columns matching Streamlit) ── */}
                  <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
                    <div className="text-[13px] font-semibold mb-2 flex items-center gap-2">
                      <span>📋</span> Transaction History
                    </div>
                    {(() => {
                      // rowData (LIFO walk) is precomputed at the .map level so
                      // the Realized P&L tile and this table draw from one
                      // source. We still own the unrealized-P&L pass below
                      // since it depends on livePrice / isOpen.
                      const rowData = lifoRowData;

                      // Compute unrealized P&L and return % for open buy rows using live price
                      const currentPrice = isOpen ? livePrice : 0;
                      if (currentPrice > 0) {
                        rowData.forEach(row => {
                          if (!row.isSell && row.remaining > 0) {
                            const buyPrice = parseFloat(String(row.tx.amount || 0)) || enrichedEntry;
                            row.unrealizedPl = (currentPrice - buyPrice) * row.remaining * multiplier;
                            // Return % = simple price change from entry
                            if (buyPrice > 0) {
                              row.returnPct = ((currentPrice - buyPrice) / buyPrice) * 100;
                            }
                          }
                        });
                      }

                      // Filter state for transaction table
                      const [txFilter, setTxFilter] = [txnFilter, setTxnFilter];
                      const filteredRows = txFilter === "all" ? rowData
                        : txFilter === "open" ? rowData.filter(r => r.status === "Open" && !r.isSell)
                        : rowData.filter(r => r.status === "Closed");

                      const mono = "var(--font-jetbrains), monospace";
                      return rowData.length > 0 ? (
                      <div>
                        {/* Filter toggle */}
                        <div className="flex items-center gap-1 mb-3">
                          <span className="text-[11px] mr-1" style={{ color: "var(--ink-4)" }}>Filter Status</span>
                          {(["all", "open", "closed"] as const).map(f => (
                            <label key={f} className="flex items-center gap-1 cursor-pointer text-[11px]" style={{ color: txFilter === f ? "var(--ink)" : "var(--ink-4)" }}>
                              <input type="radio" name={`txfilter-${trade.trade_id}`} checked={txFilter === f} onChange={() => setTxnFilter(f)}
                                className="w-3 h-3 accent-[#6366f1]" />
                              <span className="capitalize">{f}</span>
                            </label>
                          ))}
                        </div>
                        <div className="overflow-x-auto rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                        <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                          <thead>
                            <tr>
                              {["Trx ID", "Date", "Ticker", "Action", "Status", unitLabel, "Remaining", "Amount", "Exit Price", "Stop Loss", "Value", "Realized PL", "Unrealized PL", "Return %", "Rule", "Notes", "Edit"].map(h => (
                                <th key={h} className="text-left px-2.5 py-2 text-[9px] uppercase tracking-[0.06em] font-semibold whitespace-nowrap"
                                    style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {filteredRows.map((row, i) => {
                              const tx = row.tx;
                              const txStop = parseFloat(String(tx.stop_loss || 0));
                              return (
                                <tr key={i} style={{ borderBottom: i < filteredRows.length - 1 ? "1px solid var(--border)" : "none" }}>
                                  <td className="px-2.5 py-2 font-semibold" style={{ fontFamily: mono, fontSize: 10, color: "var(--ink-3)" }}>{(tx as any).trx_id || ""}</td>
                                  <td className="px-2.5 py-2 whitespace-nowrap" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(tx.date || "").slice(0, 10)}</td>
                                  <td className="px-2.5 py-2 font-semibold" style={{ fontFamily: mono, fontSize: 11 }}>{tx.ticker || trade.ticker}</td>
                                  <td className="px-2.5 py-2">
                                    <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold"
                                          style={{ background: row.isSell ? "color-mix(in oklab, #e5484d 12%, var(--surface))" : "color-mix(in oklab, #08a86b 12%, var(--surface))", color: row.isSell ? "#dc2626" : "#16a34a" }}>
                                      {tx.action}
                                    </span>
                                  </td>
                                  <td className="px-2.5 py-2 text-[10px]" style={{ color: row.status === "Open" ? "#08a86b" : "var(--ink-4)" }}>{row.status}</td>
                                  <td className="px-2.5 py-2" style={{ fontFamily: mono, color: row.displayShares < 0 ? "#e5484d" : "var(--ink)" }}>{row.displayShares}</td>
                                  <td className="px-2.5 py-2" style={{ fontFamily: mono, color: "var(--ink-4)" }}>{!row.isSell ? (row.remaining > 0 ? row.remaining : 0) : ""}</td>
                                  <td className="px-2.5 py-2 privacy-mask" style={{ fontFamily: mono }}>${parseFloat(String(tx.amount || 0)).toFixed(2)}</td>
                                  <td className="px-2.5 py-2 privacy-mask" style={{ fontFamily: mono, color: "var(--ink-4)" }}>{!row.isSell && row.exitPrice > 0 ? `$${row.exitPrice.toFixed(2)}` : "—"}</td>
                                  <td className="px-2.5 py-2" style={{ fontFamily: mono, color: "var(--ink-4)" }}>{txStop > 0 ? `$${txStop.toFixed(2)}` : "—"}</td>
                                  <td className="px-2.5 py-2 privacy-mask" style={{ fontFamily: mono, color: row.value < 0 ? "#e5484d" : "var(--ink)" }}>${Math.abs(row.value).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                                  <td className="px-2.5 py-2 privacy-mask" style={{ fontFamily: mono, fontWeight: 600, color: PLColor(row.realizedPl) }}>{!row.isSell && row.realizedPl !== 0 ? `$${row.realizedPl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : !row.isSell ? "$0.00" : ""}</td>
                                  <td className="px-2.5 py-2 privacy-mask" style={{ fontFamily: mono, color: PLColor(row.unrealizedPl || 0) }}>{!row.isSell && row.unrealizedPl !== 0 ? `$${row.unrealizedPl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : !row.isSell ? "$0.00" : ""}</td>
                                  <td className="px-2.5 py-2" style={{ fontFamily: mono, fontWeight: 600, color: PLColor(row.returnPct) }}>{!row.isSell ? `${row.returnPct.toFixed(2)}%` : ""}</td>
                                  <td className="px-2.5 py-2 text-[10px]" style={{ color: "var(--ink-3)" }}>{tx.rule || ""}</td>
                                  <td className="px-2.5 py-2 text-[10px]" style={{ color: "var(--ink-4)", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{(tx as any).notes || ""}</td>
                                  <td className="px-2.5 py-2">
                                    <button onClick={() => openEditModal(tx)}
                                            className="text-[10px] px-2 py-0.5 rounded-[4px] cursor-pointer transition-colors hover:brightness-95"
                                            style={{ background: "var(--surface-2)", color: "var(--ink-3)", border: "1px solid var(--border)" }}>
                                      Edit
                                    </button>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        </div>
                      </div>
                      ) : (
                        <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No transaction details available</div>
                      );
                    })()}
                  </div>

                  {/* ── Interactive Trade Chart (collapsible) ── */}
                  <div style={{ borderTop: "1px solid var(--border)" }}>
                    <button onClick={() => setLiveChartOpen(liveChartOpen === trade.trade_id ? null : trade.trade_id)}
                            className="w-full flex items-center gap-2 px-5 py-3 text-left cursor-pointer transition-colors hover:brightness-95"
                            style={{ background: "var(--surface-2)" }}>
                      <span className="text-[10px] transition-transform" style={{ transform: liveChartOpen === trade.trade_id ? "rotate(90deg)" : "none", color: "var(--ink-4)" }}>▶</span>
                      <span className="text-[12px] font-semibold" style={{ color: "var(--ink-3)" }}>Interactive Chart</span>
                      <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>(daily · weekly · monthly)</span>
                    </button>
                    {liveChartOpen === trade.trade_id && (
                      <div style={{ animation: "slide-up 0.12s ease-out" }}>
                        <InteractiveChart
                          ticker={trade.ticker}
                          tradeId={trade.trade_id}
                          openDate={trade.open_date}
                          closedDate={trade.closed_date}
                          details={allDetails}
                          navColor={navColor}
                        />
                      </div>
                    )}
                  </div>

                  {/* ── Section 2: Charts, Fundamentals, Notes (post-analysis — collapsed by default) ── */}
                  <div style={{ borderTop: "1px solid var(--border)" }}>
                    <button onClick={() => setAnalysisOpen(analysisOpen === trade.trade_id ? null : trade.trade_id)}
                            className="w-full flex items-center gap-2 px-5 py-3 text-left cursor-pointer transition-colors hover:brightness-95"
                            style={{ background: "var(--surface-2)" }}>
                      <span className="text-[10px] transition-transform" style={{ transform: analysisOpen === trade.trade_id ? "rotate(90deg)" : "none", color: "var(--ink-4)" }}>▶</span>
                      <span className="text-[12px] font-semibold" style={{ color: "var(--ink-3)" }}>Charts, Fundamentals & Notes</span>
                      <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>(post-analysis)</span>
                    </button>

                    {analysisOpen === trade.trade_id && (
                      <div style={{ animation: "slide-up 0.12s ease-out" }}>
                        {/* Charts */}
                        <TradeCharts tradeId={trade.trade_id} ticker={trade.ticker} />

                        {/* Trade Notes */}
                        <div className="px-5 py-4">
                          <div className="text-[13px] font-semibold mb-3 flex items-center gap-2">
                            <span>📝</span> Trade Notes
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 rounded-[8px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Entry Notes</div>
                              <div className="text-[12px]" style={{ color: "var(--ink-3)" }}>
                                {trade.buy_notes || "_No entry notes_"}
                              </div>
                            </div>
                            <div className="p-3 rounded-[8px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
                                {trade.sell_rule ? `Sell Rule: ${trade.sell_rule}` : "Setup/Rule"}
                              </div>
                              <div className="text-[12px]" style={{ color: "var(--ink-3)" }}>
                                {trade.sell_notes || trade.rule || "_No notes_"}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-16 text-sm" style={{ color: "var(--ink-4)" }}>No trades match your filters</div>
      )}
      </>
      )}

      {/* Phase 2 — right-click context menu for retroactive strategy
          tagging on Trade Journal cards. Same flyout/flat-list split
          as ACS and All Campaigns. */}
      {tjCtxMenu && strategies.length > 0 && (
        <div className="fixed z-50 rounded-[10px] py-1.5 min-w-[200px] overflow-hidden"
             data-testid="tj-ctx-menu"
             style={{
               left: tjCtxMenu.x,
               top: tjCtxMenu.y,
               background: "var(--surface)",
               border: "1px solid var(--border)",
               boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
             }}>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>
            {tjCtxMenu.trade.ticker} · {tjCtxMenu.trade.trade_id}
          </div>
          {coarsePointer ? (
            <StrategyFlatList
              strategies={strategies}
              currentStrategy={(tjCtxMenu.trade as any).strategy}
              onPick={(name) => setTradeStrategyOptimistic(tjCtxMenu.trade.trade_id, name)}
            />
          ) : (
            <StrategyFlyout
              strategies={strategies}
              currentStrategy={(tjCtxMenu.trade as any).strategy}
              onPick={(name) => setTradeStrategyOptimistic(tjCtxMenu.trade.trade_id, name)}
            />
          )}
        </div>
      )}

      {/* Closed-trade edit modal — opens from the Edit button in any
          Transaction History row. Same recipe as active-campaign.tsx's
          EOD modal: fixed-position backdrop + stop-propagation card +
          ESC handler in useEffect above.

          z-[100] matches the EOD modal in active-campaign.tsx. The
          chart lightbox uses z-[200] but cannot open simultaneously
          with this modal under current UI flows. If a future feature
          opens both, this layering needs revisiting. */}
      {editingTxn && (
        <div data-testid="tj-edit-modal-backdrop"
             className="fixed inset-0 z-[100] grid place-items-start justify-center pt-[10vh]"
             style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
             onClick={() => { if (!editLoading) closeEditModal(); }}>
          <div className="w-[640px] max-w-[92vw] rounded-[14px] overflow-hidden"
               style={{ background: "var(--surface)", boxShadow: "0 20px 48px rgba(0,0,0,0.2), 0 0 0 1px var(--border)", animation: "cmdk-rise 0.22s cubic-bezier(.2,.9,.3,1.1)" }}
               onClick={e => e.stopPropagation()}>
            <div className="px-[18px] py-3.5 flex items-center" style={{ borderBottom: "1px solid var(--border)" }}>
              <div>
                <div className="text-[14px] font-semibold">
                  Edit · {editingTxn.action} · {editingTxn.ticker}
                </div>
                <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                  Trade {editingTxn.trade_id} · transaction {(editingTxn as any).trx_id || `#${(editingTxn as any).detail_id}`}
                </div>
              </div>
              <kbd className="ml-auto text-[10px] rounded px-1.5 py-0.5"
                   style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>ESC</kbd>
            </div>
            <div className="p-4 flex flex-col gap-3 max-h-[70vh] overflow-y-auto">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Date / Time</label>
                  <input type="datetime-local" value={editForm.date}
                         onChange={e => setEditForm(f => ({ ...f, date: e.target.value }))}
                         disabled={editLoading}
                         className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                         style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Trx ID</label>
                  {/* Read-only — server generates trx_ids collision-safely
                      (db_layer.generate_unique_trx_id + migration 018 UNIQUE).
                      Editing here would just be a way to create new collisions. */}
                  <input type="text" value={(editingTxn as any).trx_id || ""} readOnly
                         className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                         style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", opacity: 0.6, cursor: "not-allowed", fontFamily: "var(--font-jetbrains), monospace" }} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Shares</label>
                  <input type="number" step="any" value={editForm.shares}
                         onChange={e => setEditForm(f => ({ ...f, shares: e.target.value }))}
                         disabled={editLoading}
                         className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                         style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Price ($)</label>
                  <input type="number" step="0.01" value={editForm.amount}
                         onChange={e => setEditForm(f => ({ ...f, amount: e.target.value }))}
                         disabled={editLoading}
                         className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                         style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Stop Loss ($)</label>
                  <input type="number" step="0.01" value={editForm.stop_loss}
                         onChange={e => setEditForm(f => ({ ...f, stop_loss: e.target.value }))}
                         disabled={editLoading}
                         className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                         style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Rule (Strategy)</label>
                  <select value={editForm.rule}
                          onChange={e => setEditForm(f => ({ ...f, rule: e.target.value }))}
                          disabled={editLoading}
                          className="w-full h-[38px] px-3 rounded-[8px] text-[12px] outline-none"
                          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", WebkitAppearance: "none", MozAppearance: "none", appearance: "none" }}>
                    <option value="">Select...</option>
                    {(String(editingTxn.action).toUpperCase() === "SELL" ? SELL_RULES : BUY_RULES).map(r => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Notes</label>
                <textarea value={editForm.notes}
                          onChange={e => setEditForm(f => ({ ...f, notes: e.target.value }))}
                          disabled={editLoading} rows={2}
                          className="w-full px-3 py-2 rounded-[8px] text-[12px] outline-none resize-none"
                          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
              </div>
              {editError && (
                <div className="px-3 py-2 rounded-[8px] text-[11px] leading-relaxed"
                     style={{
                       background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
                       border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
                       color: "#991b1b",
                     }}>
                  {editError}
                </div>
              )}
            </div>
            <div className="px-[18px] py-3 flex items-center gap-2" style={{ borderTop: "1px solid var(--border)" }}>
              <button onClick={deleteTxn}
                      disabled={editLoading}
                      className="h-[32px] px-3 rounded-md text-[12px] font-medium transition-colors hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                      style={{
                        background: confirmingDelete ? "#e5484d" : "var(--surface)",
                        color: confirmingDelete ? "white" : "#e5484d",
                        border: confirmingDelete ? "1px solid #e5484d" : "1px solid color-mix(in oklab, #e5484d 35%, var(--border))",
                      }}>
                {confirmingDelete ? "Confirm Delete" : "Delete Transaction"}
              </button>
              <button onClick={closeEditModal}
                      disabled={editLoading}
                      className="ml-auto h-[32px] px-3 rounded-md text-[12px] font-medium hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
                Cancel
              </button>
              <button onClick={saveEdit}
                      disabled={editLoading}
                      className="h-[32px] px-3.5 rounded-md text-[12px] font-medium text-white flex items-center gap-1.5 hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                      style={{ background: navColor }}>
                {editLoading && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
                    <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
                  </svg>
                )}
                {editLoading ? "Saving…" : "Save Changes"}
              </button>
            </div>
          </div>
          <style jsx global>{`@keyframes cmdk-rise { from { transform: translateY(-10px) scale(0.97); opacity: 0; } }`}</style>
        </div>
      )}
    </div>
  );
}
