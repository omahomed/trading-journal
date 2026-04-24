"use client";

import { useState, useEffect, useMemo } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";
import { InteractiveChart } from "./interactive-chart";

type SortKey = "newest" | "oldest" | "best" | "worst" | "ticker";
type StatusFilter = "all" | "open" | "closed";
type DateRange = "all" | "7d" | "30d" | "90d" | "ytd" | "recent10";

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
  const [allTrades, setAllTrades] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [equity, setEquity] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("open");
  const [sort, setSort] = useState<SortKey>("newest");
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tickerDropdownOpen, setTickerDropdownOpen] = useState(false);
  const [tickerQuery, setTickerQuery] = useState("");
  const [dateRange, setDateRange] = useState<DateRange>("all");
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [scaleOutOpen, setScaleOutOpen] = useState<string | null>(null);
  const [txnFilter, setTxnFilter] = useState<"all" | "open" | "closed">("all");
  const [analysisOpen, setAnalysisOpen] = useState<string | null>(null);
  const [liveChartOpen, setLiveChartOpen] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      api.tradesClosed(getActivePortfolio(), 500).catch(() => []),
      api.tradesRecent(getActivePortfolio(), 5000).catch(() => []),
      api.journalLatest(getActivePortfolio()).catch(() => ({ end_nlv: 100000 })),
    ]).then(async ([open, closed, details, journal]) => {
      const openArr = open as TradePosition[];
      setAllTrades([...openArr, ...closed as TradePosition[]]);
      setAllDetails(details as TradeDetail[]);
      setEquity(parseFloat(String((journal as any).end_nlv || 100000)));

      // Fetch live prices for open trades
      const tickers = openArr.map(t => t.ticker).filter(Boolean);
      if (tickers.length > 0) {
        try {
          const prices = await api.batchPrices(tickers);
          if (prices && !("error" in prices)) setLivePrices(prices);
        } catch { /* fall back */ }
      }
      setLoading(false);

      // Prefill from Active Campaign right-click (via localStorage)
      try {
        const raw = localStorage.getItem("journal_prefill");
        if (raw) {
          localStorage.removeItem("journal_prefill");
          const data = JSON.parse(raw);
          if (data.ticker) setSelectedTickers([data.ticker]);
          if (data.trade_id) setExpandedCard(data.trade_id);
          setStatusFilter("open");
        }
      } catch { /* ignore */ }
    });
  }, []);

  const filtered = useMemo(() => {
    let result = [...allTrades];

    // Status
    if (statusFilter !== "all") {
      result = result.filter(t => (t.status || "").toUpperCase() === statusFilter.toUpperCase());
    }

    // Ticker multi-select
    if (selectedTickers.length > 0) {
      result = result.filter(t => selectedTickers.includes(t.ticker || ""));
    }

    // Date range / Recent filter
    if (dateRange === "recent10") {
      // "Recently added or closed" = latest of open_date / closed_date.
      // Sort by that desc and slice to 10. Overrides the sort dropdown.
      const activityTs = (t: TradePosition) => {
        const o = new Date(String(t.open_date || 0)).getTime() || 0;
        const c = new Date(String(t.closed_date || 0)).getTime() || 0;
        return Math.max(o, c);
      };
      result.sort((a, b) => activityTs(b) - activityTs(a));
      result = result.slice(0, 10);
      return result;
    }
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
  }, [allTrades, statusFilter, sort, selectedTickers, dateRange]);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  const openCount = allTrades.filter(t => (t.status || "").toUpperCase() === "OPEN").length;
  const closedCount = allTrades.filter(t => (t.status || "").toUpperCase() === "CLOSED").length;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Trade <em className="italic" style={{ color: navColor }}>Journal</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          {allTrades.length} campaigns ({openCount} open, {closedCount} closed)
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
                       if (match && !selectedTickers.includes(match)) {
                         setSelectedTickers(prev => [...prev, match]);
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
          <option value="recent10">Recent 10 (added or closed)</option>
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
          <option value="90d">Last 90 Days</option>
          <option value="ytd">This Year</option>
        </select>

        <span className="text-[12px] ml-auto" style={{ color: "var(--ink-4)" }}>{filtered.length} results</span>
      </div>

      {/* Trade cards */}
      <div className="flex flex-col gap-4">
        {filtered.map(trade => {
          const isOpen = (trade.status || "").toUpperCase() === "OPEN";
          const shares = trade.shares || 0;
          const avgEntry = parseFloat(String(trade.avg_entry || 0));
          const avgExit = parseFloat(String(trade.avg_exit || 0));
          const totalCost = parseFloat(String(trade.total_cost || 0));
          const realizedBank = parseFloat(String(trade.realized_pl || 0));

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

          const livePrice = isOpen ? (livePrices[trade.ticker] || 0) : enrichedExit;
          const unrealizedPl = isOpen && livePrice > 0 ? (livePrice - enrichedEntry) * shares : 0;
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
          const corePl = b1Price > 0 && avgExit > 0 ? (avgExit - b1Price) * b1Shares : 0;

          const isExpanded = expandedCard === trade.trade_id;

          return (
            <div key={trade.trade_id} className="rounded-[14px] overflow-hidden transition-all"
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
                        // optimistic local update
                        setAllTrades(prev => prev.map(t => t.trade_id === trade.trade_id ? { ...t, grade: v } as any : t));
                        api.setTradeGrade({ portfolio: getActivePortfolio(), trade_id: trade.trade_id, grade: v })
                          .catch(() => {
                            // revert on failure
                            setAllTrades(prev => prev.map(t => t.trade_id === trade.trade_id ? { ...t, grade: (trade as any).grade ?? null } as any : t));
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
                  <CardMetric label="Shares" value={String(shares)} />
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

                {/* Footer: dates + rule + expand toggle */}
                <div className="flex items-center justify-between mt-3 pt-2">
                  <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
                    <span>{trade.rule}</span>
                    <span className="mx-2">·</span>
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
                    <div className="grid grid-cols-7 gap-2">
                      {(() => {
                        const curPrice = isOpen ? (livePrices[trade.ticker] || avgEntry) : avgExit;
                        const mktVal = shares * curPrice;
                        const unreal = isOpen ? (curPrice - avgEntry) * shares : 0;
                        const unrealPct = avgEntry > 0 ? ((curPrice - avgEntry) / avgEntry) * 100 : 0;
                        const posSizePct = equity > 0 ? (mktVal / equity) * 100 : 0;
                        return [
                        { label: "Current Price", value: `$${curPrice.toFixed(2)}`, sub: undefined },
                        { label: "Orig Cost", value: b1Price > 0 ? `$${b1Price.toFixed(2)}` : `$${avgEntry.toFixed(2)}`, sub: undefined },
                        { label: "Avg Cost", value: `$${avgEntry.toFixed(2)}`, sub: undefined },
                        { label: "Shares Held", value: String(isOpen ? shares : 0), sub: undefined },
                        { label: "Unrealized P&L", value: `$${unreal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, sub: isOpen ? `${unrealPct.toFixed(2)}%` : undefined, color: PLColor(unreal) },
                        { label: "Realized P&L", value: `$${realizedBank.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, sub: undefined, color: PLColor(realizedBank) },
                        { label: "Total Equity", value: `$${mktVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: `${posSizePct.toFixed(1)}% Size` },
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
                      // LIFO processing — P&L attributed to BUY rows (matching Streamlit)
                      const sorted = [...txns].sort((a, b) => {
                        const da = String(a.date || "");
                        const db2 = String(b.date || "");
                        if (da !== db2) return da.localeCompare(db2);
                        return String(a.action).toUpperCase() === "BUY" ? -1 : 1;
                      });

                      const inventory: { idx: number; qty: number; price: number }[] = [];
                      const rowData: {
                        tx: typeof txns[0]; displayShares: number; remaining: number;
                        exitPrice: number; realizedPl: number; returnPct: number;
                        unrealizedPl: number; status: string; value: number; isSell: boolean;
                      }[] = [];

                      for (let i = 0; i < sorted.length; i++) {
                        const tx = sorted[i];
                        const action = String(tx.action || "").toUpperCase();
                        const txShares = Math.abs(parseFloat(String(tx.shares || 0)));
                        const txAmount = parseFloat(String(tx.amount || 0));
                        const txValue = parseFloat(String(tx.value || 0));

                        if (action === "BUY") {
                          inventory.push({ idx: i, qty: txShares, price: txAmount || enrichedEntry });
                          rowData.push({
                            tx, displayShares: txShares, remaining: txShares,
                            exitPrice: 0, realizedPl: 0, returnPct: 0, unrealizedPl: 0,
                            status: "Open", value: txValue, isSell: false,
                          });
                        } else if (action === "SELL") {
                          let toSell = txShares;
                          const sellPrice = txAmount;

                          while (toSell > 0 && inventory.length > 0) {
                            const last = inventory[inventory.length - 1];
                            const take = Math.min(toSell, last.qty);
                            const costBasis = take * last.price;
                            const revenue = take * sellPrice;
                            const rpl = revenue - costBasis;
                            const retPct = costBasis > 0 ? (rpl / costBasis) * 100 : 0;

                            last.qty -= take;
                            toSell -= take;

                            // Attribute P&L to the BUY row
                            const buyRow = rowData[last.idx];
                            if (buyRow) {
                              buyRow.remaining = last.qty;
                              buyRow.realizedPl += rpl;
                              buyRow.exitPrice = sellPrice;
                              // For fully closed rows, return % = exit vs entry price
                              const buyRowPrice = parseFloat(String(buyRow.tx.amount || 0)) || enrichedEntry;
                              buyRow.returnPct = buyRowPrice > 0 ? ((sellPrice - buyRowPrice) / buyRowPrice) * 100 : 0;
                              if (last.qty < 0.00001) buyRow.status = "Closed";
                            }
                            if (last.qty < 0.00001) inventory.pop();
                          }

                          // Sell row — no P&L (attributed to buy rows above)
                          rowData.push({
                            tx, displayShares: -txShares, remaining: 0,
                            exitPrice: 0, realizedPl: 0, returnPct: 0, unrealizedPl: 0,
                            status: "Closed", value: -txValue, isSell: true,
                          });
                        }
                      }

                      // Compute unrealized P&L and return % for open buy rows using live price
                      const currentPrice = isOpen ? livePrice : 0;
                      if (currentPrice > 0) {
                        rowData.forEach(row => {
                          if (!row.isSell && row.remaining > 0) {
                            const buyPrice = parseFloat(String(row.tx.amount || 0)) || enrichedEntry;
                            row.unrealizedPl = (currentPrice - buyPrice) * row.remaining;
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
                              {["Trx ID", "Date", "Ticker", "Action", "Status", "Shares", "Remaining", "Amount", "Exit Price", "Stop Loss", "Value", "Realized PL", "Unrealized PL", "Return %", "Rule", "Notes"].map(h => (
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
    </div>
  );
}
