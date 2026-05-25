"use client";

import { forwardRef, useEffect, useMemo, useRef, useState } from "react";
import { Search, TriangleAlert, X } from "lucide-react";
import {
  api,
  getActivePortfolio,
  type LotClosure,
  type TradeDetail,
  type TradePosition,
} from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { computeEnrichedPositions, type EnrichedPosition } from "@/lib/positions";
import { classifyTradeState, type TradeState } from "@/lib/trade-state";
import { lotClosuresToLifoRows, type LifoRow } from "@/lib/lifo-closures";
import { SELL_RULES } from "@/lib/trade-rules";
import type { SellRuleTier } from "@/lib/sell-rule";
import { ImageLightbox, type LightboxImage } from "@/components/image-lightbox";

/**
 * Mobile Trade Journal — Phase 2 Step 4. Combines two desktop workflows
 * into one mobile surface:
 *   - Active Campaign Summary (ACS) → at-a-glance open positions list
 *   - Trade Journal → per-trade detail with lots + uploaded charts
 *
 * Reads same hooks the desktop ACS/Journal reads (`api.tradesOpen`,
 * `api.tradesOpenDetails`, `api.batchPrices`, `api.journalLatest`,
 * `api.config("pyramid_rules")`). Card-level derivations come from
 * `computeEnrichedPositions` (lib/positions.ts) — single source of
 * truth for sell_rule_tier, return %, position size %, signed risk, etc.
 *
 * Per-trade lazy fetches: `api.tradeImages` fires on detail-sheet open,
 * cached per-trade per-session so reopens don't refetch.
 *
 * Read-only end-to-end. No edit, no upload, no transaction
 * modification — desktop remains the editing surface.
 */

type TradeImage = {
  id?: number | string;
  view_url?: string;
  image_url?: string;
  file_name?: string;
  image_type?: string;
  uploaded_at?: string;
};

type FilterKey = "all" | "ready" | "winners" | "losers" | "atrisk" | "options";

type Props = {
  initialTradeId?: string;
  onTradeConsumed?: () => void;
};

// Default pyramid config — fallback when the /api/config fetch fails.
// Same defensive defaults the mobile sizer uses (PR2 commit b6c2e13).
const DEFAULT_PYRAMID_RULES = { trigger_pct: 5, alloc_pct: 20 };

// Number of transactions visible by default in the detail sheet before
// the "Show N more" expander reveals the rest. Matches desktop trade-
// journal's per-card layout density on a phone-width column.
const TRANSACTIONS_DEFAULT_LIMIT = 5;

export function MobileTradeJournal({ initialTradeId, onTradeConsumed }: Props) {
  const { activePortfolio } = usePortfolio();

  // ── Mount-fetched state ─────────────────────────────────────────
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [allClosures, setAllClosures] = useState<LotClosure[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [equity, setEquity] = useState<number | null>(null);
  const [pyramidRules, setPyramidRules] = useState(DEFAULT_PYRAMID_RULES);
  const [loading, setLoading] = useState(true);

  // ── UI state ─────────────────────────────────────────────────────
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterKey>("all");
  const [detailTradeId, setDetailTradeId] = useState<string | null>(null);
  const [transactionsExpanded, setTransactionsExpanded] = useState(false);

  // ── Lazy image cache ─────────────────────────────────────────────
  const [imagesByTradeId, setImagesByTradeId] = useState<
    Record<string, TradeImage[]>
  >({});
  const [imagesLoading, setImagesLoading] = useState<Set<string>>(new Set());
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  // Refs for the card elements, keyed by trade_id, so deep-link can
  // scrollIntoView the matching card before opening its detail sheet.
  const cardRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

  // ── Mount fetch ──────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const portfolio = getActivePortfolio();
    Promise.all([
      api.tradesOpen(portfolio).catch((err) => {
        log.error("mobile-trade-journal", "tradesOpen fetch failed", err);
        return [] as TradePosition[];
      }),
      api.tradesOpenDetails(portfolio).catch((err) => {
        log.error("mobile-trade-journal", "tradesOpenDetails fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
      api.journalLatest(portfolio).catch((err) => {
        log.error("mobile-trade-journal", "journalLatest fetch failed", err);
        return null;
      }),
      api.config("pyramid_rules").catch((err) => {
        log.error("mobile-trade-journal", "config pyramid_rules fetch failed", err);
        return { key: "pyramid_rules", value: DEFAULT_PYRAMID_RULES };
      }),
    ]).then(([open, openDet, journal, pyrCfg]) => {
      if (cancelled) return;
      const openArr = (open ?? []) as TradePosition[];
      setOpenTrades(openArr);
      const bundle = (openDet as { details?: TradeDetail[]; lot_closures?: LotClosure[] }) ?? {
        details: [],
        lot_closures: [],
      };
      setAllDetails(bundle.details ?? []);
      setAllClosures(bundle.lot_closures ?? []);
      const endNlv = journal
        ? parseFloat(String((journal as { end_nlv?: number | string }).end_nlv ?? 0))
        : 0;
      setEquity(Number.isFinite(endNlv) && endNlv > 0 ? endNlv : null);
      const cfgVal = (pyrCfg as { value?: { trigger_pct?: number; alloc_pct?: number } } | null)?.value;
      if (cfgVal && typeof cfgVal.trigger_pct === "number" && typeof cfgVal.alloc_pct === "number") {
        setPyramidRules({ trigger_pct: cfgVal.trigger_pct, alloc_pct: cfgVal.alloc_pct });
      }
      setLoading(false);

      // Live prices — fire-and-forget so price absence never blocks the
      // first paint. The shared price provider caches per ticker so
      // subsequent Dashboard/ACS/Journal nav share the result.
      const tickers = openArr.map((t) => t.ticker).filter(Boolean);
      if (tickers.length > 0) {
        api
          .batchPrices(tickers, portfolio)
          .then((prices) => {
            if (cancelled) return;
            if (prices && !("error" in prices)) {
              setLivePrices(prices as Record<string, number>);
            }
          })
          .catch(() => {
            /* fall back to entry prices via computeEnrichedPositions */
          });
      }
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Enriched positions (single source of truth for card data) ───
  const enrichedAll = useMemo(
    () =>
      computeEnrichedPositions(openTrades, allDetails, equity ?? 0, livePrices),
    [openTrades, allDetails, equity, livePrices],
  );

  // Lookup keyed by trade_id for O(1) access by detail sheet + deep-link.
  const enrichedById = useMemo(() => {
    const m = new Map<string, EnrichedPosition>();
    for (const p of enrichedAll) m.set(p.trade_id, p);
    return m;
  }, [enrichedAll]);

  // ── Filter + search ─────────────────────────────────────────────
  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    return enrichedAll.filter((p) => {
      // Ticker substring (case-insensitive). Empty query matches all.
      if (q && !p.ticker.toUpperCase().includes(q)) return false;
      switch (filter) {
        case "all":
          return true;
        case "ready":
          return classifyTradeState(p, allDetails, pyramidRules) === "ready";
        case "winners":
          return p.overall_pl > 0;
        case "losers":
          return p.overall_pl < 0;
        case "atrisk":
          // Current Risk % < 0 ⇒ live price below safe stop ⇒ at risk.
          return p.risk_pct < 0;
        case "options":
          return p.is_option;
      }
    });
  }, [enrichedAll, filter, query, allDetails, pyramidRules]);

  // Chip counts use the same predicates as the filter but ignore
  // `filter` itself (so each chip always shows the count of trades
  // that match THAT chip's predicate, not the active narrowing).
  const counts = useMemo(() => {
    const all = enrichedAll;
    let ready = 0;
    let winners = 0;
    let losers = 0;
    let atrisk = 0;
    let options = 0;
    for (const p of all) {
      if (p.is_option) options++;
      if (p.overall_pl > 0) winners++;
      if (p.overall_pl < 0) losers++;
      if (p.risk_pct < 0) atrisk++;
      if (classifyTradeState(p, allDetails, pyramidRules) === "ready") ready++;
    }
    return { all: all.length, ready, winners, losers, atrisk, options };
  }, [enrichedAll, allDetails, pyramidRules]);

  // ── Summary tile totals (mirrors active-campaign.tsx:641-670) ───
  const totals = useMemo(() => {
    const eq = equity ?? 0;
    const equities = enrichedAll.filter((p) => !p.is_option);
    const options = enrichedAll.filter((p) => p.is_option);
    const equityExposureDollar = equities.reduce((a, p) => a + p.current_value, 0);
    const optionsExposureDollar = options.reduce((a, p) => a + p.current_value, 0);
    const equityExposurePct = eq > 0 ? (equityExposureDollar / eq) * 100 : 0;
    const optionsExposurePct = eq > 0 ? (optionsExposureDollar / eq) * 100 : 0;
    const equityPlSum = equities.reduce((a, p) => a + p.overall_pl, 0);
    const optionsPlSum = options.reduce((a, p) => a + p.overall_pl, 0);
    const equityCostSum = equities.reduce((a, p) => a + p.total_cost, 0);
    const equityPlPct = equityCostSum > 0 ? (equityPlSum / equityCostSum) * 100 : 0;
    const capitalAtRiskTotal = [...equities, ...options].reduce(
      (sum, p) => sum + Math.max(0, -p.projected_pl),
      0,
    );
    const capitalAtRiskPct = eq > 0 ? (capitalAtRiskTotal / eq) * 100 : 0;
    const openRiskTotal = equities.reduce((sum, p) => {
      const safeStop = p.avg_stop > 0 ? p.avg_stop : p.avg_entry;
      return sum + (p.current_price - safeStop) * p.shares * p.multiplier;
    }, 0);
    const openRiskPct = eq > 0 ? (openRiskTotal / eq) * 100 : 0;
    return {
      equityExposureDollar,
      equityExposurePct,
      optionsExposureDollar,
      optionsExposurePct,
      equityPlSum,
      optionsPlSum,
      equityPlPct,
      capitalAtRiskTotal,
      capitalAtRiskPct,
      openRiskTotal,
      openRiskPct,
    };
  }, [enrichedAll, equity]);

  // ── Deep-link receiver (?trade_id=) ─────────────────────────────
  useEffect(() => {
    if (!initialTradeId || loading) return;
    const found = enrichedById.get(initialTradeId);
    // Silent fall-through when trade isn't in this portfolio's open
    // list — covers cross-portfolio links + closed-since-link-generated
    // races. No error UI per directive.
    if (!found) {
      onTradeConsumed?.();
      return;
    }
    setDetailTradeId(initialTradeId);
    // scrollIntoView is missing in jsdom; guarding so tests don't blow up.
    const node = cardRefs.current.get(initialTradeId);
    if (node && typeof node.scrollIntoView === "function") {
      node.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    onTradeConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialTradeId, loading, enrichedById]);

  // ── Lazy image fetch on detail sheet open ───────────────────────
  useEffect(() => {
    if (!detailTradeId) return;
    if (imagesByTradeId[detailTradeId]) return; // cached, skip
    if (imagesLoading.has(detailTradeId)) return; // in-flight
    const tradeId = detailTradeId;
    setImagesLoading((prev) => {
      const next = new Set(prev);
      next.add(tradeId);
      return next;
    });
    api
      .tradeImages(tradeId)
      .then((imgs) => {
        const arr = Array.isArray(imgs) ? (imgs as TradeImage[]) : [];
        setImagesByTradeId((prev) => ({ ...prev, [tradeId]: arr }));
      })
      .catch((err) => {
        log.debug("mobile-trade-journal", "tradeImages fetch failed", err);
        setImagesByTradeId((prev) => ({ ...prev, [tradeId]: [] }));
      })
      .finally(() => {
        setImagesLoading((prev) => {
          const next = new Set(prev);
          next.delete(tradeId);
          return next;
        });
      });
  }, [detailTradeId, imagesByTradeId, imagesLoading]);

  // Reset transactions expansion when switching trades.
  useEffect(() => {
    setTransactionsExpanded(false);
  }, [detailTradeId]);

  // Body-scroll lock + Escape close while detail sheet is open.
  useEffect(() => {
    if (!detailTradeId) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (lightboxIndex != null) setLightboxIndex(null);
        else setDetailTradeId(null);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      document.removeEventListener("keydown", onKey);
    };
  }, [detailTradeId, lightboxIndex]);

  const equityCount = enrichedAll.filter((p) => !p.is_option).length;
  const optionsCount = enrichedAll.filter((p) => p.is_option).length;

  // ── Render ──────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex flex-col gap-3 pt-2">
        <div className="h-9 animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-28 animate-pulse rounded-m-lg bg-m-surface-2" />
        <div className="h-9 animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-24 animate-pulse rounded-m-lg bg-m-surface-2" />
      </div>
    );
  }

  if (enrichedAll.length === 0) {
    return (
      <div className="flex flex-col gap-2 pt-2">
        <Header equityCount={0} optionsCount={0} />
        <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-12 text-center">
          <div className="text-[14px] font-medium text-m-text">
            No open positions in {activePortfolio?.name ?? "this portfolio"}
          </div>
          <div className="mt-1.5 text-[12px] text-m-text-dim">
            Switch portfolios or check back later.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 pt-2">
      <Header equityCount={equityCount} optionsCount={optionsCount} />

      <SummaryStrip
        nlv={equity ?? 0}
        equityPl={totals.equityPlSum}
        equityPlPct={totals.equityPlPct}
        equityExposurePct={totals.equityExposurePct}
        optionsExposurePct={totals.optionsExposurePct}
        openRiskPct={totals.openRiskPct}
      />

      <SearchBar value={query} onChange={setQuery} />

      <FilterChips active={filter} onChange={setFilter} counts={counts} />

      {filtered.length === 0 ? (
        <div className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-5 py-8 text-center text-[12px] text-m-text-dim">
          No positions match.
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {filtered.map((p) => (
            <PositionCard
              key={p.trade_id}
              ref={(node) => {
                if (node) cardRefs.current.set(p.trade_id, node);
                else cardRefs.current.delete(p.trade_id);
              }}
              enriched={p}
              state={classifyTradeState(p, allDetails, pyramidRules)}
              onTap={() => setDetailTradeId(p.trade_id)}
            />
          ))}
        </div>
      )}

      {detailTradeId && enrichedById.get(detailTradeId) && (
        <DetailSheet
          enriched={enrichedById.get(detailTradeId)!}
          details={allDetails}
          closures={allClosures}
          state={classifyTradeState(
            enrichedById.get(detailTradeId)!,
            allDetails,
            pyramidRules,
          )}
          images={imagesByTradeId[detailTradeId]}
          imagesLoading={imagesLoading.has(detailTradeId)}
          transactionsExpanded={transactionsExpanded}
          onToggleTransactions={() => setTransactionsExpanded((v) => !v)}
          onOpenLightbox={setLightboxIndex}
          onClose={() => setDetailTradeId(null)}
        />
      )}

      {detailTradeId && imagesByTradeId[detailTradeId] && (
        <ImageLightbox
          images={
            (imagesByTradeId[detailTradeId] ?? [])
              .map((img): LightboxImage => ({
                url: img.view_url ?? img.image_url ?? "",
                alt: img.file_name ?? "Trade chart",
              }))
              .filter((i) => i.url.length > 0)
          }
          activeIndex={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
          onNavigate={setLightboxIndex}
          ariaLabel="Trade chart"
        />
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────

function Header({
  equityCount,
  optionsCount,
}: {
  equityCount: number;
  optionsCount: number;
}) {
  return (
    <div className="flex flex-col gap-0.5 pt-1">
      <h1 className="text-[28px] font-medium tracking-[-0.02em] text-m-text">
        Trade{" "}
        <em
          className="not-italic font-normal text-m-accent"
          style={{ fontFamily: "var(--font-fraunces), Georgia, serif", fontStyle: "italic" }}
        >
          Journal
        </em>
      </h1>
      <div className="text-[12px] text-m-text-dim">
        {equityCount} {equityCount === 1 ? "equity" : "equities"} ·{" "}
        {optionsCount} {optionsCount === 1 ? "option" : "options"}
      </div>
    </div>
  );
}

function SummaryStrip({
  nlv,
  equityPl,
  equityPlPct,
  equityExposurePct,
  optionsExposurePct,
  openRiskPct,
}: {
  nlv: number;
  equityPl: number;
  equityPlPct: number;
  equityExposurePct: number;
  optionsExposurePct: number;
  openRiskPct: number;
}) {
  const plClass = equityPl >= 0 ? "text-m-accent" : "text-m-down";
  const optionsOver = optionsExposurePct > 10;
  const openRiskClass = openRiskPct >= 0 ? "text-m-accent" : "text-m-down";
  return (
    <div className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-[18px] py-[14px]">
      <div className="mb-0.5 text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
        NLV
      </div>
      <div className="font-m-num text-[28px] font-medium tabular-nums tracking-[-0.02em] text-m-text privacy-mask">
        {formatCurrency(nlv, { decimals: 0 })}
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className={`font-m-num text-sm font-medium tabular-nums ${plClass} privacy-mask`}>
          {formatCurrency(equityPl, { showSign: true, signGlyph: "unicode", decimals: 0 })}
        </span>
        <span className={`font-m-num text-[12px] tabular-nums ${plClass}`}>
          {equityPlPct >= 0 ? "+" : ""}
          {equityPlPct.toFixed(2)}%
        </span>
        <span className="text-[11px] text-m-text-dim">equity P&amp;L</span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <SummaryMetric label="Equity exp" value={`${equityExposurePct.toFixed(1)}%`} />
        <SummaryMetric
          label="Options exp"
          value={`${optionsExposurePct.toFixed(1)}%`}
          tone={optionsOver ? "down" : "default"}
          sub={optionsOver ? "over 10%" : "≤10% cap"}
        />
        <SummaryMetric
          label="Open risk"
          value={`${openRiskPct >= 0 ? "+" : ""}${openRiskPct.toFixed(2)}%`}
          tone={openRiskPct < 0 ? "down" : "up"}
        />
      </div>
    </div>
  );
}

function SummaryMetric({
  label,
  value,
  tone = "default",
  sub,
}: {
  label: string;
  value: string;
  tone?: "default" | "up" | "down";
  sub?: string;
}) {
  const valueClass =
    tone === "up" ? "text-m-accent" : tone === "down" ? "text-m-down" : "text-m-text";
  return (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
        {label}
      </div>
      <div className={`font-m-num text-[14px] font-medium tabular-nums ${valueClass}`}>
        {value}
      </div>
      {sub && (
        <div className="font-m-num text-[10px] tabular-nums text-m-text-dim">{sub}</div>
      )}
    </div>
  );
}

function SearchBar({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-2.5 rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <Search size={15} strokeWidth={1.5} className="text-m-text-dim" aria-hidden="true" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search ticker"
        aria-label="Search ticker"
        autoCapitalize="characters"
        autoCorrect="off"
        spellCheck={false}
        className="flex-1 bg-transparent text-sm text-m-text placeholder:text-m-text-dim focus:outline-none"
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label="Clear search"
          className="text-m-text-dim"
        >
          <X size={14} strokeWidth={1.5} aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

function FilterChips({
  active,
  onChange,
  counts,
}: {
  active: FilterKey;
  onChange: (k: FilterKey) => void;
  counts: { all: number; ready: number; winners: number; losers: number; atrisk: number; options: number };
}) {
  const chips: { key: FilterKey; label: string; count: number; tone: "primary" | "accent" | "warn" | "neutral" }[] = [
    { key: "all", label: "All", count: counts.all, tone: "primary" },
    { key: "ready", label: "Ready", count: counts.ready, tone: "accent" },
    { key: "winners", label: "Winners", count: counts.winners, tone: "neutral" },
    { key: "losers", label: "Losers", count: counts.losers, tone: "neutral" },
    { key: "atrisk", label: "At risk", count: counts.atrisk, tone: "warn" },
    { key: "options", label: "Options", count: counts.options, tone: "neutral" },
  ];
  return (
    <div className="-mx-5 overflow-x-auto whitespace-nowrap px-5">
      {chips.map((c) => (
        <FilterChip
          key={c.key}
          label={c.label}
          count={c.count}
          tone={c.tone}
          active={c.key === active}
          onClick={() => onChange(c.key)}
        />
      ))}
    </div>
  );
}

function FilterChip({
  label,
  count,
  active,
  tone,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  tone: "primary" | "accent" | "warn" | "neutral";
  onClick: () => void;
}) {
  const baseLayout = "mr-1.5 inline-block rounded-m-pill px-[14px] py-1.5 text-xs";
  let className: string;
  if (active) {
    className = `${baseLayout} bg-m-accent font-medium text-m-accent-text-on`;
  } else if (tone === "accent") {
    className = `${baseLayout} border-[0.5px] border-m-accent-border bg-m-accent-tint font-medium text-m-accent`;
  } else if (tone === "warn") {
    className = `${baseLayout} border-[0.5px] border-m-warn-border-soft bg-m-warn-tint font-medium text-m-warn`;
  } else {
    className = `${baseLayout} border-[0.5px] border-m-border bg-m-surface text-m-text-muted`;
  }
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={className}
    >
      {label} · {count}
    </button>
  );
}

type PositionCardProps = {
  enriched: EnrichedPosition;
  state: TradeState;
  onTap: () => void;
};

const PositionCard = forwardRef<HTMLButtonElement, PositionCardProps>(
  function PositionCard({ enriched, state, onTap }, ref) {
    const pnlPositive = enriched.overall_pl >= 0;
    const borderColor = pnlPositive ? "var(--m-accent)" : "var(--m-down)";
    const returnClass = pnlPositive ? "text-m-accent" : "text-m-down";
    const riskAtRisk = enriched.risk_pct < 0;
    const riskClass = riskAtRisk ? "text-m-down" : "text-m-accent";
    return (
      <button
        ref={ref}
        type="button"
        onClick={onTap}
        data-trade-id={enriched.trade_id}
        className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-4 py-3 text-left"
        style={{ borderLeft: `3px solid ${borderColor}` }}
      >
        <div className="mb-1 flex items-baseline justify-between gap-2">
          <div className="flex min-w-0 items-baseline gap-1.5">
            <span className="font-m-num text-[17px] font-medium tabular-nums text-m-text">
              {enriched.ticker}
            </span>
            <StatePill state={state} />
            <SellRuleChip tier={enriched.sell_rule_tier} />
          </div>
          <span
            className={`font-m-num text-sm font-medium tabular-nums shrink-0 ${returnClass}`}
          >
            {enriched.return_pct >= 0 ? "+" : ""}
            {enriched.return_pct.toFixed(1)}%
          </span>
        </div>
        <div className="mb-1 flex items-baseline justify-between gap-2">
          <span className="font-m-num text-[12px] tabular-nums text-m-text-muted">
            {enriched.shares.toLocaleString("en-US")}
            {enriched.is_option ? " ct" : " sh"}
            {" · "}
            {enriched.pos_size_pct.toFixed(1)}% size · D{enriched.days_held}
          </span>
          <span
            className={`font-m-num text-[13px] font-medium tabular-nums shrink-0 ${returnClass} privacy-mask`}
          >
            {formatCurrency(enriched.overall_pl, {
              showSign: true,
              signGlyph: "unicode",
              decimals: 0,
            })}
          </span>
        </div>
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            {enriched.avg_stop > 0
              ? `stop ${formatCurrency(enriched.avg_stop)}`
              : "no stop"}
          </span>
          <span
            className={`font-m-num text-[11px] font-medium tabular-nums ${riskClass}`}
          >
            risk {enriched.risk_pct >= 0 ? "+" : ""}
            {enriched.risk_pct.toFixed(2)}%
          </span>
        </div>
        <StateFooter state={state} enriched={enriched} />
      </button>
    );
  },
);

function StatePill({ state }: { state: TradeState }) {
  if (state === "ready") {
    return (
      <span className="rounded-[4px] bg-m-accent px-1.5 py-px text-[10px] font-semibold tracking-wider text-m-accent-text-on">
        READY
      </span>
    );
  }
  if (state === "added") {
    return (
      <span className="rounded-[4px] border-[0.5px] border-m-accent-border bg-m-accent-tint px-1.5 py-px text-[10px] font-medium tracking-wide text-m-accent">
        ADDED
      </span>
    );
  }
  if (state === "call") {
    return (
      <span className="rounded-[4px] bg-m-purple-tint px-1.5 py-px text-[10px] font-medium tracking-wide text-m-purple-text">
        CALLS
      </span>
    );
  }
  return null;
}

function SellRuleChip({ tier }: { tier: SellRuleTier | null }) {
  if (!tier) return null;
  const rule = SELL_RULES.find((r) => r.code === tier);
  const label = tier.toUpperCase();
  const tone: { bg: string; fg: string } =
    tier === "sr1"
      ? { bg: "color-mix(in oklab, var(--m-down) 14%, var(--m-surface))", fg: "var(--m-down)" }
      : tier === "sr11"
        ? { bg: "color-mix(in oklab, var(--m-warn) 14%, var(--m-surface))", fg: "var(--m-warn)" }
        : { bg: "color-mix(in oklab, var(--m-accent) 14%, var(--m-surface))", fg: "var(--m-accent)" };
  return (
    <span
      className="rounded-full px-1.5 py-px text-[10px] font-semibold tracking-wide"
      style={{ background: tone.bg, color: tone.fg }}
      title={rule ? `${label} ${rule.description}` : label}
    >
      {label}
    </span>
  );
}

function StateFooter({ state, enriched }: { state: TradeState; enriched: EnrichedPosition }) {
  if (state === "original") return null;
  if (state === "ready") {
    return (
      <div className="mt-2 border-t-[0.5px] border-m-border pt-1.5 font-m-num text-[11px] tabular-nums text-m-accent">
        Pyramid threshold met · last lot +{enriched.pyramid_pct.toFixed(1)}%
      </div>
    );
  }
  if (state === "added") {
    return (
      <div className="mt-2 border-t-[0.5px] border-m-border pt-1.5 font-m-num text-[11px] tabular-nums text-m-text-dim">
        Adds in place · last lot {enriched.pyramid_pct >= 0 ? "+" : ""}
        {enriched.pyramid_pct.toFixed(1)}% cushion
      </div>
    );
  }
  if (state === "call") {
    const exp = enriched.expiration;
    const expStr = exp
      ? exp.toLocaleDateString("en-US", { month: "short", day: "numeric" })
      : "—";
    return (
      <div className="mt-2 border-t-[0.5px] border-m-border pt-1.5 font-m-num text-[11px] tabular-nums text-m-text-dim">
        Expires {expStr} · {enriched.shares} contract{enriched.shares === 1 ? "" : "s"}
      </div>
    );
  }
  return null;
}

// ── Detail sheet ─────────────────────────────────────────────────

function DetailSheet({
  enriched,
  details,
  closures,
  state,
  images,
  imagesLoading,
  transactionsExpanded,
  onToggleTransactions,
  onOpenLightbox,
  onClose,
}: {
  enriched: EnrichedPosition;
  details: TradeDetail[];
  closures: LotClosure[];
  state: TradeState;
  images: TradeImage[] | undefined;
  imagesLoading: boolean;
  transactionsExpanded: boolean;
  onToggleTransactions: () => void;
  onOpenLightbox: (idx: number) => void;
  onClose: () => void;
}) {
  const tradeDetails = useMemo(
    () => details.filter((d) => d.trade_id === enriched.trade_id),
    [details, enriched.trade_id],
  );
  const tradeClosures = useMemo(
    () => closures.filter((c) => c.trade_id === enriched.trade_id),
    [closures, enriched.trade_id],
  );
  const { rowData } = useMemo(
    () =>
      lotClosuresToLifoRows(
        tradeDetails,
        tradeClosures,
        enriched.avg_entry,
        enriched.multiplier,
      ),
    [tradeDetails, tradeClosures, enriched.avg_entry, enriched.multiplier],
  );

  const visibleTransactions = transactionsExpanded
    ? rowData
    : rowData.slice(0, TRANSACTIONS_DEFAULT_LIMIT);
  const hiddenCount = Math.max(0, rowData.length - TRANSACTIONS_DEFAULT_LIMIT);

  const returnClass = enriched.return_pct >= 0 ? "text-m-accent" : "text-m-down";

  // Inline sheet chrome (pattern mirrors MobileSelectSheet, 242621b
  // overflow fix). Detail sheet uses 95vh instead of 85vh because the
  // content (Flight Deck + Charts + Transactions) is denser than the
  // option-list sheets MobileSelectSheet was built for.
  return (
    <>
      <button
        type="button"
        aria-label={`Close ${enriched.ticker} details`}
        onClick={onClose}
        className="fixed inset-0 z-40 bg-black/50"
        style={{ animation: "m-backdrop-enter var(--m-duration-tap) ease-out" }}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`${enriched.ticker} details`}
        className="fixed inset-x-0 bottom-0 z-50 flex max-h-[95vh] flex-col border-t-[0.5px] border-m-border bg-m-bg"
        style={{
          borderTopLeftRadius: "var(--m-radius-xl)",
          borderTopRightRadius: "var(--m-radius-xl)",
          animation: "m-sheet-enter var(--m-duration-sheet) var(--m-ease-spring)",
        }}
      >
        {/* Drag handle + header (fixed) */}
        <div className="shrink-0">
          <div className="flex justify-center pt-2 pb-1">
            <div className="h-1 w-9 rounded-full bg-m-border" aria-hidden="true" />
          </div>
          <div className="flex items-start justify-between gap-3 border-b-[0.5px] border-m-border px-5 pb-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2">
                <span className="font-m-num text-[22px] font-medium tabular-nums text-m-text">
                  {enriched.ticker}
                </span>
                <StatePill state={state} />
                <SellRuleChip tier={enriched.sell_rule_tier} />
              </div>
              <div className="mt-0.5 flex items-baseline gap-2">
                <span className={`font-m-num text-[22px] font-medium tabular-nums ${returnClass}`}>
                  {enriched.return_pct >= 0 ? "+" : ""}
                  {enriched.return_pct.toFixed(2)}%
                </span>
                <span
                  className={`font-m-num text-[13px] font-medium tabular-nums ${returnClass} privacy-mask`}
                >
                  {formatCurrency(enriched.overall_pl, {
                    showSign: true,
                    signGlyph: "unicode",
                    decimals: 0,
                  })}
                </span>
              </div>
              <div className="mt-0.5 font-m-num text-[11px] tabular-nums text-m-text-dim">
                {enriched.shares.toLocaleString("en-US")}
                {enriched.is_option ? " ct" : " sh"} · current{" "}
                {formatCurrency(enriched.current_price)} · D{enriched.days_held}
              </div>
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-m-surface text-m-text-dim"
            >
              <X size={18} strokeWidth={1.5} aria-hidden="true" />
            </button>
          </div>
        </div>

        {/* Body (scrollable) */}
        <div
          className="min-h-0 flex-1 overflow-y-auto"
          style={{
            paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))",
            WebkitOverflowScrolling: "touch",
          }}
        >
          <FlightDeck enriched={enriched} />
          <ChartsSection
            images={images}
            loading={imagesLoading}
            onOpenLightbox={onOpenLightbox}
          />
          <TransactionsSection
            rows={visibleTransactions}
            totalCount={rowData.length}
            hiddenCount={hiddenCount}
            expanded={transactionsExpanded}
            onToggleExpanded={onToggleTransactions}
            multiplier={enriched.multiplier}
          />
        </div>
      </div>
    </>
  );
}

function FlightDeck({ enriched }: { enriched: EnrichedPosition }) {
  const tradeRisk = Math.abs(enriched.signed_risk);
  const unrealClass = enriched.unrealized_pl >= 0 ? "text-m-accent" : "text-m-down";
  const realClass =
    enriched.realized_bank > 0
      ? "text-m-accent"
      : enriched.realized_bank < 0
        ? "text-m-down"
        : "text-m-text";
  return (
    <section className="px-5 py-4">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
        Flight Deck
      </h3>
      <div className="grid grid-cols-2 gap-2">
        <FlightTile label="Current price" value={formatCurrency(enriched.current_price)} />
        <FlightTile label="Avg cost" value={formatCurrency(enriched.avg_entry)} />
        <FlightTile
          label="Total cost"
          value={formatCurrency(enriched.total_cost, { decimals: 0 })}
          privacyMask
        />
        <FlightTile
          label="Trade risk"
          value={formatCurrency(tradeRisk, { decimals: 0 })}
          privacyMask
        />
        <FlightTile
          label="Unrealized P&L"
          value={formatCurrency(enriched.unrealized_pl, {
            showSign: true,
            signGlyph: "unicode",
            decimals: 0,
          })}
          valueClass={unrealClass}
          privacyMask
        />
        <FlightTile
          label="Realized P&L"
          value={formatCurrency(enriched.realized_bank, {
            showSign: true,
            signGlyph: "unicode",
            decimals: 0,
          })}
          valueClass={realClass}
          privacyMask
        />
        <FlightTile
          label="Current value"
          value={formatCurrency(enriched.current_value, { decimals: 0 })}
          privacyMask
        />
        <FlightTile label="Position size" value={`${enriched.pos_size_pct.toFixed(1)}%`} />
      </div>
    </section>
  );
}

function FlightTile({
  label,
  value,
  valueClass,
  privacyMask,
}: {
  label: string;
  value: string;
  valueClass?: string;
  privacyMask?: boolean;
}) {
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-m-text-dim">
        {label}
      </div>
      <div
        className={`mt-0.5 font-m-num text-[15px] font-medium tabular-nums ${valueClass ?? "text-m-text"} ${privacyMask ? "privacy-mask" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function ChartsSection({
  images,
  loading,
  onOpenLightbox,
}: {
  images: TradeImage[] | undefined;
  loading: boolean;
  onOpenLightbox: (idx: number) => void;
}) {
  const validImages = (images ?? []).filter((img) => img.view_url || img.image_url);
  return (
    <section className="border-t-[0.5px] border-m-border px-5 py-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Charts
        </h3>
        {!loading && (
          <span className="font-m-num text-[10px] tabular-nums text-m-text-dim">
            {validImages.length} uploaded
          </span>
        )}
      </div>
      {loading ? (
        <div className="h-[110px] animate-pulse rounded-m-md bg-m-surface-2" />
      ) : validImages.length === 0 ? (
        <div className="flex h-[110px] items-center justify-center rounded-m-md border-[0.5px] border-dashed border-m-border bg-m-surface px-4 text-center text-[11px] text-m-text-dim">
          Charts uploaded on desktop appear here.
        </div>
      ) : (
        <div
          className="-mx-5 flex gap-2 overflow-x-auto whitespace-nowrap px-5"
          style={{ WebkitOverflowScrolling: "touch" }}
        >
          {validImages.map((img, i) => (
            <button
              key={img.id ?? i}
              type="button"
              onClick={() => onOpenLightbox(i)}
              className="inline-block shrink-0 overflow-hidden rounded-m-md border-[0.5px] border-m-border bg-m-surface text-left"
              style={{ width: 180, height: 110 }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={img.view_url ?? img.image_url ?? ""}
                alt={img.file_name ?? "Trade chart"}
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function TransactionsSection({
  rows,
  totalCount,
  hiddenCount,
  expanded,
  onToggleExpanded,
  multiplier,
}: {
  rows: LifoRow[];
  totalCount: number;
  hiddenCount: number;
  expanded: boolean;
  onToggleExpanded: () => void;
  multiplier: number;
}) {
  return (
    <section className="border-t-[0.5px] border-m-border px-5 py-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Transactions (LIFO)
        </h3>
        <span className="font-m-num text-[10px] tabular-nums text-m-text-dim">
          {totalCount} total
        </span>
      </div>
      {rows.length === 0 ? (
        <div className="rounded-m-md border-[0.5px] border-dashed border-m-border bg-m-surface px-4 py-6 text-center text-[11px] text-m-text-dim">
          No transactions on file.
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          {rows.map((row, i) => (
            <TransactionRow key={`${row.tx.trade_id}-${i}`} row={row} multiplier={multiplier} />
          ))}
        </div>
      )}
      {!expanded && hiddenCount > 0 && (
        <button
          type="button"
          onClick={onToggleExpanded}
          className="mt-2 w-full rounded-m-md border-[0.5px] border-m-border bg-m-surface py-2 text-[12px] font-medium text-m-text-muted"
        >
          Show {hiddenCount} more
        </button>
      )}
    </section>
  );
}

function TransactionRow({ row, multiplier }: { row: LifoRow; multiplier: number }) {
  const action = String(row.tx.action || "").toUpperCase();
  const isBuy = !row.isSell;
  const actionClass = isBuy ? "text-m-accent" : "text-m-down";
  const dateStr = String(row.tx.date || "").slice(0, 10);
  const trxId = String((row.tx as unknown as { trx_id?: string }).trx_id || "");
  const muted = row.isSell || (row.remaining === 0 && !row.isSell);
  const containerClass = muted
    ? "rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2 opacity-70"
    : "rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2";
  const plClass =
    row.realizedPl > 0
      ? "text-m-accent"
      : row.realizedPl < 0
        ? "text-m-down"
        : "text-m-text-dim";

  return (
    <div className={containerClass}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="flex items-baseline gap-1.5 font-m-num text-[11px] tabular-nums">
          {trxId && <span className="text-m-text-dim">{trxId}</span>}
          <span className={`font-semibold ${actionClass}`}>{action}</span>
          <span className="text-m-text-dim">{dateStr}</span>
        </span>
        {!row.isSell && row.realizedPl !== 0 && (
          <span className={`font-m-num text-[11px] font-medium tabular-nums ${plClass}`}>
            {formatCurrency(row.realizedPl, {
              showSign: true,
              signGlyph: "unicode",
              decimals: 0,
            })}{" "}
            ({row.returnPct >= 0 ? "+" : ""}
            {row.returnPct.toFixed(1)}%)
          </span>
        )}
      </div>
      <div className="mt-0.5 font-m-num text-[11px] tabular-nums text-m-text-muted">
        {Math.abs(row.displayShares)}
        {multiplier > 1 ? " ct" : " sh"} @ {formatCurrency(parseFloat(String(row.tx.amount || 0)))}
        {row.tx.rule && <> · {row.tx.rule}</>}
      </div>
    </div>
  );
}

