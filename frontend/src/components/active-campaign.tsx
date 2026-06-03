"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type TradeDetailsBundle, type Strategy } from "@/lib/api";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { readCache, writeCache } from "@/lib/session-cache";
import { parseOptionTicker, daysUntilExpiration } from "@/lib/options";
import { computeEnrichedPositions, type EnrichedPosition } from "@/lib/positions";
import { formatCurrency } from "@/lib/format";
import { SELL_RULE_TIER_ORDER } from "@/lib/sell-rule";
import { CaptureSnapshotButton } from "./capture-snapshot";
import { StrategyChip } from "./strategy-chip";
import { StrategyFlyout, StrategyFlatList, useCoarsePointer } from "./strategy-flyout";
import { SellRuleBadge } from "./sell-rule-badge";
import { SR8TrimCalculator } from "./sr8-trim-calculator";

// Bump whenever the cached payload shape (or its derived EnrichedPosition)
// changes. v3: signed_risk + multiplier-aware option Risk $ — old caches
// would feed the legacy understated values to the new UI.
const ACS_CACHE_VERSION = 3;
const acsCacheName = (portfolioId: number) => `active-campaign::${portfolioId}`;
const STALE_THROTTLE_MS = 20_000;
const MULTIPLIER_NOTICE_KEY = "acs-v2-multiplier-notice-dismissed";

interface ACSCache {
  openTrades: TradePosition[];
  details: TradeDetail[];
  equity: number;
  livePrices: Record<string, number>;
}

function KPITile({ label, value, sub, gradient }: { label: string; value: string; sub: string; gradient: string }) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between h-[90px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full" style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      </div>
      <div className="relative z-10 text-[10px] font-medium opacity-80 privacy-mask">{sub}</div>
    </div>
  );
}

type SortDir = "asc" | "desc";

function compareRows(a: EnrichedPosition, b: EnrichedPosition, key: string, dir: SortDir): number {
  let av: any;
  let bv: any;
  if (key === "expiration") {
    av = a.expiration ? a.expiration.getTime() : Number.MAX_SAFE_INTEGER;
    bv = b.expiration ? b.expiration.getTime() : Number.MAX_SAFE_INTEGER;
  } else if (key === "dte") {
    av = a.expiration ? daysUntilExpiration(a.expiration) : Number.MAX_SAFE_INTEGER;
    bv = b.expiration ? daysUntilExpiration(b.expiration) : Number.MAX_SAFE_INTEGER;
  } else if (key === "cost_pct") {
    // Derived field — sorted via total_cost since cost_pct = total_cost/nlv*100
    // is monotonic in total_cost when nlv is the same constant for all rows.
    av = a.total_cost;
    bv = b.total_cost;
  } else if (key === "signed_risk") {
    // Risk $ column displays projected_pl (total exposure including realized
    // losses on closed lots). Sort by the displayed value, not the legacy
    // open-only signed_risk that the column key still references.
    av = a.projected_pl;
    bv = b.projected_pl;
  } else if (key === "risk_pct") {
    // Risk % column displays projected_pct. Mirror the cell's value so sort
    // order matches what the user sees.
    av = a.projected_pct;
    bv = b.projected_pct;
  } else if (key === "sell_rule_tier") {
    // Sort by tier order (sr1 < sr11 < sr8). null tiers sort last in both
    // directions — null handling below treats null as "missing", so flip
    // the comparison sign for desc but keep null pinned to the bottom.
    const aRank = a.sell_rule_tier ? SELL_RULE_TIER_ORDER[a.sell_rule_tier] : null;
    const bRank = b.sell_rule_tier ? SELL_RULE_TIER_ORDER[b.sell_rule_tier] : null;
    if (aRank === null && bRank === null) return 0;
    if (aRank === null) return 1;
    if (bRank === null) return -1;
    const cmp = aRank - bRank;
    return dir === "desc" ? -cmp : cmp;
  } else {
    av = (a as any)[key];
    bv = (b as any)[key];
  }
  let cmp: number;
  if (av == null && bv == null) cmp = 0;
  else if (av == null) cmp = -1;
  else if (bv == null) cmp = 1;
  else if (typeof av === "string" && typeof bv === "string") cmp = av.localeCompare(bv);
  else cmp = (av as number) - (bv as number);
  return dir === "desc" ? -cmp : cmp;
}

// Shared column width used by both the Equities and Options <colgroup>
// blocks. Both tables have 14 columns; giving every column an identical
// percentage width forces position-N in both tables to land at the same
// x-offset, regardless of cell content. tableLayout:"fixed" on each table
// is what makes the browser honor these colgroup widths.
const COL_WIDTH = "calc(100% / 14)";

const EQUITY_COLS: { key: string; label: string; align: "left" | "center" | "right" }[] = [
  { key: "ticker", label: "Ticker", align: "left" },
  { key: "days_held", label: "Days", align: "right" },
  { key: "sell_rule_tier", label: "Sell Rule", align: "center" },
  { key: "pyramid_pct", label: "Pyramid", align: "center" },
  { key: "return_pct", label: "Return %", align: "right" },
  { key: "pos_size_pct", label: "Pos Size %", align: "right" },
  { key: "shares", label: "Shares", align: "right" },
  { key: "avg_entry", label: "Avg Entry", align: "right" },
  { key: "avg_stop", label: "Avg Stop", align: "right" },
  { key: "current_value", label: "Current Value", align: "right" },
  { key: "signed_risk", label: "Current Risk $", align: "right" },
  { key: "risk_pct", label: "Current Risk %", align: "right" },
  { key: "overall_pl", label: "Overall P&L", align: "right" },
  { key: "risk_budget", label: "Trade Risk $", align: "right" },
];

// Column ordering for the Options table — 14 columns to mirror EQUITY_COLS
// position-for-position, so both tables auto-size identically under w-full
// without colgroup pixel hacks. Conceptual mapping:
//   1 Ticker       ↔ Contract
//   2 Days         ↔ Days
//   3 Sell Rule    ↔ Exp Date
//   4 Pyramid      ↔ DTE
//   5 Return %     ↔ Return %
//   6 Pos Size %   ↔ Pos Size %
//   7 Shares       ↔ Qty
//   8 Avg Entry    ↔ Entry
//   9 Avg Stop     ↔ Current Price (inline-editable)
//  10 Current Value↔ Value
//  11 Current Risk $ ↔ Current Risk $
//  12 Current Risk % ↔ Current Risk %
//  13 Overall P&L    ↔ Overall P&L
//  14 Trade Risk $   ↔ Trade Risk $  (both tables — historical risk_budget)
// Keep the cell counts in lock-step with the body render below — header
// position must equal cell position or alignment breaks.
const OPTION_COLS: { key: string; label: string; align: "left" | "center" | "right" }[] = [
  { key: "ticker",        label: "Contract",      align: "left" },
  { key: "days_held",     label: "Days",          align: "right" },
  { key: "expiration",    label: "Exp Date",      align: "right" },
  { key: "dte",           label: "DTE",           align: "center" },
  { key: "return_pct",    label: "Return %",      align: "right" },
  { key: "pos_size_pct",  label: "Pos Size %",    align: "right" },
  { key: "shares",        label: "Qty",           align: "right" },
  { key: "avg_entry",     label: "Entry",         align: "right" },
  { key: "current_price", label: "Current Price", align: "right" },
  { key: "current_value", label: "Value",         align: "right" },
  { key: "signed_risk",   label: "Current Risk $", align: "right" },
  { key: "risk_pct",      label: "Current Risk %", align: "right" },
  { key: "overall_pl",    label: "Overall P&L",   align: "right" },
  { key: "risk_budget",   label: "Trade Risk $",  align: "right" },
];

export function ActiveCampaign({ navColor, onNavigate }: { navColor: string; onNavigate?: (page: string) => void }) {
  const router = useRouter();
  const { activePortfolio } = usePortfolio();
  const [positions, setPositions] = useState<EnrichedPosition[]>([]);
  const [equity, setEquity] = useState(0);
  const [loading, setLoading] = useState(true);
  const [lastUpdateMs, setLastUpdateMs] = useState<number | null>(null);
  const [, setFreshnessTick] = useState(0);
  const [refetching, setRefetching] = useState(false);
  const [refetchError, setRefetchError] = useState(false);
  const lastFetchAtRef = useRef<number>(0);
  const inFlightRef = useRef(false);
  const lastActiveIdRef = useRef<number | undefined>(undefined);
  // Migration 036 — per-mount dedupe for the b1_max_return_pct
  // auto-promote POST. Each trade_id fires at most one promotion per
  // component lifetime; the backend SQL guard handles cross-tab races.
  const promotedRef = useRef<Set<string>>(new Set());
  const [riskMonitorOpen, setRiskMonitorOpen] = useState(false);

  // Independent sort state for each section. No localStorage persistence —
  // resets to "Return % desc" on every mount per spec.
  const [eqSortKey, setEqSortKey] = useState<string>("return_pct");
  const [eqSortDir, setEqSortDir] = useState<SortDir>("desc");
  const [optSortKey, setOptSortKey] = useState<string>("return_pct");
  const [optSortDir, setOptSortDir] = useState<SortDir>("desc");

  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; position: EnrichedPosition } | null>(null);
  // SR8 Trim modal — null = closed. Pre-selects this trade_id in the
  // calculator. Modal closes on backdrop click, X button, or Esc.
  const [sr8TrimModalTradeId, setSr8TrimModalTradeId] = useState<string | null>(null);
  // Phase 2 — list of active strategies for the right-click "Set strategy"
  // submenu. Loaded once on mount; refresh after a successful retag is
  // unnecessary because the source of truth (trades_summary.strategy) is
  // re-read by loadData().
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const coarsePointer = useCoarsePointer();

  // Inline option price editor (preserved from v1).
  const [editingPriceTradeId, setEditingPriceTradeId] = useState<string | null>(null);
  const [editPriceValue, setEditPriceValue] = useState<string>("");
  const [savingPrice, setSavingPrice] = useState(false);

  // EOD batch-edit modal.
  const [eodModalOpen, setEodModalOpen] = useState(false);
  const [eodEdits, setEodEdits] = useState<Record<string, string>>({});
  const [eodSaving, setEodSaving] = useState(false);
  const [eodErrors, setEodErrors] = useState<string[]>([]);

  // Exercise-option modal — single-record. Set to a position to open;
  // null to close. Mirrors the trade-journal Edit modal recipe (state
  // tuple + ESC gate + error-stays-open).
  const [exerciseTarget, setExerciseTarget] = useState<EnrichedPosition | null>(null);
  const [exerciseDate, setExerciseDate] = useState<string>("");
  const [exerciseNotes, setExerciseNotes] = useState<string>("");
  const [exerciseSaving, setExerciseSaving] = useState(false);
  const [exerciseError, setExerciseError] = useState<string | null>(null);

  // Multiplier-fix banner. Default to dismissed during SSR; the localStorage
  // read in useEffect below is the authoritative resolution.
  const [bannerDismissed, setBannerDismissed] = useState(true);
  useEffect(() => {
    if (typeof window === "undefined") return;
    setBannerDismissed(window.localStorage.getItem(MULTIPLIER_NOTICE_KEY) === "1");
  }, []);
  const dismissBanner = useCallback(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(MULTIPLIER_NOTICE_KEY, "1");
    }
    setBannerDismissed(true);
  }, []);

  const hydrateFromPayload = useCallback((payload: ACSCache) => {
    setEquity(payload.equity);
    const enriched = computeEnrichedPositions(
      payload.openTrades,
      payload.details,
      payload.equity,
      payload.livePrices,
    );
    setPositions(enriched);
  }, []);

  // Auto-promote b1_max_return_pct whenever a position's CURRENT B1 return
  // exceeds its STORED max. Fire-and-forget: the backend SQL guard makes
  // the call idempotent (lower / equal values are no-ops), and the
  // promotedRef Set dedupes within a session so the request fires once
  // per trade_id per mount. Silent on failure — the next mount will retry.
  useEffect(() => {
    const portfolioName = activePortfolio?.name;
    if (!portfolioName) return;
    positions.forEach((p) => {
      if (promotedRef.current.has(p.trade_id)) return;
      const current = p.b1_return_pct;
      if (current === null || !Number.isFinite(current)) return;
      const stored = p.b1_max_return_pct;
      const needsPromote = stored === null || current > stored;
      if (!needsPromote) return;
      promotedRef.current.add(p.trade_id);
      fetch(`/api/trades/${encodeURIComponent(p.trade_id)}/update-b1-max`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ portfolio: portfolioName, new_max_pct: current }),
      }).catch(() => { /* silent — backend guard + dedupe ref make retries safe */ });
    });
  }, [positions, activePortfolio?.name]);

  const loadData = useCallback(async (opts?: { force?: boolean }) => {
    const force = !!opts?.force;
    const activeId = activePortfolio?.id;

    if (lastActiveIdRef.current !== activeId) {
      lastActiveIdRef.current = activeId;
      lastFetchAtRef.current = 0;
      if (activeId != null) {
        const cached = readCache<ACSCache>(acsCacheName(activeId), ACS_CACHE_VERSION);
        if (cached) {
          hydrateFromPayload(cached.payload);
          setLastUpdateMs(cached.saved_at);
          setLoading(false);
          lastFetchAtRef.current = cached.saved_at;
        }
      }
    }

    if (!force && Date.now() - lastFetchAtRef.current < STALE_THROTTLE_MS) {
      setLoading(false);
      return;
    }

    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setRefetching(true);

    try {
      const [openTrades, details, nlv, journal] = await Promise.all([
        api.tradesOpen(getActivePortfolio()).catch((err) => {
          log.error("active-campaign", "tradesOpen fetch failed", err);
          return [];
        }),
        api.tradesOpenDetails(getActivePortfolio()).catch((err) => {
          log.error("active-campaign", "tradesOpenDetails fetch failed", err);
          return { details: [], lot_closures: [] };
        }),
        activeId != null ? api.portfolioNlv(activeId).catch((err) => {
          log.error("active-campaign", "portfolioNlv fetch failed", err);
          return null;
        }) : Promise.resolve(null),
        api.journalLatest(getActivePortfolio()).catch((err) => {
          log.error("active-campaign", "journalLatest fetch failed", err);
          return null;
        }),
      ]);
      const journalNlv = journal ? parseFloat(String((journal as any).end_nlv || 0)) : 0;
      const derivedNlv = nlv && typeof nlv === "object" && !("error" in nlv)
        ? (nlv as { nlv: number }).nlv
        : null;
      const eq = journalNlv > 0 ? journalNlv : (derivedNlv ?? 0);

      const tickers = (openTrades as TradePosition[]).map(t => t.ticker).filter(Boolean);
      let prices: Record<string, number> = {};
      if (tickers.length > 0) {
        try {
          const result = await api.batchPrices(tickers, getActivePortfolio());
          if (result && !("error" in result)) {
            prices = result;
          }
        } catch (e) {
          log.error("active-campaign", "price fetch failed", e);
          /* keep entry prices as fallback */
        }
      }

      const payload: ACSCache = {
        openTrades: openTrades as TradePosition[],
        details: (details as TradeDetailsBundle).details,
        equity: eq,
        livePrices: prices,
      };
      hydrateFromPayload(payload);
      if (activeId != null) {
        writeCache(acsCacheName(activeId), ACS_CACHE_VERSION, payload);
      }
      const now = Date.now();
      setLastUpdateMs(now);
      lastFetchAtRef.current = now;
      setRefetchError(false);
    } catch (e) {
      log.error("active-campaign", "refresh failed", e);
      setRefetchError(true);
    } finally {
      setLoading(false);
      setRefetching(false);
      inFlightRef.current = false;
    }
  }, [activePortfolio?.id, hydrateFromPayload]);

  useEffect(() => { loadData(); }, [loadData]);

  // Load strategies once. Phase 2 admin UI mutations don't auto-propagate
  // here — user can refresh if they just added a strategy. Acceptable for
  // a low-mutation lookup.
  useEffect(() => {
    api.listStrategies({ active: true, portfolio: getActivePortfolio() }).then(setStrategies).catch(() => setStrategies([]));
  }, []);

  useEffect(() => {
    const id = setInterval(() => setFreshnessTick(t => t + 1), 30_000);
    return () => clearInterval(id);
  }, []);

  // Inline option price editor handlers.
  const startEditPrice = useCallback((p: EnrichedPosition) => {
    setEditingPriceTradeId(p.trade_id);
    setEditPriceValue(
      p.manual_price !== null
        ? String(p.manual_price)
        : (p.current_price > 0 ? p.current_price.toFixed(2) : "")
    );
  }, []);

  const cancelEditPrice = useCallback(() => {
    setEditingPriceTradeId(null);
    setEditPriceValue("");
  }, []);

  const commitEditPrice = useCallback(async (p: EnrichedPosition) => {
    if (savingPrice) return;
    const trimmed = editPriceValue.trim();
    let newPrice: number | null;
    if (trimmed === "") {
      newPrice = null;
    } else {
      const n = parseFloat(trimmed);
      if (!isFinite(n) || n <= 0) {
        cancelEditPrice();
        return;
      }
      newPrice = n;
    }
    if (newPrice === p.manual_price) {
      cancelEditPrice();
      return;
    }
    setSavingPrice(true);
    try {
      await api.setManualPrice({
        portfolio: getActivePortfolio(),
        trade_id: p.trade_id,
        manual_price: newPrice,
      });
      setEditingPriceTradeId(null);
      setEditPriceValue("");
      await loadData({ force: true });
    } finally {
      setSavingPrice(false);
    }
  }, [editPriceValue, savingPrice, cancelEditPrice, loadData]);

  // Context menu close.
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [ctxMenu]);

  // Esc closes the SR8 Trim modal. Backdrop click also closes (handled
  // inline in the modal JSX).
  useEffect(() => {
    if (!sr8TrimModalTradeId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSr8TrimModalTradeId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sr8TrimModalTradeId]);

  const ctxViewJournal = useCallback((p: EnrichedPosition) => {
    localStorage.setItem("journal_prefill", JSON.stringify({ ticker: p.ticker, trade_id: p.trade_id }));
    if (onNavigate) onNavigate("journal");
  }, [onNavigate]);

  const ctxOpenPyramid = useCallback((trade_id: string) => {
    router.push(`/position-sizer?tab=pyramid&trade_id=${encodeURIComponent(trade_id)}`);
  }, [router]);

  // Increase position → opens Log Buy in scale-in mode with the campaign
  // pre-selected via the existing ps_prefill handshake (log-buy.tsx reads
  // the {action: "scale_in", trade_id} payload on mount and clears it).
  const ctxIncreasePosition = useCallback((p: EnrichedPosition) => {
    localStorage.setItem("ps_prefill", JSON.stringify({ action: "scale_in", trade_id: p.trade_id }));
    if (onNavigate) onNavigate("logbuy");
  }, [onNavigate]);

  // Decrease position → opens Log Sell with the campaign pre-selected via
  // ps_prefill_sell (log-sell.tsx reads {trade_id} on mount and clears it).
  const ctxDecreasePosition = useCallback((p: EnrichedPosition) => {
    localStorage.setItem("ps_prefill_sell", JSON.stringify({ trade_id: p.trade_id }));
    if (onNavigate) onNavigate("logsell");
  }, [onNavigate]);

  // Phase 2 — retag the campaign via PATCH /api/trades/{id}/strategy.
  // Optimistic UX: close the menu, fire the request, refresh on success.
  // The TradePosition.strategy on positions[] is computed from
  // trades_summary, so the refresh covers the round-trip.
  const ctxSetStrategy = useCallback(async (trade_id: string, strategy: string) => {
    setCtxMenu(null);
    try {
      const r = await api.setTradeStrategy(trade_id, { strategy });
      if (!("error" in r) || !r.error) {
        await loadData({ force: true });
      }
    } catch {
      /* swallow — user can retry from the menu */
    }
  }, [loadData]);

  // Exercise-option modal handlers.
  const ctxOpenExercise = useCallback((p: EnrichedPosition) => {
    setExerciseTarget(p);
    setExerciseDate(new Date().toISOString().slice(0, 10));
    setExerciseNotes("");
    setExerciseError(null);
  }, []);

  const closeExerciseModal = useCallback(() => {
    setExerciseTarget(null);
    setExerciseDate("");
    setExerciseNotes("");
    setExerciseError(null);
  }, []);

  const submitExercise = useCallback(async () => {
    if (!exerciseTarget || exerciseSaving) return;
    setExerciseSaving(true);
    setExerciseError(null);
    try {
      const res = await api.exerciseOption({
        trade_id: exerciseTarget.trade_id,
        date: exerciseDate,
        notes: exerciseNotes,
      });
      if (res.error) {
        setExerciseError(res.error);
        return;
      }
      // Force-refresh: option flips CLOSED (drops from open list); stock
      // either appears or scales in. The cached snapshot is now stale.
      await loadData({ force: true });
      closeExerciseModal();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Network error";
      setExerciseError(msg);
    } finally {
      setExerciseSaving(false);
    }
  }, [exerciseTarget, exerciseSaving, exerciseDate, exerciseNotes,
      loadData, closeExerciseModal]);

  // ESC dismiss for the exercise modal — gated on !saving so an in-flight
  // submit can't be cancelled mid-write.
  useEffect(() => {
    if (!exerciseTarget) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !exerciseSaving) closeExerciseModal();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exerciseTarget, exerciseSaving, closeExerciseModal]);

  // EOD modal.
  const openEodModal = useCallback(() => {
    setEodEdits({});
    setEodErrors([]);
    setEodModalOpen(true);
  }, []);

  const closeEodModal = useCallback(() => {
    setEodModalOpen(false);
    setEodEdits({});
    setEodErrors([]);
  }, []);

  useEffect(() => {
    if (!eodModalOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape" && !eodSaving) closeEodModal(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [eodModalOpen, eodSaving, closeEodModal]);

  const handleEqSort = useCallback((key: string) => {
    setEqSortDir(prev => (eqSortKey === key ? (prev === "desc" ? "asc" : "desc") : "desc"));
    setEqSortKey(key);
  }, [eqSortKey]);
  const handleOptSort = useCallback((key: string) => {
    setOptSortDir(prev => (optSortKey === key ? (prev === "desc" ? "asc" : "desc") : "desc"));
    setOptSortKey(key);
  }, [optSortKey]);

  const equities = useMemo(() => positions.filter(p => !p.is_option), [positions]);
  const options = useMemo(() => positions.filter(p => p.is_option), [positions]);
  // O(1) name → color lookup for the per-row strategy dot. Mirrors the
  // pattern used by Trade Journal's pill so colors stay consistent across
  // surfaces.
  const strategyByName = useMemo(() => {
    const m = new Map<string, Strategy>();
    for (const s of strategies) m.set(s.name, s);
    return m;
  }, [strategies]);

  const sortedEquities = useMemo(
    () => [...equities].sort((a, b) => compareRows(a, b, eqSortKey, eqSortDir)),
    [equities, eqSortKey, eqSortDir],
  );
  const sortedOptions = useMemo(
    () => [...options].sort((a, b) => compareRows(a, b, optSortKey, optSortDir)),
    [options, optSortKey, optSortDir],
  );

  const saveEodPrices = useCallback(async () => {
    if (eodSaving) return;
    setEodSaving(true);
    setEodErrors([]);

    // Collect submissions paired with their identity so we can map results
    // back to specific contracts after Promise.allSettled. setManualPrice
    // returns either { status: "ok", … } or { error: string } — backend
    // failures don't reject the promise, so we also inspect the resolved
    // value for an `error` key.
    const submissions: { trade_id: string; ticker: string; promise: Promise<unknown> }[] = [];
    for (const opt of options) {
      const raw = eodEdits[opt.trade_id];
      if (raw === undefined || raw.trim() === "") continue;
      const n = parseFloat(raw.trim());
      if (!isFinite(n) || n <= 0) continue;
      if (n === opt.manual_price) continue;
      submissions.push({
        trade_id: opt.trade_id,
        ticker: opt.ticker,
        promise: api.setManualPrice({
          portfolio: getActivePortfolio(),
          trade_id: opt.trade_id,
          manual_price: n,
        }),
      });
    }

    try {
      if (submissions.length === 0) {
        closeEodModal();
        return;
      }

      const results = await Promise.allSettled(submissions.map(s => s.promise));
      const failedTickers: string[] = [];
      const succeededIds: string[] = [];
      results.forEach((r, idx) => {
        const sub = submissions[idx];
        if (r.status === "rejected") {
          failedTickers.push(sub.ticker);
        } else if (r.value && typeof r.value === "object" && "error" in (r.value as object)) {
          failedTickers.push(sub.ticker);
        } else {
          succeededIds.push(sub.trade_id);
        }
      });

      // Always refresh — successful saves are committed server-side and
      // should be reflected even when the modal stays open for retries.
      await loadData({ force: true });

      if (failedTickers.length > 0) {
        // Clear inputs for the successful saves so the user only sees
        // pending edits for what still needs to be retried.
        setEodEdits(prev => {
          const next = { ...prev };
          for (const id of succeededIds) delete next[id];
          return next;
        });
        setEodErrors(failedTickers);
      } else {
        closeEodModal();
      }
    } finally {
      setEodSaving(false);
    }
  }, [eodEdits, options, eodSaving, closeEodModal, loadData]);

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-[90px] rounded-[14px] mb-6" style={{ background: "var(--bg-2)" }} />
      </div>
    );
  }

  // ── Header KPIs ──
  const equityExposureDollar = equities.reduce((a, p) => a + p.current_value, 0);
  const optionsExposureDollar = options.reduce((a, p) => a + p.current_value, 0);
  const equityExposurePct = equity > 0 ? (equityExposureDollar / equity) * 100 : 0;
  const optionsExposurePct = equity > 0 ? (optionsExposureDollar / equity) * 100 : 0;
  const equityPlSum = equities.reduce((a, p) => a + p.overall_pl, 0);
  const optionsPlSum = options.reduce((a, p) => a + p.overall_pl, 0);
  const equityCostSum = equities.reduce((a, p) => a + p.total_cost, 0);
  const optionsCostSum = options.reduce((a, p) => a + p.total_cost, 0);
  const equityPlPct = equityCostSum > 0 ? (equityPlSum / equityCostSum) * 100 : 0;
  const optionsPlPct = optionsCostSum > 0 ? (optionsPlSum / optionsCostSum) * 100 : 0;

  // Capital at Risk + Open Risk (Heat).
  // Capital at Risk (Group 7-5): currently-at-risk capital across BOTH
  // equities and options, floored at zero per position. Sums max(0, -projected_pl)
  // — projected_pl is the LIFO floor (open-to-stop on stocks, premium-paid
  // on options + realized bank), so its signed negation is the live
  // exposure dollars. Cross-instrument because options now report a real
  // Current Risk $ post Group 7-5.
  // Open Risk (Heat) — equity-only, current heat: distance from live
  // price to safe-stop, times shares, times multiplier.
  const capitalAtRiskTotal = [...equities, ...options].reduce(
    (sum, p) => sum + Math.max(0, -p.projected_pl),
    0,
  );
  const capitalAtRiskPct = equity > 0 ? (capitalAtRiskTotal / equity) * 100 : 0;
  const openRiskTotal = equities.reduce((sum, p) => {
    const safeStop = p.avg_stop > 0 ? p.avg_stop : p.avg_entry;
    return sum + (p.current_price - safeStop) * p.shares * p.multiplier;
  }, 0);
  const openRiskPct = equity > 0 ? (openRiskTotal / equity) * 100 : 0;

  const kpis = [
    {
      label: "NLV",
      value: formatCurrency(equity),
      sub: equity > 0
        ? `Holdings Value · ${formatCurrency(equityExposureDollar + optionsExposureDollar)}`
        : "—",
      gradient: "linear-gradient(135deg, #7c3aed, #a78bfa)",
    },
    {
      label: "EQUITY EXPOSURE",
      value: `${equityExposurePct.toFixed(1)}%`,
      sub: formatCurrency(equityExposureDollar),
      gradient: "linear-gradient(135deg, #1e40af, #3b82f6)",
    },
    {
      label: "OPTIONS EXPOSURE",
      value: `${optionsExposurePct.toFixed(1)}%`,
      sub: `${formatCurrency(optionsExposureDollar)} · cap ~10%`,
      gradient: optionsExposurePct > 10
        ? "linear-gradient(135deg, #e5484d, #f87171)"
        : "linear-gradient(135deg, #f97316, #fb923c)",
    },
    {
      label: "CAPITAL AT RISK",
      value: formatCurrency(capitalAtRiskTotal),
      sub: `${capitalAtRiskPct.toFixed(2)}% of NLV`,
      gradient: "linear-gradient(135deg, #1e40af, #3b82f6)",
    },
    {
      label: "OPEN RISK (HEAT)",
      value: formatCurrency(openRiskTotal, { showSign: true, signGlyph: "unicode" }),
      sub: `${openRiskPct >= 0 ? "+" : ""}${openRiskPct.toFixed(2)}% of NLV`,
      gradient: "linear-gradient(135deg, #e5484d, #f87171)",
    },
    {
      label: "EQUITY P&L",
      value: formatCurrency(equityPlSum, { showSign: true, signGlyph: "unicode" }),
      sub: equityCostSum > 0 ? `${equityPlPct >= 0 ? "+" : ""}${equityPlPct.toFixed(2)}% vs cost` : "—",
      gradient: equityPlSum >= 0
        ? "linear-gradient(135deg, #10b981, #34d399)"
        : "linear-gradient(135deg, #e5484d, #f472b6)",
    },
    {
      label: "OPTIONS P&L",
      value: formatCurrency(optionsPlSum, { showSign: true, signGlyph: "unicode" }),
      sub: optionsCostSum > 0 ? `${optionsPlPct >= 0 ? "+" : ""}${optionsPlPct.toFixed(2)}% vs cost` : "—",
      gradient: optionsPlSum >= 0
        ? "linear-gradient(135deg, #10b981, #34d399)"
        : "linear-gradient(135deg, #e5484d, #f472b6)",
    },
  ];

  // ── Risk Monitor (equity-only per spec H) ──
  const monitorAlerts: { type: "error" | "warn" | "info" | "success"; ticker: string; msg: string }[] = [];
  const freeRollTickers: string[] = [];
  for (const p of equities) {
    if (p.risk_budget > 0 && p.risk_dollars > (p.risk_budget + 5)) {
      const rbmStop = p.total_cost > 0 && p.shares > 0
        ? (p.total_cost - p.risk_budget) / p.shares
        : 0;
      monitorAlerts.push({
        type: "warn",
        ticker: p.ticker,
        msg: `Current Risk (${formatCurrency(p.risk_dollars, { decimals: 0 })}) > Trade Risk $ (${formatCurrency(p.risk_budget, { decimals: 0 })}). Tighten stops to ${formatCurrency(rbmStop)} to bring exposure back to the locked Trade Risk $.`,
      });
    }
    if (p.return_pct <= -7.0) {
      monitorAlerts.push({
        type: "error",
        ticker: p.ticker,
        msg: `Down ${p.return_pct.toFixed(2)}%. Violates Stop Rule.`,
      });
    }
    if (p.return_pct >= 10.0 && p.avg_stop > 0 && p.avg_stop < (p.avg_entry - 0.01)) {
      monitorAlerts.push({
        type: "info",
        ticker: p.ticker,
        msg: `Up ${p.return_pct.toFixed(2)}%. Consider moving stop to BE (${formatCurrency(p.avg_entry)}). Current stop: ${formatCurrency(p.avg_stop)}.`,
      });
    }
    if (p.risk_status === "Free Roll") {
      freeRollTickers.push(p.ticker);
    }
  }

  const mono = "var(--font-jetbrains), monospace";
  const totalPositions = positions.length;
  const showBanner = !bannerDismissed && options.length > 0;

  return (
    <div id="campaign-capture-root" style={{ animation: "slide-up 0.18s ease-out" }}>
      {showBanner && (
        <div className="mb-4 rounded-[10px] px-4 py-3 flex items-start gap-3"
             style={{
               background: "color-mix(in oklab, #3b82f6 8%, var(--surface))",
               border: "1px solid color-mix(in oklab, #3b82f6 30%, var(--border))",
             }}>
          <span className="text-[14px] mt-0.5">ℹ</span>
          <div className="flex-1 text-[12px] leading-relaxed" style={{ color: "var(--ink-2)" }}>
            <strong>Option risk numbers now correctly include the 100× contract multiplier.</strong>{" "}
            Past Risk $ values for option positions were understated.
          </div>
          <button onClick={dismissBanner}
                  className="text-[11px] font-medium px-2 py-1 rounded-md hover:brightness-95"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            Dismiss
          </button>
        </div>
      )}

      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
              Active Campaign <em className="italic" style={{ color: navColor }}>Summary</em>
            </h1>
            <div className="text-[13px] mt-1.5 flex items-center gap-2" style={{ color: "var(--ink-3)" }}>
              <span>{totalPositions} open · {equities.length} equit{equities.length === 1 ? "y" : "ies"} · {options.length} option{options.length === 1 ? "" : "s"}</span>
              {lastUpdateMs !== null && (() => {
                const ageSec = Math.max(0, Math.floor((Date.now() - lastUpdateMs) / 1000));
                let label: string;
                if (ageSec < 30) label = "Updated just now";
                else if (ageSec < 60) label = `Updated ${ageSec}s ago`;
                else if (ageSec < 3600) label = `Updated ${Math.floor(ageSec / 60)} min ago`;
                else if (ageSec < 86400) label = `Updated ${Math.floor(ageSec / 3600)}h ago`;
                else label = `Updated ${Math.floor(ageSec / 86400)}d ago`;
                const stale = ageSec >= 600;
                const absolute = new Date(lastUpdateMs).toLocaleString();
                const dotColor = refetching ? "#3b82f6" : refetchError ? "#e5484d" : null;
                return (
                  <span className="text-[12px] font-medium flex items-center gap-1.5"
                        style={{ color: stale ? "#f59f00" : "var(--ink-3)" }}
                        title={refetchError ? `Couldn't refresh — showing data from ${absolute}` : `Last loaded ${absolute}`}>
                    {dotColor && (
                      <span className={`inline-block w-1.5 h-1.5 rounded-full ${refetching ? "animate-pulse" : ""}`}
                            style={{ background: dotColor }} />
                    )}
                    · {label}
                    {refetching && <span className="text-[var(--ink-4)]">· refreshing</span>}
                    {refetchError && !refetching && <span style={{ color: "#e5484d" }}>· refresh failed</span>}
                  </span>
                );
              })()}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => loadData({ force: true })}
                    disabled={refetching}
                    className="flex items-center gap-1.5 h-[32px] px-3.5 rounded-[10px] text-xs font-medium transition-colors hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                   className={refetching ? "animate-spin" : undefined}>
                <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
              </svg>
              {refetching ? "Refreshing…" : "Refresh"}
            </button>
            <CaptureSnapshotButton targetSelector="#campaign-capture-root" snapshotType="campaign" label="Capture EOD Snapshot" />
          </div>
        </div>
      </div>

      {/* KPI tiles */}
      <div className="grid grid-cols-7 gap-3 mb-6">
        {kpis.map(k => <KPITile key={k.label} {...k} />)}
      </div>

      {/* ── Equities Section ── */}
      {equities.length > 0 && (
        <div className="rounded-[14px] overflow-hidden mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Equities ({equities.length})</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>Click headers to sort · right-click row for actions · right-click Pyramid cell for sizer</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0, tableLayout: "fixed" }}>
              {/* Equal-percentage colgroup — must mirror the Options table's
                  colgroup so position-N aligns at the same x in both. See
                  COL_WIDTH constant above. */}
              <colgroup>
                {Array.from({ length: 14 }).map((_, i) => (
                  <col key={i} style={{ width: COL_WIDTH }} />
                ))}
              </colgroup>
              <thead>
                <tr>
                  {EQUITY_COLS.map(h => {
                    const active = eqSortKey === h.key;
                    return (
                      <th key={h.key}
                          className={`text-${h.align} text-[10px] uppercase tracking-[0.08em] font-semibold px-2.5 py-2.5 whitespace-nowrap sticky top-0`}
                          style={{
                            color: active ? "var(--ink)" : "var(--ink-4)",
                            background: "var(--surface-2)",
                            borderBottom: "1px solid var(--border)",
                            cursor: "pointer",
                            userSelect: "none",
                          }}
                          onClick={() => handleEqSort(h.key)}>
                        {h.label}
                        {active && <span className="ml-1 text-[9px]">{eqSortDir === "desc" ? "▼" : "▲"}</span>}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {sortedEquities.map((p, i) => {
                  const plColor = p.overall_pl >= 0 ? "#08a86b" : "#e5484d";
                  // Current Risk $ / Current Risk % columns show total exposure
                  // (projected_pl) — realized losses on closed lots + open-to-stop
                  // risk — so coloring tracks projected_pl, not signed_risk.
                  const riskColor = p.projected_pl > 0 ? "#08a86b" : p.projected_pl < 0 ? "#e5484d" : "var(--ink-3)";
                  // Group 7-2: divergence flag — current open-only risk exceeds
                  // the historical Trade Risk $ by more than $5. Same trigger as
                  // the Risk Monitor warning at line ~657 so they fire together.
                  // Amber tint matches the At-Risk Risk-Status pill (line 859);
                  // unobtrusive flag-style, not emergency-style.
                  const riskDivergent = p.risk_budget > 0 && p.risk_dollars > (p.risk_budget + 5);
                  const riskCellBg = riskDivergent
                    ? "color-mix(in oklab, #f59f00 12%, var(--surface))"
                    : undefined;

                  let retBg: string;
                  let retText: string;
                  if (p.return_pct >= 5) { retBg = "linear-gradient(135deg, #16a34a, #22c55e)"; retText = "#fff"; }
                  else if (p.return_pct > 0) { retBg = "linear-gradient(135deg, #a3e635, #84cc16)"; retText = "#1a2e05"; }
                  else if (p.return_pct > -3) { retBg = "linear-gradient(135deg, #fbbf24, #f59e0b)"; retText = "#451a03"; }
                  else { retBg = "linear-gradient(135deg, #f87171, #ef4444)"; retText = "#fff"; }

                  const pyramidReady = p.pyramid_pct >= 5;

                  // Tooltip "Current" mirrors the Current Risk $ cell value: the
                  // magnitude of projected_pl (total exposure — realized losses on
                  // closed lots plus open-to-stop on the rest). "De-risked %" —
                  // fraction of Trade Risk $ (locked at buy events) that current
                  // stops have eliminated — is only meaningful for fresh opens;
                  // for partially-closed trades, realized losses aren't de-risked,
                  // they're locked in, so we drop that component rather than show
                  // a misleading number.
                  const isPartiallyClosed = p.realized_bank !== 0;
                  const dRiskedPct = p.risk_budget > 0
                    ? Math.max(0, Math.min(100, (1 - Math.abs(p.signed_risk) / p.risk_budget) * 100))
                    : 0;
                  const riskTooltip = isPartiallyClosed
                    ? `Trade Risk $: ${formatCurrency(p.risk_budget)} · Current: ${formatCurrency(Math.abs(p.projected_pl))}`
                    : `Trade Risk $: ${formatCurrency(p.risk_budget)} · Current: ${formatCurrency(Math.abs(p.projected_pl))} · De-risked: ${dRiskedPct.toFixed(1)}%`;

                  return (
                    <tr key={p.trade_id} className="transition-colors"
                        style={{ borderBottom: i < sortedEquities.length - 1 ? "1px solid var(--border)" : "none" }}
                        onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                        onContextMenu={e => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY, position: p }); }}>
                      <td className="px-2.5 py-2.5 font-semibold whitespace-nowrap" style={{ fontFamily: mono }} title={`Trade ID: ${p.trade_id}`}>
                        <span className="inline-flex items-center" style={{ gap: 6 }}>
                          {p.ticker}
                          {p.strategy && (
                            <StrategyChip name={p.strategy} color={strategyByName.get(p.strategy)?.color ?? "var(--ink-4)"} size="sm" showName={false} />
                          )}
                        </span>
                      </td>
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, fontSize: 11, color: "var(--ink-4)" }}>
                        {p.days_held}
                      </td>
                      <td className="px-2.5 py-2.5 text-center">
                        <SellRuleBadge tier={p.sell_rule_tier} />
                      </td>
                      <td className="px-2.5 py-2.5 text-center"
                          onContextMenu={e => { e.preventDefault(); e.stopPropagation(); setCtxMenu({ x: e.clientX, y: e.clientY, position: p }); }}
                          title="Right-click for actions (View in Journal · Open Position Sizer Pyramid)">
                        {pyramidReady ? (
                          <span className="inline-block px-2 py-0.5 rounded text-[10px] font-bold"
                                style={{ background: "linear-gradient(135deg, #16a34a, #22c55e)", color: "#fff", minWidth: 40, textAlign: "center" }}>
                            Ready
                          </span>
                        ) : (
                          <span style={{ color: "var(--ink-4)", fontSize: 11 }}>—</span>
                        )}
                      </td>
                      <td className="px-2.5 py-2.5 text-right">
                        <span className="inline-block px-2.5 py-0.5 rounded text-[11px] font-bold"
                              style={{ background: retBg, color: retText, minWidth: 62, textAlign: "center", fontFamily: mono }}>
                          {p.return_pct >= 0 ? "+" : ""}{p.return_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                        {p.pos_size_pct.toFixed(1)}%
                      </td>
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono }}>
                        {p.shares.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                        {formatCurrency(p.avg_entry)}
                      </td>
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: p.avg_stop > 0 ? "var(--ink)" : "var(--ink-4)" }}>
                        {p.avg_stop > 0 ? formatCurrency(p.avg_stop) : "—"}
                      </td>
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                        {formatCurrency(p.current_value)}
                      </td>
                      <td className="px-2.5 py-2.5 text-right privacy-mask"
                          style={{ fontFamily: mono, color: riskColor, fontWeight: 600, background: riskCellBg }}
                          title={riskTooltip}>
                        {formatCurrency(p.projected_pl, { showSign: true, signGlyph: "unicode" })}
                      </td>
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: riskColor }}>
                        {p.projected_pct >= 0 ? "+" : ""}{p.projected_pct.toFixed(2)}%
                      </td>
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, fontWeight: 700, color: plColor }}>
                        {formatCurrency(p.overall_pl, { showSign: true, signGlyph: "unicode" })}
                      </td>
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, fontWeight: 600 }}>
                        {formatCurrency(p.risk_budget)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Options Section ── */}
      {options.length > 0 && (
        <div className="rounded-[14px] overflow-hidden mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Options ({options.length})</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>Click cell to set price · click headers to sort</span>
            <button onClick={openEodModal}
                    className="ml-auto text-[11px] font-medium h-[26px] px-2.5 rounded-md hover:brightness-95"
                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
              Update EOD Prices
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0, tableLayout: "fixed" }}>
              {/* Equal-percentage colgroup — mirrors the Equities table so
                  position-N aligns at the same x in both. See COL_WIDTH
                  constant above. */}
              <colgroup>
                {Array.from({ length: 14 }).map((_, i) => (
                  <col key={i} style={{ width: COL_WIDTH }} />
                ))}
              </colgroup>
              <thead>
                <tr>
                  {OPTION_COLS.map(h => {
                    // Group 7-2: Pos 14 is now Trade Risk $ (historical
                    // risk_budget) on both tables, so all 14 columns are
                    // sortable — the prior projected_pl-only no-op cell
                    // is gone.
                    const active = optSortKey === h.key;
                    return (
                      <th key={h.key}
                          className={`text-${h.align} text-[10px] uppercase tracking-[0.08em] font-semibold px-2.5 py-2.5 whitespace-nowrap sticky top-0`}
                          style={{
                            color: active ? "var(--ink)" : "var(--ink-4)",
                            background: "var(--surface-2)",
                            borderBottom: "1px solid var(--border)",
                            cursor: "pointer",
                            userSelect: "none",
                          }}
                          onClick={() => handleOptSort(h.key)}>
                        {h.label}
                        {active && <span className="ml-1 text-[9px]">{optSortDir === "desc" ? "▼" : "▲"}</span>}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {sortedOptions.map((p, i) => {
                  let retBg: string;
                  let retText: string;
                  if (p.return_pct >= 5) { retBg = "linear-gradient(135deg, #16a34a, #22c55e)"; retText = "#fff"; }
                  else if (p.return_pct > 0) { retBg = "linear-gradient(135deg, #a3e635, #84cc16)"; retText = "#1a2e05"; }
                  else if (p.return_pct > -3) { retBg = "linear-gradient(135deg, #fbbf24, #f59e0b)"; retText = "#451a03"; }
                  else { retBg = "linear-gradient(135deg, #f87171, #ef4444)"; retText = "#fff"; }

                  // DTE — recomputed every render so it stays correct without
                  // any manual refresh. Coloring per spec D.
                  const dte = p.expiration ? daysUntilExpiration(p.expiration) : null;
                  const dteExpired = dte !== null && dte < 0;
                  let dteColor: string = "var(--ink)";
                  if (dte !== null && !dteExpired) {
                    if (dte > 45) dteColor = "#16a34a";
                    else if (dte >= 30) dteColor = "#d97706";
                    else dteColor = "#e5484d";
                  }
                  const expDateLabel = p.expiration
                    ? p.expiration.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit", timeZone: "UTC" })
                    : "—";

                  // Current Risk $ / % cells — mirror the equity render at
                  // lines ~810-821. projected_pl is now option-aware (Group
                  // 7-5: lifo treats option stops as 0, so projected_pl =
                  // -cost on a fresh open and moves toward zero/positive as
                  // partial closes bank profits). Divergence flag is
                  // computed inline against |projected_pl| since
                  // risk_dollars is intentionally kept open-only for
                  // equity-row Risk Monitor semantics (audit §D).
                  const optRiskColor = p.projected_pl > 0 ? "#08a86b" : p.projected_pl < 0 ? "#e5484d" : "var(--ink-3)";
                  const optRiskDivergent = p.risk_budget > 0 && Math.max(0, -p.projected_pl) > (p.risk_budget + 5);
                  const optRiskCellBg = optRiskDivergent
                    ? "color-mix(in oklab, #f59f00 12%, var(--surface))"
                    : undefined;
                  const optRiskTooltip = `Trade Risk $: ${formatCurrency(p.risk_budget)} · Current: ${formatCurrency(Math.abs(p.projected_pl))}`;

                  return (
                    <tr key={p.trade_id} className="transition-colors"
                        style={{ borderBottom: i < sortedOptions.length - 1 ? "1px solid var(--border)" : "none" }}
                        onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                        onContextMenu={e => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY, position: p }); }}>
                      {/* Contract */}
                      <td className="px-2.5 py-2.5 font-semibold whitespace-nowrap" style={{ fontFamily: mono }} title={`Trade ID: ${p.trade_id}`}>
                        <span className="inline-flex items-center" style={{ gap: 6 }}>
                          {p.ticker}
                          {p.strategy && (
                            <StrategyChip name={p.strategy} color={strategyByName.get(p.strategy)?.color ?? "var(--ink-4)"} size="sm" showName={false} />
                          )}
                        </span>
                      </td>
                      {/* Days */}
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, fontSize: 11, color: "var(--ink-4)" }}>
                        {p.days_held}
                      </td>
                      {/* Exp Date */}
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                        {expDateLabel}
                      </td>
                      {/* DTE */}
                      <td className="px-2.5 py-2.5 text-center" style={{ fontFamily: mono, fontWeight: 600, color: dteColor }}>
                        {dte === null ? (
                          <span style={{ color: "var(--ink-4)" }}>—</span>
                        ) : dteExpired ? (
                          <span className="inline-block px-2 py-0.5 rounded text-[10px] font-bold"
                                style={{ background: "linear-gradient(135deg, #f87171, #ef4444)", color: "#fff" }}>
                            EXPIRED
                          </span>
                        ) : (
                          `${dte}d`
                        )}
                      </td>
                      {/* Return % */}
                      <td className="px-2.5 py-2.5 text-right">
                        <span className="inline-block px-2.5 py-0.5 rounded text-[11px] font-bold"
                              style={{ background: retBg, color: retText, minWidth: 62, textAlign: "center", fontFamily: mono }}>
                          {p.return_pct >= 0 ? "+" : ""}{p.return_pct.toFixed(2)}%
                        </span>
                      </td>
                      {/* Pos Size % */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                        {p.pos_size_pct.toFixed(1)}%
                      </td>
                      {/* Qty */}
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono }}>
                        {p.shares.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      {/* Entry */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                        {formatCurrency(p.avg_entry)}
                      </td>
                      {/* Current Price (inline-editable, pos 9) */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask"
                          style={{ fontFamily: mono, cursor: editingPriceTradeId !== p.trade_id ? "text" : "default" }}
                          onClick={editingPriceTradeId !== p.trade_id ? () => startEditPrice(p) : undefined}
                          title={p.manual_price !== null
                            ? `Manual override: ${formatCurrency(p.manual_price)}. Click to change or clear.`
                            : "Click to set a manual price (yfinance is unreliable for option chains)."}>
                        {editingPriceTradeId === p.trade_id ? (
                          <input
                            type="number" step="0.01" autoFocus disabled={savingPrice}
                            value={editPriceValue}
                            onChange={e => setEditPriceValue(e.target.value)}
                            onBlur={() => commitEditPrice(p)}
                            onKeyDown={e => {
                              if (e.key === "Enter") { e.preventDefault(); (e.currentTarget as HTMLInputElement).blur(); }
                              else if (e.key === "Escape") { e.preventDefault(); cancelEditPrice(); }
                            }}
                            placeholder="(blank to clear)"
                            className="w-[80px] text-right rounded-[6px] px-1.5 py-0.5 text-[12px]"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }}
                          />
                        ) : (
                          <span style={{
                            fontStyle: p.manual_price !== null ? "italic" : "normal",
                            color: p.manual_price !== null ? "var(--ink-2)" : "inherit",
                          }}>
                            {formatCurrency(p.current_price)}
                            {p.manual_price !== null && (
                              <span className="ml-1 inline-block px-1 rounded-[3px] text-[8px] font-bold align-middle"
                                    style={{ background: "color-mix(in oklab, #3b82f6 15%, transparent)", color: "#3b82f6", letterSpacing: "0.05em" }}
                                    title="Manual override">M</span>
                            )}
                          </span>
                        )}
                      </td>
                      {/* Value (pos 10) */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                        {formatCurrency(p.current_value)}
                      </td>
                      {/* Current Risk $ (pos 11) — signed projected_pl, multiplier-aware via lifo (Group 7-5). */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask"
                          style={{ fontFamily: mono, color: optRiskColor, fontWeight: 600, background: optRiskCellBg }}
                          title={optRiskTooltip}>
                        {formatCurrency(p.projected_pl, { showSign: true, signGlyph: "unicode" })}
                      </td>
                      {/* Current Risk % (pos 12) — projected_pct = projected_pl / NLV × 100. Same coloring as pos 11. */}
                      <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: optRiskColor }}>
                        {p.projected_pct >= 0 ? "+" : ""}{p.projected_pct.toFixed(2)}%
                      </td>
                      {/* Overall P&L (pos 13) — same red/green coloring as the equity column. */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, fontWeight: 700, color: p.overall_pl >= 0 ? "#08a86b" : "#e5484d" }}>
                        {formatCurrency(p.overall_pl, { showSign: true, signGlyph: "unicode" })}
                      </td>
                      {/* Trade Risk $ (pos 14) — historical risk_budget locked at
                          buy events; equivalent to the equity column. */}
                      <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, fontWeight: 600 }}>
                        {formatCurrency(p.risk_budget)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty state — both sections empty */}
      {equities.length === 0 && options.length === 0 && (
        <div className="rounded-[14px] p-8 text-center text-[13px]"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
          No open positions.
        </div>
      )}

      {/* Right-click row context menu */}
      {ctxMenu && (
        <div className="fixed z-50 rounded-[10px] py-1.5 min-w-[180px] overflow-hidden"
             style={{
               left: ctxMenu.x,
               top: ctxMenu.y,
               background: "var(--surface)",
               border: "1px solid var(--border)",
               boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
             }}>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>
            {ctxMenu.position.ticker}
          </div>
          <button className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                  style={{ color: "var(--ink)" }}
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  onClick={e => { e.stopPropagation(); ctxViewJournal(ctxMenu.position); setCtxMenu(null); }}>
            <span style={{ color: "var(--ink-4)" }}>&#x1F4CB;</span> View in Journal
          </button>
          <button className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                  style={{ color: "var(--ink)" }}
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  onClick={e => { e.stopPropagation(); ctxOpenPyramid(ctxMenu.position.trade_id); setCtxMenu(null); }}>
            <span style={{ color: "var(--ink-4)" }}>&#x1F53A;</span> Open Position Sizer Pyramid
          </button>
          <button className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                  style={{ color: "var(--ink)" }}
                  data-testid="ctx-increase-position"
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  onClick={e => { e.stopPropagation(); ctxIncreasePosition(ctxMenu.position); setCtxMenu(null); }}>
            <span style={{ color: "var(--ink-4)" }}>&#x2795;</span> Increase position
          </button>
          <button className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                  style={{ color: "var(--ink)" }}
                  data-testid="ctx-decrease-position"
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  onClick={e => { e.stopPropagation(); ctxDecreasePosition(ctxMenu.position); setCtxMenu(null); }}>
            <span style={{ color: "var(--ink-4)" }}>&#x2796;</span> Decrease position
          </button>
          {/* Phase 2 — Set strategy. Desktop: hover-flyout. Touch: flat list. */}
          {strategies.length > 0 && (coarsePointer ? (
            <StrategyFlatList
              strategies={strategies}
              currentStrategy={(ctxMenu.position as any).strategy}
              onPick={(name) => ctxSetStrategy(ctxMenu.position.trade_id, name)}
            />
          ) : (
            <StrategyFlyout
              strategies={strategies}
              currentStrategy={(ctxMenu.position as any).strategy}
              onPick={(name) => ctxSetStrategy(ctxMenu.position.trade_id, name)}
            />
          ))}
          {ctxMenu.position.is_option && (
            <button className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                    style={{ color: "var(--ink)" }}
                    data-testid="ctx-exercise-option"
                    onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                    onClick={e => { e.stopPropagation(); ctxOpenExercise(ctxMenu.position); setCtxMenu(null); }}>
              <span style={{ color: "var(--ink-4)" }}>&#x26A1;</span> Exercise option
            </button>
          )}
          {ctxMenu.position.sell_rule_tier === "sr8" && (
            <button className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2 transition-colors hover:brightness-95"
                    style={{ color: "var(--ink)" }}
                    data-testid="ctx-sr8-trim"
                    onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                    onClick={e => { e.stopPropagation(); setSr8TrimModalTradeId(ctxMenu.position.trade_id); setCtxMenu(null); }}>
              <span style={{ color: "var(--ink-4)" }}>&#x2702;&#xFE0F;</span> Calculate SR8 Trim
            </button>
          )}
        </div>
      )}

      {/* SR8 Trim modal. Renders the shared SR8TrimCalculator with the
          right-clicked position pre-selected. Backdrop click + Esc close.
          The same calculator powers the Trade Manager SR8 Trim tab —
          a single component drives both surfaces. */}
      {sr8TrimModalTradeId && (() => {
        const sr8 = positions.filter((pp) => pp.sell_rule_tier === "sr8");
        return (
          <div
            role="dialog"
            aria-label="SR8 Trim Calculator"
            onClick={() => setSr8TrimModalTradeId(null)}
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 60,
              background: "rgba(0,0,0,0.55)",
              display: "grid",
              placeItems: "center",
              padding: "32px",
            }}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              className="rounded-[14px] p-6"
              style={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                width: "min(900px, 100%)",
                maxHeight: "90vh",
                overflowY: "auto",
                boxShadow: "0 20px 50px rgba(0,0,0,0.30)",
              }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-[18px] font-semibold tracking-tight" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
                  SR8 Trim Calculator
                </h2>
                <button
                  onClick={() => setSr8TrimModalTradeId(null)}
                  className="w-8 h-8 rounded-full flex items-center justify-center text-[16px] transition-colors hover:brightness-95"
                  style={{ background: "var(--surface-2)", color: "var(--ink-3)" }}
                  aria-label="Close"
                >
                  ×
                </button>
              </div>
              <SR8TrimCalculator positions={sr8} preselectedTradeId={sr8TrimModalTradeId} />
            </div>
          </div>
        );
      })()}

      {/* Exercise-option modal. Preview math is computed client-side from
          the option summary's avg_entry (which the backend keeps as the
          weighted avg of remaining open lots) — close to the server's
          authoritative computation, may differ by a fraction if there are
          un-recomputed edits in flight. The backend always recomputes
          atomically, so the on-disk numbers are exact. */}
      {exerciseTarget && (() => {
        const parsed = parseOptionTicker(exerciseTarget.ticker);
        const contracts = exerciseTarget.shares;
        const multiplier = exerciseTarget.multiplier || 100;
        const sharesAcquired = contracts * multiplier;
        const previewEntry = parsed ? parsed.strike + exerciseTarget.avg_entry : null;
        const previewCost = previewEntry != null ? sharesAcquired * previewEntry : null;
        return (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
               data-testid="exercise-modal"
               style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
               onClick={() => { if (!exerciseSaving) closeExerciseModal(); }}>
            <div className="rounded-[14px] w-full max-w-[480px] overflow-hidden"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)",
                          boxShadow: "0 12px 32px rgba(0,0,0,0.18)" }}
                 onClick={e => e.stopPropagation()}>
              <div className="px-[18px] py-3 flex items-center gap-2"
                   style={{ borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: "var(--ink-4)" }}>&#x26A1;</span>
                <span className="text-[13px] font-semibold">Exercise option</span>
                <span className="text-xs ml-auto" style={{ color: "var(--ink-4)" }}>
                  {exerciseTarget.ticker}
                </span>
              </div>

              <div className="px-[18px] py-4 flex flex-col gap-3.5">
                {/* Preview math — read-only summary so the user can sanity-
                    check the conversion before committing. */}
                <div className="rounded-[10px] p-3 text-[12px]"
                     data-testid="exercise-preview"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  <div className="font-semibold mb-1.5" style={{ color: "var(--ink)" }}>
                    Preview
                  </div>
                  <div className="grid grid-cols-2 gap-y-1" style={{ color: "var(--ink-3)" }}>
                    <div>Will close:</div>
                    <div className="text-right font-medium privacy-mask" style={{ color: "var(--ink)" }}>
                      {contracts.toLocaleString()} contract{contracts === 1 ? "" : "s"}
                    </div>
                    {parsed ? (
                      <>
                        <div>Will open:</div>
                        <div className="text-right font-medium privacy-mask" style={{ color: "var(--ink)" }}>
                          {sharesAcquired.toLocaleString()} {parsed.underlying} sh.
                        </div>
                        <div>Entry price:</div>
                        <div className="text-right font-medium privacy-mask" style={{ color: "var(--ink)" }}>
                          ≈ {formatCurrency(previewEntry ?? 0)}
                        </div>
                        <div>Cost basis:</div>
                        <div className="text-right font-medium privacy-mask" style={{ color: "var(--ink)" }}>
                          ≈ {formatCurrency(previewCost ?? 0)}
                        </div>
                      </>
                    ) : (
                      <div className="col-span-2 text-[11px]" style={{ color: "#dc2626" }}>
                        Cannot parse option ticker — backend will reject.
                      </div>
                    )}
                  </div>
                  <div className="text-[10px] mt-1.5" style={{ color: "var(--ink-4)" }}>
                    Approximate · backend computes exact values from current transactions on save
                  </div>
                </div>

                <div>
                  <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1"
                         style={{ color: "var(--ink-4)" }}>
                    Date
                  </label>
                  <input type="date" value={exerciseDate}
                         disabled={exerciseSaving}
                         onChange={e => setExerciseDate(e.target.value)}
                         data-testid="exercise-date"
                         className="w-full px-3 py-2 rounded-[8px] text-[13px]"
                         style={{ background: "var(--bg)", border: "1px solid var(--border)",
                                  color: "var(--ink)" }} />
                </div>

                <div>
                  <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1"
                         style={{ color: "var(--ink-4)" }}>
                    Notes (optional)
                  </label>
                  <textarea value={exerciseNotes}
                            disabled={exerciseSaving}
                            onChange={e => setExerciseNotes(e.target.value)}
                            data-testid="exercise-notes"
                            rows={2}
                            placeholder="e.g. Exercising before expiry"
                            className="w-full px-3 py-2 rounded-[8px] text-[13px] resize-none"
                            style={{ background: "var(--bg)", border: "1px solid var(--border)",
                                     color: "var(--ink)" }} />
                </div>

                {exerciseError && (
                  <div className="rounded-[8px] px-3 py-2 text-[12px]"
                       data-testid="exercise-error"
                       style={{ background: "color-mix(in oklab, #dc2626 8%, var(--surface))",
                                border: "1px solid color-mix(in oklab, #dc2626 30%, var(--border))",
                                color: "#dc2626" }}>
                    {exerciseError}
                  </div>
                )}
              </div>

              <div className="px-[18px] py-3 flex items-center justify-end gap-2"
                   style={{ borderTop: "1px solid var(--border)", background: "var(--surface-2)" }}>
                <button onClick={closeExerciseModal}
                        disabled={exerciseSaving}
                        data-testid="exercise-cancel"
                        className="px-3 py-1.5 rounded-[8px] text-[12px] font-medium transition-colors hover:brightness-95"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)",
                                 color: "var(--ink-3)",
                                 opacity: exerciseSaving ? 0.5 : 1 }}>
                  Cancel
                </button>
                <button onClick={submitExercise}
                        disabled={exerciseSaving || !parsed}
                        data-testid="exercise-confirm"
                        className="px-3 py-1.5 rounded-[8px] text-[12px] font-semibold transition-colors hover:brightness-95"
                        style={{ background: "var(--ink)", color: "var(--surface)",
                                 opacity: (exerciseSaving || !parsed) ? 0.5 : 1 }}>
                  {exerciseSaving ? "Exercising…" : "Confirm exercise"}
                </button>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Risk Monitor (equity-only) */}
      <div className="mt-6 rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <button onClick={() => setRiskMonitorOpen(!riskMonitorOpen)}
                className="w-full flex items-center gap-2 px-[18px] py-3 text-left cursor-pointer transition-colors hover:brightness-95"
                style={{ borderBottom: riskMonitorOpen ? "1px solid var(--border)" : "none", background: "var(--surface-2)" }}>
          <span className="text-[10px] transition-transform" style={{ transform: riskMonitorOpen ? "rotate(90deg)" : "none", color: "var(--ink-4)" }}>▶</span>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Risk Monitor</span>
          <span className="text-xs" style={{ color: "var(--ink-4)" }}>
            Equity alerts · {equities.length} position{equities.length === 1 ? "" : "s"}
            {!riskMonitorOpen && monitorAlerts.length > 0 && ` · ${monitorAlerts.length} alert${monitorAlerts.length === 1 ? "" : "s"}`}
            {!riskMonitorOpen && " · click to expand"}
          </span>
        </button>
        {riskMonitorOpen && (
        <div className="p-4 flex flex-col gap-2.5">
          {monitorAlerts.filter(a => a.type === "info").map((a, i) => (
            <div key={`info-${i}`} className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">📈</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#3b82f6" }}>
                <strong>{a.ticker}</strong>: {a.msg}
              </div>
            </div>
          ))}
          {monitorAlerts.filter(a => a.type === "warn").map((a, i) => (
            <div key={`warn-${i}`} className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">⚠️</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#854d0e" }}>
                <strong>{a.ticker}</strong>: {a.msg}
              </div>
            </div>
          ))}
          {monitorAlerts.filter(a => a.type === "error").map((a, i) => (
            <div key={`err-${i}`} className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">🔴</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#991b1b" }}>
                <strong>{a.ticker}</strong>: {a.msg}
              </div>
            </div>
          ))}
          {freeRollTickers.length > 0 && (
            <div className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">🆓</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#166534" }}>
                <strong>{freeRollTickers.join(", ")}</strong> — Free Roll — stops above entry. 0 risk.
              </div>
            </div>
          )}
          {monitorAlerts.length === 0 && freeRollTickers.length === 0 && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-[10px] text-[12px] font-medium"
                 style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              System Health Good — all positions within risk parameters
            </div>
          )}
        </div>
        )}
      </div>

      {/* EOD batch-edit modal */}
      {eodModalOpen && (
        <div className="fixed inset-0 z-[100] grid place-items-start justify-center pt-[12vh]"
             style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
             onClick={() => { if (!eodSaving) closeEodModal(); }}>
          <div className="w-[640px] max-w-[92vw] rounded-[14px] overflow-hidden"
               style={{ background: "var(--surface)", boxShadow: "0 20px 48px rgba(0,0,0,0.2), 0 0 0 1px var(--border)", animation: "cmdk-rise 0.22s cubic-bezier(.2,.9,.3,1.1)" }}
               onClick={e => e.stopPropagation()}>
            <div className="px-[18px] py-3.5 flex items-center" style={{ borderBottom: "1px solid var(--border)" }}>
              <div>
                <div className="text-[14px] font-semibold">Update End-of-Day Option Prices</div>
                <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
                  Leave a field blank to keep the current value. Tab moves between inputs.
                </div>
              </div>
              <kbd className="ml-auto text-[10px] rounded px-1.5 py-0.5"
                   style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-4)", fontFamily: mono }}>ESC</kbd>
            </div>
            {eodErrors.length > 0 && (
              <div className="mx-2 mt-2 px-3 py-2 rounded-[8px] text-[11px] leading-relaxed"
                   style={{
                     background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
                     border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
                     color: "#991b1b",
                   }}>
                Failed to save {eodErrors.length} contract{eodErrors.length === 1 ? "" : "s"}: <strong>{eodErrors.join(", ")}</strong>. Edit and retry.
              </div>
            )}
            <div className="max-h-[60vh] overflow-y-auto p-2">
              {options.length === 0 ? (
                <div className="px-3 py-6 text-center text-[12px]" style={{ color: "var(--ink-4)" }}>No open option positions.</div>
              ) : [...options].sort((a, b) => a.ticker.localeCompare(b.ticker)).map(opt => (
                <div key={opt.trade_id} className="grid grid-cols-[1fr_auto_140px] items-center gap-3 px-3 py-2 rounded-[8px]"
                     style={{ background: "var(--surface)" }}>
                  <div className="text-[12px] font-semibold truncate" style={{ fontFamily: mono }} title={opt.ticker}>
                    {opt.ticker}
                  </div>
                  <div className="text-[11px] tabular-nums whitespace-nowrap" style={{ color: "var(--ink-4)", fontFamily: mono }}>
                    Current: {opt.manual_price !== null ? formatCurrency(opt.manual_price) : "—"}
                  </div>
                  <input
                    type="number" step="0.01"
                    placeholder="(blank = no change)"
                    disabled={eodSaving}
                    value={eodEdits[opt.trade_id] ?? ""}
                    onChange={e => setEodEdits(prev => ({ ...prev, [opt.trade_id]: e.target.value }))}
                    className="w-full text-right rounded-[6px] px-2 py-1 text-[12px]"
                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }}
                  />
                </div>
              ))}
            </div>
            <div className="px-[18px] py-3 flex items-center gap-2" style={{ borderTop: "1px solid var(--border)" }}>
              <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{options.length} option position{options.length === 1 ? "" : "s"}</span>
              <button onClick={closeEodModal}
                      disabled={eodSaving}
                      className="ml-auto h-[30px] px-3 rounded-md text-[12px] font-medium hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
                Cancel
              </button>
              <button onClick={saveEodPrices}
                      disabled={eodSaving || options.length === 0}
                      className="h-[30px] px-3.5 rounded-md text-[12px] font-medium text-white flex items-center gap-1.5 hover:brightness-95 disabled:opacity-60 disabled:cursor-wait"
                      style={{ background: navColor }}>
                {eodSaving && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
                    <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
                  </svg>
                )}
                {eodSaving ? "Saving…" : "Save All"}
              </button>
            </div>
          </div>
          <style jsx global>{`@keyframes cmdk-rise { from { transform: translateY(-10px) scale(0.97); opacity: 0; } }`}</style>
        </div>
      )}
    </div>
  );
}
