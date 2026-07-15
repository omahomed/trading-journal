"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type Strategy, type AddEffectivenessResponse } from "@/lib/api";
import { computeEnrichedPositions, type EnrichedPosition } from "@/lib/positions";
import { LESSON_CATEGORIES, CAT_COLORS } from "@/lib/lesson-categories";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { StrategyChip } from "./strategy-chip";
import { StrategyFlyout, StrategyFlatList, useCoarsePointer } from "./strategy-flyout";
import { SearchSelect } from "./search-select";
import {
  computeWinRate,
  computeProfitFactor,
  computeHoldRatio,
  computeOnePctCompliance,
  getPriorDayNlv,
  tradeWasOpenInYear,
  availableTradeYears,
  paretoDistribution,
  holdTimeBuckets,
  brandtNormalized,
  stopCapScenario,
  fixedSizeScenario,
  regimeCrossTab,
  makeMctStateResolver,
  setupScorecard,
  type SetupScorecardRow,
  type SetupVerdict,
  riskMetrics,
  repeatOffenders,
  type RepeatOffender,
  generateInsights,
  type Insight,
  winnerMaeDistribution,
  loserMfeDistribution,
  entryQualityBySetup,
  type EntryQualityRow,
  confluenceAnalysis,
  type ConfluenceRow,
} from "@/lib/analytics-stats";
// Pure CSS bar chart — no Recharts dependency

type Tab = "overview" | "scenarios" | "buyrules" | "sellrules" | "drawdown" | "review" | "campaigns" | "add-effectiveness";

function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }

function HeroCard({ label, value, sub, ok }: { label: string; value: string; sub: string; ok: boolean }) {
  const color = ok ? "#08a86b" : "#e5484d";
  return (
    <div className="p-5 rounded-[16px] transition-transform duration-200 hover:scale-[1.02]"
         style={{ background: `color-mix(in oklab, ${color} 8%, var(--surface))`, border: "1px solid var(--border)", boxShadow: "0 2px 8px rgba(0,0,0,0.04)" }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.10em]" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[34px] font-extrabold mt-2 privacy-mask" style={{ color, lineHeight: 1, fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      <div className="text-[12px] mt-2 font-medium" style={{ color: "var(--ink-4)" }}>{sub}</div>
    </div>
  );
}

function QualityTile({ label, value, status, ok }: { label: string; value: string; status: string; ok: boolean }) {
  const color = ok ? "#08a86b" : "#d97706";
  return (
    <div className="p-4 rounded-[12px] transition-all duration-200 hover:shadow-md"
         style={{ background: `color-mix(in oklab, ${color} 6%, var(--surface))`, borderLeft: `4px solid ${color}`, border: "1px solid var(--border)" }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[24px] font-extrabold mt-1.5" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{value}</div>
      <div className="text-[11px] font-semibold mt-1" style={{ color }}>{ok ? "✅" : "⚠️"} {status}</div>
    </div>
  );
}

export function Analytics({ navColor, initialTab, initialTradeId, onTabConsumed, onTradeIdConsumed }: {
  navColor: string;
  initialTab?: string;
  initialTradeId?: string;
  onTabConsumed?: () => void;
  onTradeIdConsumed?: () => void;
}) {
  const [allTrades, setAllTrades] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [openCount, setOpenCount] = useState(0);
  const router = useRouter();
  const [journalHistory, setJournalHistory] = useState<any[]>([]);
  const [mctStates, setMctStates] = useState<Array<{ trade_date: string; state: string }>>([]);
  const [loading, setLoading] = useState(true);
  // Phase 2 — All Campaigns retroactive tagging (right-click flyout
  // only; the bulk-select toolbar was removed in feat/all-campaigns-
  // filter-row in favour of filter pills).
  // - strategies: active-only, used by the right-click flyout (matches
  //   log_buy / patch_trade_strategy validation: can't tag via PATCH
  //   with an inactive strategy).
  // - allStrategies: all rows including inactive, used by the Strategy
  //   filter dropdown so existing tagged trades stay filterable even
  //   after their strategy is deactivated.
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [allStrategies, setAllStrategies] = useState<Strategy[]>([]);
  const [campCtxMenu, setCampCtxMenu] = useState<{ x: number; y: number; trade: TradePosition } | null>(null);
  const coarsePointer = useCoarsePointer();
  // All Campaigns filter state. New in feat/all-campaigns-filter-row:
  // four pill filters (Strategy, Instrument, Buy Rule, Sell Rule) plus
  // open/close state + outside-click ref for the custom Strategy
  // dropdown (the only one of the four that needs a custom dropdown,
  // because <select> can't render the colored swatch).
  const [campStrategy, setCampStrategy] = useState<string>("");
  const [campInstrument, setCampInstrument] = useState<string>("");
  const [campBuyRule, setCampBuyRule] = useState<string>("");
  const [campSellRule, setCampSellRule] = useState<string>("");
  const [strategyFilterOpen, setStrategyFilterOpen] = useState(false);
  const strategyFilterRef = useRef<HTMLDivElement>(null);
  const [tab, setTab] = useState<Tab>((initialTab as Tab) || "overview");
  // Trade Review deep-link target. Captured from ?trade_id=<id> on URL
  // entry; we scroll to and auto-expand its lesson card once the closed
  // trades have loaded. Held as state so the URL prop can be consumed
  // (cleared in the parent) without losing the in-component target.
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(initialTradeId ?? null);

  useEffect(() => {
    if (initialTab && ["overview", "scenarios", "buyrules", "sellrules", "drawdown", "review", "campaigns", "add-effectiveness"].includes(initialTab)) {
      setTab(initialTab as Tab);
      onTabConsumed?.();
    }
  }, [initialTab, onTabConsumed]);

  useEffect(() => {
    if (initialTradeId) {
      setSelectedTradeId(initialTradeId);
      onTradeIdConsumed?.();
    }
  }, [initialTradeId, onTradeIdConsumed]);

  // Add effectiveness tab state. Default date window = year-to-date.
  // Stored as YYYY-MM-DD strings so the date inputs can bind directly.
  const _today = new Date();
  const _yearStart = `${_today.getFullYear()}-01-01`;
  const _todayIso = _today.toISOString().slice(0, 10);
  const [aeStart, setAeStart] = useState<string>(_yearStart);
  const [aeEnd, setAeEnd] = useState<string>(_todayIso);
  const [aeStrategy, setAeStrategy] = useState<string>("");
  const [aeData, setAeData] = useState<AddEffectivenessResponse | null>(null);
  const [aeLoading, setAeLoading] = useState(false);
  const [aeError, setAeError] = useState<string | null>(null);
  const [aeSortKey, setAeSortKey] = useState<keyof AddEffectivenessResponse["rules"][number]>("add_count");
  const [aeSortDir, setAeSortDir] = useState<"asc" | "desc">("desc");
  // Edge Report top-of-page controls (replaces the LTD/2026 scope pill):
  //   * yearFilter — the year the trade must have been "open at any
  //     point during." Defaults to the most recent year with data
  //     (availableTradeYears takes care of the calendar-year rollover
  //     edge case). Null means "not initialized yet" — set once trades
  //     load, per the useEffect below.
  //   * cohort — "closed" | "at-mark". "at-mark" unions open positions
  //     (marked to live price via computeEnrichedPositions) into the
  //     stats set with pnl = overall_pl and closed_date = today.
  const [yearFilter, setYearFilter] = useState<number | null>(null);
  const [cohort, setCohort] = useState<"closed" | "at-mark">("closed");

  // Open-position state hoisted above the year/cohort memos so
  // availableTradeYears and the at-mark cohort layer can reference it.
  // (Was previously declared alongside the All Campaigns state block
  // further down; the mount fetch below still populates all of it in
  // one shot.)
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [pricesStale, setPricesStale] = useState(false);
  // drillRule kept for sell rules tab (TODO)

  useEffect(() => {
    Promise.all([
      api.tradesClosed(getActivePortfolio(), 1000).catch((err) => {
        log.error("analytics", "tradesClosed fetch failed", err);
        return [];
      }),
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("analytics", "tradesOpen fetch failed", err);
        return [];
      }),
      api.journalHistory(getActivePortfolio(), 0).catch((err) => {
        log.error("analytics", "journalHistory fetch failed", err);
        return [];
      }),
      api.tradesRecent(getActivePortfolio(), 2000).catch((err) => {
        log.error("analytics", "tradesRecent fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
    ]).then(([closed, open, journal, details]) => {
      setAllTrades(closed as TradePosition[]);
      const openArr = open as TradePosition[];
      setOpenCount(openArr.length);
      setOpenTrades(openArr);
      setJournalHistory(journal as any[]);
      setAllDetails(details.details);
      // Fetch trade lessons
      api.getTradeLessons(getActivePortfolio()).then(r => { if (r.lessons) setLessons(r.lessons); }).catch((err) => {
        log.error("analytics", "getTradeLessons fetch failed", err);
      });
      setLoading(false);
    });
    // MCT states — powers the Regime × Month cross-tab. Wide window
    // covers every historical trade; endpoint returns one row per day
    // so payload stays small.
    const today = new Date().toISOString().slice(0, 10);
    api.mctStateByDateRange("2025-01-01", today).then(r => {
      setMctStates(r.states.map(s => ({ trade_date: s.trade_date, state: s.state })));
    }).catch((err) => {
      log.error("analytics", "mctStateByDateRange fetch failed", err);
    });
  }, []);

  // Phase 2 — active strategies for the right-click flyout. Inactive
  // strategies are excluded here because PATCH /api/trades/{id}/strategy
  // rejects them (matches log_buy's contract).
  useEffect(() => {
    api.listStrategies({ active: true, portfolio: getActivePortfolio() }).then(setStrategies).catch((err) => {
      log.error("analytics", "listStrategies (active) fetch failed", err);
      setStrategies([]);
    });
  }, []);

  // All strategies (including inactive) for the Strategy filter
  // dropdown. NOT portfolio-scoped (Migration 038): historical trades
  // may carry strategy tags no longer "allowed" in the active
  // portfolio (e.g., the 13 CanSlim-tagged trades in 457B Plan).
  // Filter UX must surface every strategy ever used in this portfolio's
  // trades, including ones outside the current allow list.
  // Two separate fetches by design: each consumer's purpose is obvious
  // from its source-of-truth name, and the endpoint is tiny enough
  // that doubling the request cost is negligible.
  useEffect(() => {
    api.listStrategies({ active: false }).then(setAllStrategies).catch((err) => {
      log.error("analytics", "listStrategies (all) fetch failed", err);
      setAllStrategies([]);
    });
  }, []);

  // Refresh just the campaign rows after a tagging action — cheaper than
  // reloading journal history + trade lessons. Open + closed combined is
  // what the campaigns tab reads from.
  const refreshCampaigns = async () => {
    const [closed, open] = await Promise.all([
      api.tradesClosed(getActivePortfolio(), 1000).catch((err) => {
        log.error("analytics", "tradesClosed refresh failed", err);
        return [];
      }),
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("analytics", "tradesOpen refresh failed", err);
        return [];
      }),
    ]);
    setAllTrades(closed as TradePosition[]);
    setOpenTrades(open as TradePosition[]);
  };

  // Close the right-click context menu on outside click / Escape.
  useEffect(() => {
    if (!campCtxMenu) return;
    const close = () => setCampCtxMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [campCtxMenu]);

  // Close the Strategy filter dropdown on outside click.
  useEffect(() => {
    if (!strategyFilterOpen) return;
    const handler = (e: MouseEvent) => {
      if (strategyFilterRef.current && !strategyFilterRef.current.contains(e.target as Node)) {
        setStrategyFilterOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [strategyFilterOpen]);

  // Add effectiveness fetch. Fires whenever the user enters the tab or
  // changes a filter. The backend caches load_summary / load_details,
  // so quick filter toggles stay snappy.
  useEffect(() => {
    if (tab !== "add-effectiveness") return;
    let cancelled = false;
    setAeLoading(true);
    setAeError(null);
    api.addEffectiveness(getActivePortfolio(), aeStart, aeEnd, aeStrategy)
      .then(res => {
        if (cancelled) return;
        if (res && "error" in res) {
          setAeError(String(res.error));
          setAeData(null);
        } else {
          setAeData(res as AddEffectivenessResponse);
        }
      })
      .catch(err => {
        if (cancelled) return;
        log.error("analytics", "addEffectiveness fetch failed", err);
        setAeError(err instanceof Error ? err.message : String(err));
        setAeData(null);
      })
      .finally(() => { if (!cancelled) setAeLoading(false); });
    return () => { cancelled = true; };
  }, [tab, aeStart, aeEnd, aeStrategy]);

  // Single-trade retag from the right-click menu. Refreshes after success
  // so the table re-renders with the new strategy on the row.
  const setOneStrategy = async (trade_id: string, strategy: string) => {
    setCampCtxMenu(null);
    const r = await api.setTradeStrategy(trade_id, { strategy }).catch(() => ({ error: "network" }));
    if (!("error" in r) || !r.error) await refreshCampaigns();
  };

  // Available years for the year picker; derived from loaded data. The
  // default year is set once (when data first arrives) so the user's
  // subsequent selection isn't overwritten on re-render.
  const yearData = useMemo(
    () => availableTradeYears(allTrades, openTrades),
    [allTrades, openTrades],
  );
  useEffect(() => {
    if (yearFilter == null && (allTrades.length > 0 || openTrades.length > 0)) {
      setYearFilter(yearData.defaultYear);
    }
  }, [yearData, yearFilter, allTrades.length, openTrades.length]);
  const year = yearFilter ?? yearData.defaultYear;

  // Closed cohort filtered by year — the "in-scope closed trades" for
  // the selected year. Bare year filter, no at-mark logic (that layer
  // is applied by the trades memo below).
  const closedInYear = useMemo(
    () => allTrades.filter(t => tradeWasOpenInYear(t, year)),
    [allTrades, year],
  );

  // Enriched opens are the source of overall_pl for the at-mark cohort
  // shim. Declared here (rather than alongside the All Campaigns state
  // it originally served) so the trades memo below and every
  // downstream stats memo see it.
  const enrichedOpen = useMemo(
    () => computeEnrichedPositions(openTrades, allDetails, 0, livePrices),
    [openTrades, allDetails, livePrices]
  );
  const enrichedById = useMemo(
    () => Object.fromEntries(enrichedOpen.map(p => [p.trade_id, p])) as Record<string, EnrichedPosition>,
    [enrichedOpen]
  );

  // Cohort-aware final trade set. The "closed" cohort is byte-for-byte
  // the pre-Edge-Report behavior; "at-mark" unions currently-open
  // positions in scope for the year, valued at overall_pl (realized
  // bank + unrealized at live price) and pretend-closed today.
  const trades = useMemo(() => {
    if (cohort === "closed") return closedInYear;
    const todayIso = new Date().toISOString().slice(0, 10);
    const openAsClosed: TradePosition[] = openTrades
      .filter(t => tradeWasOpenInYear(t, year))
      .map(t => {
        const enriched = enrichedById[t.trade_id];
        const overallPl = enriched ? enriched.overall_pl : 0;
        return {
          ...t,
          realized_pl: overallPl as any,
          closed_date: todayIso as any,
        } as TradePosition;
      });
    return [...closedInYear, ...openAsClosed];
  }, [closedInYear, openTrades, enrichedById, cohort, year]);

  const stats = useMemo(() => {
    const closed = trades;
    const wins = closed.filter(t => parseFloat(String(t.realized_pl || 0)) > 0);
    const losses = closed.filter(t => parseFloat(String(t.realized_pl || 0)) < 0);
    const breakEven = closed.filter(t => parseFloat(String(t.realized_pl || 0)) === 0);
    const grossProfit = wins.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
    const grossLoss = Math.abs(losses.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0));
    const pf = computeProfitFactor(closed);
    const winRate = computeWinRate(closed);
    const avgWin = wins.length > 0 ? grossProfit / wins.length : 0;
    const avgLoss = losses.length > 0 ? -grossLoss / losses.length : 0;
    const avgTrade = closed.length > 0 ? (grossProfit - grossLoss) / closed.length : 0;
    const wlRatio = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;
    const netPl = grossProfit - grossLoss;
    const expectancy = (winRate / 100 * avgWin) + ((100 - winRate) / 100 * avgLoss);
    const largestWin = wins.length > 0 ? Math.max(...wins.map(t => parseFloat(String(t.realized_pl || 0)))) : 0;
    const largestLoss = losses.length > 0 ? Math.min(...losses.map(t => parseFloat(String(t.realized_pl || 0)))) : 0;

    // R-multiple
    const withR = closed.filter(t => parseFloat(String(t.risk_budget || 0)) > 0);
    const avgR = withR.length > 0 ? withR.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)) / parseFloat(String(t.risk_budget || 1)), 0) / withR.length : 0;
    const maxR = withR.length > 0 ? Math.max(...withR.map(t => parseFloat(String(t.realized_pl || 0)) / parseFloat(String(t.risk_budget || 1)))) : 0;

    // Hold days — guard against empty/invalid dates. avgHold stays inline
    // because avgHoldAll is unique to the analytics overview; lib only
    // needs the W/L pieces.
    const holdDays = (t: TradePosition) => {
      const oStr = String(t.open_date || "").trim();
      const cStr = String(t.closed_date || "").trim();
      if (!oStr || !cStr) return null;
      const open = new Date(oStr);
      const close = new Date(cStr);
      if (isNaN(open.getTime()) || isNaN(close.getTime())) return null;
      return Math.max(0, Math.floor((close.getTime() - open.getTime()) / 86400000));
    };
    const avgHold = (arr: TradePosition[]) => {
      const valid = arr.map(holdDays).filter((d): d is number => d !== null);
      return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
    };
    const avgHoldAll = avgHold(closed);
    const hr = computeHoldRatio(closed);
    const winnersHold = hr.winnersHold;
    const losersHold = hr.losersHold;
    const holdRatio = hr.ratio;

    // Consecutive streaks
    const sorted = [...closed].sort((a, b) => String(a.closed_date || "").localeCompare(String(b.closed_date || "")));
    let maxWinStreak = 0, maxLossStreak = 0, ws = 0, ls = 0;
    for (const t of sorted) {
      if (parseFloat(String(t.realized_pl || 0)) > 0) { ws++; ls = 0; maxWinStreak = Math.max(maxWinStreak, ws); }
      else { ls++; ws = 0; maxLossStreak = Math.max(maxLossStreak, ls); }
    }

    // Monthly performance
    const monthMap: Record<string, number> = {};
    for (const t of closed) {
      const m = String(t.closed_date || "").slice(0, 7);
      if (m) monthMap[m] = (monthMap[m] || 0) + parseFloat(String(t.realized_pl || 0));
    }
    const monthVals = Object.values(monthMap);
    const bestMonth = monthVals.length > 0 ? Math.max(...monthVals) : 0;
    const worstMonth = monthVals.length > 0 ? Math.min(...monthVals) : 0;
    const avgMonth = monthVals.length > 0 ? monthVals.reduce((a, b) => a + b, 0) / monthVals.length : 0;
    const bestMonthKey = Object.entries(monthMap).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
    const worstMonthKey = Object.entries(monthMap).sort((a, b) => a[1] - b[1])[0]?.[0] || "";

    return {
      total: closed.length, wins: wins.length, losses: losses.length, breakEven: breakEven.length,
      grossProfit, grossLoss, pf, winRate, avgWin, avgLoss, avgTrade, wlRatio, netPl, expectancy,
      largestWin, largestLoss, avgR, maxR,
      winnersHold, losersHold, avgHoldAll, holdRatio,
      maxWinStreak, maxLossStreak, bestMonth, worstMonth, avgMonth, bestMonthKey, worstMonthKey,
    };
  }, [trades]);

  // Buy rules sort
  const [brSort, setBrSort] = useState("Total P&L");
  const [brDrill, setBrDrill] = useState("");
  const [brNoteText, setBrNoteText] = useState("");
  const [brNoteStatus, setBrNoteStatus] = useState("— no status —");
  // "closed" = byte-for-byte the historical behavior (closed-in-2026 only).
  // "open"   = opened-in-2026 open positions, marked to latest price.
  // "all"    = both, summed on the unified ternary the All Campaigns table
  //            uses (overall_pl for open, realized_pl for closed). Same R
  //            denominator (risk_budget) — see [L1683-1686] in this file.
  const [ruleStatus, setRuleStatus] = useState<"all" | "closed" | "open">("closed");

  // Sell rules sort
  const [srSort, setSrSort] = useState("Total P&L");
  const [srDrill, setSrDrill] = useState("");
  const [srNoteText, setSrNoteText] = useState("");
  const [srNoteStatus, setSrNoteStatus] = useState("— no status —");

  // Trade Review
  const [trRange, setTrRange] = useState("2026 YTD");
  const [topN, setTopN] = useState(10);
  const [lessons, setLessons] = useState<Record<string, { note: string; category: string }>>({});
  const [lessonEdits, setLessonEdits] = useState<Record<string, string>>({});

  // All Campaigns
  const [campStatus, setCampStatus] = useState<"all" | "open" | "closed">("all");
  const [campTicker, setCampTicker] = useState("");
  const [campDateRange, setCampDateRange] = useState("YTD");
  const [campResult, setCampResult] = useState<"all" | "winners" | "losers">("all");
  const [campGrade, setCampGrade] = useState<"all" | "unrated" | "1" | "2" | "3" | "4" | "5">("all");
  const [campSort, setCampSort] = useState<{ col: string; asc: boolean }>({ col: "open", asc: false });
  // (openTrades / livePrices / pricesStale hoisted to top of component
  // so the Edge Report year picker + at-mark cohort can reference them.)

  // All Campaigns filter — derived option lists (Buy Rule, Sell Rule,
  // Instrument). Computed at component level via useMemo so React-hooks
  // ordering stays stable across renders (the campaigns tab is a
  // conditional sub-render, so hooks can't live inside its IIFE).
  const buyRuleOptions = useMemo(() => {
    const all = [...openTrades, ...allTrades];
    return [...new Set(all.map(t => String((t as any).buy_rule || t.rule || "").trim()).filter(Boolean))].sort();
  }, [openTrades, allTrades]);
  const sellRuleOptions = useMemo(() => {
    const all = [...openTrades, ...allTrades];
    return [...new Set(all.map(t => String((t as any).sell_rule || "").trim()).filter(Boolean))].sort();
  }, [openTrades, allTrades]);
  const instrumentOptions = useMemo(() => {
    const all = [...openTrades, ...allTrades];
    return [...new Set(all.map(t => String(t.instrument_type || "STOCK")))].sort();
  }, [openTrades, allTrades]);

  useEffect(() => {
    if (openTrades.length === 0) return;
    const tickers = [...new Set(openTrades.map(t => t.ticker).filter(Boolean))];
    if (tickers.length === 0) return;
    api.batchPrices(tickers, getActivePortfolio())
      .then(prices => { setLivePrices(prices); setPricesStale(false); })
      .catch(() => setPricesStale(true));
  }, [openTrades]);

  // (enrichedOpen / enrichedById / trades hoisted up next to
  // closedInYear so downstream stats can read them.)

  // Rule stats — always 2026 for buy/sell rules (matching Streamlit).
  // Sell-rules path is closed-only (open positions have no sell_rule yet).
  // Buy-rules path supports All / Closed / Open via ruleStatus:
  //   closed: closed-in-2026 closed trades, P&L = realized_pl (historical).
  //   open:   opened-in-2026 open trades, P&L = enrichedById[id].overall_pl
  //           (marked to latest price; folds in any already-realized partials).
  //   all:    union of both, summed on the same unified ternary the All
  //           Campaigns table uses at [L1683-1686]. Denominator (risk_budget)
  //           is unchanged across statuses.
  const ruleStats = useMemo(() => {
    const isSell = tab === "sellrules";
    const col = isSell ? "sell_rule" : "buy_rule";
    const includeClosed = isSell ? true : (ruleStatus !== "open");
    const includeOpen = isSell ? false : (ruleStatus !== "closed");
    const closedSource = includeClosed
      ? allTrades.filter(t => String(t.closed_date || "").startsWith("2026"))
      : [];
    const openSource = includeOpen
      ? openTrades.filter(t => String(t.open_date || "").startsWith("2026"))
      : [];
    type Bucket = { rule: string; count: number; wins: number; totalPl: number; rValues: number[]; trades: TradePosition[] };
    const map: Record<string, Bucket> = {};
    const tally = (t: TradePosition, pl: number) => {
      const rule = String((t as any)[col] || (t as any).rule || "").trim();
      if (!rule || rule === "nan" || rule === "undefined") return;
      if (!map[rule]) map[rule] = { rule, count: 0, wins: 0, totalPl: 0, rValues: [], trades: [] };
      const rb = parseFloat(String(t.risk_budget || 0));
      map[rule].count++;
      if (pl > 0) map[rule].wins++;
      map[rule].totalPl += pl;
      if (rb > 0) map[rule].rValues.push(pl / rb);
      map[rule].trades.push(t);
    };
    for (const t of closedSource) tally(t, parseFloat(String(t.realized_pl || 0)));
    for (const t of openSource) tally(t, enrichedById[t.trade_id]?.overall_pl ?? 0);
    const arr = Object.values(map).map(r => ({
      ...r,
      avgPl: r.count > 0 ? r.totalPl / r.count : 0,
      winRate: r.count > 0 ? (r.wins / r.count) * 100 : 0,
      avgR: r.rValues.length > 0 ? r.rValues.reduce((a, b) => a + b, 0) / r.rValues.length : null as number | null,
    }));
    // Sort
    const key = brSort === "Win Rate %" ? "winRate" : brSort === "Avg P&L" ? "avgPl" : brSort === "Trades" ? "count" : "totalPl";
    return arr.sort((a, b) => (b as any)[key] - (a as any)[key]);
  }, [allTrades, openTrades, enrichedById, tab, brSort, ruleStatus]);

  // Sell rule stats — separate because it includes Hold days
  const sellRuleStats = useMemo(() => {
    const source = allTrades.filter(t => String(t.closed_date || "").startsWith("2026"));
    const map: Record<string, { rule: string; count: number; wins: number; totalPl: number; rValues: number[]; holdDays: number[]; trades: TradePosition[] }> = {};
    for (const t of source) {
      const rule = String((t as any).sell_rule || "").trim();
      if (!rule || rule === "nan" || rule === "undefined") continue;
      if (!map[rule]) map[rule] = { rule, count: 0, wins: 0, totalPl: 0, rValues: [], holdDays: [], trades: [] };
      const pl = parseFloat(String(t.realized_pl || 0));
      const rb = parseFloat(String(t.risk_budget || 0));
      map[rule].count++;
      if (pl > 0) map[rule].wins++;
      map[rule].totalPl += pl;
      if (rb > 0) map[rule].rValues.push(pl / rb);
      // Hold days
      const oStr = String(t.open_date || "").trim();
      const cStr = String(t.closed_date || "").trim();
      if (oStr && cStr) {
        const od = new Date(oStr); const cd = new Date(cStr);
        if (!isNaN(od.getTime()) && !isNaN(cd.getTime())) map[rule].holdDays.push(Math.max(0, Math.floor((cd.getTime() - od.getTime()) / 86400000)));
      }
      map[rule].trades.push(t);
    }
    const arr = Object.values(map).map(r => ({
      ...r,
      avgPl: r.count > 0 ? r.totalPl / r.count : 0,
      winRate: r.count > 0 ? (r.wins / r.count) * 100 : 0,
      avgR: r.rValues.length > 0 ? r.rValues.reduce((a, b) => a + b, 0) / r.rValues.length : null as number | null,
      avgHold: r.holdDays.length > 0 ? r.holdDays.reduce((a, b) => a + b, 0) / r.holdDays.length : null as number | null,
    }));
    const key = srSort === "Uses" ? "count" : srSort === "Avg P&L" ? "avgPl" : srSort === "Winners %" ? "winRate" : "totalPl";
    return arr.sort((a, b) => (b as any)[key] - (a as any)[key]);
  }, [allTrades, srSort]);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  const mono = "var(--font-jetbrains), monospace";
  const s = stats;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Edge <em className="italic" style={{ color: navColor }}>Report</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>{openCount} open · {allTrades.length} closed · {openCount + allTrades.length} total</div>
      </div>

      {/* Tabs */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex gap-1 pb-0.5 flex-wrap" style={{ borderBottom: "2px solid var(--border)" }}>
          {([ { key: "overview" as Tab, label: "🎯 Overview" }, { key: "scenarios" as Tab, label: "🏆 Setup Scorecard" }, { key: "buyrules" as Tab, label: "🟢 Buy Rules" }, { key: "sellrules" as Tab, label: "🔴 Sell Rules" }, { key: "drawdown" as Tab, label: "🛡️ Drawdown" }, { key: "review" as Tab, label: "🔬 Trade Review" }, { key: "campaigns" as Tab, label: "📋 All Campaigns" }, { key: "add-effectiveness" as Tab, label: "➕ Add effectiveness" } ]).map(t => (
            <button key={t.key} onClick={() => { setTab(t.key); setBrDrill(""); }}
                    className="px-4 py-2 text-[12px] font-medium transition-all"
                    style={{ color: tab === t.key ? navColor : "var(--ink-4)", borderBottom: tab === t.key ? `2px solid ${navColor}` : "2px solid transparent", marginBottom: -2 }}>
              {t.label}
            </button>
          ))}
        </div>
        {(tab === "overview" || tab === "scenarios") && (
          <div className="flex items-center gap-3">
            {/* Year picker */}
            <div className="flex items-center gap-1.5">
              <label htmlFor="edge-year" className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Year</label>
              <select id="edge-year" value={year} onChange={e => setYearFilter(Number(e.target.value))}
                      className="h-[28px] px-2 rounded-md text-[11px] font-medium"
                      style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
                {yearData.years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            {/* Cohort toggle */}
            <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              {(["closed", "at-mark"] as const).map(c => (
                <button key={c} onClick={() => setCohort(c)}
                        title={c === "closed" ? "Only actually-closed campaigns count toward the stats." : "Open positions ALSO count, valued at their live batch-fetched price (marked-to-market)."}
                        className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                        style={{ background: cohort === c ? "var(--surface)" : "transparent", color: cohort === c ? "var(--ink)" : "var(--ink-4)", cursor: "help" }}>
                  {c === "closed" ? "Closed only" : "Include open @ mark"}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ═══ OVERVIEW ═══ */}
      {tab === "overview" && (
        <>
          <div className="text-[13px] mb-4" style={{ color: "var(--ink-3)" }}>The headline numbers across every closed trade — start here for a quick health check.</div>

          {/* Hero Row */}
          <div className="grid grid-cols-4 gap-3 mb-4">
            <HeroCard label="Total P&L" value={formatCurrency(s.netPl, { decimals: 0 })} sub={`${s.total} closed trades`} ok={s.netPl >= 0} />
            <HeroCard label="Win Rate" value={`${s.winRate.toFixed(1)}%`} sub={`${s.wins}W · ${s.losses}L`} ok={s.winRate >= 40} />
            <HeroCard label="Profit Factor" value={s.pf.toFixed(2)} sub={s.pf >= 1.5 ? "≥1.5 healthy" : "target ≥1.5"} ok={s.pf >= 1.5} />
            <HeroCard label="Expectancy / Trade" value={formatCurrency(s.expectancy, { decimals: 0 })} sub="avg $ per trade" ok={s.expectancy >= 0} />
          </div>

          {/* Win/Loss visual bar */}
          <div className="mb-2 flex items-center gap-3">
            <div className="flex-1 h-3 rounded-full overflow-hidden flex" style={{ background: "var(--bg)" }}>
              <div style={{ width: `${s.total > 0 ? (s.wins / s.total) * 100 : 0}%`, background: "#08a86b", transition: "width 0.8s ease" }} />
              <div style={{ width: `${s.total > 0 ? (s.losses / s.total) * 100 : 0}%`, background: "#e5484d", transition: "width 0.8s ease" }} />
            </div>
            <span className="text-[11px] font-semibold" style={{ color: "var(--ink-4)", whiteSpace: "nowrap" }}>
              {s.wins}W · {s.losses}L · {s.breakEven}BE
            </span>
          </div>

          {/* Winners vs Losers */}
          <div className="grid grid-cols-2 gap-4 mb-5">
            {[
              { title: "✅ WINNERS", color: "#08a86b", count: s.wins, avg: s.avgWin, largest: s.largestWin, hold: s.winnersHold },
              { title: "❌ LOSERS", color: "#e5484d", count: s.losses, avg: s.avgLoss, largest: s.largestLoss, hold: s.losersHold },
            ].map(side => (
              <div key={side.title} className="p-5 rounded-[14px] transition-all duration-200 hover:shadow-md"
                   style={{ background: `color-mix(in oklab, ${side.color} 6%, var(--surface))`, borderLeft: `5px solid ${side.color}`, border: "1px solid var(--border)" }}>
                <div className="text-[12px] font-bold mb-3 uppercase tracking-[0.08em]" style={{ color: side.color }}>{side.title}</div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                  {[
                    { k: "Count", v: String(side.count) },
                    { k: "Avg", v: formatCurrency(side.avg, { decimals: 0 }) },
                    { k: "Largest", v: formatCurrency(side.largest, { decimals: 0 }) },
                    { k: "Avg Hold", v: `${side.hold.toFixed(0)}d` },
                  ].map(m => (
                    <div key={m.k}>
                      <div className="text-[10px] font-medium" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                      <div className="text-[18px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono }}>{m.v}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Quality Indicators */}
          <div className="text-[13px] font-semibold mb-2">Quality Indicators</div>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <QualityTile label="Win/Loss Ratio" value={`${s.wlRatio.toFixed(2)}x`} status="≥2.0 target" ok={s.wlRatio >= 2} />
            <QualityTile label="Hold Ratio (W/L)" value={`${s.holdRatio.toFixed(2)}x`} status={s.holdRatio >= 1 ? "letting winners run" : "holding losers too long"} ok={s.holdRatio >= 1} />
            <QualityTile label="Avg Trade" value={formatCurrency(s.avgTrade, { decimals: 0 })} status={s.avgTrade >= 0 ? "positive" : "negative"} ok={s.avgTrade >= 0} />
            <QualityTile label="Avg R-Multiple" value={`${s.avgR.toFixed(2)}R`} status={`max ${s.maxR.toFixed(1)}R`} ok={s.avgR >= 1} />
          </div>

          {/* Loss Discipline — respects the selected year via the shared
              tradeWasOpenInYear helper. Per-trade NLV lookup is now the
              strict-<-priority getPriorDayNlv, which was `<=` inline
              here previously — numbers may shift SLIGHTLY stricter on
              same-day trades (using yesterday's end_nlv as the
              denominator instead of today's, which included the loss
              itself and understated impact %). */}
          {(() => {
            const yearLosses = allTrades.filter(t =>
              parseFloat(String(t.realized_pl || 0)) < 0
              && tradeWasOpenInYear(t, year)
            );
            const compliance = computeOnePctCompliance(yearLosses, journalHistory as any);
            const { passRate, withinRule, breaches, totalLosses } = compliance;
            const impactTrades = yearLosses.map(t => {
              const nlv = getPriorDayNlv(journalHistory as any, t.open_date);
              const impact = nlv && nlv > 0 ? (parseFloat(String(t.realized_pl || 0)) / nlv) * 100 : null;
              return { ticker: t.ticker, trade_id: t.trade_id, closed_date: t.closed_date, realized_pl: t.realized_pl, open_date: t.open_date, nlvAtOpen: nlv, impactPct: impact };
            }).filter(t => t.impactPct !== null);

            // Buckets
            const bucketDefs = [
              { label: "0 to −0.25%", lo: -0.25, hi: 0, color: "#08a86b", sub: "Minor nicks" },
              { label: "−0.25 to −0.50%", lo: -0.50, hi: -0.25, color: "#65a30d", sub: "Small" },
              { label: "−0.50 to −1.00%", lo: -1.00, hi: -0.50, color: "#d97706", sub: "Borderline" },
              { label: "Over −1.00%", lo: -9999, hi: -1.00, color: "#e5484d", sub: "🚨 BREACH" },
            ];

            return totalLosses > 0 ? (
              <div className="mb-4">
                <div className="text-[15px] font-semibold mb-1">🛡️ Loss Discipline <span className="text-[13px] font-normal" style={{ color: "var(--ink-4)" }}>— {year} only</span></div>

                {/* Score card */}
                <div className="p-5 rounded-[14px] mb-3 flex items-center justify-between"
                     style={{ background: `color-mix(in oklab, ${passRate >= 95 ? "#08a86b" : passRate >= 85 ? "#d97706" : "#e5484d"} 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                  <div>
                    <div className="text-[11px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>1% Rule Compliance</div>
                    <div className="text-[30px] font-extrabold mt-1" style={{
                      color: passRate >= 95 ? "#08a86b" : passRate >= 85 ? "#d97706" : "#e5484d", lineHeight: 1.1
                    }}>
                      {passRate >= 95 ? "✅" : passRate >= 85 ? "⚠️" : "🚨"} {passRate.toFixed(1)}% within rule
                    </div>
                    <div className="text-[12px] mt-1" style={{ color: "var(--ink-4)" }}>{withinRule} of {totalLosses} closed losses held under −1% account impact</div>
                  </div>
                  <div className="text-right">
                    <div className="text-[11px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Breaches</div>
                    <div className="text-[36px] font-extrabold" style={{ color: breaches > 0 ? "#e5484d" : "#08a86b", lineHeight: 1 }}>{breaches}</div>
                  </div>
                </div>

                {/* Buckets */}
                <div className="grid grid-cols-4 gap-3 mb-3">
                  {bucketDefs.map(b => {
                    const count = impactTrades.filter(t => (t.impactPct || 0) > b.lo && (t.impactPct || 0) <= b.hi).length;
                    const dollarSum = impactTrades.filter(t => (t.impactPct || 0) > b.lo && (t.impactPct || 0) <= b.hi)
                      .reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
                    return (
                      <div key={b.label} className="p-3.5 rounded-[10px]" style={{ background: `color-mix(in oklab, ${b.color} 6%, var(--surface))`, borderLeft: `4px solid ${b.color}`, border: "1px solid var(--border)" }}>
                        <div className="text-[11px] font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--ink-4)" }}>{b.label}</div>
                        <div className="text-[26px] font-extrabold mt-1">{count}</div>
                        <div className="text-[12px] font-semibold" style={{ color: b.color }}>{b.sub}</div>
                        <div className="text-[11px] mt-1 privacy-mask" style={{ color: "var(--ink-4)" }}>{formatCurrency(dollarSum, { decimals: 0 })} total</div>
                      </div>
                    );
                  })}
                </div>

                {/* Worst offenders */}
                {breaches > 0 && (
                  <details className="rounded-[10px] overflow-hidden mb-3" style={{ border: "1px solid var(--border)" }}>
                    <summary className="px-4 py-2.5 cursor-pointer text-[12px] font-semibold">⚠️ Worst {Math.min(5, breaches)} Offenders</summary>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                        <thead><tr>
                          {["Ticker", "Trade ID", "Closed", "P&L", "Impact %"].map(h => (
                            <th key={h} className="text-left px-3 py-2 text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr></thead>
                        <tbody>
                          {impactTrades.sort((a, b) => (a.impactPct || 0) - (b.impactPct || 0)).slice(0, 5).map((t, i) => (
                            <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                              <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                              <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{t.trade_id}</td>
                              <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(t.closed_date || "").slice(0, 10)}</td>
                              <td className="px-3 py-2 privacy-mask" style={{ fontFamily: mono, color: "#e5484d" }}>{formatCurrency(parseFloat(String(t.realized_pl || 0)))}</td>
                              <td className="px-3 py-2 font-bold" style={{ fontFamily: mono, color: "#e5484d" }}>{(t.impactPct || 0).toFixed(2)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                )}
              </div>
            ) : (
              <div className="mb-4 px-4 py-3 rounded-[10px] text-[12px]" style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
                No 2026 closed losses with journal NLV data yet.
              </div>
            );
          })()}

          {/* Streaks & Activity */}
          <div className="rounded-[14px] overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="grid grid-cols-5 divide-x" style={{ borderColor: "var(--border)" }}>
              {[
                { k: "Max Win Streak", v: s.maxWinStreak, color: "#08a86b" },
                { k: "Max Loss Streak", v: s.maxLossStreak, color: "#e5484d" },
                { k: "Avg Hold (all)", v: `${s.avgHoldAll.toFixed(0)}d`, color: "var(--ink)" },
                { k: "Open Positions", v: openCount, color: "var(--ink)" },
                { k: "Break-Even", v: s.breakEven, color: "var(--ink-3)" },
              ].map(m => (
                <div key={m.k} className="p-4 text-center">
                  <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                  <div className="text-[24px] font-extrabold mt-1.5" style={{ fontFamily: mono, color: m.color }}>{m.v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Monthly Performance */}
          <div className="text-[13px] font-semibold mb-3">📅 Monthly Performance</div>
          <div className="grid grid-cols-3 gap-4 mb-5">
            {[
              { k: "Best Month", v: formatCurrency(s.bestMonth, { decimals: 0 }), sub: s.bestMonthKey, color: "#08a86b", icon: "📈" },
              { k: "Worst Month", v: formatCurrency(s.worstMonth, { decimals: 0 }), sub: s.worstMonthKey, color: "#e5484d", icon: "📉" },
              { k: "Average Month", v: formatCurrency(s.avgMonth, { decimals: 0 }), sub: undefined, color: "var(--ink)", icon: "📊" },
            ].map(m => (
              <div key={m.k} className="p-5 rounded-[14px] transition-all duration-200 hover:shadow-md"
                   style={{ background: `color-mix(in oklab, ${m.color === "var(--ink)" ? "#888" : m.color} 5%, var(--surface))`, border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                <div className="text-[26px] font-extrabold mt-2 privacy-mask" style={{ fontFamily: mono, color: m.color }}>{m.v}</div>
                {m.sub && <div className="text-[12px] mt-1 font-medium" style={{ color: "var(--ink-4)" }}>{m.icon} {m.sub}</div>}
              </div>
            ))}
          </div>

          {/* How to read */}
          <details className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <summary className="px-5 py-3 cursor-pointer text-[13px] font-semibold">📖 How to read these stats</summary>
            <div className="p-5 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
              <p><strong>Hero row</strong> — your 4 most important numbers. Green = healthy.</p>
              <ul className="list-disc pl-5 mb-3">
                <li><strong>Total P&L</strong>: closed-trade realized profit</li>
                <li><strong>Win Rate</strong>: ≥40% is good for a trend-following system</li>
                <li><strong>Profit Factor</strong>: gross profit ÷ gross loss. ≥1.5 = healthy, ≥2.0 = excellent</li>
                <li><strong>Expectancy</strong>: average P&L per trade. Must be positive long-term</li>
              </ul>
              <p><strong>Winners vs Losers</strong> — symmetric breakdown. Look for:</p>
              <ul className="list-disc pl-5 mb-3">
                <li>Avg win <strong>bigger</strong> than avg loss (confirms W/L ratio)</li>
                <li>Avg hold on winners <strong>longer</strong> than on losers (confirms you cut losses fast)</li>
              </ul>
              <p><strong>Quality Indicators</strong></p>
              <ul className="list-disc pl-5 mb-3">
                <li><strong>W/L Ratio ≥2.0x</strong>: you make $2+ for every $1 lost per trade on average</li>
                <li><strong>Hold Ratio ≥1.0x</strong>: you hold winners longer than losers (discipline)</li>
                <li><strong>Avg R-Multiple</strong>: actual reward for each unit of risk taken</li>
              </ul>
              <p><strong>Streaks</strong>: max consecutive wins/losses show variance — use to gut-check position sizing.</p>
              <p><strong>Monthly Performance</strong>: gives context on seasonality and consistency.</p>
            </div>
          </details>

          {/* ═══ EDGE REPORT — new cards ═══ */}
          <EdgeCards trades={trades} journal={journalHistory as any} mctStates={mctStates} year={year} cohort={cohort} />
        </>
      )}

      {/* ═══ SCENARIOS ═══ */}
      {tab === "scenarios" && (
        <ScenariosTab trades={trades} journal={journalHistory as any} year={year} cohort={cohort} navColor={navColor} />
      )}

      {/* ═══ BUY RULES ═══ */}
      {tab === "buyrules" && (
        <>
          <div className="flex items-start justify-between mb-1 gap-3">
            <div>
              <div className="text-[14px] font-semibold">🟢 Buy Rules — What{"'"}s Working in 2026</div>
              <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>Study your entry rules. Sort by the metric you care about, click any rule to drill into individual trades.</div>
            </div>
            <div className="flex p-0.5 rounded-[8px] gap-0.5 shrink-0" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              {(["all", "closed", "open"] as const).map(st => (
                <button key={st} onClick={() => setRuleStatus(st)}
                        className="px-3 py-1 rounded-md text-[11px] font-medium transition-all capitalize"
                        style={{ background: ruleStatus === st ? "var(--surface)" : "transparent", color: ruleStatus === st ? "var(--ink)" : "var(--ink-4)" }}>
                  {st}
                </button>
              ))}
            </div>
          </div>
          <div className="mb-4" />

          {ruleStats.length > 0 ? (
            <>
              {/* Insight cards */}
              {(() => {
                const profitable = ruleStats.filter(r => r.totalPl > 0);
                const unprofitable = ruleStats.filter(r => r.totalPl < 0);
                const best = profitable[0];
                const worst = [...ruleStats].sort((a, b) => a.totalPl - b.totalPl)[0];
                return (
                  <div className="grid grid-cols-3 gap-3 mb-5">
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #08a86b 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#08a86b" }}>💰 Best Rule</div>
                      <div className="text-[14px] font-bold mt-1">{best?.rule || "—"}</div>
                      {best && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#08a86b" }}>{formatCurrency(best.totalPl, { decimals: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{best.count} trades · {best.winRate.toFixed(0)}% win rate</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #e5484d 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#e5484d" }}>🚨 Worst Rule</div>
                      <div className="text-[14px] font-bold mt-1">{worst?.rule || "—"}</div>
                      {worst && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#e5484d" }}>{formatCurrency(worst.totalPl, { decimals: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{worst.count} trades · {worst.winRate.toFixed(0)}% win rate</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #3b82f6 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#3b82f6" }}>📊 Rules Used</div>
                      <div className="text-[28px] font-extrabold mt-1">{ruleStats.length}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{profitable.length} profitable · {unprofitable.length} losing</div>
                    </div>
                  </div>
                );
              })()}

              {/* P&L by Rule — pure CSS bar chart */}
              <div className="rounded-[14px] overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="px-5 py-3 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="text-[13px] font-semibold">P&L by Buy Rule</span>
                  <select value={brSort} onChange={e => setBrSort(e.target.value)}
                          className="h-[30px] px-2.5 rounded-[8px] text-[11px]"
                          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                    {["Total P&L", "Win Rate %", "Avg P&L", "Trades"].map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <div className="px-5 py-4">
                  {(() => {
                    const maxAbs = Math.max(...ruleStats.map(r => Math.abs(r.totalPl)), 1);
                    return ruleStats.map((r, i) => {
                      const pct = (Math.abs(r.totalPl) / maxAbs) * 100;
                      const isPos = r.totalPl >= 0;
                      const selected = brDrill === r.rule;
                      return (
                        <div key={i} className="flex items-center gap-3 py-[6px] cursor-pointer transition-all rounded-[6px] px-1 -mx-1"
                             style={{ background: selected ? `color-mix(in oklab, ${navColor} 8%, var(--surface))` : "transparent" }}
                             onClick={() => setBrDrill(selected ? "" : r.rule)}>
                          <span className="text-[11px] font-medium text-right shrink-0" style={{ width: 130, color: "var(--ink-3)" }}>
                            {r.rule.replace(/^br\d+\.\d+ /, "")}
                          </span>
                          <div className="flex-1 flex items-center" style={{ height: 20 }}>
                            {isPos ? (
                              <div className="h-full rounded-r-[4px] transition-all duration-500"
                                   style={{ width: `${pct}%`, background: "#08a86b", minWidth: pct > 0 ? 3 : 0 }} />
                            ) : (
                              <div className="h-full rounded-l-[4px] transition-all duration-500 ml-auto"
                                   style={{ width: `${pct}%`, background: "#e5484d", minWidth: pct > 0 ? 3 : 0 }} />
                            )}
                          </div>
                          <span className="text-[11px] font-bold shrink-0 privacy-mask" style={{ width: 65, textAlign: "right", fontFamily: mono, color: isPos ? "#08a86b" : "#e5484d" }}>
                            ${r.totalPl >= 0 ? "" : ""}{(r.totalPl / 1000).toFixed(1)}k
                          </span>
                        </div>
                      );
                    });
                  })()}
                </div>
              </div>
              {ruleStatus !== "closed" && (
                <div className="text-[11px] mb-4 -mt-3 px-1" style={{ color: "var(--ink-4)" }}>
                  Open positions marked to latest price — figures are live and unrealized.
                </div>
              )}

              {/* Drill-down — two columns: stats left, trades right.
                  effectivePl(t) keeps the panel honest under the All / Open
                  toggle: open rows use overall_pl (marked to latest price),
                  closed rows use realized_pl. Same denominator (risk_budget)
                  for R, matching the unified ternary in the All Campaigns
                  table at [L1683-1686] and in ruleStats above. */}
              {brDrill && (() => {
                const rs = ruleStats.find(r => r.rule === brDrill);
                const rt = rs?.trades || [];
                const effectivePl = (t: TradePosition) => {
                  const isOpen = (t.status || "").toUpperCase() === "OPEN";
                  return isOpen
                    ? (enrichedById[t.trade_id]?.overall_pl ?? 0)
                    : parseFloat(String(t.realized_pl || 0));
                };
                const wins = rt.filter(t => effectivePl(t) > 0);
                const losses = rt.filter(t => effectivePl(t) < 0);
                const totalPl = rt.reduce((a, t) => a + effectivePl(t), 0);
                const avgPl = rt.length > 0 ? totalPl / rt.length : 0;
                const winRate = rt.length > 0 ? (wins.length / rt.length) * 100 : 0;
                const grossW = wins.reduce((a, t) => a + effectivePl(t), 0);
                const grossL = Math.abs(losses.reduce((a, t) => a + effectivePl(t), 0));
                const pf = grossL > 0 ? grossW / grossL : 0;
                const avgR = rs?.avgR;

                return (
                  <div className="grid grid-cols-2 gap-4 mb-5" style={{ alignItems: "start" }}>
                    {/* Left: Rule stats */}
                    <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="px-5 py-3 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                        <span className="text-[13px] font-semibold">{brDrill}</span>
                        <button onClick={() => setBrDrill("")} className="text-[11px] font-medium" style={{ color: "var(--ink-4)" }}>Close ×</button>
                      </div>
                      <div className="p-5">
                        <div className="grid grid-cols-2 gap-3 mb-4">
                          {[
                            { k: "Total P&L", v: formatCurrency(totalPl, { decimals: 0 }), color: pctColor(totalPl) },
                            { k: "Trades", v: String(rt.length) },
                            { k: "Win Rate", v: `${winRate.toFixed(0)}%`, color: winRate >= 50 ? "#08a86b" : "#e5484d" },
                            { k: "Profit Factor", v: pf.toFixed(2) },
                            { k: "Avg P&L", v: formatCurrency(avgPl, { decimals: 0 }), color: pctColor(avgPl) },
                            { k: "Avg R", v: avgR != null ? `${avgR.toFixed(2)}R` : "—" },
                            { k: "Winners", v: String(wins.length), color: "#08a86b" },
                            { k: "Losers", v: String(losses.length), color: "#e5484d" },
                          ].map(m => (
                            <div key={m.k} className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                              <div className="text-[9px] uppercase tracking-[0.06em] font-bold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                              <div className="text-[18px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono, color: (m as any).color || "var(--ink)" }}>{m.v}</div>
                            </div>
                          ))}
                        </div>

                        {/* Rule Observations */}
                        <div className="pt-3" style={{ borderTop: "1px solid var(--border)" }}>
                          <div className="text-[12px] font-semibold mb-2">📝 Observations</div>
                          <div className="flex gap-2 mb-2">
                            <select value={brNoteStatus} onChange={e => setBrNoteStatus(e.target.value)}
                                    className="h-[32px] px-2.5 rounded-[8px] text-[11px]"
                                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                              {["— no status —", "✅ Validated", "✏️ Modify", "⚠️ Review", "🛑 Avoid"].map(o => <option key={o} value={o}>{o}</option>)}
                            </select>
                          </div>
                          <textarea value={brNoteText} onChange={e => setBrNoteText(e.target.value)} rows={3}
                                    placeholder="What's working or not working?"
                                    className="w-full px-3 py-2 rounded-[8px] text-[11px] outline-none resize-none mb-2"
                                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                          <button onClick={() => alert("Backend endpoint needed")}
                                  className="h-[30px] px-4 rounded-[8px] text-[11px] font-semibold"
                                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                            Save
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Right: Trade list */}
                    <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="px-5 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
                        Trades — {rt.length}
                      </div>
                      <div className="overflow-y-auto" style={{ maxHeight: 400 }}>
                        <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                          <thead><tr>
                            {["Ticker", "Opened", "Closed", "P&L", "R"].map(h => (
                              <th key={h} className="text-left px-3 py-2 text-[9px] uppercase font-semibold sticky top-0"
                                  style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                            ))}
                          </tr></thead>
                          <tbody>{[...rt].sort((a, b) => effectivePl(b) - effectivePl(a)).map((t, i) => {
                            const isOpen = (t.status || "").toUpperCase() === "OPEN";
                            const pl = effectivePl(t);
                            const rb = parseFloat(String(t.risk_budget || 0));
                            const rMult = rb > 0 ? pl / rb : null;
                            const closedCell = isOpen || !t.closed_date ? "—" : String(t.closed_date).slice(5, 10);
                            return (
                              <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                                <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                                <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(t.open_date || "").slice(5, 10)}</td>
                                <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{closedCell}</td>
                                <td className="px-3 py-2 font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>{formatCurrency(pl, { decimals: 0 })}</td>
                                <td className="px-3 py-2" style={{ fontFamily: mono }}>{rMult != null ? `${rMult.toFixed(2)}R` : "—"}</td>
                              </tr>
                            );
                          })}</tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </>
          ) : (
            <div className="px-4 py-3 rounded-[10px] text-[12px]" style={{ background: "var(--bg)", color: "var(--ink-4)" }}>
              No 2026 closed trades with buy rule data yet.
            </div>
          )}
        </>
      )}

      {/* ═══ SELL RULES ═══ */}
      {tab === "sellrules" && (
        <>
          <div className="text-[14px] font-semibold mb-1">🔴 Sell Rules — Exit Quality in 2026</div>
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>Which are protecting capital, which are capturing profits, which are hurting you.</div>
          <div className="text-[11px] mb-4 mt-1" style={{ color: "var(--ink-4)" }}>Closed only — open positions haven{"'"}t sold yet.</div>

          {sellRuleStats.length > 0 ? (
            <>
              {/* Insight cards */}
              {(() => {
                const negRules = sellRuleStats.filter(r => r.avgPl < 0).sort((a, b) => b.avgPl - a.avgPl);
                const posRules = sellRuleStats.filter(r => r.avgPl > 0).sort((a, b) => b.totalPl - a.totalPl);
                const mostUsed = [...sellRuleStats].sort((a, b) => b.count - a.count)[0];
                return (
                  <div className="grid grid-cols-3 gap-3 mb-5">
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #08a86b 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#08a86b" }}>🛡️ Best Protector</div>
                      <div className="text-[14px] font-bold mt-1">{negRules[0]?.rule || "No losing exits yet"}</div>
                      {negRules[0] && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#08a86b" }}>{formatCurrency(negRules[0].avgPl, { decimals: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>smallest avg loss · {negRules[0].count} uses</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #3b82f6 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#3b82f6" }}>💰 Top Profit Capture</div>
                      <div className="text-[14px] font-bold mt-1">{posRules[0]?.rule || "No winning exits yet"}</div>
                      {posRules[0] && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#3b82f6" }}>{formatCurrency(posRules[0].totalPl, { decimals: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>avg {formatCurrency(posRules[0].avgPl, { decimals: 0 })} · {posRules[0].count} uses</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #64748b 6%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-3)" }}>📊 Most Used Exit</div>
                      <div className="text-[14px] font-bold mt-1">{mostUsed?.rule || "—"}</div>
                      {mostUsed && <><div className="text-[28px] font-extrabold">{mostUsed.count} uses</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>avg {formatCurrency(mostUsed.avgPl, { decimals: 0 })}</div></>}
                    </div>
                  </div>
                );
              })()}

              {/* P&L by Sell Rule — CSS bar chart */}
              <div className="rounded-[14px] overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="px-5 py-3 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="text-[13px] font-semibold">P&L by Sell Rule</span>
                  <select value={srSort} onChange={e => setSrSort(e.target.value)}
                          className="h-[30px] px-2.5 rounded-[8px] text-[11px]"
                          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                    {["Total P&L", "Uses", "Avg P&L", "Winners %"].map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
                <div className="px-5 py-4">
                  {(() => {
                    const maxAbs = Math.max(...sellRuleStats.map(r => Math.abs(r.totalPl)), 1);
                    return sellRuleStats.map((r, i) => {
                      const pct = (Math.abs(r.totalPl) / maxAbs) * 100;
                      const isPos = r.totalPl >= 0;
                      const selected = srDrill === r.rule;
                      return (
                        <div key={i} className="flex items-center gap-3 py-[6px] cursor-pointer transition-all rounded-[6px] px-1 -mx-1"
                             style={{ background: selected ? `color-mix(in oklab, ${navColor} 8%, var(--surface))` : "transparent" }}
                             onClick={() => setSrDrill(selected ? "" : r.rule)}>
                          <span className="text-[11px] font-medium text-right shrink-0" style={{ width: 140, color: "var(--ink-3)" }}>
                            {r.rule.replace(/^sr\d+ /, "")}
                          </span>
                          <div className="flex-1 flex items-center" style={{ height: 20 }}>
                            {isPos ? (
                              <div className="h-full rounded-r-[4px] transition-all duration-500"
                                   style={{ width: `${pct}%`, background: "#08a86b", minWidth: pct > 0 ? 3 : 0 }} />
                            ) : (
                              <div className="h-full rounded-l-[4px] transition-all duration-500 ml-auto"
                                   style={{ width: `${pct}%`, background: "#e5484d", minWidth: pct > 0 ? 3 : 0 }} />
                            )}
                          </div>
                          <span className="text-[11px] font-bold shrink-0 privacy-mask" style={{ width: 65, textAlign: "right", fontFamily: mono, color: isPos ? "#08a86b" : "#e5484d" }}>
                            {formatCurrency(r.totalPl, { compact: true })}
                          </span>
                        </div>
                      );
                    });
                  })()}
                </div>
              </div>

              {/* Drill-down */}
              {srDrill && (() => {
                const rs = sellRuleStats.find(r => r.rule === srDrill);
                const rt = rs?.trades || [];
                const wins = rt.filter(t => parseFloat(String(t.realized_pl || 0)) > 0);
                const losses = rt.filter(t => parseFloat(String(t.realized_pl || 0)) < 0);
                const totalPl = rt.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
                const avgPl = rt.length > 0 ? totalPl / rt.length : 0;
                const winRate = rt.length > 0 ? (wins.length / rt.length) * 100 : 0;
                const grossW = wins.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
                const grossL = Math.abs(losses.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0));
                const pf = grossL > 0 ? grossW / grossL : 0;

                // Status badge
                let statusLabel: string;
                let statusColor: string;
                if (avgPl > 0) { statusLabel = "💰 Capturing"; statusColor = "#08a86b"; }
                else if (rs?.avgR != null && rs.avgR < -1.0) { statusLabel = "🚨 Hurting"; statusColor = "#e5484d"; }
                else if (avgPl < 0) { statusLabel = "🛡️ Protecting"; statusColor = "#08a86b"; }
                else { statusLabel = "— Flat"; statusColor = "var(--ink-4)"; }

                return (
                  <div className="grid grid-cols-2 gap-4 mb-5" style={{ alignItems: "start" }}>
                    {/* Left: Stats */}
                    <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="px-5 py-3 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                        <span className="text-[13px] font-semibold">{srDrill}</span>
                        <button onClick={() => setSrDrill("")} className="text-[11px] font-medium" style={{ color: "var(--ink-4)" }}>Close ×</button>
                      </div>
                      <div className="p-5">
                        {/* Status badge */}
                        <div className="mb-3">
                          <span className="text-[11px] px-2.5 py-1 rounded-[6px] font-bold" style={{
                            background: `color-mix(in oklab, ${statusColor} 10%, var(--surface))`, color: statusColor,
                          }}>{statusLabel}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 mb-4">
                          {[
                            { k: "Total P&L", v: formatCurrency(totalPl, { decimals: 0 }), color: pctColor(totalPl) },
                            { k: "Uses", v: String(rt.length) },
                            { k: "Win Rate", v: `${winRate.toFixed(0)}%`, color: winRate >= 50 ? "#08a86b" : "#e5484d" },
                            { k: "Profit Factor", v: pf.toFixed(2) },
                            { k: "Avg P&L", v: formatCurrency(avgPl, { decimals: 0 }), color: pctColor(avgPl) },
                            { k: "Avg R", v: rs?.avgR != null ? `${rs.avgR.toFixed(2)}R` : "—" },
                            { k: "Avg Hold", v: rs?.avgHold != null ? `${rs.avgHold.toFixed(0)}d` : "—" },
                            { k: "Winners", v: `${wins.length}W / ${losses.length}L` },
                          ].map(m => (
                            <div key={m.k} className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                              <div className="text-[9px] uppercase tracking-[0.06em] font-bold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                              <div className="text-[18px] font-bold mt-0.5 privacy-mask" style={{ fontFamily: mono, color: (m as any).color || "var(--ink)" }}>{m.v}</div>
                            </div>
                          ))}
                        </div>
                        {/* Observations */}
                        <div className="pt-3" style={{ borderTop: "1px solid var(--border)" }}>
                          <div className="text-[12px] font-semibold mb-2">📝 Observations</div>
                          <select value={srNoteStatus} onChange={e => setSrNoteStatus(e.target.value)}
                                  className="h-[32px] px-2.5 rounded-[8px] text-[11px] mb-2"
                                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                            {["— no status —", "✅ Validated", "✏️ Modify", "⚠️ Review", "🛑 Avoid"].map(o => <option key={o} value={o}>{o}</option>)}
                          </select>
                          <textarea value={srNoteText} onChange={e => setSrNoteText(e.target.value)} rows={3}
                                    placeholder="Is this exit protecting capital or cutting winners short?"
                                    className="w-full px-3 py-2 rounded-[8px] text-[11px] outline-none resize-none mb-2"
                                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                          <button onClick={() => alert("Backend endpoint needed")}
                                  className="h-[30px] px-4 rounded-[8px] text-[11px] font-semibold"
                                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                            Save
                          </button>
                        </div>
                      </div>
                    </div>
                    {/* Right: Trades */}
                    <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="px-5 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Exits — {rt.length}</div>
                      <div className="overflow-y-auto" style={{ maxHeight: 400 }}>
                        <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                          <thead><tr>
                            {["Ticker", "Opened", "Closed", "P&L", "R", "Hold"].map(h => (
                              <th key={h} className="text-left px-3 py-2 text-[9px] uppercase font-semibold sticky top-0"
                                  style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                            ))}
                          </tr></thead>
                          <tbody>{[...rt].sort((a, b) => parseFloat(String(b.realized_pl || 0)) - parseFloat(String(a.realized_pl || 0))).map((t, i) => {
                            const pl = parseFloat(String(t.realized_pl || 0));
                            const rb = parseFloat(String(t.risk_budget || 0));
                            const rMult = rb > 0 ? pl / rb : null;
                            const oStr = String(t.open_date || "").trim();
                            const cStr = String(t.closed_date || "").trim();
                            const hold = (oStr && cStr) ? Math.max(0, Math.floor((new Date(cStr).getTime() - new Date(oStr).getTime()) / 86400000)) : null;
                            return (
                              <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                                <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                                <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{oStr.slice(5, 10)}</td>
                                <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{cStr.slice(5, 10)}</td>
                                <td className="px-3 py-2 font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>{formatCurrency(pl, { decimals: 0 })}</td>
                                <td className="px-3 py-2" style={{ fontFamily: mono }}>{rMult != null ? `${rMult.toFixed(2)}R` : "—"}</td>
                                <td className="px-3 py-2" style={{ fontFamily: mono, color: "var(--ink-4)" }}>{hold != null ? `${hold}d` : "—"}</td>
                              </tr>
                            );
                          })}</tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </>
          ) : (
            <div className="px-4 py-3 rounded-[10px] text-[12px]" style={{ background: "var(--bg)", color: "var(--ink-4)" }}>
              No 2026 closed trades with sell rule data yet.
            </div>
          )}
        </>
      )}

      {/* ═══ DRAWDOWN DISCIPLINE ═══ */}
      {tab === "drawdown" && (() => {
        // === FULL DRAWDOWN DISCIPLINE (matching Streamlit) ===
        const DECKS = [
          { key: "L1", pct: 7.5, action: "Remove margin", color: "#f59f00" },
          { key: "L2", pct: 12.5, action: "Max 30% invested", color: "#f97316" },
          { key: "L3", pct: 15.0, action: "Go to cash", color: "#e5484d" },
        ];
        const jSorted = [...journalHistory].sort((a: any, b: any) => String(a.day).localeCompare(String(b.day)));
        let _peak = 0;
        const ddSeries = jSorted.map((h: any) => { if (h.end_nlv > _peak) _peak = h.end_nlv; return { day: h.day, nlv: h.end_nlv, peak: _peak, ddPct: _peak > 0 ? ((h.end_nlv - _peak) / _peak) * 100 : 0, exposure: h.pct_invested || 0 }; });
        const curr = ddSeries.length > 0 ? ddSeries[ddSeries.length - 1] : { ddPct: 0, nlv: 0, peak: 0, exposure: 0 };

        // Detect crossings
        const crossings: any[] = [];
        for (const deck of DECKS) {
          const thresh = -deck.pct;
          let inB = false, sIdx = 0;
          for (let i = 0; i < ddSeries.length; i++) {
            if (ddSeries[i].ddPct <= thresh && !inB) { inB = true; sIdx = i; }
            if ((ddSeries[i].ddPct > thresh || i === ddSeries.length - 1) && inB) {
              const eIdx = ddSeries[i].ddPct > thresh ? i - 1 : i;
              const grp = ddSeries.slice(sIdx, eIdx + 1);
              const maxD = Math.min(...grp.map(g => g.ddPct));
              const mdi = grp.findIndex(g => g.ddPct === maxD) + sIdx;
              const exS = sIdx > 0 ? ddSeries[sIdx - 1].exposure : ddSeries[sIdx].exposure;
              const exT = ddSeries[mdi].exposure;
              const pAS = ddSeries[sIdx].peak;
              let rec: number | null = null;
              for (let k = eIdx + 1; k < ddSeries.length; k++) { if (ddSeries[k].nlv >= pAS) { rec = Math.floor((new Date(ddSeries[k].day).getTime() - new Date(ddSeries[sIdx].day).getTime()) / 86400000); break; } }
              const sd = String(ddSeries[sIdx].day).slice(0, 10);
              const ed = String(ddSeries[eIdx].day).slice(0, 10);
              const wL = allTrades.filter(t => { const cd = String(t.closed_date || "").slice(0, 10); return cd >= sd && cd <= ed && parseFloat(String(t.realized_pl || 0)) < 0; }).reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
              const drop = exS - exT;
              let v = "";
              if (deck.key === "L1") v = exT <= exS ? "L1_Aware" : (exT - exS) < 10 ? "L1_Drifted" : "L1_Leveraged";
              else if (deck.key === "L2") v = (drop >= 20 || exT <= 50) ? "L2_Reducing" : drop >= 5 ? "L2_Partial" : "L2_NotReducing";
              else v = exT <= 20 ? "L3_Exited" : exT <= 50 ? "L3_PartialExit" : "L3_StillIn";
              crossings.push({ deck: deck.key, thresh, maxDepth: maxD, expStart: exS, expTrough: exT, recoveryDays: rec, lossesInWindow: wL, verdict: v, startDay: sd });
              inB = false;
            }
          }
        }
        const l2l3 = crossings.filter(c => c.deck !== "L1").sort((a: any, b: any) => b.startDay.localeCompare(a.startDay));
        const vs: Record<string, { color: string; label: string }> = {
          L1_Aware: { color: "#08a86b", label: "✅ Aware" }, L1_Drifted: { color: "#d97706", label: "⚠️ Drifted" }, L1_Leveraged: { color: "#e5484d", label: "🚨 Leveraged" },
          L2_Reducing: { color: "#08a86b", label: "✅ Reducing" }, L2_Partial: { color: "#d97706", label: "⚠️ Partial" }, L2_NotReducing: { color: "#e5484d", label: "🚨 Not Reducing" },
          L3_Exited: { color: "#08a86b", label: "✅ Exited" }, L3_PartialExit: { color: "#d97706", label: "⚠️ Partial Exit" }, L3_StillIn: { color: "#e5484d", label: "🚨 Still In" },
        };
        const bestC = l2l3.filter(c => ["L2_Reducing", "L3_Exited"].includes(c.verdict));
        const partC = l2l3.filter(c => ["L2_Partial", "L3_PartialExit"].includes(c.verdict));
        const worstC = l2l3.filter(c => ["L2_NotReducing", "L3_StillIn"].includes(c.verdict));
        const dW: Record<string, number> = { L1: 0, L2: 1, L3: 3 };
        const vS: Record<string, number> = { L1_Aware: 1, L1_Drifted: 0.5, L1_Leveraged: 0, L2_Reducing: 1, L2_Partial: 0.5, L2_NotReducing: 0, L3_Exited: 1, L3_PartialExit: 0.5, L3_StillIn: 0 };
        const tW = crossings.reduce((a: number, c: any) => a + (dW[c.deck] || 0), 0);
        const wS = crossings.reduce((a: number, c: any) => a + (dW[c.deck] || 0) * (vS[c.verdict] || 0), 0);
        const cPct = tW > 0 ? (wS / tW) * 100 : 100;
        const grd = cPct >= 90 ? "A" : cPct >= 75 ? "B" : cPct >= 60 ? "C" : cPct >= 40 ? "D" : "F";
        const gC = grd <= "B" ? "#08a86b" : grd <= "C" ? "#d97706" : "#e5484d";

        return (
          <>
            <div className="text-[14px] font-semibold mb-1">🛡️ Drawdown Discipline</div>
            <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>Each historical crossing of L1 (−7.5%), L2 (−12.5%), or L3 (−15.0%) is logged with a pass/fail verdict.</div>

            {/* Live Status */}
            <div className="text-[13px] font-semibold mb-2">Live Status</div>
            <div className="grid grid-cols-3 gap-3 mb-3">
              {DECKS.map(d => {
                const t = -d.pct; const dist = curr.ddPct - t; const br = curr.ddPct <= t; const cl = !br && dist < 2;
                const sc = br ? "#e5484d" : cl ? "#d97706" : "#08a86b";
                return (
                  <div key={d.key} className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, ${sc} 6%, var(--surface))`, borderLeft: `4px solid ${d.color}`, border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>{d.key} · {t.toFixed(1)}%</div>
                    <div className="text-[12px] font-semibold mt-0.5">{d.action}</div>
                    <div className="text-[18px] font-extrabold mt-1" style={{ color: sc }}>{br ? "🚨 BREACHED" : cl ? "⚠️ Close" : "✅ Safe"}</div>
                    <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{br ? `${Math.abs(dist).toFixed(2)}% into breach` : `${dist.toFixed(2)}% from deck`}</div>
                  </div>
                );
              })}
            </div>
            <div className="text-[12px] mb-5 px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", color: "var(--ink-3)" }}>
              Current DD: <strong>{curr.ddPct.toFixed(2)}%</strong> · NLV: <strong className="privacy-mask">{formatCurrency(curr.nlv, { decimals: 0 })}</strong> · Peak: <strong className="privacy-mask">{formatCurrency(curr.peak, { decimals: 0 })}</strong> · Exposure: <strong>{curr.exposure.toFixed(1)}%</strong>
            </div>

            {/* Crossings Log */}
            <div className="text-[13px] font-semibold mb-2">Deck Crossings Log <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>— L2 & L3 only</span></div>
            {l2l3.length === 0 ? (
              <div className="mb-5 px-4 py-3 rounded-[10px] text-[12px]" style={{ background: `color-mix(in oklab, #08a86b 6%, var(--surface))`, color: "#08a86b", border: "1px solid var(--border)" }}>No L2 or L3 crossings. Keep it up.</div>
            ) : (
              <div className="flex flex-col gap-3 mb-5">
                {l2l3.map((c: any, i: number) => {
                  const v = vs[c.verdict] || { color: "var(--ink-4)", label: c.verdict };
                  return (
                    <div key={i} className="rounded-[12px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                        <div className="text-[14px] font-bold">{c.deck} · {c.thresh.toFixed(1)}% <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>({c.startDay})</span></div>
                        <span className="text-[11px] px-2.5 py-1 rounded-[6px] font-bold" style={{ background: `color-mix(in oklab, ${v.color} 10%, var(--surface))`, color: v.color }}>{v.label}</span>
                      </div>
                      <div className="grid grid-cols-4 gap-3 px-4 py-3">
                        <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Max Depth</div><div className="text-[15px] font-bold" style={{ color: "#e5484d" }}>{c.maxDepth.toFixed(2)}%</div></div>
                        <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Exposure</div><div className="text-[15px] font-bold">{c.expStart.toFixed(0)}% → {c.expTrough.toFixed(0)}%</div><div className="text-[10px]" style={{ color: "var(--ink-4)" }}>Δ {(c.expStart - c.expTrough).toFixed(0)}pp</div></div>
                        <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Recovery</div><div className="text-[15px] font-bold">{c.recoveryDays != null ? `${c.recoveryDays}d` : "ongoing"}</div></div>
                        <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Realized</div><div className="text-[15px] font-bold privacy-mask" style={{ color: "#e5484d" }}>{formatCurrency(c.lossesInWindow, { decimals: 0 })}</div></div>
                      </div>
                      {/* Lessons & notes */}
                      <details className="mx-4 mb-3">
                        <summary className="text-[12px] font-medium cursor-pointer px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                          📝 Lessons & notes — {c.deck} {c.startDay}
                        </summary>
                        <div className="mt-2 p-3 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                          <div className="text-[11px] font-medium mb-1.5" style={{ color: "var(--ink-3)" }}>What happened? What would you do differently?</div>
                          <textarea rows={3} placeholder="e.g. I held through L1 because I thought the market would bounce — instead it kept falling."
                                    className="w-full px-3 py-2 rounded-[8px] text-[11px] outline-none resize-none mb-2"
                                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                          <button onClick={() => alert("Backend endpoint needed: POST /api/analytics/drawdown-note")}
                                  className="h-[30px] px-4 rounded-[8px] text-[11px] font-semibold"
                                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                            Save note
                          </button>
                        </div>
                      </details>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Cost of Non-Compliance + Report Card */}
            {l2l3.length > 0 && (
              <>
                <div className="text-[13px] font-semibold mb-2">Cost of Non-Compliance</div>
                <div className="grid grid-cols-3 gap-3 mb-5">
                  {[
                    { label: "🚨 Non-Compliance", data: worstC, color: "#e5484d" },
                    { label: "⚠️ Partial", data: partC, color: "#d97706" },
                    { label: "✅ Rule-Respected", data: bestC, color: "#08a86b" },
                  ].map(g => (
                    <div key={g.label} className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, ${g.color} 6%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase font-bold" style={{ color: g.color }}>{g.label}</div>
                      <div className="text-[24px] font-extrabold mt-1 privacy-mask" style={{ color: g.color }}>{formatCurrency(Math.abs(g.data.reduce((a: number, c: any) => a + c.lossesInWindow, 0)), { decimals: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{g.data.length} crossing(s)</div>
                    </div>
                  ))}
                </div>

                <div className="text-[13px] font-semibold mb-2">Discipline Report Card</div>
                <div className="grid grid-cols-[1fr_2fr] gap-4">
                  <div className="p-5 rounded-[14px] text-center" style={{ background: `color-mix(in oklab, ${gC} 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                    <div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Behavior Grade</div>
                    <div className="text-[72px] font-black" style={{ color: gC, lineHeight: 1 }}>{grd}</div>
                    <div className="text-[13px] font-semibold" style={{ color: gC }}>{cPct.toFixed(0)}% compliance</div>
                  </div>
                  <div className="p-5 rounded-[14px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                    <div className="grid grid-cols-2 gap-4">
                      <div><div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Total Crossings</div><div className="text-[22px] font-extrabold">{crossings.length}</div></div>
                      <div><div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg Depth</div><div className="text-[22px] font-extrabold" style={{ color: "#e5484d" }}>{l2l3.length > 0 ? (l2l3.reduce((a: number, c: any) => a + c.maxDepth, 0) / l2l3.length).toFixed(2) : 0}%</div></div>
                      <div><div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg Recovery</div><div className="text-[22px] font-extrabold">{(() => { const r = l2l3.filter((c: any) => c.recoveryDays != null); return r.length > 0 ? `${(r.reduce((a: number, c: any) => a + c.recoveryDays, 0) / r.length).toFixed(0)}d` : "—"; })()}</div></div>
                      <div><div className="text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Breakdown</div><div className="text-[13px] font-bold mt-1"><span style={{ color: "#08a86b" }}>{bestC.length} ✅</span> · <span style={{ color: "#d97706" }}>{partC.length} ⚠️</span> · <span style={{ color: "#e5484d" }}>{worstC.length} 🚨</span></div></div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </>
        );
      })()}

      {/* ═══ TRADE REVIEW ═══ */}
      {tab === "review" && (() => {
        // Filter by time range
        const trFiltered = allTrades.filter(t => {
          const cd = String(t.closed_date || "").slice(0, 10);
          if (!cd) return false;
          if (trRange === "2026 YTD") return cd.startsWith("2026");
          if (trRange === "Last 30 days") return cd >= new Date(Date.now() - 30 * 86400000).toISOString().slice(0, 10);
          if (trRange === "Last 90 days") return cd >= new Date(Date.now() - 90 * 86400000).toISOString().slice(0, 10);
          return true;
        });

        const topWinners = [...trFiltered].filter(t => parseFloat(String(t.realized_pl || 0)) > 0)
          .sort((a, b) => parseFloat(String(b.realized_pl || 0)) - parseFloat(String(a.realized_pl || 0))).slice(0, topN);
        const worstLosers = [...trFiltered].filter(t => parseFloat(String(t.realized_pl || 0)) < 0)
          .sort((a, b) => parseFloat(String(a.realized_pl || 0)) - parseFloat(String(b.realized_pl || 0))).slice(0, topN);

        // Deep-link target. Looked up against the FULL closed-trade list
        // (not trFiltered) so explicit user nav overrides the date filter.
        // If the target sits inside Top/Worst we just auto-open + scroll
        // the existing card; otherwise we render an extra "Selected
        // Campaign" section above the lists.
        const selectedTrade = selectedTradeId
          ? allTrades.find(t => t.trade_id === selectedTradeId) ?? null
          : null;
        const selectedInLists = !!selectedTrade && (
          topWinners.some(t => t.trade_id === selectedTrade.trade_id) ||
          worstLosers.some(t => t.trade_id === selectedTrade.trade_id)
        );
        const showSelectedSection = !!selectedTrade && !selectedInLists;
        if (selectedTradeId && !selectedTrade) {
          log.warn("analytics", "trade not found in loaded closed trades", {
            trade_id: selectedTradeId,
          });
        }

        // Ref callback wired to the targeted card. Fires once when the
        // element mounts (after trades load and the card renders),
        // scrolls it into view, then no-ops on subsequent renders.
        const onSelectedCardRef = (el: HTMLDivElement | null) => {
          if (el && selectedTradeId) {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        };

        const TradeCard = ({ rank, t, isWinner, autoOpenLesson, refCallback }: {
          rank: number | null;
          t: TradePosition;
          isWinner: boolean;
          autoOpenLesson?: boolean;
          refCallback?: (el: HTMLDivElement | null) => void;
        }) => {
          const pl = parseFloat(String(t.realized_pl || 0));
          const ret = parseFloat(String(t.return_pct || 0));
          const rb = parseFloat(String(t.risk_budget || 0));
          const rMult = rb > 0 ? pl / rb : null;
          const oStr = String(t.open_date || "").trim(); const cStr = String(t.closed_date || "").trim();
          const hold = (oStr && cStr && !isNaN(new Date(oStr).getTime()) && !isNaN(new Date(cStr).getTime())) ? Math.max(0, Math.floor((new Date(cStr).getTime() - new Date(oStr).getTime()) / 86400000)) : null;
          const borderColor = isWinner ? "#08a86b" : "#e5484d";
          const plColor = isWinner ? "#08a86b" : "#e5484d";
          return (
            <div ref={refCallback} className="rounded-[12px] overflow-hidden mb-3" style={{ background: "var(--surface)", borderLeft: `4px solid ${borderColor}`, border: "1px solid var(--border)" }}>
              <div className="px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-[15px] font-extrabold">{rank != null ? `#${rank} · ` : ""}{t.ticker} <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>({t.trade_id})</span></div>
                  <div className="text-[18px] font-extrabold privacy-mask" style={{ color: plColor }}>{formatCurrency(pl, { showSign: true, decimals: 0 })}</div>
                </div>
                {/* Category pills */}
                {(() => { const cats = (lessons[t.trade_id]?.category || "").split("|").filter(Boolean); return cats.length > 0 ? (
                  <div className="flex flex-wrap gap-1 mb-2">{cats.map(c => { const cc = CAT_COLORS[c] || { bg: "var(--bg-2)", fg: "var(--ink-3)" }; return <span key={c} className="text-[10px] font-bold px-2 py-0.5 rounded-full" style={{ background: cc.bg, color: cc.fg }}>✓ {c}</span>; })}</div>
                ) : null; })()}
                <div className="grid grid-cols-5 gap-3">
                  <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Return</div><div className="text-[13px] font-bold" style={{ color: plColor }}>{ret >= 0 ? "+" : ""}{ret.toFixed(1)}%</div></div>
                  <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>R-Multiple</div><div className="text-[13px] font-bold">{rMult != null ? `${rMult.toFixed(2)}R` : "—"}</div></div>
                  <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Held</div><div className="text-[13px] font-bold">{hold != null ? `${hold}d` : "—"}</div></div>
                  <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Opened → Closed</div><div className="text-[13px] font-bold">{oStr.slice(5, 10)} → {cStr.slice(5, 10)}</div></div>
                  <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Rules</div><div className="text-[10px] font-semibold">B: {(t as any).buy_rule || t.rule || "—"}</div><div className="text-[10px] font-semibold">S: {(t as any).sell_rule || "—"}</div></div>
                </div>
              </div>
              {/* Transaction Trail */}
              {(() => {
                const txns = allDetails.filter(d => d.trade_id === t.trade_id).sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
                const buys = txns.filter(d => String(d.action).toUpperCase() === "BUY");
                const sells = txns.filter(d => String(d.action).toUpperCase() === "SELL");
                // LIFO to compute per-buy Return %
                const inventory: { idx: number; qty: number; price: number; origQty: number }[] = [];
                const buyRealized: Record<number, number> = {};
                txns.forEach((tx, j) => {
                  const action = String(tx.action || "").toUpperCase();
                  const shs = parseFloat(String(tx.shares || 0));
                  const px = parseFloat(String(tx.amount || 0));
                  if (action === "BUY") {
                    inventory.push({ idx: j, qty: shs, price: px, origQty: shs });
                    buyRealized[j] = 0;
                  } else if (action === "SELL") {
                    let toSell = shs;
                    while (toSell > 0 && inventory.length > 0) {
                      const lot = inventory[inventory.length - 1];
                      const take = Math.min(toSell, lot.qty);
                      buyRealized[lot.idx] = (buyRealized[lot.idx] || 0) + take * (px - lot.price);
                      toSell -= take;
                      lot.qty -= take;
                      if (lot.qty < 0.0001) inventory.pop();
                    }
                  }
                });

                return txns.length > 0 ? (
                  <details className="mx-4 mb-2">
                    <summary className="text-[11px] font-medium cursor-pointer px-3 py-1.5 rounded-[6px]" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                      📋 Transaction Trail — {buys.length} buy(s) · {sells.length} sell(s)
                    </summary>
                    <div className="mt-2 overflow-x-auto rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                      <table className="w-full text-[10px]" style={{ borderCollapse: "collapse" }}>
                        <thead><tr>
                          {["Date", "Trx", "Action", "Shares", "Price", "Return %", "Value", "Rule"].map(h => (
                            <th key={h} className="text-left px-2.5 py-1.5 text-[9px] uppercase font-semibold"
                                style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                          ))}
                        </tr></thead>
                        <tbody>{txns.map((tx, j) => {
                          const isSell = String(tx.action).toUpperCase() === "SELL";
                          const shs = parseFloat(String(tx.shares || 0));
                          const px = parseFloat(String(tx.amount || 0));
                          // Return % for BUY rows: LIFO-attributed return
                          let retPct = 0;
                          if (!isSell && px > 0 && shs > 0) {
                            const costBasis = px * shs;
                            retPct = costBasis > 0 ? ((buyRealized[j] || 0) / costBasis) * 100 : 0;
                          }
                          return (
                            <tr key={j} style={{ borderBottom: "1px solid var(--border)" }}>
                              <td className="px-2.5 py-1.5" style={{ fontFamily: mono, color: "var(--ink-4)" }}>{String(tx.date || "").slice(0, 16)}</td>
                              <td className="px-2.5 py-1.5 font-semibold" style={{ fontFamily: mono }}>{tx.trx_id || ""}</td>
                              <td className="px-2.5 py-1.5"><span className="px-1.5 py-0.5 rounded text-[9px] font-bold" style={{ background: `color-mix(in oklab, ${isSell ? "#e5484d" : "#08a86b"} 12%, var(--surface))`, color: isSell ? "#e5484d" : "#08a86b" }}>{tx.action}</span></td>
                              <td className="px-2.5 py-1.5" style={{ fontFamily: mono, color: isSell ? "#e5484d" : "var(--ink)" }}>{isSell ? -shs : shs}</td>
                              <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>{formatCurrency(px)}</td>
                              <td className="px-2.5 py-1.5 font-semibold" style={{ fontFamily: mono, color: !isSell && retPct !== 0 ? pctColor(retPct) : "var(--ink-4)" }}>
                                {!isSell && retPct !== 0 ? `${retPct >= 0 ? "+" : ""}${retPct.toFixed(2)}%` : isSell ? "" : "0.00%"}
                              </td>
                              <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>{formatCurrency(shs * px)}</td>
                              <td className="px-2.5 py-1.5 text-[9px]" style={{ color: "var(--ink-3)" }}>{tx.rule || ""}</td>
                            </tr>
                          );
                        })}</tbody>
                      </table>
                    </div>
                  </details>
                ) : null;
              })()}

              {/* Lesson with categories */}
              {(() => {
                const existing = lessons[t.trade_id];
                const editKey = t.trade_id;
                const currentText = lessonEdits[editKey] ?? existing?.note ?? "";
                const existingCats = (existing?.category || "").split("|").filter(Boolean);
                const catKey = `cat_${t.trade_id}`;
                const editCats = (lessonEdits[catKey] !== undefined) ? lessonEdits[catKey].split("|").filter(Boolean) : existingCats;

                const toggleCat = (cat: string) => {
                  const newCats = editCats.includes(cat) ? editCats.filter(c => c !== cat) : [...editCats, cat];
                  setLessonEdits(prev => ({ ...prev, [catKey]: newCats.join("|") }));
                };

                return (
                  <details className="mx-4 mb-3" {...(autoOpenLesson ? { open: true } : {})}>
                    <summary className="text-[11px] font-medium cursor-pointer px-3 py-1.5 rounded-[6px]" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                      📝 Lesson — {t.ticker} {t.trade_id} {existing?.note ? "✅" : ""}
                    </summary>
                    <div className="mt-2 p-3 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                      {/* Category pills */}
                      <div className="text-[10px] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Category (pick one or more)</div>
                      <div className="flex flex-wrap gap-1.5 mb-3">
                        {LESSON_CATEGORIES.map(cat => {
                          const active = editCats.includes(cat);
                          const cc = CAT_COLORS[cat] || { bg: "var(--bg-2)", fg: "var(--ink-3)" };
                          return (
                            <button key={cat} onClick={() => toggleCat(cat)}
                                    className="text-[10px] font-bold px-2.5 py-1 rounded-full transition-all"
                                    style={{ background: active ? cc.bg : "var(--bg)", color: active ? cc.fg : "var(--ink-4)", border: `1px solid ${active ? cc.fg + "40" : "var(--border)"}` }}>
                              {active ? "✓ " : ""}{cat}
                            </button>
                          );
                        })}
                      </div>
                      <div className="text-[10px] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>What did you learn from this trade?</div>
                      <textarea rows={2} value={currentText}
                                onChange={e => setLessonEdits(prev => ({ ...prev, [editKey]: e.target.value }))}
                                placeholder="e.g. Scaled in too fast on the third add..."
                                className="w-full px-3 py-2 rounded-[8px] text-[11px] outline-none resize-none mb-2"
                                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                      <button onClick={async () => {
                                const saveCat = editCats.join("|");
                                const result = await api.saveTradeLessons({ portfolio: getActivePortfolio(), trade_id: t.trade_id, note: currentText, category: saveCat });
                                if (result.status === "ok") {
                                  setLessons(prev => ({ ...prev, [t.trade_id]: { note: currentText, category: saveCat } }));
                                }
                              }}
                              className="h-[28px] px-3 rounded-[6px] text-[10px] font-semibold"
                              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>Save lesson</button>
                    </div>
                  </details>
                );
              })()}
            </div>
          );
        };

        // Pattern snapshot
        const patternStats = (group: TradePosition[]) => {
          if (group.length === 0) return null;
          const avgPl = group.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0) / group.length;
          const holdVals = group.map(t => { const o = new Date(String(t.open_date || "")); const c = new Date(String(t.closed_date || "")); return (!isNaN(o.getTime()) && !isNaN(c.getTime())) ? Math.floor((c.getTime() - o.getTime()) / 86400000) : 0; }).filter(v => v > 0);
          const avgHold = holdVals.length > 0 ? holdVals.reduce((a, b) => a + b, 0) / holdVals.length : 0;
          const rVals = group.map(t => { const rb = parseFloat(String(t.risk_budget || 0)); return rb > 0 ? parseFloat(String(t.realized_pl || 0)) / rb : null; }).filter((v): v is number => v !== null);
          const avgR = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : null;
          const buyRules = group.map(t => (t as any).buy_rule || t.rule || "").filter(Boolean);
          const sellRules = group.map(t => (t as any).sell_rule || "").filter(Boolean);
          const topBuy = buyRules.length > 0 ? buyRules.sort((a, b) => buyRules.filter(r => r === b).length - buyRules.filter(r => r === a).length)[0] : "—";
          const topSell = sellRules.length > 0 ? sellRules.sort((a, b) => sellRules.filter(r => r === b).length - sellRules.filter(r => r === a).length)[0] : "—";
          return { avgPl, avgHold, avgR, topBuy, topSell };
        };

        return (
          <>
            <div className="text-[14px] font-semibold mb-1">🔬 Trade Review — Top Winners & Worst Losers</div>
            <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>Study your best and worst trades. Tag each one with what you learned.</div>

            {/* Filter bar */}
            <div className="flex items-center gap-3 mb-5">
              <select value={trRange} onChange={e => setTrRange(e.target.value)}
                      className="h-[36px] px-3 rounded-[10px] text-[12px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                {["2026 YTD", "Last 30 days", "Last 90 days", "All time"].map(o => <option key={o} value={o}>{o}</option>)}
              </select>
              <select value={String(topN)} onChange={e => setTopN(parseInt(e.target.value))}
                      className="h-[36px] px-3 rounded-[10px] text-[12px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                {[5, 10, 15, 20].map(n => <option key={n} value={n}>Show top/bottom {n}</option>)}
              </select>
              <div className="ml-auto p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Closed in Window</div>
                <div className="text-[18px] font-extrabold" style={{ fontFamily: mono }}>{trFiltered.length}</div>
              </div>
            </div>

            {/* Selected Campaign — deep-link target rendered above the
                top/worst lists when the targeted trade isn't already in
                either list (cold trade, or sitting outside the active
                date-range filter). Explicit user nav overrides the
                filter for the targeted trade only. */}
            {showSelectedSection && selectedTrade && (
              <>
                <div className="text-[15px] font-bold mb-3">🎯 Selected Campaign</div>
                <TradeCard
                  key={`selected-${selectedTrade.trade_id}`}
                  rank={null}
                  t={selectedTrade}
                  isWinner={parseFloat(String(selectedTrade.realized_pl || 0)) >= 0}
                  autoOpenLesson
                  refCallback={onSelectedCardRef}
                />
              </>
            )}

            {/* Top Winners */}
            <div className="text-[15px] font-bold mb-3">🏆 Top {topN} Winners</div>
            {topWinners.length > 0 ? topWinners.map((t, i) => {
              const isTarget = selectedInLists && t.trade_id === selectedTradeId;
              return (
                <TradeCard
                  key={t.trade_id}
                  rank={i + 1}
                  t={t}
                  isWinner
                  autoOpenLesson={isTarget}
                  refCallback={isTarget ? onSelectedCardRef : undefined}
                />
              );
            }) : (
              <div className="mb-4 px-4 py-3 rounded-[10px] text-[12px]" style={{ background: "var(--bg)", color: "var(--ink-4)" }}>No profitable trades in this window.</div>
            )}

            {/* Worst Losers */}
            <div className="text-[15px] font-bold mb-3 mt-5">⚠️ Worst {topN} Losers</div>
            {worstLosers.length > 0 ? worstLosers.map((t, i) => {
              const isTarget = selectedInLists && t.trade_id === selectedTradeId;
              return (
                <TradeCard
                  key={t.trade_id}
                  rank={i + 1}
                  t={t}
                  isWinner={false}
                  autoOpenLesson={isTarget}
                  refCallback={isTarget ? onSelectedCardRef : undefined}
                />
              );
            }) : (
              <div className="mb-4 px-4 py-3 rounded-[10px] text-[12px]" style={{ background: `color-mix(in oklab, #08a86b 6%, var(--surface))`, color: "#08a86b" }}>No losing trades in this window.</div>
            )}

            {/* Pattern Snapshot */}
            <div className="text-[15px] font-bold mb-3 mt-5">📊 Pattern Snapshot</div>
            <div className="grid grid-cols-2 gap-4">
              {[
                { title: "🏆 Top Winners Pattern", data: patternStats(topWinners), color: "#08a86b" },
                { title: "⚠️ Worst Losers Pattern", data: patternStats(worstLosers), color: "#e5484d" },
              ].map(p => (
                <div key={p.title} className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, ${p.color} 5%, var(--surface))`, borderLeft: `3px solid ${p.color}`, border: "1px solid var(--border)" }}>
                  <div className="text-[11px] uppercase font-bold mb-2" style={{ color: p.color }}>{p.title}</div>
                  {p.data ? (
                    <div className="grid grid-cols-2 gap-2 text-[12px]">
                      <div><span style={{ color: "var(--ink-4)" }}>Avg P&L:</span> <strong className="privacy-mask">{formatCurrency(p.data.avgPl, { decimals: 0 })}</strong></div>
                      <div><span style={{ color: "var(--ink-4)" }}>Avg Hold:</span> <strong>{p.data.avgHold.toFixed(0)}d</strong></div>
                      <div><span style={{ color: "var(--ink-4)" }}>Avg R:</span> <strong>{p.data.avgR != null ? `${p.data.avgR.toFixed(2)}R` : "—"}</strong></div>
                      <div><span style={{ color: "var(--ink-4)" }}>Top Buy Rule:</span> <strong>{p.data.topBuy}</strong></div>
                      <div className="col-span-2"><span style={{ color: "var(--ink-4)" }}>Top Sell Rule:</span> <strong>{p.data.topSell}</strong></div>
                    </div>
                  ) : <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No trades</div>}
                </div>
              ))}
            </div>
          </>
        );
      })()}

      {/* ═══ ALL CAMPAIGNS ═══ */}
      {tab === "campaigns" && (() => {
        // Combine open + closed
        const allCampaigns = [...openTrades, ...allTrades];

        // Filter
        const filtered = allCampaigns.filter(t => {
          // Status
          if (campStatus === "open" && (t.status || "").toUpperCase() !== "OPEN") return false;
          if (campStatus === "closed" && (t.status || "").toUpperCase() !== "CLOSED") return false;
          // Ticker
          if (campTicker && !(t.ticker || "").toUpperCase().includes(campTicker.toUpperCase())) return false;
          // Date
          const d = String(t.open_date || "").slice(0, 10);
          if (campDateRange === "YTD" && !d.startsWith("2026")) return false;
          if (campDateRange === "This Month") {
            const now = new Date();
            const monthStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
            if (!d.startsWith(monthStr)) return false;
          }
          if (campDateRange === "Last 30d" && d < new Date(Date.now() - 30 * 86400000).toISOString().slice(0, 10)) return false;
          if (campDateRange === "Last 90d" && d < new Date(Date.now() - 90 * 86400000).toISOString().slice(0, 10)) return false;
          // Result — open trades classified by overall_pl (unrealized + realized
          // partial closures); closed trades by realized_pl (the only thing left
          // once shares hit zero).
          const isOpenRow = (t.status || "").toUpperCase() === "OPEN";
          const resultPl = isOpenRow
            ? (enrichedById[t.trade_id]?.overall_pl ?? 0)
            : parseFloat(String(t.realized_pl || 0));
          if (campResult === "winners" && resultPl <= 0) return false;
          if (campResult === "losers" && resultPl >= 0) return false;
          // Grade
          if (campGrade !== "all") {
            const g = (t as any).grade;
            if (campGrade === "unrated" && g != null) return false;
            if (campGrade !== "unrated" && g !== parseInt(campGrade, 10)) return false;
          }
          // Strategy ("" = no filter, includes inactive strategies so
          // existing tagged trades stay filterable post-deactivation).
          if (campStrategy && String((t as any).strategy || "") !== campStrategy) return false;
          // Instrument (STOCK / OPTION). Default "STOCK" for legacy
          // pre-Migration-016 rows that lack the column.
          if (campInstrument && String(t.instrument_type || "STOCK") !== campInstrument) return false;
          // Buy Rule — match on the same buy_rule || rule fallback used
          // for the option list above so the filter compares like-with-
          // like.
          if (campBuyRule && String((t as any).buy_rule || t.rule || "").trim() !== campBuyRule) return false;
          // Sell Rule — only set on closed trades; matching against ""
          // for an open trade would never fire because campSellRule is
          // also "" (no filter) in that case.
          if (campSellRule && String((t as any).sell_rule || "").trim() !== campSellRule) return false;
          return true;
        });

        // Sortable columns
        const sortKey = campSort.col;
        const sortAsc = campSort.asc;
        const getSortVal = (t: TradePosition): number | string => {
          const txns = allDetails.filter(d => d.trade_id === t.trade_id);
          const buyTxns = txns.filter(d => String(d.action).toUpperCase() === "BUY");
          const sellTxns = txns.filter(d => String(d.action).toUpperCase() === "SELL");
          let entry = parseFloat(String(t.avg_entry || 0));
          if (entry === 0 && buyTxns.length > 0) {
            const tv = buyTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0);
            const ts = buyTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
            entry = ts > 0 ? tv / ts : 0;
          }
          let exit = parseFloat(String(t.avg_exit || 0));
          if (exit === 0 && sellTxns.length > 0) {
            const tv = sellTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0);
            const ts = sellTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
            exit = ts > 0 ? tv / ts : 0;
          }
          let ret = parseFloat(String(t.return_pct || 0));
          if (ret === 0 && entry > 0 && exit > 0) ret = ((exit - entry) / entry) * 100;
          const rb = parseFloat(String(t.risk_budget || 0));
          const realizedPl = parseFloat(String(t.realized_pl || 0));
          const isOpen = (t.status || "").toUpperCase() === "OPEN";
          const pl = isOpen ? (enrichedById[t.trade_id]?.overall_pl ?? 0) : realizedPl;
          const rMult = rb > 0 ? pl / rb : 0;
          switch (sortKey) {
            case "ticker": return (t.ticker || "").toUpperCase();
            case "trade_id": return t.trade_id || "";
            case "status": return (t.status || "").toUpperCase();
            case "buy_rule": return String((t as any).buy_rule || t.rule || "").toUpperCase();
            case "sell_rule": return String((t as any).sell_rule || "").toUpperCase();
            case "open": return t.open_date || "";
            case "close": return t.closed_date || "";
            case "shares": return t.shares || 0;
            case "entry": return entry;
            case "exit": return exit;
            case "pl": return pl;
            case "return": return ret;
            case "r": return rMult;
            default: return t.open_date || "";
          }
        };
        filtered.sort((a, b) => {
          const va = getSortVal(a);
          const vb = getSortVal(b);
          const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
          return sortAsc ? cmp : -cmp;
        });

        // Flight deck — adaptive by visible filtered set
        const filteredOpen = filtered.filter(t => (t.status || "").toUpperCase() === "OPEN");
        const filteredClosed = filtered.filter(t => (t.status || "").toUpperCase() === "CLOSED");
        const enrichedFilteredOpen = filteredOpen
          .map(t => enrichedById[t.trade_id])
          .filter(Boolean) as EnrichedPosition[];

        const fdMode: "open" | "closed" | "mixed" =
          filteredClosed.length === 0 && filteredOpen.length > 0 ? "open"
          : filteredOpen.length === 0 && filteredClosed.length > 0 ? "closed"
          : "mixed";

        // Closed-mode metrics (scoped to filteredClosed only)
        const cWins = filteredClosed.filter(t => parseFloat(String(t.realized_pl || 0)) > 0);
        const cLosses = filteredClosed.filter(t => parseFloat(String(t.realized_pl || 0)) < 0);
        const cPl = filteredClosed.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
        const cWinRate = filteredClosed.length > 0 ? (cWins.length / filteredClosed.length) * 100 : 0;
        const cAvgPl = filteredClosed.length > 0 ? cPl / filteredClosed.length : 0;
        const cGrossW = cWins.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
        const cGrossL = Math.abs(cLosses.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0));
        const cPf = cGrossL > 0 ? cGrossW / cGrossL : 0;

        // Open-mode metrics
        const oUnrealized = enrichedFilteredOpen.reduce((a, p) => a + p.unrealized_pl, 0);
        const oInProfit = enrichedFilteredOpen.filter(p => p.unrealized_pl > 0).length;
        const oInLoss = enrichedFilteredOpen.filter(p => p.unrealized_pl < 0).length;
        const oAvgUnrl = enrichedFilteredOpen.length > 0 ? oUnrealized / enrichedFilteredOpen.length : 0;
        const oCurrentVal = enrichedFilteredOpen.reduce((a, p) => a + p.current_value, 0);
        const oAvgDays = enrichedFilteredOpen.length > 0
          ? enrichedFilteredOpen.reduce((a, p) => a + p.days_held, 0) / enrichedFilteredOpen.length : 0;

        // Mixed-mode metrics — total economic outcome across both
        const mRealized = cPl + enrichedFilteredOpen.reduce((a, p) => a + p.realized_bank, 0);
        const mUnrealized = enrichedFilteredOpen.reduce((a, p) => a + p.unrealized_pl, 0);
        const mTotal = mRealized + mUnrealized;

        type Tile = { k: string; v: string; sub?: string; color?: string };
        const tiles: Tile[] = fdMode === "closed" ? [
          { k: "Trades",        v: String(filtered.length) },
          { k: "Net P&L",       v: formatCurrency(cPl, { showSign: true, decimals: 0 }), color: pctColor(cPl) },
          { k: "Win Rate",      v: `${cWinRate.toFixed(0)}%`, color: cWinRate >= 50 ? "#08a86b" : "#e5484d" },
          { k: "Avg P&L",       v: formatCurrency(cAvgPl, { showSign: true, decimals: 0 }), color: pctColor(cAvgPl) },
          { k: "Profit Factor", v: cPf.toFixed(2) },
          { k: "W / L",         v: `${cWins.length}W · ${cLosses.length}L` },
        ] : fdMode === "open" ? [
          { k: "Trades",             v: String(filtered.length) },
          { k: "Unrealized P&L",     v: formatCurrency(oUnrealized, { showSign: true, decimals: 0 }), color: pctColor(oUnrealized) },
          { k: "In Profit / Loss",   v: `${oInProfit} · ${oInLoss}` },
          { k: "Avg Unrealized P&L", v: formatCurrency(oAvgUnrl, { showSign: true, decimals: 0 }), color: pctColor(oAvgUnrl) },
          { k: "Total Value",        v: formatCurrency(oCurrentVal, { decimals: 0 }) },
          { k: "Avg Days Held",      v: oAvgDays.toFixed(0) },
        ] : [
          { k: "Trades",         v: String(filtered.length) },
          { k: "Closed / Open",  v: `${filteredClosed.length} · ${filteredOpen.length}` },
          { k: "Win Rate",       v: `${cWinRate.toFixed(0)}%`, sub: "closed only", color: cWinRate >= 50 ? "#08a86b" : "#e5484d" },
          { k: "Net Realized",   v: formatCurrency(mRealized, { showSign: true, decimals: 0 }), color: pctColor(mRealized) },
          { k: "Net Unrealized", v: formatCurrency(mUnrealized, { showSign: true, decimals: 0 }), color: pctColor(mUnrealized) },
          { k: "Total P&L",      v: formatCurrency(mTotal, { showSign: true, decimals: 0 }), color: pctColor(mTotal) },
        ];

        // Unique tickers for filter dropdown
        const allTickers = [...new Set(allCampaigns.map(t => t.ticker).filter(Boolean))].sort();

        return (
          <>
            <div className="text-[14px] font-semibold mb-1">📋 All Campaigns</div>
            <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>Every trade ever. Filter by status, ticker, date, or result.</div>

            {/* Filters */}
            <div className="flex items-center gap-3 mb-4 flex-wrap">
              {/* Status */}
              <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                {(["all", "open", "closed"] as const).map(s => (
                  <button key={s} onClick={() => setCampStatus(s)}
                          className="px-3 py-1 rounded-md text-[11px] font-medium transition-all capitalize"
                          style={{ background: campStatus === s ? "var(--surface)" : "transparent", color: campStatus === s ? "var(--ink)" : "var(--ink-4)" }}>
                    {s}
                  </button>
                ))}
              </div>

              {/* Ticker */}
              <input type="text" value={campTicker} onChange={e => setCampTicker(e.target.value.toUpperCase())}
                     placeholder="Ticker..." className="h-[32px] px-3 rounded-[8px] text-[11px] w-[100px]"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />

              {/* Date range */}
              <select value={campDateRange} onChange={e => setCampDateRange(e.target.value)}
                      className="h-[32px] px-2.5 rounded-[8px] text-[11px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                {["LTD", "YTD", "This Month", "Last 30d", "Last 90d"].map(o => <option key={o} value={o}>{o}</option>)}
              </select>

              {/* Result */}
              <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                {(["all", "winners", "losers"] as const).map(r => (
                  <button key={r} onClick={() => setCampResult(r)}
                          className="px-3 py-1 rounded-md text-[11px] font-medium transition-all capitalize"
                          style={{ background: campResult === r ? "var(--surface)" : "transparent", color: campResult === r ? "var(--ink)" : "var(--ink-4)" }}>
                    {r}
                  </button>
                ))}
              </div>

              {/* Grade */}
              <select value={campGrade} onChange={e => setCampGrade(e.target.value as typeof campGrade)}
                      className="h-[32px] px-2.5 rounded-[8px] text-[11px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                <option value="all">All grades</option>
                <option value="unrated">Unrated</option>
                <option value="5">★★★★★ (5)</option>
                <option value="4">★★★★ (4)</option>
                <option value="3">★★★ (3)</option>
                <option value="2">★★ (2)</option>
                <option value="1">★ (1)</option>
              </select>

              {/* Strategy filter — custom dropdown with StrategyChip
                  swatches per option. Native <select> can't render the
                  swatch, so this is the one filter that needs a
                  bespoke control. Inactive strategies render with an
                  "(inactive)" suffix so the user knows the option
                  refers to a deactivated strategy that some trades
                  may still be tagged with. */}
              <div ref={strategyFilterRef} className="relative" data-testid="campaigns-strategy-filter">
                <button type="button"
                        onClick={() => setStrategyFilterOpen(o => !o)}
                        className="h-[32px] px-3 rounded-[8px] text-[11px] flex items-center gap-2 cursor-pointer"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                  {campStrategy ? (
                    (() => {
                      const s = allStrategies.find(x => x.name === campStrategy);
                      return s
                        ? <StrategyChip name={s.name} color={s.color} size="sm" />
                        : <span>{campStrategy}</span>;
                    })()
                  ) : (
                    <span>All Strategies</span>
                  )}
                  <span style={{ color: "var(--ink-4)" }}>▾</span>
                </button>
                {strategyFilterOpen && (
                  <div className="absolute top-full mt-1 left-0 z-40 rounded-[10px] py-1.5 overflow-hidden"
                       style={{
                         minWidth: 200,
                         background: "var(--surface)",
                         border: "1px solid var(--border)",
                         boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
                       }}>
                    <button type="button"
                            onClick={() => { setCampStrategy(""); setStrategyFilterOpen(false); }}
                            className="w-full text-left px-3 py-2 text-[12px] transition-colors hover:brightness-95"
                            style={{ background: campStrategy === "" ? "var(--surface-2)" : "transparent", color: "var(--ink)" }}
                            onMouseEnter={e => { if (campStrategy !== "") e.currentTarget.style.background = "var(--surface-2)"; }}
                            onMouseLeave={e => { if (campStrategy !== "") e.currentTarget.style.background = "transparent"; }}>
                      All Strategies
                    </button>
                    {allStrategies.map(s => (
                      <button key={s.name} type="button"
                              onClick={() => { setCampStrategy(s.name); setStrategyFilterOpen(false); }}
                              className="w-full text-left px-3 py-2 text-[12px] flex items-center gap-2 transition-colors hover:brightness-95"
                              style={{ background: campStrategy === s.name ? "var(--surface-2)" : "transparent", color: "var(--ink)" }}
                              onMouseEnter={e => { if (campStrategy !== s.name) e.currentTarget.style.background = "var(--surface-2)"; }}
                              onMouseLeave={e => { if (campStrategy !== s.name) e.currentTarget.style.background = "transparent"; }}>
                        <StrategyChip name={s.name} color={s.color} size="md" />
                        {!s.is_active && (
                          <span className="ml-auto text-[10px]" style={{ color: "var(--ink-4)" }}>(inactive)</span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Instrument filter (STOCK / OPTION). Native <select>;
                  options derived from the loaded data so a future third
                  instrument type appears automatically. */}
              <select value={campInstrument} onChange={e => setCampInstrument(e.target.value)}
                      data-testid="campaigns-instrument-filter"
                      className="h-[32px] px-2.5 rounded-[8px] text-[11px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                <option value="">All Instruments</option>
                {instrumentOptions.map(o => <option key={o} value={o}>{o === "STOCK" ? "Stocks" : o === "OPTION" ? "Options" : o}</option>)}
              </select>

              {/* Buy Rule filter — option list derived from data, not
                  the master BUY_RULES list, so the dropdown only shows
                  rules actually used by the user's trades. Searchable
                  via SearchSelect because the rule catalog is long
                  enough to make a native scroll painful. Empty value
                  ("") is the "show everything" sentinel, surfaced as
                  the first option so the SearchSelect label resolves
                  cleanly when nothing is filtered. */}
              <div className="w-[180px]" data-testid="campaigns-buy-rule-filter">
                <SearchSelect
                  value={campBuyRule}
                  onChange={setCampBuyRule}
                  options={[{ value: "", label: "All Buy Rules" }, ...buyRuleOptions.map(o => ({ value: o, label: o }))]}
                  placeholder="All Buy Rules"
                />
              </div>

              {/* Sell Rule filter — only populated by closed trades
                  (sell_rule is set when the campaign closes). On a
                  fresh portfolio with only open trades, this dropdown
                  shows just "All Sell Rules". */}
              <div className="w-[180px]" data-testid="campaigns-sell-rule-filter">
                <SearchSelect
                  value={campSellRule}
                  onChange={setCampSellRule}
                  options={[{ value: "", label: "All Sell Rules" }, ...sellRuleOptions.map(o => ({ value: o, label: o }))]}
                  placeholder="All Sell Rules"
                />
              </div>

              <span className="ml-auto text-[11px]" style={{ color: "var(--ink-4)" }}>{filtered.length} results</span>
            </div>

            {/* Flight Deck */}
            {pricesStale && fdMode !== "closed" && (
              <div className="text-[11px] mb-2 px-3 py-1.5 rounded-[8px]"
                   style={{ background: "color-mix(in oklab, #d97706 10%, var(--surface))", color: "#92400e", border: "1px solid color-mix(in oklab, #d97706 20%, var(--border))" }}>
                ⚠️ Live prices unavailable — using cost basis for open trades
              </div>
            )}
            <div className="rounded-[14px] overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-5 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
                Flight Deck {campTicker ? `— ${campTicker}` : "— All"} · {fdMode === "open" ? "Open" : fdMode === "closed" ? "Closed" : "Mixed"}
              </div>
              <div className="grid grid-cols-6 divide-x" style={{ borderColor: "var(--border)" }}>
                {tiles.map(m => (
                  <div key={m.k} className="p-4 text-center">
                    <div className="text-[9px] uppercase tracking-[0.06em] font-bold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                    <div className="text-[20px] font-extrabold mt-1 privacy-mask" style={{ fontFamily: mono, color: m.color || "var(--ink)" }}>{m.v}</div>
                    {m.sub && (
                      <div className="text-[9px] mt-0.5" style={{ color: "var(--ink-4)" }}>{m.sub}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Trade table */}
            {/* Phase 2 — right-click context menu for single-row retag,
                plus "Open in Trade Journal" drill-in. Mirrors the ACS menu
                style; renders the StrategyFlyout on desktop and
                StrategyFlatList on touch (no hover state). The retag block
                is hidden when no strategies are loaded, but the menu still
                renders so the journal drill-in is always available. */}
            {campCtxMenu && (
              <div className="fixed z-50 rounded-[10px] py-1.5 min-w-[200px] overflow-hidden"
                   data-testid="campaigns-ctx-menu"
                   style={{
                     left: campCtxMenu.x,
                     top: campCtxMenu.y,
                     background: "var(--surface)",
                     border: "1px solid var(--border)",
                     boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
                   }}>
                <div className="px-3 py-1.5 text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>
                  {campCtxMenu.trade.ticker} · {campCtxMenu.trade.trade_id}
                </div>
                <button type="button"
                        data-testid="campaigns-ctx-open-journal"
                        onClick={() => {
                          const id = encodeURIComponent(campCtxMenu.trade.trade_id);
                          router.push(`/trade-journal?trade_id=${id}`);
                          setCampCtxMenu(null);
                        }}
                        className="w-full text-left px-3 py-1.5 text-[12px] transition-colors hover:brightness-95"
                        style={{ color: "var(--ink)" }}
                        onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                  Open in Trade Journal
                </button>
                {strategies.length > 0 && (
                  <>
                    <div className="my-1 mx-2" style={{ borderTop: "1px solid var(--border)" }} />
                    {coarsePointer ? (
                      <StrategyFlatList
                        strategies={strategies}
                        currentStrategy={(campCtxMenu.trade as any).strategy}
                        onPick={(name) => setOneStrategy(campCtxMenu.trade.trade_id, name)}
                      />
                    ) : (
                      <StrategyFlyout
                        strategies={strategies}
                        currentStrategy={(campCtxMenu.trade as any).strategy}
                        onPick={(name) => setOneStrategy(campCtxMenu.trade.trade_id, name)}
                      />
                    )}
                  </>
                )}
              </div>
            )}

            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="overflow-x-auto" style={{ maxHeight: 600 }}>
                <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                  <thead><tr>
                    {([
                      { label: "Ticker", key: "ticker" }, { label: "Trade ID", key: "trade_id" },
                      { label: "Open", key: "open" }, { label: "Close", key: "close" },
                      { label: "Status", key: "status" },
                      { label: "Shares", key: "shares" }, { label: "Entry", key: "entry" }, { label: "Exit", key: "exit" },
                      { label: "P&L", key: "pl" }, { label: "Return %", key: "return" }, { label: "R", key: "r" },
                      { label: "Buy Rule", key: "buy_rule" }, { label: "Sell Rule", key: "sell_rule" },
                    ] as const).map(h => (
                      <th key={h.key}
                          className="text-left px-3 py-2.5 text-[9px] uppercase font-semibold whitespace-nowrap sticky top-0 cursor-pointer select-none"
                          style={{ color: campSort.col === h.key ? "var(--ink)" : "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}
                          onClick={() => setCampSort(prev => prev.col === h.key ? { col: h.key, asc: !prev.asc } : { col: h.key, asc: h.key === "ticker" || h.key === "trade_id" })}>
                        {h.label} {campSort.col === h.key ? (campSort.asc ? "▲" : "▼") : ""}
                      </th>
                    ))}
                  </tr></thead>
                  <tbody>{filtered.map((t, i) => {
                    const isOpen = (t.status || "").toUpperCase() === "OPEN";
                    const realizedPl = parseFloat(String(t.realized_pl || 0));
                    const pl = isOpen ? (enrichedById[t.trade_id]?.overall_pl ?? 0) : realizedPl;
                    const ret = parseFloat(String(t.return_pct || 0));
                    const rb = parseFloat(String(t.risk_budget || 0));
                    const rMult = rb > 0 ? pl / rb : null;

                    // Enrich from details if summary has missing data
                    const txns = allDetails.filter(d => d.trade_id === t.trade_id);
                    const sellTxns = txns.filter(d => String(d.action).toUpperCase() === "SELL");
                    const buyTxns = txns.filter(d => String(d.action).toUpperCase() === "BUY");

                    // Shares: use summary if > 0, else compute from buys
                    const displayShares = t.shares > 0 ? t.shares : buyTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);

                    // Avg entry: use summary if > 0, else compute from buys
                    let displayEntry = parseFloat(String(t.avg_entry || 0));
                    if (displayEntry === 0 && buyTxns.length > 0) {
                      const totalBuyVal = buyTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0);
                      const totalBuyShs = buyTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
                      displayEntry = totalBuyShs > 0 ? totalBuyVal / totalBuyShs : 0;
                    }

                    // Avg exit: use summary if > 0, else compute from sells
                    let displayExit = parseFloat(String(t.avg_exit || 0));
                    if (displayExit === 0 && sellTxns.length > 0) {
                      const totalSellVal = sellTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)) * parseFloat(String(d.amount || 0)), 0);
                      const totalSellShs = sellTxns.reduce((a, d) => a + parseFloat(String(d.shares || 0)), 0);
                      displayExit = totalSellShs > 0 ? totalSellVal / totalSellShs : 0;
                    }

                    // Return %: use summary if != 0, else compute from entry/exit
                    let displayRet = ret;
                    if (displayRet === 0 && displayEntry > 0 && displayExit > 0) {
                      displayRet = ((displayExit - displayEntry) / displayEntry) * 100;
                    }

                    // Closed date: use summary if present, else last sell date
                    let displayCloseDate = String(t.closed_date || "").slice(0, 10);
                    if (!displayCloseDate && sellTxns.length > 0) {
                      displayCloseDate = String(sellTxns[sellTxns.length - 1].date || "").slice(0, 10);
                    }

                    return (
                      <tr key={`${t.trade_id}-${i}`} style={{ borderBottom: "1px solid var(--border)" }}
                          className="transition-colors"
                          onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                          onContextMenu={e => { e.preventDefault(); setCampCtxMenu({ x: e.clientX, y: e.clientY, trade: t }); }}>
                        <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                        <td className="px-3 py-2" style={{ fontFamily: mono, fontSize: 10, color: "var(--ink-4)" }}>{t.trade_id}</td>
                        <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(t.open_date || "").slice(0, 10)}</td>
                        <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{isOpen ? "—" : (displayCloseDate || "—")}</td>
                        <td className="px-3 py-2">
                          <span className="text-[9px] px-1.5 py-0.5 rounded-full font-semibold"
                                style={{ background: `color-mix(in oklab, ${isOpen ? "#08a86b" : "#888"} 10%, var(--surface))`, color: isOpen ? "#08a86b" : "var(--ink-4)" }}>
                            {t.status}
                          </span>
                        </td>
                        <td className="px-3 py-2" style={{ fontFamily: mono }}>{displayShares > 0 ? displayShares : "—"}</td>
                        <td className="px-3 py-2 privacy-mask" style={{ fontFamily: mono }}>{displayEntry > 0 ? formatCurrency(displayEntry) : "—"}</td>
                        <td className="px-3 py-2 privacy-mask" style={{ fontFamily: mono }}>{!isOpen && displayExit > 0 ? formatCurrency(displayExit) : "—"}</td>
                        <td className="px-3 py-2 font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>
                          {formatCurrency(pl, { showSign: true, decimals: 0 })}
                        </td>
                        <td className="px-3 py-2" style={{ fontFamily: mono, color: pctColor(isOpen ? 0 : displayRet) }}>{!isOpen && displayRet !== 0 ? `${displayRet >= 0 ? "+" : ""}${displayRet.toFixed(1)}%` : "—"}</td>
                        <td className="px-3 py-2" style={{ fontFamily: mono }}>{rMult != null ? `${rMult.toFixed(2)}R` : "—"}</td>
                        <td className="px-3 py-2 text-[10px]" style={{ color: "var(--ink-3)" }}>{(t as any).buy_rule || t.rule || "—"}</td>
                        <td className="px-3 py-2 text-[10px]" style={{ color: "var(--ink-3)" }}>{isOpen ? "—" : ((t as any).sell_rule || "—")}</td>
                      </tr>
                    );
                  })}</tbody>
                </table>
              </div>
            </div>
          </>
        );
      })()}

      {/* ═══ ADD EFFECTIVENESS ═══ */}
      {tab === "add-effectiveness" && (() => {
        const totals = aeData?.totals ?? {
          total_adds: 0, total_realized_pl: 0, total_unrealized_pl: 0,
          overall_win_rate: 0, avg_realized_per_add: 0,
        };
        const rules = aeData?.rules ?? [];
        const avgDownCount = aeData?.discipline?.average_down_count ?? 0;
        const winRatePct = totals.overall_win_rate * 100;
        // Sort rules per current sort state. Strings sort case-insensitive
        // alphabetically; numbers numerically. Sort is stable across keys
        // because the backend orders by add_count desc as a tiebreak.
        const sortedRules = [...rules].sort((a, b) => {
          const va = a[aeSortKey]; const vb = b[aeSortKey];
          const cmp = (typeof va === "string" && typeof vb === "string")
            ? va.localeCompare(vb)
            : ((va as number) - (vb as number));
          return aeSortDir === "asc" ? cmp : -cmp;
        });
        const onSort = (k: keyof AddEffectivenessResponse["rules"][number]) => {
          if (aeSortKey === k) { setAeSortDir(d => d === "asc" ? "desc" : "asc"); }
          else { setAeSortKey(k); setAeSortDir(k === "rule" ? "asc" : "desc"); }
        };
        const sortArrow = (k: keyof AddEffectivenessResponse["rules"][number]) =>
          aeSortKey === k ? (aeSortDir === "asc" ? " ▲" : " ▼") : "";

        return (
          <>
            <div className="text-[14px] font-semibold mb-1">➕ Add Effectiveness</div>
            <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>
              Group your scale-in (add) buys by their buy rule. Pyramid-up adds (above blended cost) build winners; average-down adds erode them.
            </div>

            {/* Filters: date range + strategy pills. Mirrors the All
                Campaigns filter row's idiom — same control sizes,
                spacing, and pill-toggle pattern. */}
            <div className="flex items-center gap-3 mb-4 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-medium" style={{ color: "var(--ink-4)" }}>From</span>
                <input type="date" value={aeStart} onChange={e => setAeStart(e.target.value)}
                       data-testid="ae-start"
                       className="h-[32px] px-2 rounded-[8px] text-[11px]"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
                <span className="text-[11px] font-medium" style={{ color: "var(--ink-4)" }}>to</span>
                <input type="date" value={aeEnd} onChange={e => setAeEnd(e.target.value)}
                       data-testid="ae-end"
                       className="h-[32px] px-2 rounded-[8px] text-[11px]"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
              </div>

              {/* Strategy pill row — same idiom as the All-Campaigns
                  status pill toggle. "All" maps to empty string, which
                  the backend reads as "no filter". */}
              <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <button onClick={() => setAeStrategy("")}
                        className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                        style={{ background: aeStrategy === "" ? "var(--surface)" : "transparent", color: aeStrategy === "" ? "var(--ink)" : "var(--ink-4)" }}>
                  All
                </button>
                {allStrategies.map(s => (
                  <button key={s.name} onClick={() => setAeStrategy(s.name)}
                          className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                          style={{ background: aeStrategy === s.name ? "var(--surface)" : "transparent", color: aeStrategy === s.name ? "var(--ink)" : "var(--ink-4)" }}>
                    {s.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Loading / error states — match the existing skeleton +
                inline error idiom used elsewhere in the file. */}
            {aeError && (
              <div className="mb-4 px-4 py-3 rounded-[10px]"
                   style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
                Failed to load add effectiveness: {aeError}
              </div>
            )}
            {aeLoading && !aeData && (
              <div className="animate-pulse"><div className="h-[120px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>
            )}

            {/* Hero cards — same component as the Overview tab so the
                section reads as part of the same flight deck. */}
            {!aeLoading && !aeError && (
              <>
                <div className="grid grid-cols-4 gap-3 mb-4">
                  <HeroCard label="Total Adds" value={String(totals.total_adds)} sub={`in window`} ok={totals.total_adds > 0} />
                  <HeroCard label="Realized P&L · Adds" value={formatCurrency(totals.total_realized_pl, { decimals: 0 })} sub={`unrealized ${formatCurrency(totals.total_unrealized_pl, { decimals: 0 })}`} ok={totals.total_realized_pl >= 0} />
                  <HeroCard label="Win Rate" value={`${winRatePct.toFixed(1)}%`} sub={`on adds with closures`} ok={winRatePct >= 40} />
                  <HeroCard label="Avg P&L / Add" value={formatCurrency(totals.avg_realized_per_add, { decimals: 0 })} sub={`per closed add`} ok={totals.avg_realized_per_add >= 0} />
                </div>

                {/* Discipline line — single line, success-styled at zero,
                    warning-styled when >0 since these are the exceptions
                    to investigate. */}
                <div className="mb-5 px-4 py-3 rounded-[12px] text-[12px] font-semibold flex items-center gap-2"
                     data-testid="ae-discipline"
                     style={{
                       background: avgDownCount === 0
                         ? "color-mix(in oklab, #08a86b 8%, var(--surface))"
                         : "color-mix(in oklab, #d97706 8%, var(--surface))",
                       border: "1px solid var(--border)",
                       color: avgDownCount === 0 ? "#08a86b" : "#d97706",
                     }}>
                  <span>{avgDownCount === 0 ? "✅" : "⚠️"}</span>
                  <span>
                    {avgDownCount} of {totals.total_adds} adds were average-downs
                    {avgDownCount === 0 ? " — all adds extended at or above blended cost." : "."}
                  </span>
                </div>

                {/* Rule-effectiveness table. Same shape + styling as the
                    Buy Rules drill-down table — sortable headers, mono
                    numeric cells, success/danger color by sign. */}
                {sortedRules.length === 0 ? (
                  <div className="px-4 py-8 text-center text-[12px] rounded-[14px]"
                       data-testid="ae-empty-table"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
                    No adds in this window. Try widening the date range or switching strategies.
                  </div>
                ) : (
                  <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                    <table className="w-full text-[11px]" data-testid="ae-table">
                      <thead>
                        <tr style={{ background: "var(--bg)", borderBottom: "1px solid var(--border)" }}>
                          {([
                            ["rule", "Add rule", "left"],
                            ["add_count", "Adds", "right"],
                            ["realized_pl", "Realized", "right"],
                            ["unrealized_pl", "Open P&L", "right"],
                            ["avg_extension_at_add", "Avg ext at add", "right"],
                            ["win_rate", "Win", "right"],
                            ["avg_realized_per_add", "Avg / add", "right"],
                          ] as const).map(([k, label, align]) => (
                            <th key={k as string}
                                onClick={() => onSort(k as keyof AddEffectivenessResponse["rules"][number])}
                                data-testid={`ae-th-${k}`}
                                className="px-3 py-2 text-[10px] font-bold uppercase tracking-[0.06em] cursor-pointer select-none"
                                style={{ color: "var(--ink-4)", textAlign: align }}>
                              {label}{sortArrow(k as keyof AddEffectivenessResponse["rules"][number])}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sortedRules.map((r, i) => (
                          <tr key={r.rule + i} style={{ borderBottom: "1px solid var(--border)" }}>
                            <td className="px-3 py-2 text-[11px]" style={{ color: "var(--ink-2)" }}>{r.rule}</td>
                            <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{r.add_count}</td>
                            <td className="px-3 py-2 text-right font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(r.realized_pl) }}>
                              {formatCurrency(r.realized_pl, { showSign: true, decimals: 0 })}
                            </td>
                            <td className="px-3 py-2 text-right privacy-mask" style={{ fontFamily: mono, color: pctColor(r.unrealized_pl) }}>
                              {formatCurrency(r.unrealized_pl, { showSign: true, decimals: 0 })}
                            </td>
                            <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: pctColor(r.avg_extension_at_add) }}>
                              {r.avg_extension_at_add >= 0 ? "+" : ""}{r.avg_extension_at_add.toFixed(1)}%
                            </td>
                            <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
                              {r.closed_count > 0 ? `${(r.win_rate * 100).toFixed(0)}%` : "—"}
                            </td>
                            <td className="px-3 py-2 text-right privacy-mask" style={{ fontFamily: mono, color: pctColor(r.avg_realized_per_add) }}>
                              {r.closed_count > 0 ? formatCurrency(r.avg_realized_per_add, { showSign: true, decimals: 0 }) : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </>
        );
      })()}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════════
// Edge Report — Overview additions
//
// Pareto card + Hold-time buckets + Brandt NAV normalization + Regime
// cross-tab. All read from the cohort-aware `trades` set already
// filtered by year + cohort by the parent.
// ═══════════════════════════════════════════════════════════════════════

function EdgeCards({ trades, journal, mctStates, year, cohort }: {
  trades: TradePosition[];
  journal: any[];
  mctStates: Array<{ trade_date: string; state: string }>;
  year: number;
  cohort: "closed" | "at-mark";
}) {
  const pareto = useMemo(() => paretoDistribution(trades), [trades]);
  const holdBuckets = useMemo(() => holdTimeBuckets(trades), [trades]);
  const brandt = useMemo(() => brandtNormalized(trades, journal), [trades, journal]);
  const mctResolver = useMemo(() => makeMctStateResolver(mctStates), [mctStates]);
  const regime = useMemo(
    () => regimeCrossTab(trades, journal, undefined, { regimeOf: mctResolver }),
    [trades, journal, mctResolver],
  );
  const risk = useMemo(() => riskMetrics(trades, journal), [trades, journal]);
  const offenders = useMemo(() => repeatOffenders(trades), [trades]);
  const insights = useMemo(
    () => generateInsights(trades, { mctStateResolver: mctResolver }),
    [trades, mctResolver],
  );
  const confluence = useMemo(() => confluenceAnalysis(trades), [trades]);

  const yearBadge = (
    <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>
      — {year} · {cohort === "closed" ? "closed only" : "including open @ mark"}
    </span>
  );

  return (
    <>
      {/* ─────────── Insights (auto-generated Action List) ─────────── */}
      {insights.length > 0 && (
        <div className="mt-6 mb-5 p-5 rounded-[14px]"
             style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <div className="text-[15px] font-semibold mb-1">
            💡 Insights (Auto Action List) {yearBadge}
          </div>
          <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
            Rules automatically detected from this cohort. Each finding is data-derived — no hard-coded picks.
          </div>
          <div className="flex flex-col gap-3">
            {insights.map(ins => <InsightCard key={ins.id} insight={ins} />)}
          </div>
        </div>
      )}

      {/* ─────────── Confluence Effect ─────────── */}
      <ConfluenceCard rows={confluence} yearBadge={yearBadge} />

      {/* ─────────── Risk Metrics ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          🛡️ Risk Metrics {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          Actualized risk shape of the cohort — matches the New Entry page formula (position × stop = risk budget).
        </div>
        <div className="grid grid-cols-4 gap-3">
          <RiskTile label="Avg Stop Distance"
                    value={risk.avgStopDistancePct}
                    suffix="%"
                    tooltip="Average (entry − stop) / entry × 100 across trades with stop_loss set. Populated on ~55% of legacy trades; higher on new entries."
                    coverage={`${risk.nWithStop} / ${risk.n}`} />
          <RiskTile label="Avg Realized Loss"
                    value={risk.avgRealizedLossPct}
                    suffix="%"
                    tooltip="Average return_pct across losing trades. Signed (negative)."
                    coverage={`${risk.nLosers} losers`}
                    negative />
          <RiskTile label="Median Position Size"
                    value={risk.medianPositionSizePct}
                    suffix="%"
                    tooltip="Median total_cost / prior-day NAV × 100. Excludes trades without a prior-day journal row."
                    coverage={`${risk.nWithPositionSize} / ${risk.n}`} />
          <RiskTile label="Avg Risk / Trade"
                    value={risk.avgRiskPerTradePct}
                    suffix="%"
                    tooltip="Theoretical risk budget = avg position % × avg stop distance % / 100. Compares to the New Entry page's 0.5%-of-NAV target."
                    coverage="of NAV" />
        </div>
      </div>

      {/* ─────────── Pareto card ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-3">
          📊 Concentration of P&L (Pareto) {yearBadge}
        </div>
        {pareto.ranks.length === 0 ? (
          <div className="text-[13px]" style={{ color: "var(--ink-4)" }}>No trades in this cohort.</div>
        ) : (
          <ParetoBody pareto={pareto} trades={trades} />
        )}
      </div>

      {/* ─────────── Hold-time buckets ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-3">
          ⏱️ Hold-Time Buckets {yearBadge}
        </div>
        <div className="grid grid-cols-5 gap-3">
          {holdBuckets.map(b => {
            const color = b.n === 0 ? "var(--ink-4)" : b.netPl > 0 ? "#08a86b" : b.netPl < 0 ? "#e5484d" : "var(--ink-3)";
            return (
              <div key={b.label} className="p-3 rounded-[10px]"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{b.label}</div>
                <div className="text-[22px] font-extrabold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{b.n}</div>
                <div className="text-[11px] font-medium mt-0.5" style={{ color: "var(--ink-3)" }}>
                  {b.n > 0 ? `${b.winRate.toFixed(0)}% win` : "no trades"}
                </div>
                <div className="text-[11px] font-semibold mt-1 privacy-mask" style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
                  {formatCurrency(b.netPl, { showSign: true, decimals: 0 })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ─────────── Brandt NAV normalization ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-3">
          🧮 Brandt Normalization (% of prior-day NAV) {yearBadge}
        </div>
        {brandt.nWithNlv === 0 ? (
          <div className="text-[13px]" style={{ color: "var(--ink-4)" }}>
            No trades have a prior-day journal row to normalize against.
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3">
            <BrandtTile label="Avg trade (expectancy)" pct={brandt.avgTradePctNav} />
            <BrandtTile label="Avg winner" pct={brandt.avgWinPctNav} />
            <BrandtTile label="Avg loser" pct={brandt.avgLossPctNav} />
          </div>
        )}
        {brandt.nWithNlv < brandt.n && (
          <div className="text-[11px] mt-2" style={{ color: "var(--ink-4)" }}>
            {brandt.n - brandt.nWithNlv} of {brandt.n} trades excluded — no prior-day journal row.
          </div>
        )}
      </div>

      {/* ─────────── Repeat-Offenders ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          🔁 Repeat-Offender Tickers {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          Tickers attempted ≥ 3 times in this cohort, sorted by net P&L asc (biggest bleeders first).
          A name that stops you out repeatedly is telling you something about the name — or the read of it. Penalty-box candidates surface here.
        </div>
        {offenders.length === 0 ? (
          <div className="text-[13px]" style={{ color: "var(--ink-4)" }}>
            No tickers with ≥ 3 attempts in this cohort.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--bg)" }}>
                  <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Ticker</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Attempts</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>W / L</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Net P&L</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg / attempt</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Best</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Worst</th>
                </tr>
              </thead>
              <tbody>
                {offenders.map(o => {
                  const netColor = o.netPl > 0 ? "#08a86b" : o.netPl < 0 ? "#e5484d" : "var(--ink-3)";
                  const avgColor = o.avgPl > 0 ? "#08a86b" : o.avgPl < 0 ? "#e5484d" : "var(--ink-3)";
                  return (
                    <tr key={o.ticker} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-3 py-2 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{o.ticker}</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{o.attempts}</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-3)" }}>{o.wins}W / {o.losses}L</td>
                      <td className="px-3 py-2 text-right font-semibold privacy-mask"
                          style={{ color: netColor, fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(o.netPl, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ color: avgColor, fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(o.avgPl, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ color: "#08a86b", fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(o.bestPl, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ color: "#e5484d", fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(o.worstPl, { showSign: true, decimals: 0 })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ─────────── Regime cross-tab (open month × market window) ─────────── */}
      <div className="mb-4 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          🌦️ MCT × Month {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          Trades bucketed by M Factor state at entry (POWERTREND / UPTREND / CORRECTION / RALLY MODE). The PDF's April +$342K window sits under POWERTREND / UPTREND; June–July's −$33K bleed sits under CORRECTION.
        </div>
        {regime.length === 0 ? (
          <div className="text-[13px]" style={{ color: "var(--ink-4)" }}>No trades in this cohort.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--bg)" }}>
                  <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Month</th>
                  <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>MCT State</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>n</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Win %</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Net P&L</th>
                </tr>
              </thead>
              <tbody>
                {regime.map(r => {
                  const color = r.netPl > 0 ? "#08a86b" : r.netPl < 0 ? "#e5484d" : "var(--ink-3)";
                  return (
                    <tr key={`${r.month}|${r.window}`} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-3 py-2" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.month}</td>
                      <td className="px-3 py-2" style={{ color: "var(--ink-2)" }}>{r.window}</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.n}</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.winRate.toFixed(0)}%</td>
                      <td className="px-3 py-2 text-right font-semibold privacy-mask" style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
                        {formatCurrency(r.netPl, { showSign: true, decimals: 0 })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}


type ParetoSlice = "top5" | "top10" | "top20" | "rest";

function ParetoBody({ pareto, trades }: {
  pareto: ReturnType<typeof paretoDistribution>;
  trades: TradePosition[];
}) {
  const [selected, setSelected] = useState<ParetoSlice | null>(null);
  const top5 = pareto.topN(5);
  const top10 = pareto.topN(10);
  const top20 = pareto.topN(20);
  const total = pareto.ranks.length;
  const pctOfCount = (n: number) => total > 0 ? (n / total) * 100 : 0;
  const rest = useMemo(() => {
    if (total <= 10) return { net: 0, pctOfNet: 0, count: 0 };
    const restRanks = pareto.ranks.slice(10);
    const net = restRanks.reduce((a, r) => a + r.pnl, 0);
    return {
      net,
      pctOfNet: pareto.netPl !== 0 ? (net / pareto.netPl) * 100 : 0,
      count: restRanks.length,
    };
  }, [pareto, total]);

  const slice = useMemo(() => {
    if (!selected) return null;
    if (selected === "rest") return pareto.ranks.slice(10);
    const n = selected === "top5" ? 5 : selected === "top10" ? 10 : 20;
    return pareto.ranks.slice(0, Math.min(n, pareto.ranks.length));
  }, [selected, pareto.ranks]);

  const tradeById = useMemo(() => {
    const m = new Map<string, TradePosition>();
    for (const t of trades) m.set(String(t.trade_id || ""), t);
    return m;
  }, [trades]);

  return (
    <>
      <div className="grid grid-cols-4 gap-3 mb-4">
        <ParetoTile label="Top 5 trades" pct={pctOfCount(5)} value={top5.pctOfNet} netPl={top5.net}
                   selected={selected === "top5"} onClick={() => setSelected(selected === "top5" ? null : "top5")} />
        <ParetoTile label="Top 10 trades" pct={pctOfCount(10)} value={top10.pctOfNet} netPl={top10.net}
                   selected={selected === "top10"} onClick={() => setSelected(selected === "top10" ? null : "top10")} />
        <ParetoTile label="Top 20 trades" pct={pctOfCount(20)} value={top20.pctOfNet} netPl={top20.net}
                   selected={selected === "top20"} onClick={() => setSelected(selected === "top20" ? null : "top20")} />
        <ParetoTile label={`Trades 11+ (${rest.count})`} pct={total > 0 ? (rest.count / total) * 100 : 0} value={rest.pctOfNet} netPl={rest.net}
                   selected={selected === "rest"} onClick={() => setSelected(selected === "rest" ? null : "rest")} />
      </div>
      {pareto.netPl > 0 && pareto.breakevenRank != null && (
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
          Break-even at trade rank <strong style={{ fontFamily: "var(--font-jetbrains), monospace" }}>#{pareto.breakevenRank}</strong> — everything ranked past that adds no net P&L to the year. <span style={{ opacity: 0.6 }}>Click any tile to see the constituent trades.</span>
        </div>
      )}
      {/* Drill-down table — shows constituent trades for the selected tile. */}
      {slice && (
        <div className="mb-4 rounded-[10px]" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <div className="px-3 py-2 text-[11px] uppercase font-bold" style={{ color: "var(--ink-4)", borderBottom: "1px solid var(--border)" }}>
            {slice.length} trades · click Trade Journal to open a trade
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--surface)" }}>
                  <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Rank</th>
                  <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Ticker</th>
                  <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Trade ID</th>
                  <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Setup</th>
                  <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Open</th>
                  <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Closed</th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>P&L</th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>% of Net</th>
                </tr>
              </thead>
              <tbody>
                {slice.map((r, i) => {
                  const t = tradeById.get(r.trade_id);
                  const rank = selected === "rest" ? 11 + i : i + 1;
                  const pctOfNet = pareto.netPl !== 0 ? (r.pnl / pareto.netPl) * 100 : 0;
                  return (
                    <tr key={r.trade_id + "-" + i} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="px-2 py-1 text-left" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-3)" }}>#{rank}</td>
                      <td className="px-2 py-1 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{r.ticker}</td>
                      <td className="px-2 py-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-3)" }}>{r.trade_id}</td>
                      <td className="px-2 py-1" style={{ color: "var(--ink-2)" }}>{t?.rule || "—"}</td>
                      <td className="px-2 py-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-3)" }}>{String(t?.open_date || "").slice(0, 10)}</td>
                      <td className="px-2 py-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-3)" }}>{String(t?.closed_date || "").slice(0, 10) || "—"}</td>
                      <td className="px-2 py-1 text-right font-semibold privacy-mask"
                          style={{ fontFamily: "var(--font-jetbrains), monospace", color: r.pnl >= 0 ? "#08a86b" : "#e5484d" }}>
                        {formatCurrency(r.pnl, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-2 py-1 text-right"
                          style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctOfNet >= 0 ? "#08a86b" : "#e5484d" }}>
                        {pctOfNet >= 0 ? "+" : ""}{pctOfNet.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {/* Cumulative curve — pure CSS. Show as a small horizontal bar per
          trade with height proportional to cum% of net. Capped at 100
          rendered bars to keep the row cheap. */}
      <div className="mt-2 flex items-end gap-[1px]" style={{ height: 60, borderBottom: "1px solid var(--border)" }}>
        {pareto.ranks.slice(0, 100).map((r, i) => {
          // Clamp cumPctOfNet to [-100, 200] for visual — anything past
          // net (>100%) reads as "adding P&L we later gave back."
          const raw = r.cumPctOfNet;
          const capped = Math.min(200, Math.max(-100, raw));
          const height = Math.abs(capped) / 200 * 100;
          const color = capped >= 100 ? "#08a86b" : capped >= 0 ? "#65a30d" : "#e5484d";
          return (
            <div key={i} className="flex-1"
                 title={`#${i + 1} ${r.ticker} ${r.trade_id} · cum ${r.cumPctOfNet.toFixed(0)}% of net`}
                 style={{ height: `${Math.max(2, height)}%`, background: color, opacity: 0.85 }} />
          );
        })}
      </div>
    </>
  );
}


function ParetoTile({ label, pct, value, netPl, selected, onClick }: {
  label: string;
  pct: number;
  value: number;
  netPl: number;
  selected: boolean;
  onClick: () => void;
}) {
  const isNegative = value < 0;
  const color = isNegative ? "#e5484d"
              : value >= 100 ? "#08a86b"
              : value >= 50 ? "#d97706"
              : "var(--ink-2)";
  return (
    <button type="button"
            onClick={onClick}
            className="p-3 rounded-[10px] text-left transition-all"
            style={{
              background: selected ? "color-mix(in oklab, #0d6efd 10%, var(--bg))" : "var(--bg)",
              border: `1px solid ${selected ? "#0d6efd" : "var(--border)"}`,
              cursor: "pointer",
              boxShadow: selected ? "0 0 0 2px color-mix(in oklab, #0d6efd 20%, transparent)" : "none",
            }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>
        {label} <span style={{ opacity: 0.6 }}>({pct.toFixed(1)}%)</span>
      </div>
      <div className="text-[22px] font-extrabold mt-1" style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
        {value >= 0 ? "+" : ""}{value.toFixed(0)}%
      </div>
      <div className="text-[11px] privacy-mask" style={{ color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>
        {formatCurrency(netPl, { showSign: true, decimals: 0 })}
      </div>
    </button>
  );
}


// ═══════════════════════════════════════════════════════════════════════
// ConfluenceCard — pair analysis with click-to-drill on each row.
// Empty state renders a "no confluence yet" pitch pointing at Log Buy.
// ═══════════════════════════════════════════════════════════════════════

function ConfluenceCard({ rows, yearBadge }: {
  rows: ConfluenceRow[];
  yearBadge: React.ReactNode;
}) {
  const mono = "var(--font-jetbrains), monospace";
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="mb-5 p-5 rounded-[14px]"
         style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="text-[15px] font-semibold mb-1">
        🔀 Confluence Effect {yearBadge}
      </div>
      <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
        Do secondary tags add edge? Each row compares a (primary + confluence) pair against the primary rule alone.
        <strong> Lift columns are the answer</strong> — positive = the confluence tag helped; negative = noise or worse.
        A trade tagged with 2 confluence rules contributes to 2 rows (pair-level, not triples).
      </div>
      {rows.length === 0 ? (
        <div className="text-[12px] italic mt-2 p-3 rounded-[10px]"
             style={{ color: "var(--ink-4)", background: "var(--bg)", border: "1px solid var(--border)" }}>
          No confluence tags in this cohort. Tag secondary rules in Log Buy or Trade Manager → Edit Transaction
          to unlock this analysis. Pairs with n &lt; 3 are also filtered out to avoid single-trade noise.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--bg)" }}>
                <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Primary → Confluence</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>n</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Pair Win %</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Baseline Win %</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Δ Win %</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Pair Avg P&L</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Δ Avg P&L</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Net P&L</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => {
                const key = `${r.primary}||${r.confluence}`;
                const isOpen = expanded === key;
                const liftWColor = r.liftWinPct > 0 ? "#08a86b" : r.liftWinPct < 0 ? "#e5484d" : "var(--ink-3)";
                const liftPColor = r.liftAvgPl > 0 ? "#08a86b" : r.liftAvgPl < 0 ? "#e5484d" : "var(--ink-3)";
                const netColor = r.netPl > 0 ? "#08a86b" : r.netPl < 0 ? "#e5484d" : "var(--ink-3)";
                return (
                  <React.Fragment key={key}>
                    <tr onClick={() => setExpanded(isOpen ? null : key)}
                        style={{ borderBottom: "1px solid var(--border)", cursor: "pointer",
                                 background: isOpen ? "color-mix(in oklab, #0d6efd 5%, transparent)" : "transparent" }}>
                      <td className="px-3 py-2 font-semibold">
                        <span style={{ display: "inline-block", width: 14, textAlign: "center", color: "var(--ink-4)" }}>{isOpen ? "▾" : "▸"}</span>
                        <span style={{ fontFamily: mono, color: "var(--ink-2)" }}>{r.primary}</span>
                        <span style={{ color: "var(--ink-4)", margin: "0 4px" }}>→</span>
                        <span style={{ fontFamily: mono, color: "#6366f1" }}>+{r.confluence}</span>
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{r.n}</td>
                      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono }}>{r.winPct.toFixed(0)}%</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                        {r.primaryAloneWinPct.toFixed(0)}% <span style={{ opacity: 0.5 }}>(n={r.primaryAloneN})</span>
                      </td>
                      <td className="px-3 py-2 text-right font-semibold"
                          style={{ fontFamily: mono, color: liftWColor }}>
                        {r.liftWinPct >= 0 ? "+" : ""}{r.liftWinPct.toFixed(0)} pp
                      </td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ fontFamily: mono, color: r.avgPlPerTrade >= 0 ? "#08a86b" : "#e5484d" }}>
                        {formatCurrency(r.avgPlPerTrade, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold privacy-mask"
                          style={{ fontFamily: mono, color: liftPColor }}>
                        {formatCurrency(r.liftAvgPl, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold privacy-mask"
                          style={{ fontFamily: mono, color: netColor }}>
                        {formatCurrency(r.netPl, { showSign: true, decimals: 0 })}
                      </td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={8} style={{ padding: 0, background: "var(--bg)" }}>
                          <div className="px-6 py-4">
                            <div className="text-[11px] uppercase font-bold mb-2" style={{ color: "var(--ink-4)" }}>
                              {r.n} trades tagged {r.primary} + {r.confluence} · sorted by realized P&L desc
                            </div>
                            <MaeMfeTradesTable title="" trades={r.trades} sortBy="return-asc" compact />
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}


const SEVERITY_STYLE: Record<Insight["severity"], { bg: string; border: string; badge: string; badgeFg: string; label: string }> = {
  critical: { bg: "color-mix(in oklab, #e5484d 10%, var(--bg))", border: "#e5484d", badge: "#e5484d", badgeFg: "#fff", label: "CRITICAL" },
  warning:  { bg: "color-mix(in oklab, #f59f00 10%, var(--bg))", border: "#f59f00", badge: "#f59f00", badgeFg: "#000", label: "WARNING" },
  info:     { bg: "color-mix(in oklab, #0d6efd 8%, var(--bg))",  border: "#0d6efd", badge: "#0d6efd", badgeFg: "#fff", label: "INFO" },
};

function InsightCard({ insight }: { insight: Insight }) {
  const [open, setOpen] = useState(false);
  const style = SEVERITY_STYLE[insight.severity];
  const mono = "var(--font-jetbrains), monospace";
  const hasItems = insight.items && insight.items.length > 0;
  return (
    <div className="rounded-[10px] p-3"
         style={{ background: style.bg, border: `1px solid ${style.border}` }}>
      <div className="flex items-start gap-3">
        <div className="text-[22px] leading-none pt-0.5">{insight.icon}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span style={{
              display: "inline-block", padding: "2px 6px", borderRadius: 4,
              background: style.badge, color: style.badgeFg,
              fontSize: 9, fontWeight: 700, letterSpacing: "0.05em",
            }}>{style.label}</span>
            <span className="text-[14px] font-bold">{insight.title}</span>
            {insight.impactDollars != null && (
              <span className="text-[13px] font-semibold privacy-mask"
                    style={{ color: insight.impactDollars >= 0 ? "#08a86b" : "#e5484d", fontFamily: mono }}>
                {formatCurrency(insight.impactDollars, { showSign: true, decimals: 0 })}
              </span>
            )}
          </div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-2)" }}>{insight.detail}</div>
          {hasItems && (
            <>
              <button type="button" onClick={() => setOpen(o => !o)}
                      className="text-[11px] mt-2 underline"
                      style={{ color: "var(--ink-3)", cursor: "pointer", background: "none", border: "none", padding: 0 }}>
                {open ? "▾ Hide" : "▸ Show"} {insight.items!.length} item{insight.items!.length === 1 ? "" : "s"}
              </button>
              {open && (
                <div className="mt-2 flex flex-col gap-1">
                  {insight.items!.map((it, i) => (
                    <div key={i} className="flex items-center justify-between text-[11px] px-2 py-1 rounded"
                         style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                      <div>
                        <span className="font-semibold" style={{ fontFamily: mono }}>{it.label}</span>
                        {it.detail && <span className="ml-2" style={{ color: "var(--ink-4)" }}>{it.detail}</span>}
                      </div>
                      <div className="privacy-mask" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                        {it.netPl != null && (
                          <span style={{ color: it.netPl >= 0 ? "#08a86b" : "#e5484d", fontWeight: 600 }}>
                            {formatCurrency(it.netPl, { showSign: true, decimals: 0 })}
                          </span>
                        )}
                        {it.pct != null && <span>{it.pct >= 0 ? "+" : ""}{it.pct.toFixed(1)}%</span>}
                        {it.count != null && <span>{it.count}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}


function RiskTile({ label, value, suffix, tooltip, coverage, negative }: {
  label: string;
  value: number | null;
  suffix: string;
  tooltip: string;
  coverage: string;
  negative?: boolean;
}) {
  const color = value == null ? "var(--ink-4)"
              : negative ? "#e5484d"
              : "var(--ink)";
  return (
    <div className="p-3 rounded-[10px]" title={tooltip}
         style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[22px] font-extrabold mt-1" style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
        {value == null ? "—" : `${negative && value > 0 ? "-" : ""}${value.toFixed(2)}${suffix}`}
      </div>
      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{coverage}</div>
    </div>
  );
}


function BrandtTile({ label, pct }: { label: string; pct: number | null }) {
  const color = pct == null ? "var(--ink-4)" : pct > 0 ? "#08a86b" : pct < 0 ? "#e5484d" : "var(--ink-3)";
  return (
    <div className="p-3 rounded-[10px]"
         style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[22px] font-extrabold mt-1" style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
        {pct == null ? "—" : `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`}
      </div>
      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>of prior-day NAV</div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════════
// Setup Scorecard tab — per-rule PF + auto-verdict + drilldown
// (Old Stop-Cap and Fixed-Size scenarios retired 2026-07-14 — both were
//  mathematically tautological; setup scorecard is the actionable view.)
// ═══════════════════════════════════════════════════════════════════════

const VERDICT_STYLE: Record<SetupVerdict, { bg: string; fg: string; label: string; title: string }> = {
  "core":    { bg: "#08a86b", fg: "#fff",     label: "CORE",    title: "PF ≥ 5 with meaningful n — keep and lean into" },
  "keep":    { bg: "#0d6efd", fg: "#fff",     label: "KEEP",    title: "PF ≥ 2 — solid contributor" },
  "watch":   { bg: "#f59f00", fg: "#000",     label: "WATCH",   title: "PF between 1 and 2 — marginal, monitor" },
  "kill":    { bg: "#e5484d", fg: "#fff",     label: "KILL",    title: "PF < 1 — losing money net; strong candidate for retirement or regime gate" },
  "small-n": { bg: "#e0e0e0", fg: "#666",     label: "SMALL n", title: "Sample too small (n < 5) — no verdict rendered" },
};

function ScenariosTab({ trades, journal: _journal, year, cohort, navColor: _navColor }: {
  trades: TradePosition[];
  journal: any[];
  year: number;
  cohort: "closed" | "at-mark";
  navColor: string;
}) {
  const scorecard = useMemo(() => setupScorecard(trades), [trades]);
  const anyMae = useMemo(() => trades.some(t => (t as any).mae_pct != null), [trades]);
  const winnerDist = useMemo(() => winnerMaeDistribution(trades), [trades]);
  const loserDist = useMemo(() => loserMfeDistribution(trades), [trades]);
  const entryQuality = useMemo(() => entryQualityBySetup(trades), [trades]);
  const stopCapMae = useMemo(() => stopCapScenario(trades, [3, 5, 7, 8, 10], {
    maePctOf: (t: any) => (t.mae_pct != null ? Number(t.mae_pct) : null),
  }), [trades]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const yearBadge = (
    <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>
      — {year} · {cohort === "closed" ? "closed only" : "including open @ mark"}
    </span>
  );
  const qualifying = scorecard.filter(r => r.verdict !== "small-n");
  const smallN = scorecard.filter(r => r.verdict === "small-n");
  const topFiveIds = new Set(qualifying.slice(0, 5).map(r => r.setup));
  const bottomFiveIds = new Set(qualifying.slice(-5).map(r => r.setup));
  const mono = "var(--font-jetbrains), monospace";

  return (
    <>
      <div className="text-[13px] mb-4" style={{ color: "var(--ink-3)" }}>
        One row per setup ({`rule`} field). Sorted by profit factor. Verdict thresholds:
        <strong> Core</strong> ≥ 5 · <strong>Keep</strong> ≥ 2 · <strong>Watch</strong> ≥ 1 · <strong>Kill</strong> &lt; 1. Setups with fewer than 5 trades are parked at the bottom without a verdict.
        Click any row to see the constituent trades.
      </div>

      {/* ─────────── Setup Scorecard ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          🏆 Setup Scorecard {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          Top {Math.min(5, qualifying.length)} highlighted green, bottom {Math.min(5, qualifying.length)} highlighted red. PF = ∞ means all wins, no losses.
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--bg)" }}>
                <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Setup</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>n</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Win %</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Gross Win</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Gross Loss</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>PF</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Net P&L</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Expectancy</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg Hold</th>
                <th className="px-3 py-2 text-center text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {scorecard.map(r => {
                const isTop = topFiveIds.has(r.setup);
                const isBot = bottomFiveIds.has(r.setup);
                const rowBg = isTop ? "color-mix(in oklab, #08a86b 8%, transparent)"
                            : isBot ? "color-mix(in oklab, #e5484d 8%, transparent)"
                            : "transparent";
                const isOpen = expanded === r.setup;
                const style = VERDICT_STYLE[r.verdict];
                return (
                  <>
                    <tr key={r.setup}
                        onClick={() => setExpanded(isOpen ? null : r.setup)}
                        style={{ borderBottom: "1px solid var(--border)", background: rowBg, cursor: "pointer" }}>
                      <td className="px-3 py-2 font-semibold" style={{ color: "var(--ink)" }}>
                        <span style={{ display: "inline-block", width: 14, textAlign: "center", color: "var(--ink-4)" }}>{isOpen ? "▾" : "▸"}</span>
                        {r.setup}
                        {r.confluenceTradeCount > 0 && (
                          <span className="ml-2 inline-flex items-center h-[18px] px-1.5 rounded-[5px] text-[10px] font-medium"
                                title={`${r.confluenceTradeCount} of ${r.n} trades have confluence tags — see the Confluence Effect card on the Overview tab`}
                                style={{
                                  background: "color-mix(in oklab, #6366f1 10%, transparent)",
                                  color: "#6366f1",
                                  border: "1px solid color-mix(in oklab, #6366f1 20%, var(--border))",
                                  fontFamily: mono,
                                }}>
                            +{r.confluenceTradeCount} conf
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{r.n}</td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{r.winPct.toFixed(0)}%</td>
                      <td className="px-3 py-2 text-right privacy-mask" style={{ fontFamily: mono, color: "#08a86b" }}>{formatCurrency(r.grossWin, { showSign: false, decimals: 0 })}</td>
                      <td className="px-3 py-2 text-right privacy-mask" style={{ fontFamily: mono, color: "#e5484d" }}>{formatCurrency(-r.grossLoss, { showSign: false, decimals: 0 })}</td>
                      <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono }}>
                        {r.profitFactor === Infinity ? "∞" : r.profitFactor.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold privacy-mask"
                          style={{ fontFamily: mono, color: r.netPl >= 0 ? "#08a86b" : "#e5484d" }}>
                        {formatCurrency(r.netPl, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ fontFamily: mono, color: r.expectancy >= 0 ? "#08a86b" : "#e5484d" }}>
                        {formatCurrency(r.expectancy, { showSign: true, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                        {r.avgHoldDays == null ? "—" : `${r.avgHoldDays.toFixed(0)}d`}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span title={style.title}
                              style={{
                                display: "inline-block",
                                padding: "2px 8px",
                                borderRadius: 6,
                                background: style.bg,
                                color: style.fg,
                                fontSize: 10,
                                fontWeight: 700,
                                letterSpacing: "0.05em",
                              }}>
                          {style.label}
                        </span>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr key={r.setup + "-open"}>
                        <td colSpan={10} style={{ padding: 0, background: "var(--bg)" }}>
                          <SetupDrilldown row={r} />
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
              {scorecard.length === 0 && (
                <tr>
                  <td colSpan={10} className="px-3 py-6 text-center text-[12px]" style={{ color: "var(--ink-4)" }}>
                    No setups in this cohort.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {smallN.length > 0 && (
          <div className="text-[11px] mt-3 italic" style={{ color: "var(--ink-4)" }}>
            {smallN.length} setup{smallN.length === 1 ? "" : "s"} with n &lt; 5 shown at bottom without verdict.
          </div>
        )}
      </div>

      {/* ─────────── Excursion section — real content when MAE data is present ─────────── */}
      {!anyMae ? (
        <div className="mb-4 p-5 rounded-[14px]"
             style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <div className="text-[15px] font-semibold mb-1">
            📉 Excursion (MAE/MFE-aware) {yearBadge}
          </div>
          <div className="text-[12px] italic mt-2" style={{ color: "var(--ink-4)" }}>
            Awaiting excursion data — MAE/MFE feature runs the daily reconciler on open positions and requires a Phase-2 backfill for closed trades. This section will populate once mae_pct / mfe_pct fields are present on the trade rows in this cohort.
          </div>
        </div>
      ) : (
        <ExcursionSection
          yearBadge={yearBadge}
          winnerDist={winnerDist}
          loserDist={loserDist}
          entryQuality={entryQuality}
          stopCapMae={stopCapMae}
        />
      )}
    </>
  );
}


function ExcursionSection({ yearBadge, winnerDist, loserDist, entryQuality, stopCapMae }: {
  yearBadge: React.ReactNode;
  winnerDist: ReturnType<typeof winnerMaeDistribution>;
  loserDist: ReturnType<typeof loserMfeDistribution>;
  entryQuality: EntryQualityRow[];
  stopCapMae: ReturnType<typeof stopCapScenario>;
}) {
  const mono = "var(--font-jetbrains), monospace";
  // One expansion slot per section — click again on the same key to collapse.
  const [winnerOpen, setWinnerOpen] = useState<string | null>(null);
  const [loserOpen, setLoserOpen] = useState<string | null>(null);
  const [stopCapOpen, setStopCapOpen] = useState<number | null>(null);
  const [qualityOpen, setQualityOpen] = useState<string | null>(null);

  return (
    <>
      {/* ─────────── Winner-MAE Distribution ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          📉 Winner MAE Distribution {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          How deep did winners dip before working? Answers "how much pain do I have to tolerate to catch the average winner on this book?"
          <strong> n = {winnerDist.n} winners with MAE populated.</strong> Click any bucket to see the constituent trades.
        </div>
        {winnerDist.n === 0 ? (
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No winners with MAE data in this cohort.</div>
        ) : (
          <>
            <div className="grid grid-cols-5 gap-3">
              {winnerDist.buckets.map(b => (
                <ExcursionBucketTile key={b.label} bucket={b} color="#08a86b"
                                     selected={winnerOpen === b.label}
                                     onClick={() => setWinnerOpen(winnerOpen === b.label ? null : b.label)} />
              ))}
            </div>
            {winnerOpen && (
              <MaeMfeTradesTable
                title={`${winnerDist.buckets.find(b => b.label === winnerOpen)?.n ?? 0} winners with |MAE| in ${winnerOpen}`}
                trades={winnerDist.buckets.find(b => b.label === winnerOpen)?.trades ?? []}
                sortBy="mae-asc" />
            )}
          </>
        )}
      </div>

      {/* ─────────── Loser-MFE Distribution ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          📈 Loser MFE Distribution {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          How far did losers rally before rolling over? A big tail here is the "let winners turn into losers" pattern — candidate for a "trail once above X%" rule.
          <strong> n = {loserDist.n} losers with MFE populated.</strong> Click any bucket to see the constituent trades.
        </div>
        {loserDist.n === 0 ? (
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No losers with MFE data in this cohort.</div>
        ) : (
          <>
            <div className="grid grid-cols-5 gap-3">
              {loserDist.buckets.map(b => (
                <ExcursionBucketTile key={b.label} bucket={b} color="#e5484d"
                                     selected={loserOpen === b.label}
                                     onClick={() => setLoserOpen(loserOpen === b.label ? null : b.label)} />
              ))}
            </div>
            {loserOpen && (
              <MaeMfeTradesTable
                title={`${loserDist.buckets.find(b => b.label === loserOpen)?.n ?? 0} losers with MFE in ${loserOpen}`}
                trades={loserDist.buckets.find(b => b.label === loserOpen)?.trades ?? []}
                sortBy="mfe-desc" />
            )}
          </>
        )}
      </div>

      {/* ─────────── MAE-Aware Stop-Cap Backtest ─────────── */}
      <div className="mb-5 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          🛡️ Stop-Cap Backtest (MAE-aware) {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          For each hard stop cap X, this shows: <strong>$ saved</strong> from losers whose realized loss exceeded X (capping them there), MINUS <strong>$ foregone</strong> from winners whose MAE dipped below X (clipping them out). The <strong>Net delta</strong> is the honest answer. Click a row to see the trades on both sides.
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--bg)" }}>
                <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Cap</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Losers breaching</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>$ saved</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Winners clipped</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>$ foregone</th>
                <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Net delta</th>
              </tr>
            </thead>
            <tbody>
              {stopCapMae.map(r => {
                const net = r.dollarsSaved - r.clippedWinnerForegonePl;
                const netColor = net > 0 ? "#08a86b" : net < 0 ? "#e5484d" : "var(--ink-3)";
                const isOpen = stopCapOpen === r.capPct;
                return (
                  <>
                    <tr key={r.capPct}
                        onClick={() => setStopCapOpen(isOpen ? null : r.capPct)}
                        style={{ borderBottom: "1px solid var(--border)", cursor: "pointer",
                                 background: isOpen ? "color-mix(in oklab, #0d6efd 5%, transparent)" : "transparent" }}>
                      <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>
                        <span style={{ display: "inline-block", width: 14, textAlign: "center", color: "var(--ink-4)" }}>{isOpen ? "▾" : "▸"}</span>
                        −{r.capPct}%
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{r.breachCount}</td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ fontFamily: mono, color: "#08a86b" }}>
                        {formatCurrency(r.dollarsSaved, { showSign: false, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{r.clippedWinnerCount}</td>
                      <td className="px-3 py-2 text-right privacy-mask"
                          style={{ fontFamily: mono, color: "#e5484d" }}>
                        {formatCurrency(-r.clippedWinnerForegonePl, { showSign: false, decimals: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right font-semibold privacy-mask"
                          style={{ fontFamily: mono, color: netColor }}>
                        {formatCurrency(net, { showSign: true, decimals: 0 })}
                      </td>
                    </tr>
                    {isOpen && (
                      <tr key={r.capPct + "-open"}>
                        <td colSpan={6} style={{ padding: 0, background: "var(--bg)" }}>
                          <div className="px-6 py-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <div className="text-[11px] uppercase font-bold mb-2" style={{ color: "#e5484d" }}>
                                Losers breaching −{r.capPct}% ({r.losersBreaching.length})
                              </div>
                              <MaeMfeTradesTable title="" trades={r.losersBreaching} sortBy="return-asc" compact />
                            </div>
                            <div>
                              <div className="text-[11px] uppercase font-bold mb-2" style={{ color: "#08a86b" }}>
                                Winners clipped by −{r.capPct}% ({r.winnersClipped.length})
                              </div>
                              <MaeMfeTradesTable title="" trades={r.winnersClipped} sortBy="mae-asc" compact />
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="text-[11px] mt-3" style={{ color: "var(--ink-4)" }}>
          Winners are "clipped" when their MAE dipped below the cap during holding — that trade would have stopped out before working. Foregone P&L = the realized P&L that trade eventually delivered.
        </div>
      </div>

      {/* ─────────── Entry Quality by Setup ─────────── */}
      <div className="mb-4 p-5 rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="text-[15px] font-semibold mb-1">
          🎯 Entry Quality by Setup {yearBadge}
        </div>
        <div className="text-[12px] mb-3" style={{ color: "var(--ink-4)" }}>
          Per-setup MAE / MFE. Sorted by <strong>avg winner MAE ascending</strong> — most punishing setups first. "Winner MAE" is the drawdown a working entry subjects you to before paying off; a big number means valid setups need a wide stop or you'll shake yourself out. Click a row to see the trades tagged that setup.
        </div>
        {entryQuality.length === 0 ? (
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No setups with ≥ 5 trades and MAE data in this cohort.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--bg)" }}>
                  <th className="px-3 py-2 text-left text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Setup</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>n</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg MAE</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Winner MAE</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Worst MAE</th>
                  <th className="px-3 py-2 text-right text-[10px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Avg MFE</th>
                </tr>
              </thead>
              <tbody>
                {entryQuality.map(r => {
                  const isOpen = qualityOpen === r.setup;
                  return (
                    <>
                      <tr key={r.setup}
                          onClick={() => setQualityOpen(isOpen ? null : r.setup)}
                          style={{ borderBottom: "1px solid var(--border)", cursor: "pointer",
                                   background: isOpen ? "color-mix(in oklab, #0d6efd 5%, transparent)" : "transparent" }}>
                        <td className="px-3 py-2 font-semibold">
                          <span style={{ display: "inline-block", width: 14, textAlign: "center", color: "var(--ink-4)" }}>{isOpen ? "▾" : "▸"}</span>
                          {r.setup}
                        </td>
                        <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>{r.n}</td>
                        <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "#e5484d" }}>
                          {r.avgMae == null ? "—" : `${r.avgMae.toFixed(2)}%`}
                        </td>
                        <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: mono, color: "#e5484d" }}>
                          {r.avgMaeOnWinners == null ? "—" : `${r.avgMaeOnWinners.toFixed(2)}%`}
                        </td>
                        <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "#e5484d" }}>
                          {r.worstMae == null ? "—" : `${r.worstMae.toFixed(2)}%`}
                        </td>
                        <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "#08a86b" }}>
                          {r.avgMfe == null ? "—" : `+${r.avgMfe.toFixed(2)}%`}
                        </td>
                      </tr>
                      {isOpen && (
                        <tr key={r.setup + "-open"}>
                          <td colSpan={6} style={{ padding: 0, background: "var(--bg)" }}>
                            <div className="px-6 py-4">
                              <div className="text-[11px] uppercase font-bold mb-2" style={{ color: "var(--ink-4)" }}>
                                {r.n} trades tagged {r.setup} · sorted by MAE (worst first)
                              </div>
                              <MaeMfeTradesTable title="" trades={r.trades} sortBy="mae-asc" compact />
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}


function ExcursionBucketTile({ bucket, color, selected, onClick }: {
  bucket: { label: string; n: number; pct: number };
  color: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick}
            className="p-3 rounded-[10px] text-left transition-all"
            style={{
              background: selected ? "color-mix(in oklab, #0d6efd 10%, var(--bg))" : "var(--bg)",
              border: `1px solid ${selected ? "#0d6efd" : "var(--border)"}`,
              cursor: "pointer",
              boxShadow: selected ? "0 0 0 2px color-mix(in oklab, #0d6efd 20%, transparent)" : "none",
            }}>
      <div className="text-[10px] font-bold uppercase tracking-[0.08em]" style={{ color: "var(--ink-4)" }}>
        {bucket.label}
      </div>
      <div className="text-[22px] font-extrabold mt-1" style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
        {bucket.n}
      </div>
      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{bucket.pct.toFixed(0)}%</div>
    </button>
  );
}


/** Shared MAE/MFE-aware trades table used by all four Excursion drilldowns.
 *  sortBy:
 *    - "mae-asc"    → most negative MAE first (worst drawdown surfaces)
 *    - "mfe-desc"   → largest MFE first (biggest give-back surfaces)
 *    - "return-asc" → most negative return_pct first (biggest realized loss) */
function MaeMfeTradesTable({ title, trades, sortBy, compact }: {
  title: string;
  trades: TradePosition[];
  sortBy: "mae-asc" | "mfe-desc" | "return-asc";
  compact?: boolean;
}) {
  const mono = "var(--font-jetbrains), monospace";
  const sorted = useMemo(() => {
    const key = (t: TradePosition): number => {
      const anyT = t as any;
      if (sortBy === "mae-asc") return Number(anyT.mae_pct ?? 0);
      if (sortBy === "mfe-desc") return -Number(anyT.mfe_pct ?? 0);
      // return-asc
      return Number(anyT.return_pct ?? 0);
    };
    return [...trades].sort((a, b) => key(a) - key(b));
  }, [trades, sortBy]);
  if (sorted.length === 0) {
    return (
      <div className="text-[11px] italic px-3 py-3" style={{ color: "var(--ink-4)" }}>
        (no trades)
      </div>
    );
  }
  return (
    <div className={compact ? "" : "mt-3 rounded-[10px]"}
         style={compact ? undefined : { background: "var(--bg)", border: "1px solid var(--border)" }}>
      {!compact && title && (
        <div className="px-3 py-2 text-[11px] uppercase font-bold"
             style={{ color: "var(--ink-4)", borderBottom: "1px solid var(--border)" }}>
          {title}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: compact ? "var(--surface)" : "var(--surface)" }}>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Ticker</th>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Trade ID</th>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Setup</th>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Open</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>MAE</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>MFE</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Return %</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Net P&L</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(t => {
              const anyT = t as any;
              const mae = anyT.mae_pct != null ? Number(anyT.mae_pct) : null;
              const mfe = anyT.mfe_pct != null ? Number(anyT.mfe_pct) : null;
              const rp = anyT.return_pct != null ? Number(anyT.return_pct) : null;
              const pl = parseFloat(String(t.realized_pl || 0));
              return (
                <tr key={t.trade_id} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td className="px-2 py-1 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                  <td className="px-2 py-1" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{t.trade_id}</td>
                  <td className="px-2 py-1" style={{ color: "var(--ink-2)" }}>{t.rule || "—"}</td>
                  <td className="px-2 py-1" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{String(t.open_date || "").slice(0, 10)}</td>
                  <td className="px-2 py-1 text-right" style={{ fontFamily: mono, color: "#e5484d" }}>
                    {mae == null ? "—" : `${mae.toFixed(2)}%`}
                  </td>
                  <td className="px-2 py-1 text-right" style={{ fontFamily: mono, color: "#08a86b" }}>
                    {mfe == null ? "—" : `+${mfe.toFixed(2)}%`}
                  </td>
                  <td className="px-2 py-1 text-right"
                      style={{ fontFamily: mono, color: rp == null ? "var(--ink-4)" : rp >= 0 ? "#08a86b" : "#e5484d" }}>
                    {rp == null ? "—" : `${rp >= 0 ? "+" : ""}${rp.toFixed(2)}%`}
                  </td>
                  <td className="px-2 py-1 text-right font-semibold privacy-mask"
                      style={{ fontFamily: mono, color: pl >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(pl, { showSign: true, decimals: 0 })}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SetupDrilldown({ row }: { row: SetupScorecardRow }) {
  const mono = "var(--font-jetbrains), monospace";
  const sorted = useMemo(() =>
    [...row.trades].sort((a, b) => parseFloat(String(b.realized_pl || 0)) - parseFloat(String(a.realized_pl || 0))),
    [row.trades],
  );
  const holdDays = (t: TradePosition): number | null => {
    const o = String(t.open_date || "").trim();
    const c = String(t.closed_date || "").trim();
    if (!o || !c) return null;
    const od = new Date(o).getTime();
    const cd = new Date(c).getTime();
    if (isNaN(od) || isNaN(cd)) return null;
    return Math.max(0, Math.floor((cd - od) / 86_400_000));
  };
  return (
    <div className="px-6 py-4">
      <div className="text-[11px] uppercase font-bold mb-2" style={{ color: "var(--ink-4)" }}>
        {row.n} trades tagged {row.setup} · click ticker to open Trade Journal
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "var(--surface)" }}>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Ticker</th>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Trade ID</th>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Open</th>
              <th className="px-2 py-1 text-left text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Closed</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Hold</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Net P&L</th>
              <th className="px-2 py-1 text-right text-[10px] uppercase" style={{ color: "var(--ink-4)" }}>Return %</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(t => {
              const pl = parseFloat(String(t.realized_pl || 0));
              const rp = Number((t as any).return_pct ?? 0);
              const hd = holdDays(t);
              return (
                <tr key={t.trade_id} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td className="px-2 py-1 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                  <td className="px-2 py-1" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{t.trade_id}</td>
                  <td className="px-2 py-1" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{String(t.open_date || "").slice(0, 10)}</td>
                  <td className="px-2 py-1" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{String(t.closed_date || "").slice(0, 10) || "—"}</td>
                  <td className="px-2 py-1 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{hd == null ? "—" : `${hd}d`}</td>
                  <td className="px-2 py-1 text-right font-semibold privacy-mask"
                      style={{ fontFamily: mono, color: pl >= 0 ? "#08a86b" : "#e5484d" }}>
                    {formatCurrency(pl, { showSign: true, decimals: 0 })}
                  </td>
                  <td className="px-2 py-1 text-right"
                      style={{ fontFamily: mono, color: rp >= 0 ? "#08a86b" : "#e5484d" }}>
                    {rp >= 0 ? "+" : ""}{rp.toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
