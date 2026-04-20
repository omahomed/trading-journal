"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { api, type TradePosition, type TradeDetail } from "@/lib/api";

interface EnrichedPosition {
  trade_id: string;
  ticker: string;
  shares: number;
  avg_entry: number;
  total_cost: number;
  realized_pl: number;
  rule: string;
  buy_notes: string;
  risk_budget: number;
  open_date: string;
  days_held: number;
  avg_stop: number;
  risk_dollars: number;
  risk_pct: number;
  current_price: number;
  current_value: number;
  unrealized_pl: number;
  overall_pl: number;
  return_pct: number;
  pos_size_pct: number;
  is_option: boolean;
  opt_mult: number;
  pyramid_pct: number;
  risk_status: string;
  projected_pl: number;
  realized_bank: number;
  open_risk_equity: number;
  stop_pct: number;
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

/**
 * LIFO engine — mirrors the Streamlit run_lifo_engine exactly.
 * Processes buy/sell transactions in date order, LIFO matching for sells,
 * returns risk, avg stop, avg cost, projected floor, and realized bank.
 */
function runLifoEngine(
  tradeDetails: TradeDetail[],
  summaryEntry: number,
  summaryShares: number,
) {
  if (tradeDetails.length === 0) {
    const risk = Math.max(0, (summaryEntry - summaryEntry) * summaryShares);
    return { risk: 0, avgStop: summaryEntry, avgCost: summaryEntry, projectedPl: 0, realizedBank: 0 };
  }

  // Sort by date, buys before sells on same date
  const sorted = [...tradeDetails].sort((a, b) => {
    const da = String(a.date || "");
    const db = String(b.date || "");
    if (da !== db) return da.localeCompare(db);
    const aIsBuy = String(a.action).toUpperCase() === "BUY" ? 0 : 1;
    const bIsBuy = String(b.action).toUpperCase() === "BUY" ? 0 : 1;
    return aIsBuy - bIsBuy;
  });

  const inventory: { qty: number; price: number; stop: number }[] = [];
  let realizedBank = 0;

  for (const tx of sorted) {
    const action = String(tx.action || "").toUpperCase();
    const txShares = Math.abs(parseFloat(String(tx.shares || 0)));

    if (action === "BUY") {
      let price = parseFloat(String(tx.amount || 0));
      if (price === 0) price = summaryEntry;
      let stop = parseFloat(String(tx.stop_loss || 0));
      if (stop === 0) stop = price; // fallback: stop = entry
      inventory.push({ qty: txShares, price, stop });
    } else if (action === "SELL") {
      let toSell = txShares;
      const sellPrice = parseFloat(String(tx.amount || 0));
      let costBasis = 0;
      let soldQty = 0;

      while (toSell > 0 && inventory.length > 0) {
        const last = inventory[inventory.length - 1]; // LIFO
        const take = Math.min(toSell, last.qty);
        costBasis += take * last.price;
        soldQty += take;
        last.qty -= take;
        toSell -= take;
        if (last.qty < 0.00001) inventory.pop();
      }
      const revenue = soldQty * sellPrice;
      realizedBank += revenue - costBasis;
    }
  }

  // Calculate from remaining inventory
  let totalOpenShares = 0;
  let weightedCost = 0;
  let weightedStop = 0;
  let inventoryProjPl = 0;

  for (const item of inventory) {
    if (item.qty > 0) {
      totalOpenShares += item.qty;
      weightedCost += item.qty * item.price;
      weightedStop += item.qty * item.stop;
      inventoryProjPl += (item.stop - item.price) * item.qty;
    }
  }

  const avgCost = totalOpenShares > 0 ? weightedCost / totalOpenShares : summaryEntry;
  const avgLogStop = totalOpenShares > 0 ? weightedStop / totalOpenShares : 0;
  const masterStop = avgLogStop > 0 ? avgLogStop : avgCost;
  const initialRisk = Math.max(0, (avgCost - masterStop) * totalOpenShares);
  const projectedFloor = inventoryProjPl + realizedBank;

  return {
    risk: initialRisk,
    avgStop: masterStop,
    avgCost,
    projectedPl: projectedFloor,
    realizedBank,
  };
}

function computeEnrichedPositions(
  openTrades: TradePosition[],
  allDetails: TradeDetail[],
  equity: number,
  livePrices: Record<string, number> = {},
): EnrichedPosition[] {
  const now = new Date();

  return openTrades.map(trade => {
    const tradeDetails = allDetails.filter(d => d.trade_id === trade.trade_id);
    const buys = tradeDetails.filter(d => String(d.action).toUpperCase() === "BUY");

    // Detect options
    const ticker = trade.ticker || "";
    const isOption = /\d{6}/.test(ticker) || (trade.buy_notes || "").startsWith("OPT:");
    const optMult = isOption ? 100 : 1;

    const shares = trade.shares || 0;
    const summaryEntry = trade.avg_entry || 0;

    // Run LIFO engine (same as Streamlit)
    const lifo = runLifoEngine(tradeDetails, summaryEntry, shares);

    // Days held
    const firstDate = tradeDetails.length > 0
      ? new Date(tradeDetails[0].date)
      : new Date(trade.open_date);
    const daysHeld = Math.max(1, Math.floor((now.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24)));

    // Current price — use live price if available, else fall back to avg_entry
    const currentPrice = livePrices[ticker] || summaryEntry;

    const avgEntry = lifo.avgCost;
    const avgStop = lifo.avgStop;
    const currentValue = shares * currentPrice * optMult;
    const unrealizedPl = (currentPrice - avgEntry) * shares * optMult;
    const overallPl = unrealizedPl + lifo.realizedBank;
    const returnPct = avgEntry > 0 ? ((currentPrice - avgEntry) / avgEntry) * 100 : 0;
    const posSizePct = equity > 0 ? (currentValue / equity) * 100 : 0;
    const riskDollars = lifo.risk;
    const riskPct = equity > 0 ? (riskDollars / equity) * 100 : 0;
    const riskBudget = parseFloat(String(trade.risk_budget || 0));
    const stopPct = avgEntry > 0 && avgStop > 0 ? ((avgEntry - avgStop) / avgEntry) * 100 : 0;

    // Open risk equity (for heat KPI)
    const safeStop = avgStop > 0 ? avgStop : avgEntry;
    const openRiskEquity = (currentPrice - safeStop) * shares;

    // Pyramid: last remaining lot return %
    // Rebuild LIFO inventory to find last lot
    let pyramidPct = 0;
    if (tradeDetails.length > 0 && currentPrice > 0) {
      const sortedTx = [...tradeDetails].sort((a, b) => {
        const da = String(a.date || "");
        const db = String(b.date || "");
        if (da !== db) return da.localeCompare(db);
        const aR = String(a.action).toUpperCase() === "BUY" ? 0 : 1;
        const bR = String(b.action).toUpperCase() === "BUY" ? 0 : 1;
        return aR - bR;
      });
      const inv: { qty: number; price: number }[] = [];
      for (const tx of sortedTx) {
        const action = String(tx.action || "").toUpperCase();
        const txShares = Math.abs(parseFloat(String(tx.shares || 0)));
        if (action === "BUY") {
          let price = parseFloat(String(tx.amount || 0));
          if (price === 0) price = summaryEntry;
          inv.push({ qty: txShares, price });
        } else if (action === "SELL") {
          let toSell = txShares;
          while (toSell > 0 && inv.length > 0) {
            const last = inv[inv.length - 1];
            const take = Math.min(toSell, last.qty);
            last.qty -= take;
            toSell -= take;
            if (last.qty < 0.00001) inv.pop();
          }
        }
      }
      if (inv.length > 0) {
        const lastLotPrice = inv[inv.length - 1].price;
        if (lastLotPrice > 0) {
          pyramidPct = ((currentPrice - lastLotPrice) / lastLotPrice) * 100;
        }
      }
    }

    // Risk status
    const riskStatus = riskDollars <= 0.01 ? "Free Roll" : "At Risk";

    return {
      trade_id: trade.trade_id,
      ticker,
      shares,
      avg_entry: avgEntry,
      total_cost: parseFloat(String(trade.total_cost || 0)),
      realized_pl: parseFloat(String(trade.realized_pl || 0)),
      rule: trade.rule || "",
      buy_notes: trade.buy_notes || "",
      risk_budget: riskBudget,
      open_date: trade.open_date || "",
      days_held: daysHeld,
      avg_stop: avgStop,
      risk_dollars: riskDollars,
      risk_pct: riskPct,
      current_price: currentPrice,
      current_value: currentValue,
      unrealized_pl: unrealizedPl,
      overall_pl: overallPl,
      return_pct: returnPct,
      pos_size_pct: posSizePct,
      is_option: isOption,
      opt_mult: optMult,
      pyramid_pct: pyramidPct,
      risk_status: riskStatus,
      projected_pl: lifo.projectedPl,
      realized_bank: lifo.realizedBank,
      open_risk_equity: openRiskEquity,
      stop_pct: stopPct,
    };
  }).sort((a, b) => b.return_pct - a.return_pct);
}

type RiskFilter = "all" | "at_risk" | "free_roll";
type SortKey = keyof EnrichedPosition;
type SortDir = "asc" | "desc";

export function ActiveCampaign({ navColor }: { navColor: string }) {
  const [positions, setPositions] = useState<EnrichedPosition[]>([]);
  const [equity, setEquity] = useState(0);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>("");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("return_pct");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const loadData = useCallback(async () => {
    try {
      const [openTrades, details, journal] = await Promise.all([
        api.tradesOpen("CanSlim").catch(() => []),
        api.tradesOpenDetails("CanSlim").catch(() => []),
        api.journalLatest("CanSlim").catch(() => ({ end_nlv: 100000 })),
      ]);
      const eq = parseFloat(String(journal.end_nlv || 100000));
      setEquity(eq);

      // Fetch live prices for all open tickers
      const tickers = (openTrades as TradePosition[]).map(t => t.ticker).filter(Boolean);
      let prices: Record<string, number> = {};
      if (tickers.length > 0) {
        try {
          const result = await api.batchPrices(tickers);
          if (result && !("error" in result)) {
            prices = result;
          }
        } catch {
          // Fall back to entry prices if price fetch fails
        }
      }

      const enriched = computeEnrichedPositions(
        openTrades as TradePosition[],
        details as TradeDetail[],
        eq,
        prices,
      );
      setPositions(enriched);
      setLastUpdate(new Date().toLocaleTimeString());
      setLoading(false);
    } catch {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleSort = useCallback((key: string) => {
    if (key === "idx") return;
    const k = key as SortKey;
    setSortDir(prev => (sortKey === k ? (prev === "desc" ? "asc" : "desc") : "desc"));
    setSortKey(k);
  }, [sortKey]);

  // Filtered + sorted positions
  const filtered = useMemo(() => {
    let list = positions;
    if (riskFilter === "at_risk") list = positions.filter(p => p.risk_status === "At Risk");
    else if (riskFilter === "free_roll") list = positions.filter(p => p.risk_status === "Free Roll");

    return [...list].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      let cmp: number;
      if (typeof av === "string" && typeof bv === "string") {
        cmp = av.localeCompare(bv);
      } else {
        cmp = (av as number) - (bv as number);
      }
      return sortDir === "desc" ? -cmp : cmp;
    });
  }, [positions, riskFilter, sortKey, sortDir]);

  const atRiskCount = positions.filter(p => p.risk_status === "At Risk").length;
  const freeRollCount = positions.filter(p => p.risk_status === "Free Roll").length;

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-[90px] rounded-[14px] mb-6" style={{ background: "var(--bg-2)" }} />
      </div>
    );
  }

  // ── Aggregate KPIs ──
  const totalPositions = positions.length;
  const totalMarketValue = positions.reduce((a, p) => a + p.current_value, 0);
  const liveExposure = equity > 0 ? (totalMarketValue / equity) * 100 : 0;
  const totalOverallPl = positions.reduce((a, p) => a + p.overall_pl, 0);
  const totalProjected = positions.reduce((a, p) => a + p.projected_pl, 0);
  const totalInitialRisk = positions.reduce((a, p) => a + p.risk_dollars, 0);
  const totalOpenRiskEquity = positions.reduce((a, p) => a + p.open_risk_equity, 0);
  const irPct = equity > 0 ? (totalInitialRisk / equity) * 100 : 0;
  const orPct = equity > 0 ? (totalOpenRiskEquity / equity) * 100 : 0;

  const kpis = [
    {
      label: "OPEN POSITIONS",
      value: String(totalPositions),
      sub: `${freeRollCount} free roll · ${atRiskCount} at risk`,
      gradient: "linear-gradient(135deg, #7c3aed, #a78bfa)",
    },
    {
      label: "TOTAL MARKET VALUE",
      value: `$${totalMarketValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
      sub: "",
      gradient: "linear-gradient(135deg, #ec4899, #f472b6)",
    },
    {
      label: "LIVE EXPOSURE",
      value: `${liveExposure.toFixed(1)}%`,
      sub: `of $${equity.toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
      gradient: "linear-gradient(135deg, #f97316, #fb923c)",
    },
    {
      label: "OVERALL P&L",
      value: `$${totalOverallPl >= 0 ? "" : ""}${totalOverallPl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      sub: `Projected: $${totalProjected.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      gradient: totalOverallPl >= 0 ? "linear-gradient(135deg, #10b981, #34d399)" : "linear-gradient(135deg, #e5484d, #f472b6)",
    },
    {
      label: "INITIAL RISK",
      value: `$${totalInitialRisk.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      sub: `${irPct.toFixed(2)}% of NLV`,
      gradient: "linear-gradient(135deg, #1e40af, #3b82f6)",
    },
    {
      label: "OPEN RISK (HEAT)",
      value: `$${totalOpenRiskEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      sub: `${orPct.toFixed(2)}% of NLV`,
      gradient: "linear-gradient(135deg, #e5484d, #f87171)",
    },
  ];

  // ── Risk Monitor Alerts ──
  const monitorAlerts: { type: "error" | "warn" | "info" | "success"; ticker: string; msg: string }[] = [];
  const freeRollTickers: string[] = [];

  positions.forEach(p => {
    // Budget check: raise stop suggestion
    if (p.risk_budget > 0 && p.risk_dollars > (p.risk_budget + 5)) {
      const rbmStop = p.total_cost > 0 && p.shares > 0
        ? (p.total_cost - p.risk_budget) / p.shares
        : 0;
      monitorAlerts.push({
        type: "warn",
        ticker: p.ticker,
        msg: `Risk ($${p.risk_dollars.toFixed(0)}) > Budget ($${p.risk_budget.toFixed(0)}). Raise stop to $${rbmStop.toFixed(2)} to stay within budget.`,
      });
    }

    // Stop rule violation
    if (p.return_pct <= -7.0) {
      monitorAlerts.push({
        type: "error",
        ticker: p.ticker,
        msg: `Down ${p.return_pct.toFixed(2)}%. Violates Stop Rule.`,
      });
    }

    // Consider moving stop to BE
    if (p.return_pct >= 10.0 && p.avg_stop > 0 && p.avg_stop < (p.avg_entry - 0.01)) {
      monitorAlerts.push({
        type: "info",
        ticker: p.ticker,
        msg: `Up ${p.return_pct.toFixed(2)}%. Consider moving stop to BE ($${p.avg_entry.toFixed(2)}). Current stop: $${p.avg_stop.toFixed(2)}.`,
      });
    }

    // Track free rolls
    if (p.risk_status === "Free Roll") {
      freeRollTickers.push(p.ticker);
    }
  });

  // ── Table columns matching Streamlit exactly ──
  const COL_HEADERS = [
    { key: "idx", label: "#", align: "center" as const },
    { key: "trade_id", label: "Trade_ID", align: "left" as const },
    { key: "ticker", label: "Ticker", align: "left" as const },
    { key: "days_held", label: "Days", align: "center" as const },
    { key: "risk_status", label: "Risk Status", align: "center" as const },
    { key: "pyramid_pct", label: "Pyramid", align: "right" as const },
    { key: "return_pct", label: "Return %", align: "right" as const },
    { key: "pos_size_pct", label: "Pos Size %", align: "right" as const },
    { key: "shares", label: "Shares", align: "right" as const },
    { key: "avg_entry", label: "Avg Entry", align: "right" as const },
    { key: "current_price", label: "Current", align: "right" as const },
    { key: "avg_stop", label: "Avg Stop", align: "right" as const },
    { key: "stop_pct", label: "Stop %", align: "right" as const },
    { key: "risk_budget", label: "Risk $", align: "right" as const },
    { key: "risk_dollars", label: "Risk $", align: "right" as const },
    { key: "risk_pct", label: "Risk %", align: "right" as const },
    { key: "current_value", label: "Current Value", align: "right" as const },
    { key: "overall_pl", label: "Overall_PL", align: "right" as const },
    { key: "projected_pl", label: "Projected P&L", align: "right" as const },
  ];

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
              Active Campaign <em className="italic" style={{ color: navColor }}>Summary</em>
            </h1>
            <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
              {totalPositions} open positions · {lastUpdate && `Updated ${lastUpdate}`}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={loadData}
                    className="flex items-center gap-1.5 h-[32px] px-3.5 rounded-[10px] text-xs font-medium transition-colors hover:brightness-95"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
              </svg>
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* KPI tiles */}
      <div className="grid grid-cols-6 gap-3 mb-6">
        {kpis.map(k => <KPITile key={k.label} {...k} />)}
      </div>

      {/* Positions table */}
      <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Active Positions</span>
          <span className="text-xs" style={{ color: "var(--ink-4)" }}>Click headers to sort</span>

          {/* Filter tabs */}
          <div className="ml-auto flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
            {([
              { key: "all" as RiskFilter, label: "All", count: totalPositions },
              { key: "at_risk" as RiskFilter, label: "At Risk", count: atRiskCount },
              { key: "free_roll" as RiskFilter, label: "Free Roll", count: freeRollCount },
            ]).map(f => (
              <button key={f.key} onClick={() => setRiskFilter(f.key)}
                      className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                      style={{
                        background: riskFilter === f.key ? "var(--surface)" : "transparent",
                        color: riskFilter === f.key ? "var(--ink)" : "var(--ink-4)",
                        boxShadow: riskFilter === f.key ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
                      }}>
                {f.label} <span style={{ opacity: 0.6 }}>({f.count})</span>
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
            <thead>
              <tr>
                {COL_HEADERS.map(h => {
                  const sortable = h.key !== "idx";
                  const active = sortKey === h.key;
                  return (
                    <th key={h.key + h.label}
                        className={`text-${h.align} text-[10px] uppercase tracking-[0.08em] font-semibold px-2.5 py-2.5 whitespace-nowrap sticky top-0`}
                        style={{
                          color: active ? "var(--ink)" : "var(--ink-4)",
                          background: "var(--surface-2)",
                          borderBottom: "1px solid var(--border)",
                          cursor: sortable ? "pointer" : "default",
                          userSelect: "none",
                        }}
                        onClick={() => sortable && handleSort(h.key)}>
                      {h.label}
                      {active && (
                        <span className="ml-1 text-[9px]">{sortDir === "desc" ? "▼" : "▲"}</span>
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => {
                const plColor = p.overall_pl >= 0 ? "#08a86b" : "#e5484d";
                const projColor = p.projected_pl >= 0 ? "#08a86b" : "#e5484d";
                const mono = "var(--font-jetbrains), monospace";

                // Return % badge — gradient backgrounds like the prototype
                let retBg: string;
                let retText: string;
                if (p.return_pct >= 5) {
                  retBg = "linear-gradient(135deg, #16a34a, #22c55e)";
                  retText = "#fff";
                } else if (p.return_pct > 0) {
                  retBg = "linear-gradient(135deg, #a3e635, #84cc16)";
                  retText = "#1a2e05";
                } else if (p.return_pct > -3) {
                  retBg = "linear-gradient(135deg, #fbbf24, #f59e0b)";
                  retText = "#451a03";
                } else {
                  retBg = "linear-gradient(135deg, #f87171, #ef4444)";
                  retText = "#fff";
                }

                // Pyramid: only show "Ready" if last lot is up >= 5%
                const pyramidReady = p.pyramid_pct >= 5;

                return (
                  <tr key={p.trade_id} className="transition-colors"
                      style={{ borderBottom: i < filtered.length - 1 ? "1px solid var(--border)" : "none" }}
                      onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                      onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                    {/* # */}
                    <td className="px-2.5 py-2.5 text-center text-[11px]" style={{ color: "var(--ink-4)" }}>
                      {i + 1}
                    </td>
                    {/* Trade_ID */}
                    <td className="px-2.5 py-2.5 whitespace-nowrap" style={{ fontFamily: mono, fontSize: 11, color: "var(--ink-4)" }}>
                      {p.trade_id}
                    </td>
                    {/* Ticker */}
                    <td className="px-2.5 py-2.5 font-semibold whitespace-nowrap" style={{ fontFamily: mono }}>
                      {p.ticker}
                    </td>
                    {/* Days */}
                    <td className="px-2.5 py-2.5 text-center" style={{ fontFamily: mono, fontSize: 11, color: "var(--ink-4)" }}>
                      {p.days_held}
                    </td>
                    {/* Risk Status */}
                    <td className="px-2.5 py-2.5 text-center">
                      <span className="inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold whitespace-nowrap"
                            style={{
                              background: p.risk_status === "Free Roll" ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : "color-mix(in oklab, #f59f00 12%, var(--surface))",
                              color: p.risk_status === "Free Roll" ? "#16a34a" : "#d97706",
                            }}>
                        {p.risk_status === "Free Roll" ? "Free Roll" : "At Risk"}
                      </span>
                    </td>
                    {/* Pyramid — only show green "Ready" badge if last lot >= +5% */}
                    <td className="px-2.5 py-2.5 text-center">
                      {pyramidReady ? (
                        <span className="inline-block px-2 py-0.5 rounded text-[10px] font-bold"
                              style={{ background: "linear-gradient(135deg, #16a34a, #22c55e)", color: "#fff", minWidth: 40, textAlign: "center" }}>
                          Ready
                        </span>
                      ) : (
                        <span style={{ color: "var(--ink-4)", fontSize: 11 }}>—</span>
                      )}
                    </td>
                    {/* Return % — gradient badge like prototype */}
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
                    {/* Shares */}
                    <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono }}>
                      {p.shares.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    {/* Avg Entry */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                      ${p.avg_entry.toFixed(2)}
                    </td>
                    {/* Current Price */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                      ${p.current_price.toFixed(2)}
                    </td>
                    {/* Avg Stop */}
                    <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: p.avg_stop > 0 ? "var(--ink)" : "var(--ink-4)" }}>
                      {p.avg_stop > 0 ? `$${p.avg_stop.toFixed(2)}` : "—"}
                    </td>
                    {/* Stop % */}
                    <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: "var(--ink-4)" }}>
                      {p.stop_pct > 0 ? `${p.stop_pct.toFixed(1)}%` : "—"}
                    </td>
                    {/* Risk Budget */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                      ${p.risk_budget.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    {/* Risk $ */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, color: p.risk_dollars > 0 ? "#e5484d" : "#08a86b" }}>
                      ${p.risk_dollars.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    {/* Risk % */}
                    <td className="px-2.5 py-2.5 text-right" style={{ fontFamily: mono, color: p.risk_pct > 1 ? "#e5484d" : "var(--ink-4)" }}>
                      {p.risk_pct.toFixed(2)}%
                    </td>
                    {/* Current Value */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono }}>
                      ${p.current_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    {/* Overall P&L — BOLD + colored */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, fontWeight: 700, color: plColor }}>
                      ${p.overall_pl >= 0 ? "+" : ""}{p.overall_pl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    {/* Projected P&L */}
                    <td className="px-2.5 py-2.5 text-right privacy-mask" style={{ fontFamily: mono, fontWeight: 600, color: projColor }}>
                      ${p.projected_pl >= 0 ? "+" : ""}{p.projected_pl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Risk Monitor ── */}
      <div className="mt-6 rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Risk Monitor</span>
          <span className="text-xs" style={{ color: "var(--ink-4)" }}>Active alerts · {positions.length} positions</span>
        </div>
        <div className="p-4 flex flex-col gap-2.5">
          {/* Info alerts (move to BE) */}
          {monitorAlerts.filter(a => a.type === "info").map((a, i) => (
            <div key={`info-${i}`} className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">📈</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#3b82f6" }}>
                <strong>{a.ticker}</strong>: {a.msg}
              </div>
            </div>
          ))}

          {/* Warnings (budget exceeded) */}
          {monitorAlerts.filter(a => a.type === "warn").map((a, i) => (
            <div key={`warn-${i}`} className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">⚠️</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#854d0e" }}>
                <strong>{a.ticker}</strong>: {a.msg}
              </div>
            </div>
          ))}

          {/* Errors (stop violations) */}
          {monitorAlerts.filter(a => a.type === "error").map((a, i) => (
            <div key={`err-${i}`} className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">🔴</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#991b1b" }}>
                <strong>{a.ticker}</strong>: {a.msg}
              </div>
            </div>
          ))}

          {/* Free rolls summary */}
          {freeRollTickers.length > 0 && (
            <div className="flex items-start gap-2.5 px-4 py-3 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              <span className="text-[13px] shrink-0 mt-0.5">🆓</span>
              <div className="text-[12px] leading-relaxed" style={{ color: "#166534" }}>
                <strong>{freeRollTickers.join(", ")}</strong> — Free Roll — stops above entry. 0 risk.
              </div>
            </div>
          )}

          {/* All clear */}
          {monitorAlerts.length === 0 && freeRollTickers.length === 0 && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-[10px] text-[12px] font-medium"
                 style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              System Health Good — all positions within risk parameters
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
