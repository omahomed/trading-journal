"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type TradePosition, type TradeDetail } from "@/lib/api";
// Pure CSS bar chart — no Recharts dependency

type Tab = "overview" | "buyrules" | "sellrules" | "drawdown" | "review" | "campaigns";

function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }

const LESSON_CATEGORIES = [
  "Entry timing", "Stop placement", "Undersized", "Oversized",
  "Scaled in too fast", "Exit too early", "Exit too late",
  "Market conditions", "Rule deviation", "Other",
];
const CAT_COLORS: Record<string, { bg: string; fg: string }> = {
  "Entry timing": { bg: "color-mix(in oklab, #f59f00 12%, var(--surface))", fg: "#b45309" }, "Stop placement": { bg: "#fed7aa", fg: "#c2410c" },
  "Undersized": { bg: "#dbeafe", fg: "#3b82f6" }, "Oversized": { bg: "#ede9fe", fg: "#6d28d9" },
  "Scaled in too fast": { bg: "color-mix(in oklab, #e5484d 30%, var(--border))", fg: "#b91c1c" }, "Exit too early": { bg: "#ccfbf1", fg: "#0f766e" },
  "Exit too late": { bg: "#e0e7ff", fg: "#4338ca" }, "Market conditions": { bg: "var(--border)", fg: "var(--ink-2)" },
  "Rule deviation": { bg: "#ffe4e6", fg: "#be123c" }, "Other": { bg: "var(--bg-2)", fg: "var(--ink-3)" },
};

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

export function Analytics({ navColor }: { navColor: string }) {
  const [allTrades, setAllTrades] = useState<TradePosition[]>([]);
  const [allDetails, setAllDetails] = useState<TradeDetail[]>([]);
  const [openCount, setOpenCount] = useState(0);
  const [journalHistory, setJournalHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");
  const [scope, setScope] = useState<"ltd" | "2026">("ltd");
  // drillRule kept for sell rules tab (TODO)

  useEffect(() => {
    Promise.all([
      api.tradesClosed("CanSlim", 1000).catch(() => []),
      api.tradesOpen("CanSlim").catch(() => []),
      api.journalHistory("CanSlim", 0).catch(() => []),
      api.tradesRecent("CanSlim", 2000).catch(() => []),
    ]).then(([closed, open, journal, details]) => {
      setAllTrades(closed as TradePosition[]);
      const openArr = open as TradePosition[];
      setOpenCount(openArr.length);
      setOpenTrades(openArr);
      setJournalHistory(journal as any[]);
      setAllDetails(details as TradeDetail[]);
      // Fetch trade lessons
      api.getTradeLessons("CanSlim").then(r => { if (r.lessons) setLessons(r.lessons); }).catch(() => {});
      setLoading(false);
    });
  }, []);

  const trades = useMemo(() => {
    if (scope === "2026") return allTrades.filter(t => String(t.closed_date || t.open_date || "").startsWith("2026"));
    return allTrades;
  }, [allTrades, scope]);

  const stats = useMemo(() => {
    const closed = trades;
    const wins = closed.filter(t => parseFloat(String(t.realized_pl || 0)) > 0);
    const losses = closed.filter(t => parseFloat(String(t.realized_pl || 0)) < 0);
    const breakEven = closed.filter(t => parseFloat(String(t.realized_pl || 0)) === 0);
    const grossProfit = wins.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
    const grossLoss = Math.abs(losses.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0));
    const pf = grossLoss > 0 ? grossProfit / grossLoss : 0;
    const winRate = closed.length > 0 ? (wins.length / closed.length) * 100 : 0;
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

    // Hold days — guard against empty/invalid dates
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
    const winnersHold = avgHold(wins);
    const losersHold = avgHold(losses);
    const avgHoldAll = avgHold(closed);
    const holdRatio = losersHold > 0 ? winnersHold / losersHold : 0;

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
  const [campSort, setCampSort] = useState<{ col: string; asc: boolean }>({ col: "open", asc: false });
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);

  // Rule stats — always 2026 for buy/sell rules (matching Streamlit)
  const ruleStats = useMemo(() => {
    const isSell = tab === "sellrules";
    const col = isSell ? "sell_rule" : "buy_rule";
    // Always scope to 2026 closed for rules tabs
    const source = allTrades.filter(t => String(t.closed_date || "").startsWith("2026"));
    const map: Record<string, { rule: string; count: number; wins: number; totalPl: number; rValues: number[]; trades: TradePosition[] }> = {};
    for (const t of source) {
      const rule = String((t as any)[col] || (t as any).rule || "").trim();
      if (!rule || rule === "nan" || rule === "undefined") continue;
      if (!map[rule]) map[rule] = { rule, count: 0, wins: 0, totalPl: 0, rValues: [], trades: [] };
      const pl = parseFloat(String(t.realized_pl || 0));
      const rb = parseFloat(String(t.risk_budget || 0));
      map[rule].count++;
      if (pl > 0) map[rule].wins++;
      map[rule].totalPl += pl;
      if (rb > 0) map[rule].rValues.push(pl / rb);
      map[rule].trades.push(t);
    }
    const arr = Object.values(map).map(r => ({
      ...r,
      avgPl: r.count > 0 ? r.totalPl / r.count : 0,
      winRate: r.count > 0 ? (r.wins / r.count) * 100 : 0,
      avgR: r.rValues.length > 0 ? r.rValues.reduce((a, b) => a + b, 0) / r.rValues.length : null as number | null,
    }));
    // Sort
    const key = brSort === "Win Rate %" ? "winRate" : brSort === "Avg P&L" ? "avgPl" : brSort === "Trades" ? "count" : "totalPl";
    return arr.sort((a, b) => (b as any)[key] - (a as any)[key]);
  }, [allTrades, tab, brSort]);

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
          Analytics & <em className="italic" style={{ color: navColor }}>Audit</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>{trades.length} closed trades · {scope === "2026" ? "2026" : "Life to Date"}</div>
      </div>

      {/* Tabs */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex gap-1 pb-0.5" style={{ borderBottom: "2px solid var(--border)" }}>
          {([ { key: "overview" as Tab, label: "🎯 Overview" }, { key: "buyrules" as Tab, label: "🟢 Buy Rules" }, { key: "sellrules" as Tab, label: "🔴 Sell Rules" }, { key: "drawdown" as Tab, label: "🛡️ Drawdown" }, { key: "review" as Tab, label: "🔬 Trade Review" }, { key: "campaigns" as Tab, label: "📋 All Campaigns" } ]).map(t => (
            <button key={t.key} onClick={() => { setTab(t.key); setBrDrill(""); }}
                    className="px-4 py-2 text-[12px] font-medium transition-all"
                    style={{ color: tab === t.key ? navColor : "var(--ink-4)", borderBottom: tab === t.key ? `2px solid ${navColor}` : "2px solid transparent", marginBottom: -2 }}>
              {t.label}
            </button>
          ))}
        </div>
        {tab === "overview" && (
          <div className="flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
            {(["ltd", "2026"] as const).map(sc => (
              <button key={sc} onClick={() => setScope(sc)}
                      className="px-3 py-1 rounded-md text-[11px] font-medium transition-all"
                      style={{ background: scope === sc ? "var(--surface)" : "transparent", color: scope === sc ? "var(--ink)" : "var(--ink-4)" }}>
                {sc === "ltd" ? "LTD" : "2026"}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ═══ OVERVIEW ═══ */}
      {tab === "overview" && (
        <>
          <div className="text-[13px] mb-4" style={{ color: "var(--ink-3)" }}>The headline numbers across every closed trade — start here for a quick health check.</div>

          {/* Hero Row */}
          <div className="grid grid-cols-4 gap-3 mb-4">
            <HeroCard label="Total P&L" value={`$${s.netPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub={`${s.total} closed trades`} ok={s.netPl >= 0} />
            <HeroCard label="Win Rate" value={`${s.winRate.toFixed(1)}%`} sub={`${s.wins}W · ${s.losses}L`} ok={s.winRate >= 40} />
            <HeroCard label="Profit Factor" value={s.pf.toFixed(2)} sub={s.pf >= 1.5 ? "≥1.5 healthy" : "target ≥1.5"} ok={s.pf >= 1.5} />
            <HeroCard label="Expectancy / Trade" value={`$${s.expectancy.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="avg $ per trade" ok={s.expectancy >= 0} />
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
                    { k: "Avg", v: `$${side.avg.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                    { k: "Largest", v: `$${side.largest.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
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
            <QualityTile label="Avg Trade" value={`$${s.avgTrade.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} status={s.avgTrade >= 0 ? "positive" : "negative"} ok={s.avgTrade >= 0} />
            <QualityTile label="Avg R-Multiple" value={`${s.avgR.toFixed(2)}R`} status={`max ${s.maxR.toFixed(1)}R`} ok={s.avgR >= 1} />
          </div>

          {/* Loss Discipline — 2026 only */}
          {(() => {
            // Compute Impact_Pct = Realized_PL / NLV at trade open date
            const losses2026 = allTrades.filter(t => parseFloat(String(t.realized_pl || 0)) < 0 && String(t.closed_date || "").startsWith("2026"));

            const getNlvAtOpen = (openDate: string) => {
              const d = String(openDate).slice(0, 10);
              // Find closest journal entry on or before the open date
              const sorted = journalHistory.filter(h => String(h.day).slice(0, 10) <= d).sort((a: any, b: any) => String(b.day).localeCompare(String(a.day)));
              return sorted.length > 0 ? sorted[0].end_nlv : null;
            };

            const impactTrades = losses2026.map(t => {
              const nlv = getNlvAtOpen(t.open_date);
              const impact = nlv && nlv > 0 ? (parseFloat(String(t.realized_pl || 0)) / nlv) * 100 : null;
              return { ticker: t.ticker, trade_id: t.trade_id, closed_date: t.closed_date, realized_pl: t.realized_pl, open_date: t.open_date, nlvAtOpen: nlv, impactPct: impact };
            }).filter(t => t.impactPct !== null);

            const totalLosses = impactTrades.length;
            const withinRule = impactTrades.filter(t => (t.impactPct || 0) >= -1.0).length;
            const breaches = totalLosses - withinRule;
            const passRate = totalLosses > 0 ? (withinRule / totalLosses) * 100 : 100;

            // Buckets
            const bucketDefs = [
              { label: "0 to −0.25%", lo: -0.25, hi: 0, color: "#08a86b", sub: "Minor nicks" },
              { label: "−0.25 to −0.50%", lo: -0.50, hi: -0.25, color: "#65a30d", sub: "Small" },
              { label: "−0.50 to −1.00%", lo: -1.00, hi: -0.50, color: "#d97706", sub: "Borderline" },
              { label: "Over −1.00%", lo: -9999, hi: -1.00, color: "#e5484d", sub: "🚨 BREACH" },
            ];

            return totalLosses > 0 ? (
              <div className="mb-4">
                <div className="text-[15px] font-semibold mb-1">🛡️ Loss Discipline <span className="text-[13px] font-normal" style={{ color: "var(--ink-4)" }}>— 2026 only</span></div>

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
                        <div className="text-[11px] mt-1 privacy-mask" style={{ color: "var(--ink-4)" }}>${dollarSum.toLocaleString(undefined, { maximumFractionDigits: 0 })} total</div>
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
                              <td className="px-3 py-2 privacy-mask" style={{ fontFamily: mono, color: "#e5484d" }}>${parseFloat(String(t.realized_pl || 0)).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
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
              { k: "Best Month", v: `$${s.bestMonth.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: s.bestMonthKey, color: "#08a86b", icon: "📈" },
              { k: "Worst Month", v: `$${s.worstMonth.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: s.worstMonthKey, color: "#e5484d", icon: "📉" },
              { k: "Average Month", v: `$${s.avgMonth.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: undefined, color: "var(--ink)", icon: "📊" },
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
        </>
      )}

      {/* ═══ BUY RULES ═══ */}
      {tab === "buyrules" && (
        <>
          <div className="text-[14px] font-semibold mb-1">🟢 Buy Rules — What{"'"}s Working in 2026</div>
          <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>Study your entry rules. Sort by the metric you care about, click any rule to drill into individual trades.</div>

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
                      {best && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#08a86b" }}>${best.totalPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>{best.count} trades · {best.winRate.toFixed(0)}% win rate</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #e5484d 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#e5484d" }}>🚨 Worst Rule</div>
                      <div className="text-[14px] font-bold mt-1">{worst?.rule || "—"}</div>
                      {worst && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#e5484d" }}>${worst.totalPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
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

              {/* Drill-down — two columns: stats left, trades right */}
              {brDrill && (() => {
                const rs = ruleStats.find(r => r.rule === brDrill);
                const rt = rs?.trades || [];
                const wins = rt.filter(t => parseFloat(String(t.realized_pl || 0)) > 0);
                const losses = rt.filter(t => parseFloat(String(t.realized_pl || 0)) < 0);
                const totalPl = rt.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
                const avgPl = rt.length > 0 ? totalPl / rt.length : 0;
                const winRate = rt.length > 0 ? (wins.length / rt.length) * 100 : 0;
                const grossW = wins.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
                const grossL = Math.abs(losses.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0));
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
                            { k: "Total P&L", v: `$${totalPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: pctColor(totalPl) },
                            { k: "Trades", v: String(rt.length) },
                            { k: "Win Rate", v: `${winRate.toFixed(0)}%`, color: winRate >= 50 ? "#08a86b" : "#e5484d" },
                            { k: "Profit Factor", v: pf.toFixed(2) },
                            { k: "Avg P&L", v: `$${avgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: pctColor(avgPl) },
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
                          <tbody>{[...rt].sort((a, b) => parseFloat(String(b.realized_pl || 0)) - parseFloat(String(a.realized_pl || 0))).map((t, i) => {
                            const pl = parseFloat(String(t.realized_pl || 0));
                            const rb = parseFloat(String(t.risk_budget || 0));
                            const rMult = rb > 0 ? pl / rb : null;
                            return (
                              <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                                <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                                <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(t.open_date || "").slice(5, 10)}</td>
                                <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(t.closed_date || "").slice(5, 10)}</td>
                                <td className="px-3 py-2 font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>${pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
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
          <div className="text-[12px] mb-4" style={{ color: "var(--ink-4)" }}>Which are protecting capital, which are capturing profits, which are hurting you.</div>

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
                      {negRules[0] && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#08a86b" }}>${negRules[0].avgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>smallest avg loss · {negRules[0].count} uses</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #3b82f6 8%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "#3b82f6" }}>💰 Top Profit Capture</div>
                      <div className="text-[14px] font-bold mt-1">{posRules[0]?.rule || "No winning exits yet"}</div>
                      {posRules[0] && <><div className="text-[18px] font-bold privacy-mask" style={{ color: "#3b82f6" }}>${posRules[0].totalPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>avg ${posRules[0].avgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })} · {posRules[0].count} uses</div></>}
                    </div>
                    <div className="p-4 rounded-[12px]" style={{ background: `color-mix(in oklab, #64748b 6%, var(--surface))`, border: "1px solid var(--border)" }}>
                      <div className="text-[10px] uppercase tracking-[0.08em] font-bold" style={{ color: "var(--ink-3)" }}>📊 Most Used Exit</div>
                      <div className="text-[14px] font-bold mt-1">{mostUsed?.rule || "—"}</div>
                      {mostUsed && <><div className="text-[28px] font-extrabold">{mostUsed.count} uses</div>
                      <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>avg ${mostUsed.avgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div></>}
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
                            ${(r.totalPl / 1000).toFixed(1)}k
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
                            { k: "Total P&L", v: `$${totalPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: pctColor(totalPl) },
                            { k: "Uses", v: String(rt.length) },
                            { k: "Win Rate", v: `${winRate.toFixed(0)}%`, color: winRate >= 50 ? "#08a86b" : "#e5484d" },
                            { k: "Profit Factor", v: pf.toFixed(2) },
                            { k: "Avg P&L", v: `$${avgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: pctColor(avgPl) },
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
                                <td className="px-3 py-2 font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>${pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
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
              Current DD: <strong>{curr.ddPct.toFixed(2)}%</strong> · NLV: <strong className="privacy-mask">${curr.nlv.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong> · Peak: <strong className="privacy-mask">${curr.peak.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong> · Exposure: <strong>{curr.exposure.toFixed(1)}%</strong>
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
                        <div><div className="text-[9px] uppercase font-bold" style={{ color: "var(--ink-4)" }}>Realized</div><div className="text-[15px] font-bold privacy-mask" style={{ color: "#e5484d" }}>${c.lossesInWindow.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div></div>
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
                      <div className="text-[24px] font-extrabold mt-1 privacy-mask" style={{ color: g.color }}>${Math.abs(g.data.reduce((a: number, c: any) => a + c.lossesInWindow, 0)).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
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

        const TradeCard = ({ rank, t, isWinner }: { rank: number; t: TradePosition; isWinner: boolean }) => {
          const pl = parseFloat(String(t.realized_pl || 0));
          const ret = parseFloat(String(t.return_pct || 0));
          const rb = parseFloat(String(t.risk_budget || 0));
          const rMult = rb > 0 ? pl / rb : null;
          const oStr = String(t.open_date || "").trim(); const cStr = String(t.closed_date || "").trim();
          const hold = (oStr && cStr && !isNaN(new Date(oStr).getTime()) && !isNaN(new Date(cStr).getTime())) ? Math.max(0, Math.floor((new Date(cStr).getTime() - new Date(oStr).getTime()) / 86400000)) : null;
          const borderColor = isWinner ? "#08a86b" : "#e5484d";
          const plColor = isWinner ? "#08a86b" : "#e5484d";
          return (
            <div className="rounded-[12px] overflow-hidden mb-3" style={{ background: "var(--surface)", borderLeft: `4px solid ${borderColor}`, border: "1px solid var(--border)" }}>
              <div className="px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-[15px] font-extrabold">#{rank} · {t.ticker} <span className="text-[11px] font-normal" style={{ color: "var(--ink-4)" }}>({t.trade_id})</span></div>
                  <div className="text-[18px] font-extrabold privacy-mask" style={{ color: plColor }}>{pl >= 0 ? "+" : ""}${pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
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
                              <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>${px.toFixed(2)}</td>
                              <td className="px-2.5 py-1.5 font-semibold" style={{ fontFamily: mono, color: !isSell && retPct !== 0 ? pctColor(retPct) : "var(--ink-4)" }}>
                                {!isSell && retPct !== 0 ? `${retPct >= 0 ? "+" : ""}${retPct.toFixed(2)}%` : isSell ? "" : "0.00%"}
                              </td>
                              <td className="px-2.5 py-1.5 privacy-mask" style={{ fontFamily: mono }}>${(shs * px).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
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
                  <details className="mx-4 mb-3">
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
                                const result = await api.saveTradeLessons({ portfolio: "CanSlim", trade_id: t.trade_id, note: currentText, category: saveCat });
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

            {/* Top Winners */}
            <div className="text-[15px] font-bold mb-3">🏆 Top {topN} Winners</div>
            {topWinners.length > 0 ? topWinners.map((t, i) => <TradeCard key={t.trade_id} rank={i + 1} t={t} isWinner />) : (
              <div className="mb-4 px-4 py-3 rounded-[10px] text-[12px]" style={{ background: "var(--bg)", color: "var(--ink-4)" }}>No profitable trades in this window.</div>
            )}

            {/* Worst Losers */}
            <div className="text-[15px] font-bold mb-3 mt-5">⚠️ Worst {topN} Losers</div>
            {worstLosers.length > 0 ? worstLosers.map((t, i) => <TradeCard key={t.trade_id} rank={i + 1} t={t} isWinner={false} />) : (
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
                      <div><span style={{ color: "var(--ink-4)" }}>Avg P&L:</span> <strong className="privacy-mask">${p.data.avgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong></div>
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
          // Result
          const pl = parseFloat(String(t.realized_pl || 0));
          if (campResult === "winners" && pl <= 0) return false;
          if (campResult === "losers" && pl >= 0) return false;
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
          const pl = parseFloat(String(t.realized_pl || 0));
          const rMult = rb > 0 ? pl / rb : 0;
          switch (sortKey) {
            case "ticker": return (t.ticker || "").toUpperCase();
            case "trade_id": return t.trade_id || "";
            case "status": return (t.status || "").toUpperCase();
            case "rule": return (t.rule || "").toUpperCase();
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

        // Flight deck stats
        const fdTrades = campTicker ? filtered : filtered;
        const fdTotal = fdTrades.length;
        const fdWins = fdTrades.filter(t => parseFloat(String(t.realized_pl || 0)) > 0).length;
        const fdLosses = fdTrades.filter(t => parseFloat(String(t.realized_pl || 0)) < 0).length;
        const fdPl = fdTrades.reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
        const fdWinRate = fdTotal > 0 ? (fdWins / fdTotal) * 100 : 0;
        const fdAvgPl = fdTotal > 0 ? fdPl / fdTotal : 0;
        const fdGrossW = fdTrades.filter(t => parseFloat(String(t.realized_pl || 0)) > 0).reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0);
        const fdGrossL = Math.abs(fdTrades.filter(t => parseFloat(String(t.realized_pl || 0)) < 0).reduce((a, t) => a + parseFloat(String(t.realized_pl || 0)), 0));
        const fdPf = fdGrossL > 0 ? fdGrossW / fdGrossL : 0;

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

              <span className="ml-auto text-[11px]" style={{ color: "var(--ink-4)" }}>{filtered.length} results</span>
            </div>

            {/* Flight Deck */}
            <div className="rounded-[14px] overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-5 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
                Flight Deck {campTicker ? `— ${campTicker}` : "— All"}
              </div>
              <div className="grid grid-cols-6 divide-x" style={{ borderColor: "var(--border)" }}>
                {[
                  { k: "Trades", v: String(fdTotal) },
                  { k: "Net P&L", v: `$${fdPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: pctColor(fdPl) },
                  { k: "Win Rate", v: `${fdWinRate.toFixed(0)}%`, color: fdWinRate >= 50 ? "#08a86b" : "#e5484d" },
                  { k: "Avg P&L", v: `$${fdAvgPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: pctColor(fdAvgPl) },
                  { k: "Profit Factor", v: fdPf.toFixed(2) },
                  { k: "W / L", v: `${fdWins}W · ${fdLosses}L` },
                ].map(m => (
                  <div key={m.k} className="p-4 text-center">
                    <div className="text-[9px] uppercase tracking-[0.06em] font-bold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                    <div className="text-[20px] font-extrabold mt-1 privacy-mask" style={{ fontFamily: mono, color: (m as any).color || "var(--ink)" }}>{m.v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Trade table */}
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="overflow-x-auto" style={{ maxHeight: 600 }}>
                <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                  <thead><tr>
                    {([
                      { label: "Ticker", key: "ticker" }, { label: "Trade ID", key: "trade_id" }, { label: "Status", key: "status" },
                      { label: "Rule", key: "rule" }, { label: "Open", key: "open" }, { label: "Close", key: "close" },
                      { label: "Shares", key: "shares" }, { label: "Entry", key: "entry" }, { label: "Exit", key: "exit" },
                      { label: "P&L", key: "pl" }, { label: "Return %", key: "return" }, { label: "R", key: "r" },
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
                    const pl = parseFloat(String(t.realized_pl || 0));
                    const ret = parseFloat(String(t.return_pct || 0));
                    const rb = parseFloat(String(t.risk_budget || 0));
                    const rMult = rb > 0 ? pl / rb : null;
                    const isOpen = (t.status || "").toUpperCase() === "OPEN";

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
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                        <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>{t.ticker}</td>
                        <td className="px-3 py-2" style={{ fontFamily: mono, fontSize: 10, color: "var(--ink-4)" }}>{t.trade_id}</td>
                        <td className="px-3 py-2">
                          <span className="text-[9px] px-1.5 py-0.5 rounded-full font-semibold"
                                style={{ background: `color-mix(in oklab, ${isOpen ? "#08a86b" : "#888"} 10%, var(--surface))`, color: isOpen ? "#08a86b" : "var(--ink-4)" }}>
                            {t.status}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-[10px]" style={{ color: "var(--ink-3)" }}>{t.rule}</td>
                        <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{String(t.open_date || "").slice(0, 10)}</td>
                        <td className="px-3 py-2" style={{ fontSize: 10, color: "var(--ink-4)" }}>{displayCloseDate || "—"}</td>
                        <td className="px-3 py-2" style={{ fontFamily: mono }}>{displayShares > 0 ? displayShares : "—"}</td>
                        <td className="px-3 py-2 privacy-mask" style={{ fontFamily: mono }}>{displayEntry > 0 ? `$${displayEntry.toFixed(2)}` : "—"}</td>
                        <td className="px-3 py-2 privacy-mask" style={{ fontFamily: mono }}>{displayExit > 0 ? `$${displayExit.toFixed(2)}` : "—"}</td>
                        <td className="px-3 py-2 font-bold privacy-mask" style={{ fontFamily: mono, color: pctColor(pl) }}>
                          ${pl >= 0 ? "+" : ""}{pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-3 py-2" style={{ fontFamily: mono, color: pctColor(displayRet) }}>{displayRet !== 0 ? `${displayRet >= 0 ? "+" : ""}${displayRet.toFixed(1)}%` : "—"}</td>
                        <td className="px-3 py-2" style={{ fontFamily: mono }}>{rMult != null ? `${rMult.toFixed(2)}R` : "—"}</td>
                      </tr>
                    );
                  })}</tbody>
                </table>
              </div>
            </div>
          </>
        );
      })()}
    </div>
  );
}
