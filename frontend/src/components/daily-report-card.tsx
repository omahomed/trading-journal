"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type JournalHistoryPoint, type TradeDetail, type TradePosition } from "@/lib/api";

function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }

function windowBadge(mw: string) {
  const wl = (mw || "").toUpperCase();
  const styles: Record<string, { bg: string; fg: string }> = {
    POWERTREND: { bg: "#8b5cf6", fg: "#fff" },
    OPEN: { bg: "#08a86b", fg: "#fff" },
    NEUTRAL: { bg: "#f59f00", fg: "#000" },
    CLOSED: { bg: "#e5484d", fg: "#fff" },
  };
  const s = styles[wl] || { bg: "#888", fg: "#fff" };
  return (
    <span className="px-3 py-1 rounded-[6px] text-[12px] font-bold" style={{ background: s.bg, color: s.fg }}>{mw || "N/A"}</span>
  );
}

export function DailyReportCard({ navColor }: { navColor: string }) {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState("");

  useEffect(() => {
    Promise.all([
      api.journalHistory("CanSlim", 0).catch(() => []),
      api.tradesRecent("CanSlim", 500).catch(() => []),
      api.tradesClosed("CanSlim", 500).catch(() => []),
    ]).then(([hist, det, closed]) => {
      const h = (hist as JournalHistoryPoint[]).sort((a, b) => String(b.day).localeCompare(String(a.day)));
      setHistory(h);
      setDetails(det as TradeDetail[]);
      setClosedTrades(closed as TradePosition[]);
      if (h.length > 0) setSelectedDate(String(h[0].day).slice(0, 10));
      setLoading(false);
    });
  }, []);

  const day = useMemo(() => {
    if (!selectedDate || history.length === 0) return null;
    return history.find(h => String(h.day).slice(0, 10) === selectedDate) || null;
  }, [history, selectedDate]);

  // Previous day for SPY/NDX daily change
  const prevDay = useMemo(() => {
    if (!selectedDate || history.length === 0) return null;
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const idx = sorted.findIndex(h => String(h.day).slice(0, 10) === selectedDate);
    return idx > 0 ? sorted[idx - 1] : null;
  }, [history, selectedDate]);

  // YTD calculations
  const ytdStats = useMemo(() => {
    if (!selectedDate || history.length === 0) return { portYtd: 0, spyYtd: 0, ndxYtd: 0 };
    const year = selectedDate.slice(0, 4);
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const ytd = sorted.filter(h => String(h.day).slice(0, 4) === year && String(h.day).slice(0, 10) <= selectedDate);
    const portYtd = ytd.length > 0 ? (ytd.reduce((p, h) => p * (1 + (h.daily_pct_change || 0) / 100), 1) - 1) * 100 : 0;
    const jan1 = ytd[0];
    const curr = ytd[ytd.length - 1];
    const spyYtd = jan1 && curr && jan1.spy > 0 ? ((curr.spy / jan1.spy) - 1) * 100 : 0;
    const ndxYtd = jan1 && curr && jan1.nasdaq > 0 ? ((curr.nasdaq / jan1.nasdaq) - 1) * 100 : 0;
    return { portYtd, spyYtd, ndxYtd };
  }, [history, selectedDate]);

  // Drawdown
  const ddPct = useMemo(() => {
    if (!selectedDate || history.length === 0) return 0;
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const upTo = sorted.filter(h => String(h.day).slice(0, 10) <= selectedDate);
    if (upTo.length === 0) return 0;
    const peak = Math.max(...upTo.map(h => h.end_nlv || 0));
    const curr = upTo[upTo.length - 1].end_nlv || 0;
    return peak > 0 ? ((curr - peak) / peak) * 100 : 0;
  }, [history, selectedDate]);

  // Trades on this day
  const dayBuys = details.filter(d => String(d.date).slice(0, 10) === selectedDate && String(d.action).toUpperCase() === "BUY");
  const daySells = details.filter(d => String(d.date).slice(0, 10) === selectedDate && String(d.action).toUpperCase() === "SELL");
  const dayClosed = closedTrades.filter(t => String(t.closed_date).slice(0, 10) === selectedDate);

  // Risk status
  const riskMsg = ddPct >= -7.5 ? "GREEN LIGHT" : ddPct >= -12.5 ? "CAUTION" : ddPct >= -15 ? "MAX 30% INVESTED" : "GO TO CASH";
  const riskColor = ddPct >= -7.5 ? "#08a86b" : ddPct >= -12.5 ? "#f59f00" : "#e5484d";

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  const spyDailyPct = prevDay && prevDay.spy > 0 && day ? ((day.spy - prevDay.spy) / prevDay.spy) * 100 : 0;
  const ndxDailyPct = prevDay && prevDay.nasdaq > 0 && day ? ((day.nasdaq - prevDay.nasdaq) / prevDay.nasdaq) * 100 : 0;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Daily <em className="italic" style={{ color: navColor }}>Report</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>CanSlim · End-of-day debrief</div>
      </div>

      {/* Date selector */}
      <div className="mb-5">
        <select value={selectedDate} onChange={e => setSelectedDate(e.target.value)}
                className="h-[38px] px-3 rounded-[10px] text-[13px] w-[220px]"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any, fontFamily: "var(--font-jetbrains), monospace" }}>
          {history.map((h, i) => (
            <option key={i} value={String(h.day).slice(0, 10)}>{String(h.day).slice(0, 10)}</option>
          ))}
        </select>
      </div>

      {day && (
        <>
          {/* Header date */}
          <div className="text-[16px] font-semibold mb-4">
            {new Date(selectedDate).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" })}
          </div>

          {/* Section 1: Header Metrics */}
          <div className="grid grid-cols-4 gap-3 mb-5">
            <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Net Liquidity</div>
              <div className="text-[20px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${(day.end_nlv || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
            </div>
            <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Daily P&L</div>
              <div className="text-[20px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(day.daily_pct_change || 0) }}>
                ${(day.daily_dollar_change || 0) >= 0 ? "+" : ""}{(day.daily_dollar_change || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
              <div className="text-[11px] mt-0.5" style={{ color: pctColor(day.daily_pct_change || 0) }}>
                {(day.daily_pct_change || 0) >= 0 ? "+" : ""}{(day.daily_pct_change || 0).toFixed(2)}%
              </div>
            </div>
            <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>Market Window</div>
              {windowBadge((day as any).market_window || "")}
            </div>
            <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>Risk Status</div>
              <span className="px-3 py-1 rounded-[6px] text-[12px] font-bold" style={{ background: riskColor, color: "#fff" }}>{riskMsg}</span>
            </div>
          </div>

          {/* Section 2: Performance + Market Notes */}
          <div className="grid grid-cols-2 gap-4 mb-5">
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-4 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Performance Comparison</div>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {["", "Daily", "YTD"].map(h => (
                        <th key={h} className="text-left px-4 py-2 text-[10px] uppercase tracking-[0.06em] font-semibold"
                            style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { label: "Portfolio", daily: (day.daily_pct_change || 0), ytd: ytdStats.portYtd },
                      { label: "SPY", daily: spyDailyPct, ytd: ytdStats.spyYtd },
                      { label: "NASDAQ", daily: ndxDailyPct, ytd: ytdStats.ndxYtd },
                    ].map(r => (
                      <tr key={r.label} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td className="px-4 py-2.5 font-semibold">{r.label}</td>
                        <td className="px-4 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(r.daily) }}>{r.daily >= 0 ? "+" : ""}{r.daily.toFixed(2)}%</td>
                        <td className="px-4 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(r.ytd) }}>{r.ytd >= 0 ? "+" : ""}{r.ytd.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="px-4 py-2.5 text-[12px]" style={{ color: "var(--ink-3)" }}>
                <strong>Drawdown:</strong> {ddPct.toFixed(2)}% · <strong>Invested:</strong> {(day.pct_invested || 0).toFixed(0)}%
              </div>
            </div>

            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-4 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Market Notes</div>
              <div className="p-4">
                {(day as any).market_notes ? (
                  <div className="px-3 py-2.5 rounded-[8px] text-[12px]" style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
                    {(day as any).market_notes}
                  </div>
                ) : (
                  <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No market notes logged.</div>
                )}
                {(day as any).market_action && (
                  <div className="mt-2 text-[12px]"><strong>Actions:</strong> {(day as any).market_action}</div>
                )}
              </div>
            </div>
          </div>

          {/* Section 3: Trade Activity */}
          <div className="grid grid-cols-2 gap-4 mb-5">
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-4 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Positions Opened</div>
              <div className="p-4">
                {dayBuys.length > 0 ? dayBuys.map((b, i) => (
                  <div key={i} className="flex items-center justify-between py-2" style={{ borderBottom: i < dayBuys.length - 1 ? "1px solid var(--border)" : "none" }}>
                    <span className="text-[13px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{b.ticker}</span>
                    <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                      {b.shares} shs @ ${parseFloat(String(b.amount || 0)).toFixed(2)} · {b.rule}
                    </span>
                  </div>
                )) : <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No new positions opened.</div>}
              </div>
            </div>

            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-4 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Positions Closed</div>
              <div className="p-4">
                {dayClosed.length > 0 ? dayClosed.map((s, i) => {
                  const pl = parseFloat(String(s.realized_pl || 0));
                  const ret = parseFloat(String(s.return_pct || 0));
                  return (
                    <div key={i} className="flex items-center justify-between py-2" style={{ borderBottom: i < dayClosed.length - 1 ? "1px solid var(--border)" : "none" }}>
                      <span className="text-[13px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{s.ticker}</span>
                      <span className="text-[11px]" style={{ color: pctColor(pl) }}>
                        P&L: ${pl >= 0 ? "+" : ""}{pl.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({ret >= 0 ? "+" : ""}{ret.toFixed(2)}%) · {s.sell_rule || ""}
                      </span>
                    </div>
                  );
                }) : daySells.length > 0 ? daySells.map((s, i) => (
                  <div key={i} className="flex items-center justify-between py-2" style={{ borderBottom: i < daySells.length - 1 ? "1px solid var(--border)" : "none" }}>
                    <span className="text-[13px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{s.ticker}</span>
                    <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                      Sold {s.shares} shs @ ${parseFloat(String(s.amount || 0)).toFixed(2)}
                    </span>
                  </div>
                )) : <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No positions closed.</div>}
              </div>
            </div>
          </div>

          {/* Section 4: Daily Review */}
          {(() => {
            const score = day.score || 0;
            const highlights = (day as any).highlights || "";
            const mistakes = (day as any).mistakes || "";
            const topLesson = (day as any).top_lesson || "";
            if (!score && !highlights && !mistakes && !topLesson) return null;

            // Try to parse report card scores from highlights
            let rc: Record<string, number> | null = null;
            try { if (highlights.startsWith("{")) rc = JSON.parse(highlights); } catch { /* */ }

            const gradeLabel = score >= 5 ? "A+" : score >= 4 ? "A" : score >= 3 ? "B" : score >= 2 ? "C" : score > 0 ? "D" : "";
            const gradeColor = score >= 4 ? "#08a86b" : score >= 3 ? "#f59f00" : "#e5484d";

            return (
              <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="px-4 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Daily Review</div>
                <div className="p-4">
                  {gradeLabel && (
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[11px] font-semibold" style={{ color: "var(--ink-4)" }}>Grade:</span>
                      <span className="text-[18px] font-bold" style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: gradeColor }}>{gradeLabel}</span>
                      {rc && (
                        <div className="flex gap-2 ml-2">
                          {[
                            { k: "plan", l: "Plan" }, { k: "stops", l: "Stops" }, { k: "sized", l: "Sized" },
                            { k: "fomo", l: "FOMO" }, { k: "journaled", l: "Jrnl" },
                          ].map(cat => rc![cat.k] != null ? (
                            <span key={cat.k} className="text-[10px] px-1.5 py-0.5 rounded" style={{
                              background: rc![cat.k] >= 4 ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : rc![cat.k] >= 3 ? "color-mix(in oklab, #f59f00 10%, var(--surface))" : "color-mix(in oklab, #e5484d 12%, var(--surface))",
                              color: rc![cat.k] >= 4 ? "#16a34a" : rc![cat.k] >= 3 ? "#d97706" : "#dc2626",
                            }}>
                              {cat.l} {rc![cat.k]}/5
                            </span>
                          ) : null)}
                        </div>
                      )}
                    </div>
                  )}
                  {mistakes && mistakes !== "nan" && (
                    <div className="text-[12px] mb-1"><strong>Notes:</strong> {mistakes}</div>
                  )}
                  {topLesson && topLesson !== "nan" && (
                    <div className="text-[12px]"><strong>Top Lesson:</strong> {topLesson}</div>
                  )}
                </div>
              </div>
            );
          })()}
        </>
      )}
    </div>
  );
}
