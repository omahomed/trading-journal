"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { api, getActivePortfolio, type JournalHistoryPoint, type TradeDetail, type TradePosition } from "@/lib/api";

/** Convert GitHub-style alert blockquotes into styled callout divs.
 *  Supports both two-line form:
 *    > [!great]
 *    > content
 *  and single-line form:
 *    > [!great] content
 */
function preprocessCallouts(md: string): string {
  // Matches: start of line, optional leading ">", "[!TYPE]", optional same-line content,
  // optional continuation lines starting with ">"
  const pattern = /^> \[!(\w+)\][ \t]*(.*?)(?:\r?\n((?:> ?.*(?:\r?\n|$))+))?(?=\r?\n[^>]|\r?\n$|$)/gmi;
  return md.replace(pattern, (_m, type: string, sameLine: string, body: string | undefined) => {
    const parts: string[] = [];
    if (sameLine && sameLine.trim()) parts.push(sameLine.trim());
    if (body) {
      const cleaned = body
        .split(/\r?\n/)
        .map(l => l.replace(/^> ?/, ""))
        .join("\n")
        .trim();
      if (cleaned) parts.push(cleaned);
    }
    const content = parts.join("\n");
    const t = type.toLowerCase();
    return `<div class="callout callout-${t}">\n<div class="callout-title">${type.toUpperCase()}</div>\n\n${content}\n\n</div>\n`;
  });
}

type SnapItem = { id?: number; image_type?: string; view_url?: string; uploaded_at?: string };

function pctColor(v: number) { return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)"; }

function windowBadge(mw: string) {
  const wl = (mw || "").toUpperCase();
  const styles: Record<string, { bg: string; fg: string }> = {
    POWERTREND: { bg: "#8A2BE2", fg: "#fff" },
    OPEN: { bg: "#08a86b", fg: "#fff" },
    NEUTRAL: { bg: "#f59f00", fg: "#000" },
    CLOSED: { bg: "#e5484d", fg: "#fff" },
  };
  const s = styles[wl] || { bg: "#888", fg: "#fff" };
  return (
    <span className="px-3 py-1 rounded-[6px] text-[12px] font-bold" style={{ background: s.bg, color: s.fg }}>{(mw || "N/A").toUpperCase()}</span>
  );
}

function cycleBadge(state: string) {
  const s = (state || "").toUpperCase();
  const styles: Record<string, { bg: string; fg: string }> = {
    POWERTREND: { bg: "#8A2BE2", fg: "#fff" },
    UPTREND: { bg: "#08a86b", fg: "#fff" },
    "RALLY MODE": { bg: "#f59f00", fg: "#000" },
    CORRECTION: { bg: "#e5484d", fg: "#fff" },
  };
  const st = styles[s] || { bg: "#888", fg: "#fff" };
  return (
    <span className="px-3 py-1 rounded-[6px] text-[12px] font-bold" style={{ background: st.bg, color: st.fg }}>{state || "N/A"}</span>
  );
}

export function DailyReportCard({ navColor, initialDate }: { navColor: string; initialDate?: string }) {
  const dateParam = initialDate || "";
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState("");
  const [snapshots, setSnapshots] = useState<SnapItem[]>([]);
  const [eodOpen, setEodOpen] = useState(false);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [thoughts, setThoughts] = useState("");
  const [thoughtsDirty, setThoughtsDirty] = useState(false);
  const [savingThoughts, setSavingThoughts] = useState(false);
  const [thoughtsMsg, setThoughtsMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [thoughtsMode, setThoughtsMode] = useState<"edit" | "preview">(() => {
    if (typeof window === "undefined") return "edit";
    const v = window.localStorage.getItem("dailyReport.thoughtsMode");
    return v === "preview" ? "preview" : "edit";
  });
  useEffect(() => {
    if (typeof window !== "undefined") window.localStorage.setItem("dailyReport.thoughtsMode", thoughtsMode);
  }, [thoughtsMode]);
  const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null);
  const [imageMsg, setImageMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize the textarea to fit its content
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.max(200, ta.scrollHeight + 2) + "px";
  }, [thoughts, thoughtsMode]);

  useEffect(() => {
    Promise.all([
      api.journalHistory(getActivePortfolio(), 0).catch(() => []),
      api.tradesRecent(getActivePortfolio(), 500).catch(() => []),
      api.tradesClosed(getActivePortfolio(), 500).catch(() => []),
    ]).then(([hist, det, closed]) => {
      const h = (hist as JournalHistoryPoint[]).sort((a, b) => String(b.day).localeCompare(String(a.day)));
      setHistory(h);
      setDetails(det as TradeDetail[]);
      setClosedTrades(closed as TradePosition[]);
      if (h.length > 0) {
        const match = dateParam && h.find(d => String(d.day).slice(0, 10) === dateParam);
        setSelectedDate(match ? dateParam : String(h[0].day).slice(0, 10));
      }
      setLoading(false);
    });
  }, [dateParam]);

  // Load snapshots + thoughts when selectedDate changes
  useEffect(() => {
    if (!selectedDate) { setSnapshots([]); setThoughts(""); return; }
    api.listEodSnapshots(selectedDate, getActivePortfolio()).then(res => {
      if (Array.isArray(res)) setSnapshots(res as any);
      else setSnapshots([]);
    }).catch(() => setSnapshots([]));
    setThoughtsMsg(null);
    setThoughtsDirty(false);
  }, [selectedDate]);

  // Lazy-fill market_cycle for the selected day if the entry exists but the
  // value is missing. Fires at most once per date per session via the
  // attempted set. Same auto-compute path used by Market Window.
  const attemptedCycleFill = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!selectedDate || history.length === 0) return;
    const entry = history.find(h => String(h.day).slice(0, 10) === selectedDate) as any;
    if (!entry) return;
    if (entry.market_cycle) return;
    if (attemptedCycleFill.current.has(selectedDate)) return;
    attemptedCycleFill.current.add(selectedDate);
    api.journalEdit({ portfolio: getActivePortfolio(), day: selectedDate })
      .then(res => {
        if (res.status !== "ok") return;
        return api.journalHistory(getActivePortfolio(), 0);
      })
      .then(fresh => {
        if (!fresh) return;
        const h = (fresh as JournalHistoryPoint[]).sort((a, b) => String(b.day).localeCompare(String(a.day)));
        setHistory(h);
      })
      .catch(() => { /* ignore */ });
  }, [selectedDate, history]);

  // Hydrate thoughts from the selected journal entry (stored in lowlights field)
  useEffect(() => {
    if (!selectedDate || history.length === 0) { setThoughts(""); return; }
    const entry = history.find(h => String(h.day).slice(0, 10) === selectedDate) as any;
    setThoughts(entry?.lowlights || "");
    setThoughtsDirty(false);
  }, [selectedDate, history]);

  const reloadSnapshots = useCallback(async () => {
    if (!selectedDate) return;
    try {
      const res = await api.listEodSnapshots(selectedDate, getActivePortfolio());
      if (Array.isArray(res)) setSnapshots(res as any);
    } catch { /* ignore */ }
  }, [selectedDate]);

  const saveThoughts = async () => {
    if (!selectedDate) return;
    setSavingThoughts(true);
    setThoughtsMsg(null);
    try {
      const res = await api.journalEdit({
        portfolio: getActivePortfolio(),
        day: selectedDate,
        lowlights: thoughts,
      });
      if (res.status === "ok") {
        setThoughtsMsg({ ok: true, text: "Saved" });
        setThoughtsDirty(false);
        // Refresh history cache locally
        setHistory(prev => prev.map(h => String(h.day).slice(0, 10) === selectedDate
          ? ({ ...h, lowlights: thoughts } as any) : h));
      } else {
        setThoughtsMsg({ ok: false, text: res.detail || "Save failed" });
      }
    } catch (err: any) {
      setThoughtsMsg({ ok: false, text: err.message || "Save failed" });
    }
    setSavingThoughts(false);
    setTimeout(() => setThoughtsMsg(null), 3000);
  };

  const uploadFiles = async (files: FileList | File[]) => {
    if (!selectedDate) return;
    setUploading(true);
    try {
      const arr = Array.from(files).filter(f => f.type.startsWith("image/"));
      for (const file of arr) {
        await api.uploadEodSnapshot(file, selectedDate, "note", getActivePortfolio());
      }
      await reloadSnapshots();
    } catch (err) {
      console.error("Upload failed:", err);
    }
    setUploading(false);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) uploadFiles(e.dataTransfer.files);
  };

  // Close lightbox on Escape
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setLightbox(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  // Paste images from clipboard — works anywhere on the page when a date is selected
  useEffect(() => {
    if (!selectedDate) return;
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      const files: File[] = [];
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (it.kind === "file" && it.type.startsWith("image/")) {
          const f = it.getAsFile();
          if (f) files.push(f);
        }
      }
      if (files.length > 0) {
        e.preventDefault();
        uploadFiles(files);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [selectedDate]);

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
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>{getActivePortfolio()} · End-of-day debrief</div>
      </div>

      {history.length === 0 && (
        <div className="border-[1.5px] border-dashed rounded-[14px] p-16 text-center"
             style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <div className="w-14 h-14 rounded-[16px] flex items-center justify-center mx-auto mb-[18px] text-2xl"
               style={{ background: `color-mix(in oklab, ${navColor} 12%, transparent)`, color: navColor }}>
            ✦
          </div>
          <h2 className="text-[22px] font-normal italic m-0 mb-1.5"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            No report yet
          </h2>
          <p className="text-[13px] max-w-[440px] mx-auto leading-relaxed"
             style={{ color: "var(--ink-3)" }}>
            Fill out the <strong>Daily Routine</strong> first — your end-of-day NLV,
            scorecard, and journal entries feed this report.
          </p>
        </div>
      )}

      {/* Date selector */}
      {history.length > 0 && (() => {
        const days = history.map(h => String(h.day).slice(0, 10));
        const minDay = days.length ? days[days.length - 1] : undefined;
        const maxDay = days.length ? days[0] : undefined;
        const hasData = !!selectedDate && days.includes(selectedDate);
        const step = (delta: number) => {
          if (!selectedDate || days.length === 0) return;
          const sortedAsc = [...days].sort();
          const idx = sortedAsc.indexOf(selectedDate);
          if (idx === -1) return;
          const next = sortedAsc[idx + delta];
          if (next) setSelectedDate(next);
        };
        return (
          <div className="mb-5 flex items-center gap-2">
            <button onClick={() => step(-1)} disabled={!hasData || selectedDate === minDay}
                    className="h-[38px] w-[38px] rounded-[10px] text-[13px] font-semibold transition-all hover:brightness-110 disabled:opacity-40"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
                    title="Previous day with data">‹</button>
            <input type="date" value={selectedDate} min={minDay} max={maxDay}
                   onChange={e => setSelectedDate(e.target.value)}
                   className="h-[38px] px-3 rounded-[10px] text-[13px] w-[180px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }} />
            <button onClick={() => step(1)} disabled={!hasData || selectedDate === maxDay}
                    className="h-[38px] w-[38px] rounded-[10px] text-[13px] font-semibold transition-all hover:brightness-110 disabled:opacity-40"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
                    title="Next day with data">›</button>
            {selectedDate && !hasData && (
              <span className="text-[12px] ml-2" style={{ color: "var(--ink-4)" }}>No data for this date</span>
            )}
          </div>
        );
      })()}

      {day && (
        <>
          {/* Header date */}
          <div className="text-[16px] font-semibold mb-4">
            {(() => {
              const [y, m, d] = selectedDate.split("-").map(n => parseInt(n));
              const dt = new Date(y, m - 1, d);
              return dt.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
            })()}
          </div>

          {/* Section 1: Header Metrics */}
          <div className="grid grid-cols-5 gap-3 mb-5">
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
              <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>NASDAQ Cycle</div>
              {(day as any).market_cycle
                ? cycleBadge((day as any).market_cycle)
                : <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>—</span>}
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
                            { k: "fomo", l: "FOMO" },
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

          {/* ── EOD Snapshots (collapsible) ── */}
          {(() => {
            const eodSnaps = snapshots.filter(s => (s.image_type || "").startsWith("eod_"));
            if (eodSnaps.length === 0) return null;
            return (
              <div className="mt-6 rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                <button onClick={() => setEodOpen(!eodOpen)}
                        className="w-full flex items-center gap-2 px-[18px] py-3 text-left cursor-pointer transition-colors hover:brightness-95"
                        style={{ background: "var(--surface-2)" }}>
                  <span className="text-[10px] transition-transform" style={{ transform: eodOpen ? "rotate(90deg)" : "none", color: "var(--ink-4)" }}>▶</span>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                  <span className="text-[13px] font-semibold">End-of-Day Snapshots</span>
                  <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{eodSnaps.length} captured · click to expand</span>
                </button>
                {eodOpen && (
                  <div className="p-4 grid grid-cols-2 gap-3" style={{ animation: "slide-up 0.12s ease-out" }}>
                    {eodSnaps.map((snap, idx) => (
                      <div key={snap.id ?? idx} className="rounded-[8px] overflow-hidden" style={{ border: "1px solid var(--border)", background: "var(--bg)" }}>
                        <div className="px-2.5 py-1.5 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                          <span className="text-[10px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>
                            {snap.image_type?.replace("eod_", "") || "Snapshot"}
                          </span>
                          {snap.uploaded_at && (
                            <span className="text-[9px]" style={{ color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>
                              {String(snap.uploaded_at).slice(11, 19)}
                            </span>
                          )}
                        </div>
                        {snap.view_url && (
                          <button onClick={() => setLightbox(snap.view_url || null)}
                                  className="block w-full p-0 border-0 cursor-zoom-in"
                                  style={{ background: "transparent" }}>
                            <img src={snap.view_url} alt={snap.image_type}
                                 style={{ width: "100%", maxHeight: 220, objectFit: "contain", display: "block", background: "var(--bg-2)" }} />
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })()}

          {/* ── Daily Thoughts ── */}
          <div className="mt-6 rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Daily Thoughts</span>
              <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>markdown supported · drag/paste images to attach</span>
              <div className="ml-auto flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                {([["edit", "Edit"], ["preview", "Preview"]] as const).map(([val, label]) => (
                  <button key={val} onClick={() => setThoughtsMode(val)}
                          className="px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all"
                          style={{
                            background: thoughtsMode === val ? "var(--surface)" : "transparent",
                            color: thoughtsMode === val ? "var(--ink)" : "var(--ink-4)",
                            boxShadow: thoughtsMode === val ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
                            border: "none", cursor: "pointer",
                          }}>
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="p-4 flex flex-col gap-4">
              {thoughtsMode === "edit" ? (
                <textarea
                  ref={textareaRef}
                  value={thoughts}
                  onChange={e => { setThoughts(e.target.value); setThoughtsDirty(true); }}
                  placeholder="What did you learn today? What went well? What didn't? Decisions, mistakes, observations..."
                  className="w-full px-3.5 py-3 rounded-[10px] text-[13px] outline-none"
                  style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "inherit", lineHeight: 1.6, minHeight: 200, overflow: "hidden" }}
                />
              ) : (
                <div className="px-5 py-4 rounded-[10px] prose-custom"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", lineHeight: 1.6, minHeight: 200 }}>
                  {thoughts.trim() ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                      {preprocessCallouts(thoughts)}
                    </ReactMarkdown>
                  ) : (
                    <div style={{ color: "var(--ink-4)", fontStyle: "italic" }}>Nothing written yet. Switch to Edit to start.</div>
                  )}
                </div>
              )}

              {/* Drag-drop zone + file picker */}
              <div
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                className="rounded-[10px] px-4 py-6 text-center cursor-pointer transition-all"
                style={{
                  border: `2px dashed ${dragOver ? navColor : "var(--border)"}`,
                  background: dragOver ? `${navColor}08` : "var(--bg)",
                  color: "var(--ink-3)",
                }}
                onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" accept="image/*" multiple style={{ display: "none" }}
                       onChange={e => { if (e.target.files) uploadFiles(e.target.files); e.target.value = ""; }} />
                {uploading ? (
                  <div className="text-[12px] font-medium">Uploading...</div>
                ) : (
                  <>
                    <div className="text-[12px] font-medium mb-1">Drag &amp; drop, click to upload, or paste (⌘V)</div>
                    <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>Multiple files supported. PNG / JPG</div>
                  </>
                )}
              </div>

              {/* Uploaded notes images */}
              {(() => {
                const noteSnaps = snapshots.filter(s => (s.image_type || "") === "eod_note");
                if (noteSnaps.length === 0 && !imageMsg) return null;
                return (
                  <div className="flex flex-col gap-2">
                    {imageMsg && (
                      <div className="text-[12px] font-medium px-3 py-1.5 rounded-[6px] self-start"
                           style={{ background: "color-mix(in oklab, #08a86b 12%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))", fontFamily: "var(--font-jetbrains), monospace" }}>
                        ✓ {imageMsg}
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-3">
                      {noteSnaps.map((snap, idx) => {
                        const isPending = pendingDeleteId === snap.id;
                        return (
                          <div key={snap.id ?? idx} className="relative rounded-[8px] overflow-hidden group"
                               style={{ border: `1px solid ${isPending ? "#e5484d" : "var(--border)"}`, background: "var(--bg-2)" }}>
                            {snap.view_url && (
                              <button onClick={() => setLightbox(snap.view_url || null)}
                                      className="block w-full p-0 border-0 cursor-zoom-in"
                                      style={{ background: "transparent" }}>
                                <img src={snap.view_url} alt="note attachment"
                                     style={{ width: "100%", maxHeight: 260, objectFit: "contain", display: "block", background: "var(--bg)" }} />
                              </button>
                            )}
                            {snap.id && (
                              <button
                                onClick={async (e) => {
                                  e.stopPropagation();
                                  if (!isPending) {
                                    setPendingDeleteId(snap.id!);
                                    setTimeout(() => {
                                      setPendingDeleteId(prev => prev === snap.id ? null : prev);
                                    }, 3000);
                                    return;
                                  }
                                  try {
                                    await api.deleteImage(snap.id!);
                                    setPendingDeleteId(null);
                                    setImageMsg("Image deleted");
                                    setTimeout(() => setImageMsg(null), 2500);
                                    await reloadSnapshots();
                                  } catch {
                                    setPendingDeleteId(null);
                                    setImageMsg("Delete failed");
                                    setTimeout(() => setImageMsg(null), 2500);
                                  }
                                }}
                                className={`absolute top-2 right-2 h-7 rounded-full text-white text-[11px] font-semibold flex items-center justify-center transition-all ${isPending ? "px-3 opacity-100" : "w-7 opacity-0 group-hover:opacity-100"}`}
                                style={{ background: isPending ? "#e5484d" : "rgba(0,0,0,0.6)", border: "1px solid rgba(255,255,255,0.2)" }}
                                title={isPending ? "Click again to confirm" : "Delete"}>
                                {isPending ? "Confirm" : "✕"}
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

              {/* Save row */}
              <div className="flex items-center gap-3">
                <button onClick={saveThoughts} disabled={savingThoughts || !thoughtsDirty}
                        className="h-[38px] px-5 rounded-[10px] text-[12px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
                        style={{ background: navColor }}>
                  {savingThoughts ? "Saving..." : "Save Thoughts"}
                </button>
                {thoughtsMsg && (
                  <span className="text-[12px] font-medium" style={{ color: thoughtsMsg.ok ? "#16a34a" : "#e5484d" }}>
                    {thoughtsMsg.ok ? "✓" : "✗"} {thoughtsMsg.text}
                  </span>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Lightbox */}
      {lightbox && (
        <div onClick={() => setLightbox(null)}
             className="fixed inset-0 z-50 flex items-center justify-center cursor-zoom-out"
             style={{ background: "rgba(0,0,0,0.92)" }}>
          <img src={lightbox} alt="full size"
               onClick={e => e.stopPropagation()}
               style={{ maxWidth: "99vw", maxHeight: "99vh", objectFit: "contain", boxShadow: "0 20px 60px rgba(0,0,0,0.5)" }} />
          <button onClick={() => setLightbox(null)}
                  className="fixed top-4 right-4 w-10 h-10 rounded-full text-white text-[20px] flex items-center justify-center"
                  style={{ background: "rgba(255,255,255,0.15)", border: "1px solid rgba(255,255,255,0.25)" }}>
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
