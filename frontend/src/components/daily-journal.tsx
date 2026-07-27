"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import {
  api,
  getActivePortfolio,
  type JournalHistoryPoint,
  type TradeDetail,
  type TradePosition,
  type NotesRailItem,
  type NotesRailYtdStats,
} from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { NotesRail, type NotesRailHandle } from "./notes-rail";
import { TagPicker } from "./tag-picker";
import { DailyThoughts } from "./daily-thoughts";
import { TradingChecklist } from "./trading-checklist";
import { SectionExpander } from "./section-expander";
import { ScorecardMiniForm } from "./scorecard-mini-form";
import { autoTickByPrefix, SYSTEM_ITEM_PREFIXES, todayInChicago } from "@/lib/routine-autotick";
import { SCORECARD_CATEGORIES } from "@/lib/scorecard";
import { SnapshotGallery } from "./snapshot-gallery";

/** Convert GitHub-style alert blockquotes into styled callout divs.
 *  Supports both two-line form:
 *    > [!great]
 *    > content
 *  and single-line form:
 *    > [!great] content
 */
function preprocessCallouts(md: string): string {
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

// Exported for unit testing (daily-journal.test.tsx). The rest of
// the file continues to use this via the DailyJournal component
// below — the export is additive, no consumers outside the test.
export function cycleBadge(state: string) {
  const s = (state || "").toUpperCase();
  const styles: Record<string, { bg: string; fg: string }> = {
    POWERTREND: { bg: "#8A2BE2", fg: "#fff" },
    UPTREND: { bg: "#08a86b", fg: "#fff" },
    "UPTREND UNDER PRESSURE": { bg: "#d97706", fg: "#fff" },
    "RALLY MODE": { bg: "#f59f00", fg: "#000" },
    CORRECTION: { bg: "#e5484d", fg: "#fff" },
  };
  const st = styles[s] || { bg: "#888", fg: "#fff" };
  return (
    <span className="px-3 py-1 rounded-[6px] text-[12px] font-bold" style={{ background: st.bg, color: st.fg }}>{state || "N/A"}</span>
  );
}

export function DailyJournal({ navColor, initialDate }: { navColor: string; initialDate?: string }) {
  const dateParam = initialDate || "";
  const portfolio = getActivePortfolio();
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [closedTrades, setClosedTrades] = useState<TradePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState("");
  const [snapshots, setSnapshots] = useState<SnapItem[]>([]);
  const [lightbox, setLightbox] = useState<string | null>(null);
  // "recap" backs the existing lowlights markdown column (renamed from
  // "Daily Thoughts" to "Daily Recap" in Phase 7 — content + behavior
  // unchanged; rename only).
  const [recap, setRecap] = useState("");
  const [recapDirty, setRecapDirty] = useState(false);
  // "dailyThoughts" backs the new daily_thoughts TEXT column (rich-text
  // HTML edited via the shared <ThoughtsEditor>). Migration 031.
  const [dailyThoughts, setDailyThoughts] = useState("");
  const dailyThoughtsDirtyRef = useRef(false);
  const [savingThoughts, setSavingThoughts] = useState(false);
  const [thoughtsMsg, setThoughtsMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [recapMode, setRecapMode] = useState<"edit" | "preview">(() => {
    if (typeof window === "undefined") return "edit";
    const v = window.localStorage.getItem("dailyReport.thoughtsMode");
    return v === "preview" ? "preview" : "edit";
  });
  useEffect(() => {
    if (typeof window !== "undefined") window.localStorage.setItem("dailyReport.thoughtsMode", recapMode);
  }, [recapMode]);

  // Phase 7 — NotesRail state. Mirrors the weekly-retro rail wiring.
  const [railItems, setRailItems] = useState<NotesRailItem[]>([]);
  // Phase 8 — imperative ref so TagBar can fire a rail refetch on
  // successful tag mutations. The rail's refresh() delegates to the
  // onRefresh prop wired below.
  const railRef = useRef<NotesRailHandle>(null);
  const [railYtdStats, setRailYtdStats] = useState<NotesRailYtdStats>({
    total_weeks: 0, weeks_graded: 0, avg_grade: null, weeks_pinned: 0,
  });

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // T2-4b — market_notes inline edit state. Desktop previously had a
  // read-only Market Notes display; this adds a pencil → textarea +
  // Save / Cancel affordance so users can edit inline
  // without bouncing away. Save uses the same
  // /api/journal/edit endpoint as the other field saves.
  const [marketNotesEdit, setMarketNotesEdit] = useState(false);
  const [marketNotesValue, setMarketNotesValue] = useState("");
  const [marketNotesSaving, setMarketNotesSaving] = useState(false);
  const [marketNotesMsg, setMarketNotesMsg] = useState<{ ok: boolean; text: string } | null>(null);

  // Phase 2 merger: scorecard mini-form open state + a reload counter
  // used to remount TradingChecklist after autotick so the "Journal"
  // row's ticked state reflects the save without needing a manual
  // refresh. Also gates the fetch effect below via its dep array so
  // history refetches after a save.
  const [scorecardOpen, setScorecardOpen] = useState(false);
  const [reloadCounter, setReloadCounter] = useState(0);

  // Auto-resize the textarea to fit its content (recap markdown editor).
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.max(200, ta.scrollHeight + 2) + "px";
  }, [recap, recapMode]);

  // Phase 7 — fetch the rail envelope.
  const refreshRail = useCallback(async () => {
    if (!portfolio) return;
    try {
      const res = await api.dailyJournalList(portfolio);
      if ("error" in res) {
        log.error("daily-journal", "rail fetch failed", res.error);
        return;
      }
      setRailItems(res.days);
      setRailYtdStats(res.ytd_stats);
    } catch (err) {
      log.error("daily-journal", "rail fetch threw", err);
    }
  }, [portfolio]);

  useEffect(() => {
    if (!portfolio) return;
    refreshRail();
  }, [portfolio, refreshRail]);

  useEffect(() => {
    Promise.all([
      api.journalHistory(portfolio, 0).catch((err) => {
        log.error("daily-journal", "journalHistory fetch failed", err);
        return [];
      }),
      api.tradesRecent(portfolio, 500).catch((err) => {
        log.error("daily-journal", "tradesRecent fetch failed", err);
        return { details: [], lot_closures: [] };
      }),
      api.tradesClosed(portfolio, 500).catch((err) => {
        log.error("daily-journal", "tradesClosed fetch failed", err);
        return [];
      }),
    ]).then(([hist, det, closed]) => {
      const h = (hist as JournalHistoryPoint[]).sort((a, b) => String(b.day).localeCompare(String(a.day)));
      setHistory(h);
      setDetails(det.details);
      setClosedTrades(closed as TradePosition[]);
      if (h.length > 0) {
        const match = dateParam && h.find(d => String(d.day).slice(0, 10) === dateParam);
        setSelectedDate(match ? dateParam : String(h[0].day).slice(0, 10));
      }
      setLoading(false);
    });
  }, [dateParam, portfolio, reloadCounter]);

  // Load snapshots when selectedDate changes
  useEffect(() => {
    if (!selectedDate) {
      setSnapshots([]); setRecap(""); setDailyThoughts("");
      return;
    }
    api.listEodSnapshots(selectedDate, portfolio).then(res => {
      if (Array.isArray(res)) setSnapshots(res as any);
      else setSnapshots([]);
    }).catch((err) => {
      log.error("daily-journal", "listEodSnapshots fetch failed", err);
      setSnapshots([]);
    });
    setThoughtsMsg(null);
    setRecapDirty(false);
    dailyThoughtsDirtyRef.current = false;
  }, [selectedDate, portfolio]);

  // Lazy-fill market_cycle for the selected day if the entry exists but
  // the value is missing. Fires at most once per date per session.
  const attemptedCycleFill = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!selectedDate || history.length === 0) return;
    const entry = history.find(h => String(h.day).slice(0, 10) === selectedDate) as any;
    if (!entry) return;
    if (entry.market_cycle) return;
    if (attemptedCycleFill.current.has(selectedDate)) return;
    attemptedCycleFill.current.add(selectedDate);
    api.journalEdit({ portfolio, day: selectedDate })
      .then(res => {
        if (res.status !== "ok") return;
        return api.journalHistory(portfolio, 0);
      })
      .then(fresh => {
        if (!fresh) return;
        const h = (fresh as JournalHistoryPoint[]).sort((a, b) => String(b.day).localeCompare(String(a.day)));
        setHistory(h);
      })
      .catch(() => { /* ignore */ });
  }, [selectedDate, history, portfolio]);

  // Hydrate recap + dailyThoughts from the selected journal entry.
  // lowlights → recap (existing markdown column); daily_thoughts → the
  // new rich-text editor body (migration 031). Resetting the dirty
  // flags here makes sure the debounced auto-save effect below doesn't
  // immediately re-save on initial hydration.
  useEffect(() => {
    if (!selectedDate || history.length === 0) {
      setRecap(""); setDailyThoughts("");
      return;
    }
    const entry = history.find(h => String(h.day).slice(0, 10) === selectedDate) as any;
    setRecap(entry?.lowlights || "");
    setDailyThoughts(entry?.daily_thoughts || "");
    setRecapDirty(false);
    dailyThoughtsDirtyRef.current = false;
  }, [selectedDate, history]);

  // Auto-save dailyThoughts via debounced effect. Mirrors the weekly-
  // retro pattern: dirtyRef gates the effect so the initial hydration
  // doesn't trigger an empty-write race. The recap markdown still has
  // its explicit Save button.
  useEffect(() => {
    if (!selectedDate) return;
    if (!dailyThoughtsDirtyRef.current) return;
    const handle = window.setTimeout(() => {
      void api.journalEdit({
        portfolio,
        day: selectedDate,
        daily_thoughts: dailyThoughts,
      }).then(res => {
        if (res.status === "ok") {
          dailyThoughtsDirtyRef.current = false;
          setHistory(prev => prev.map(h => String(h.day).slice(0, 10) === selectedDate
            ? ({ ...h, daily_thoughts: dailyThoughts } as any) : h));
          // Autotick "Journal" only when the entry being edited IS today
          // — backfilling a past-date journal shouldn't count as "did
          // today's journal."
          if (selectedDate === todayInChicago()) {
            void autoTickByPrefix(SYSTEM_ITEM_PREFIXES.journal);
          }
        }
      }).catch(err => log.error("daily-journal", "daily_thoughts save failed", err));
    }, 800);
    return () => window.clearTimeout(handle);
  }, [dailyThoughts, selectedDate, portfolio]);

  const openMarketNotesEdit = () => {
    const current = String((day as any)?.market_notes ?? "");
    setMarketNotesValue(current);
    setMarketNotesEdit(true);
    setMarketNotesMsg(null);
  };

  const cancelMarketNotesEdit = () => {
    setMarketNotesEdit(false);
    setMarketNotesMsg(null);
  };

  const saveMarketNotes = async () => {
    if (!selectedDate) return;
    setMarketNotesSaving(true);
    setMarketNotesMsg(null);
    try {
      const res = await api.journalEdit({
        portfolio,
        day: selectedDate,
        market_notes: marketNotesValue,
      });
      if (res.status === "ok") {
        setHistory((prev) =>
          prev.map((h) =>
            String(h.day).slice(0, 10) === selectedDate
              ? ({ ...h, market_notes: marketNotesValue } as any)
              : h,
          ),
        );
        setMarketNotesMsg({ ok: true, text: "Saved" });
        setMarketNotesEdit(false);
      } else {
        setMarketNotesMsg({ ok: false, text: res.detail || "Save failed" });
      }
    } catch (err: any) {
      setMarketNotesMsg({ ok: false, text: err.message || "Save failed" });
    }
    setMarketNotesSaving(false);
    setTimeout(() => setMarketNotesMsg(null), 3000);
  };

  const saveRecap = async () => {
    if (!selectedDate) return;
    setSavingThoughts(true);
    setThoughtsMsg(null);
    try {
      const res = await api.journalEdit({
        portfolio,
        day: selectedDate,
        lowlights: recap,
      });
      if (res.status === "ok") {
        setThoughtsMsg({ ok: true, text: "Saved" });
        setRecapDirty(false);
        setHistory(prev => prev.map(h => String(h.day).slice(0, 10) === selectedDate
          ? ({ ...h, lowlights: recap } as any) : h));
        // Autotick "Journal" only when the entry being edited IS today.
        if (selectedDate === todayInChicago()) {
          void autoTickByPrefix(SYSTEM_ITEM_PREFIXES.journal);
        }
      } else {
        setThoughtsMsg({ ok: false, text: res.detail || "Save failed" });
      }
    } catch (err: any) {
      setThoughtsMsg({ ok: false, text: err.message || "Save failed" });
    }
    setSavingThoughts(false);
    setTimeout(() => setThoughtsMsg(null), 3000);
  };

  // Close lightbox on Escape
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setLightbox(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  const day = useMemo(() => {
    if (!selectedDate || history.length === 0) return null;
    return history.find(h => String(h.day).slice(0, 10) === selectedDate) || null;
  }, [history, selectedDate]);

  // Phase 7 — id of the daily journal row for the selected day. Drives
  // the TagPicker entityId, the DailyThoughts editor (inline image
  // uploads need it), and the SnapshotGallery FK. May be null until
  // history loads or for pre-migration rows without an id field; the
  // child components handle the disabled state.
  const dayJournalId = useMemo(() => {
    const raw = (day as any)?.id;
    if (raw == null) return null;
    const n = typeof raw === "number" ? raw : parseInt(String(raw), 10);
    return isNaN(n) ? null : n;
  }, [day]);

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
    <div className="flex" style={{ animation: "slide-up 0.18s ease-out", minHeight: "100%" }}>
      {/* Phase 7 — NotesRail (left side). Mirrors the weekly-retro
          mount. Pin toggles go through the polymorphic /api/pins/toggle
          and refresh the rail on success. */}
      <NotesRail
        ref={railRef}
        entityType="daily_journal"
        items={railItems}
        ytdStats={railYtdStats}
        currentEntityKey={selectedDate || null}
        onItemClick={(it) => setSelectedDate(it.key)}
        onPinToggle={async (entityId, _currentlyPinned) => {
          const res = await api.pinsToggle("daily_journal", entityId);
          if ("error" in res) throw new Error(res.error);
          await refreshRail();
        }}
        onRefresh={refreshRail}
      />

      <div className="flex-1 min-w-0 lg:pl-7">
        <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
          <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Daily <em className="italic" style={{ color: navColor }}>Journal</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>{portfolio} · Today&apos;s workflow + end-of-day debrief</div>
          {/* Phase 7 — TagPicker. entityId is null until the journal row
              exists (i.e., the day was logged via NLV Entry); the
              picker handles the disabled state. */}
          <TagPicker
            entityType="daily_journal"
            entityId={dayJournalId}
            portfolio={portfolio}
            onTagsChanged={() => railRef.current?.refresh()}
          />
        </div>

        {/* Date selector — comes first (before Checklist) so the "which
            day am I looking at" affordance is at the top, per user
            request 2026-07-26. Only renders when there's history to
            page through. */}
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

        {/* Human-readable date sits right below the picker, above the
            Checklist. Renders on any valid selectedDate — decoupled from
            `day` so it's visible even when the entry is missing. */}
        {selectedDate && /^\d{4}-\d{2}-\d{2}$/.test(selectedDate) && (
          <div className="text-[16px] font-semibold mb-4">
            {(() => {
              const [y, m, d] = selectedDate.split("-").map(n => parseInt(n));
              const dt = new Date(y, m - 1, d);
              return dt.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
            })()}
          </div>
        )}

        {history.length === 0 && (
          <div className="border-[1.5px] border-dashed rounded-[14px] p-8 text-center mb-5"
               style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
            <p className="text-[13px] max-w-[440px] mx-auto leading-relaxed m-0"
               style={{ color: "var(--ink-3)" }}>
              Tick the <strong>Equity routine</strong> checklist item above and log NLV
              to populate today&apos;s metrics + market notes.
            </p>
          </div>
        )}

        {day && (
          <>

            {/* Section 1: Header Metrics */}
            <div className="grid grid-cols-4 gap-3 mb-5">
              <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Net Liquidity</div>
                <div className="text-[20px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{formatCurrency(day.end_nlv || 0)}</div>
              </div>
              <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Daily P&L</div>
                <div className="text-[20px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(day.daily_pct_change || 0) }}>
                  {formatCurrency(day.daily_dollar_change || 0, { showSign: true })}
                </div>
                <div className="text-[11px] mt-0.5" style={{ color: pctColor(day.daily_pct_change || 0) }}>
                  {(day.daily_pct_change || 0) >= 0 ? "+" : ""}{(day.daily_pct_change || 0).toFixed(2)}%
                </div>
              </div>
              <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold mb-2" style={{ color: "var(--ink-4)" }}>MCT State</div>
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
                <div
                  className="flex items-center justify-between px-4 py-3 text-[13px] font-semibold"
                  style={{ borderBottom: "1px solid var(--border)" }}
                >
                  <span>Market Notes</span>
                  {!marketNotesEdit && (
                    <button
                      type="button"
                      onClick={openMarketNotesEdit}
                      aria-label="Edit market notes"
                      data-testid="market-notes-edit-button"
                      className="text-[11px] font-medium rounded-[6px] px-2 py-1"
                      style={{
                        background: "transparent",
                        color: "var(--ink-3)",
                        border: "1px solid var(--border)",
                        cursor: "pointer",
                      }}
                    >
                      Edit
                    </button>
                  )}
                </div>
                <div className="p-4">
                  {marketNotesEdit ? (
                    <div className="flex flex-col gap-2">
                      <textarea
                        value={marketNotesValue}
                        onChange={(e) => setMarketNotesValue(e.target.value)}
                        placeholder="One-line market summary — QQQ at 21EMA, strong open, etc."
                        data-testid="market-notes-textarea"
                        className="w-full px-3 py-2 rounded-[8px] text-[12px] outline-none resize-none"
                        rows={3}
                        style={{
                          background: "var(--bg)",
                          border: "1px solid var(--border)",
                          color: "var(--ink)",
                          fontFamily: "inherit",
                          lineHeight: 1.5,
                        }}
                      />
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={saveMarketNotes}
                          disabled={marketNotesSaving}
                          data-testid="market-notes-save-button"
                          className="rounded-[6px] px-3 py-1.5 text-[11px] font-medium"
                          style={{
                            background: "#08a86b",
                            color: "#fff",
                            border: "none",
                            cursor: marketNotesSaving ? "default" : "pointer",
                            opacity: marketNotesSaving ? 0.6 : 1,
                          }}
                        >
                          {marketNotesSaving ? "Saving…" : "Save"}
                        </button>
                        <button
                          type="button"
                          onClick={cancelMarketNotesEdit}
                          disabled={marketNotesSaving}
                          className="rounded-[6px] px-3 py-1.5 text-[11px]"
                          style={{
                            background: "transparent",
                            color: "var(--ink-3)",
                            border: "1px solid var(--border)",
                            cursor: "pointer",
                          }}
                        >
                          Cancel
                        </button>
                        {marketNotesMsg && (
                          <span
                            className="text-[11px] ml-1"
                            style={{ color: marketNotesMsg.ok ? "#08a86b" : "#e5484d" }}
                          >
                            {marketNotesMsg.text}
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (day as any).market_notes ? (
                    <div className="px-3 py-2.5 rounded-[8px] text-[12px]" style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
                      {(day as any).market_notes}
                    </div>
                  ) : (
                    <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No market notes logged.</div>
                  )}
                  {!marketNotesEdit && (day as any).market_action && (
                    <div className="mt-2 text-[12px]"><strong>Actions:</strong> {(day as any).market_action}</div>
                  )}
                </div>
              </div>
            </div>

            {/* Section 3: Trade Activity — SectionExpander per row so the
                cards share styling with Daily Thoughts. Grid stays 2-col
                so the two expanders sit side-by-side; each collapses
                independently. */}
            {(() => {
              const closedRows = dayClosed.length > 0 ? dayClosed.length : daySells.length;
              return (
                <div className="grid grid-cols-2 gap-4">
                  <SectionExpander
                    title="Positions Opened"
                    defaultExpanded={dayBuys.length > 0 && dayBuys.length <= 3}
                    localStorageKey="mo-daily-journal-positions-opened-expanded"
                    showDot
                    headerCaption={() => dayBuys.length === 0 ? "none" : `${dayBuys.length}`}>
                    <div className="p-4">
                      {dayBuys.length > 0 ? dayBuys.map((b, i) => (
                        <div key={i} className="flex items-center justify-between py-2" style={{ borderBottom: i < dayBuys.length - 1 ? "1px solid var(--border)" : "none" }}>
                          <span className="text-[13px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{b.ticker}</span>
                          <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                            {b.shares} shs @ {formatCurrency(parseFloat(String(b.amount || 0)))} · {b.rule}
                          </span>
                        </div>
                      )) : <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No new positions opened.</div>}
                    </div>
                  </SectionExpander>

                  <SectionExpander
                    title="Positions Closed"
                    defaultExpanded={closedRows > 0 && closedRows <= 3}
                    localStorageKey="mo-daily-journal-positions-closed-expanded"
                    showDot
                    headerCaption={() => closedRows === 0 ? "none" : `${closedRows}`}>
                    <div className="p-4">
                      {dayClosed.length > 0 ? dayClosed.map((s, i) => {
                        const pl = parseFloat(String(s.realized_pl || 0));
                        const ret = parseFloat(String(s.return_pct || 0));
                        return (
                          <div key={i} className="flex items-center justify-between py-2" style={{ borderBottom: i < dayClosed.length - 1 ? "1px solid var(--border)" : "none" }}>
                            <span className="text-[13px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{s.ticker}</span>
                            <span className="text-[11px]" style={{ color: pctColor(pl) }}>
                              P&L: {formatCurrency(pl, { showSign: true })} ({ret >= 0 ? "+" : ""}{ret.toFixed(2)}%) · {s.sell_rule || ""}
                            </span>
                          </div>
                        );
                      }) : daySells.length > 0 ? daySells.map((s, i) => (
                        <div key={i} className="flex items-center justify-between py-2" style={{ borderBottom: i < daySells.length - 1 ? "1px solid var(--border)" : "none" }}>
                          <span className="text-[13px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{s.ticker}</span>
                          <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                            Sold {s.shares} shs @ {formatCurrency(parseFloat(String(s.amount || 0)))}
                          </span>
                        </div>
                      )) : <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>No positions closed.</div>}
                    </div>
                  </SectionExpander>
                </div>
              );
            })()}
          </>
        )}

        {/* Phase 2 merger: Trading Checklist section. Renders regardless
            of NLV / journal-entry state — always today's items, never
            tied to the date picker's selected date. Sits between the
            "what happened so far" sections (Positions) and the "what
            I'm capturing about today" sections (Daily Review + notes),
            per user request 2026-07-26. */}
        <SectionExpander
          title="Checklist"
          defaultExpanded={true}
          localStorageKey="mo-daily-journal-checklist-expanded"
          showDot
          headerCaption={() => "same-day undo only"}>
          <div className="p-4">
            {/* key={reloadCounter} forces a remount after scorecard save
                so TradingChecklist re-fetches items and the "Journal"
                row reflects the auto-tick without a manual refresh. */}
            <TradingChecklist key={reloadCounter} navColor={navColor} />
          </div>
        </SectionExpander>

        {day && (
          <>
            {/* Section 4: Daily Scorecard (renamed from Daily Review
                per Phase 2 merger). Captures via ScorecardMiniForm on
                the Journal checklist item's flow; NLV Entry keeps its
                own copy of the same fields until the multi-portfolio
                form is trimmed in a follow-up. Empty state surfaces a
                "Grade today" call-to-action; populated state renders
                the grade + chips + notes with an Edit button. */}
            {(() => {
              const score = day.score || 0;
              const highlights = (day as any).highlights || "";
              const mistakes = (day as any).mistakes || "";
              const topLesson = (day as any).top_lesson || "";
              const graded = !!(score || (highlights && highlights.startsWith("{")) || mistakes);

              let rc: Record<string, number> | null = null;
              try { if (highlights.startsWith("{")) rc = JSON.parse(highlights); } catch { /* */ }

              const gradeLabel = score >= 5 ? "A+" : score >= 4 ? "A" : score >= 3 ? "B" : score >= 2 ? "C" : score > 0 ? "D" : "";
              const gradeColor = score >= 4 ? "#08a86b" : score >= 3 ? "#f59f00" : "#e5484d";

              return (
                <SectionExpander
                  title="Daily Scorecard"
                  defaultExpanded={true}
                  localStorageKey="mo-daily-journal-scorecard-expanded"
                  showDot
                  headerCaption={() => graded ? gradeLabel : "not graded"}>
                  <div className="p-4">
                    {graded ? (
                      <>
                        {gradeLabel && (
                          <div className="flex items-center gap-3 mb-3 flex-wrap">
                            <span className="text-[11px] font-semibold" style={{ color: "var(--ink-4)" }}>Grade:</span>
                            <span className="text-[18px] font-bold" style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: gradeColor }}>{gradeLabel}</span>
                            {rc && (
                              <div className="flex gap-2 ml-2">
                                {SCORECARD_CATEGORIES.map(cat => rc![cat.key] != null ? (
                                  <span key={cat.key} className="text-[10px] px-1.5 py-0.5 rounded" style={{
                                    background: rc![cat.key] >= 4 ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : rc![cat.key] >= 3 ? "color-mix(in oklab, #f59f00 10%, var(--surface))" : "color-mix(in oklab, #e5484d 12%, var(--surface))",
                                    color: rc![cat.key] >= 4 ? "#16a34a" : rc![cat.key] >= 3 ? "#d97706" : "#dc2626",
                                  }}>
                                    {cat.key === "plan" ? "Plan" : cat.key === "stops" ? "Stops" : cat.key === "sized" ? "Sized" : "FOMO"} {rc![cat.key]}/5
                                  </span>
                                ) : null)}
                              </div>
                            )}
                            <button type="button"
                                    onClick={() => setScorecardOpen(true)}
                                    className="ml-auto text-[11px] px-2 py-1 rounded-[6px] transition-colors hover:brightness-95"
                                    style={{ background: "var(--bg-2)", color: "var(--ink-3)", border: "1px solid var(--border)" }}
                                    data-testid="scorecard-edit-button">
                              ✎ Edit
                            </button>
                          </div>
                        )}
                        {mistakes && mistakes !== "nan" && (
                          <div className="text-[12px] mb-1"><strong>Notes:</strong> {mistakes}</div>
                        )}
                        {topLesson && topLesson !== "nan" && (
                          <div className="text-[12px]"><strong>Top Lesson:</strong> {topLesson}</div>
                        )}
                      </>
                    ) : (
                      <div className="flex flex-col items-start gap-2">
                        <span className="text-[12px]" style={{ color: "var(--ink-3)" }}>
                          Not graded yet. Tap to grade the day (Plan / Stops / Sized / FOMO).
                        </span>
                        <button type="button"
                                onClick={() => setScorecardOpen(true)}
                                className="px-3 py-1.5 rounded-[8px] text-[12px] font-semibold text-white transition-all"
                                style={{ background: navColor }}
                                data-testid="scorecard-grade-today-button">
                          Grade today
                        </button>
                      </div>
                    )}
                  </div>
                </SectionExpander>
              );
            })()}

            {/* ── Daily Thoughts (Phase 7 — rich-text editor) ──
                Shared <ThoughtsEditor> via <DailyThoughts> wrapper. Auto-
                saves via the debounced effect above when the dirty ref
                flips. journalId enables inline image embed; when null
                (e.g., pre-Daily-Routine days) the editor surfaces the
                "save first" inline error on image paste/drop. */}
            <div className="mt-6">
              <DailyThoughts
                value={dailyThoughts}
                onChange={(next) => { dailyThoughtsDirtyRef.current = true; setDailyThoughts(next); }}
                journalId={dayJournalId}
                portfolio={portfolio}
              />
            </div>

            {/* ── Daily Recap (renamed from "Daily Thoughts" in Phase 7) ──
                Same markdown editor + content as before. Backs the
                `lowlights` column. Explicit Save button; no auto-save.
                Phase 2 merger: wrapped in SectionExpander for consistent
                collapse styling with Daily Thoughts / Positions cards. */}
            <SectionExpander
              title="Daily Recap"
              defaultExpanded={!recap || recap.length < 500}
              localStorageKey="mo-daily-journal-recap-expanded"
              showDot
              headerCaption={() => recap.trim()
                ? `${recap.trim().split(/\s+/).length} words`
                : "empty · markdown"}>
              <div className="px-4 py-3 flex items-center gap-2" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>markdown supported</span>
                <div className="ml-auto flex p-0.5 rounded-[8px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  {([["edit", "Edit"], ["preview", "Preview"]] as const).map(([val, label]) => (
                    <button key={val} onClick={() => setRecapMode(val)}
                            className="px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all"
                            style={{
                              background: recapMode === val ? "var(--surface)" : "transparent",
                              color: recapMode === val ? "var(--ink)" : "var(--ink-4)",
                              boxShadow: recapMode === val ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
                              border: "none", cursor: "pointer",
                            }}>
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="p-4 flex flex-col gap-4">
                {recapMode === "edit" ? (
                  <textarea
                    ref={textareaRef}
                    value={recap}
                    onChange={e => { setRecap(e.target.value); setRecapDirty(true); }}
                    placeholder="Summarize the day. What went well, what didn't, decisions made, observations…"
                    className="w-full px-3.5 py-3 rounded-[10px] text-[13px] outline-none"
                    style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "inherit", lineHeight: 1.6, minHeight: 200, overflow: "hidden" }}
                  />
                ) : (
                  <div className="px-5 py-4 rounded-[10px] prose-custom"
                       style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", lineHeight: 1.6, minHeight: 200 }}>
                    {recap.trim() ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                        {preprocessCallouts(recap)}
                      </ReactMarkdown>
                    ) : (
                      <div style={{ color: "var(--ink-4)", fontStyle: "italic" }}>Nothing written yet. Switch to Edit to start.</div>
                    )}
                  </div>
                )}

                {/* Save row */}
                <div className="flex items-center gap-3">
                  <button onClick={saveRecap} disabled={savingThoughts || !recapDirty}
                          className="h-[38px] px-5 rounded-[10px] text-[12px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
                          style={{ background: navColor }}>
                    {savingThoughts ? "Saving..." : "Save Recap"}
                  </button>
                  {thoughtsMsg && (
                    <span className="text-[12px] font-medium" style={{ color: thoughtsMsg.ok ? "#16a34a" : "#e5484d" }}>
                      {thoughtsMsg.ok ? "✓" : "✗"} {thoughtsMsg.text}
                    </span>
                  )}
                </div>
              </div>
            </SectionExpander>

            {/* ── EOD Snapshots ──
                Moved 2026-07-26 to sit right before Daily Captures per
                user request. Phase 7: no user uploads (those route to
                Daily Captures below); only auto-generated eod_dashboard
                / eod_campaign rows render. Legacy `eod_note` rows are
                filtered out server-side by /api/snapshots/{day}.
                Converted to SectionExpander for consistent chrome. */}
            {(() => {
              const eodSnaps = snapshots.filter(s => (s.image_type || "").startsWith("eod_"));
              if (eodSnaps.length === 0) return null;
              return (
                <SectionExpander
                  title="End-of-Day Snapshots"
                  defaultExpanded={false}
                  localStorageKey="mo-daily-journal-eod-snapshots-expanded"
                  showDot
                  headerCaption={() => `${eodSnaps.length} captured`}>
                  <div className="p-4 grid grid-cols-2 gap-3">
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
                </SectionExpander>
              );
            })()}

            {/* ── Daily Captures (Phase 7) ──
                Shared <SnapshotGallery> with entityType="daily_journal".
                Replaces the pre-Phase-7 drag-drop zone that lived inside
                the (now-renamed) Daily Recap section. The gallery's own
                window paste handler cooperates with the DailyThoughts
                editor via the [data-thoughts-editor] check, so pastes
                inside the editor route inline; pastes outside route
                here. Wrapped in SectionExpander for consistent collapse
                styling with the other Phase 2 merger sections. */}
            <SectionExpander
              title="Daily Captures"
              defaultExpanded={false}
              localStorageKey="mo-daily-journal-captures-expanded"
              showDot
              headerCaption={() => "screenshots, charts, anything visual"}>
              <SnapshotGallery
                entityType="daily_journal"
                entityId={dayJournalId}
                portfolio={portfolio}
                disabledMessage="Save the journal entry first to add captures."
                activeMessage="Paste a screenshot or drag an image here"
                microcopy="Anything worth a second look — charts, alerts, news clips. PNG, JPEG, GIF, WEBP. Max 15MB."
                dropZoneAriaLabel="Upload capture"
                lightboxAriaLabel="Capture preview"
              />
            </SectionExpander>
          </>
        )}

        {/* Scorecard mini-form — triggered by "Grade today" empty
            state or "Edit" chip inside Daily Scorecard section. On
            successful save: bumps reloadCounter (re-fetches history +
            remounts TradingChecklist), autoticks "Journal" when
            editing today. */}
        <ScorecardMiniForm
          open={scorecardOpen}
          portfolio={portfolio}
          day={selectedDate}
          initial={{
            highlights: day ? ((day as any).highlights || null) : null,
            mistakes: day ? ((day as any).mistakes || null) : null,
          }}
          onSaved={() => {
            setScorecardOpen(false);
            setReloadCounter(c => c + 1);
            if (selectedDate === todayInChicago()) {
              void autoTickByPrefix(SYSTEM_ITEM_PREFIXES.journal);
            }
          }}
          onClose={() => setScorecardOpen(false)}
        />

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
    </div>
  );
}
