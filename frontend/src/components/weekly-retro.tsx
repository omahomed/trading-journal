"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { api, getActivePortfolio, type NotesRailItem, type NotesRailYtdStats, type TradeDetail, type WeeklyMetrics, type WeeklyRetro, type WeeklyRetroTickerGrade } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { gradeColor } from "@/lib/grade-helpers";
import { log } from "@/lib/log";
import { TagPicker } from "./tag-picker";
import { WeeklyThoughts } from "./weekly-thoughts";
import { WeeklySnapshot } from "./weekly-snapshot";
import { SectionExpander } from "./section-expander";
import { WeeklyInsightsTile } from "./weekly-insights-tile";
import { FlightDeck } from "./flight-deck";
import { NotesRail, type NotesRailHandle } from "./notes-rail";
import { CloseTheWeek, type CloseTheWeekState } from "./close-the-week";
import { Icons } from "./icons";

// Phase 2: Per-Ticker Details expander persistence. Per-USER UI preference
// (not portfolio- or week-scoped). Owned by <SectionExpander>; the key is
// held here so future tests/migrations can find the canonical name.
const TICKETS_EXPANDED_KEY = "mo-weekly-retro-tickets-expanded";
// Phase 4: Weekly Snapshot expander key.
const SNAPSHOT_EXPANDED_KEY = "mo-weekly-retro-snapshot-expanded";

const EXEC_GRADES = ["A (Perfect)", "B (Good)", "C (Sloppy)", "D (Bad)", "F (Impulse)"];
const BEHAVIOR_TAGS = [
  "Followed Plan", "FOMO Entry", "Caught Knife", "Late Stop",
  "Hesitated", "Boredom Trade", "Sized Too Big", "Revenge Trade", "Panic Sell",
];

type TickerGradeMap = Record<string, WeeklyRetroTickerGrade>;

// gradeColor — single source of truth in @/lib/grade-helpers. Used by
// ticker-card grade chip below; the Phase 4.6 CloseTheWeek surface
// imports its own copy.

export function WeeklyRetro({ navColor, initialWeek }: { navColor: string; initialWeek?: string }) {
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [weekDate, setWeekDate] = useState(() => {
    // Honor ?week=YYYY-MM-DD when present — mirrors the daily-report ?date=
    // pattern so the mobile list view's tap-to-detail navigation lands on
    // the tapped week instead of the current Monday.
    if (initialWeek && /^\d{4}-\d{2}-\d{2}$/.test(initialWeek)) return initialWeek;
    const n = new Date();
    const day = n.getDay(); // 0=Sun
    const offset = day === 0 ? -6 : 1 - day;
    const mon = new Date(n);
    mon.setDate(n.getDate() + offset);
    return `${mon.getFullYear()}-${String(mon.getMonth() + 1).padStart(2, "0")}-${String(mon.getDate()).padStart(2, "0")}`;
  });

  // Ticker-level grades
  const [ticker_grades, setTickerGrades] = useState<TickerGradeMap>({});
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  // Week summary — snake_case mirrors the wire shape (Phase 0).
  const [week_grade, setWeekGrade] = useState<string>("");
  const [best_decision, setBestDecision] = useState("");
  const [worst_decision, setWorstDecision] = useState("");
  const [rule_change, setRuleChange] = useState(false);
  const [rule_change_text, setRuleChangeText] = useState("");
  // Phase 4.6 — 3-axis grading + persisted review state. State stays in
  // this parent so the existing debounced auto-save effect keeps firing
  // on any change without CloseTheWeek owning a side-channel write path.
  const [execution_grade, setExecutionGrade] = useState<string>("");
  const [process_grade, setProcessGrade] = useState<string>("");
  const [pnl_grade, setPnlGrade] = useState<string>("");
  const [overall_override, setOverallOverride] = useState<boolean>(false);
  const [reviewed_at, setReviewedAt] = useState<string | null>(null);
  // Phase 3: HTML body of the Weekly Thoughts editor. Snake_case for wire
  // parity. Stored as HTML string; sanitized in the editor before send.
  const [weekly_thoughts, setWeeklyThoughts] = useState("");
  const [saveMsg, setSaveMsg] = useState("");

  // Phase 4: Weekly Snapshot count, lifted up so the SectionExpander
  // header caption can render "N attached" without exposing the list.
  // WeeklySnapshot fires onCountChange whenever its visible-count changes
  // (uploads, deletes, fetch). Reset on week change via the same
  // hydration effect that resets the other per-week local state.
  const [snapshotCount, setSnapshotCount] = useState(0);

  // Phase 5: server-computed performance metrics for the top tile row.
  // Refetched on portfolio or week change. `error` short-circuits the
  // tile values to a single inline message; `null` metrics + !error
  // means in-flight (renders skeleton tiles).
  const [metrics, setMetrics] = useState<WeeklyMetrics | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);

  // Saved retros, keyed by week_start. Source of truth for the form
  // hydration effect; the NotesRail uses a separate /list endpoint that
  // returns sparkline + synthetic-empty-week rows in addition to these.
  const [retros, setRetros] = useState<Record<string, WeeklyRetro>>({});

  // Phase 6 — NotesRail data. Server-shaped (live computation, no
  // snapshot columns); refetched after saves so a freshly-graded week's
  // dot fills in. The rail itself owns the optimistic pin-toggle UI.
  const [railItems, setRailItems] = useState<NotesRailItem[]>([]);
  // Phase 8 — imperative ref so TagBar can fire a rail refetch on
  // successful tag mutations.
  const railRef = useRef<NotesRailHandle>(null);
  const [railYtdStats, setRailYtdStats] = useState<NotesRailYtdStats>({
    total_weeks: 0, weeks_graded: 0, avg_grade: null, weeks_pinned: 0,
  });

  // Dirty flag gates the debounced auto-save effect so the initial mount
  // and every cross-week hydration don't fire a wasteful PUT. Mutated by
  // user-driven setters below.
  const dirtyRef = useRef(false);

  const portfolio = getActivePortfolio();

  // Week range — always snap to Monday (Mon=1...Sun=0 → treat as previous
  // week's end). Computed eagerly before the effects below; previously
  // lived further down the body, but Phase 6 deps reference monStr in
  // useEffect arrays which require TDZ-clean declaration order.
  const _wd = new Date(weekDate + "T12:00:00");
  const _dayOfWeek = _wd.getDay(); // 0=Sun, 1=Mon...6=Sat
  const _monOffset = _dayOfWeek === 0 ? -6 : 1 - _dayOfWeek;
  const monday = new Date(_wd);
  monday.setDate(_wd.getDate() + _monOffset);
  const friday = new Date(monday);
  friday.setDate(monday.getDate() + 4);
  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);
  const monStr = `${monday.getFullYear()}-${String(monday.getMonth() + 1).padStart(2, "0")}-${String(monday.getDate()).padStart(2, "0")}`;
  const sunStr = `${sunday.getFullYear()}-${String(sunday.getMonth() + 1).padStart(2, "0")}-${String(sunday.getDate()).padStart(2, "0")}`;

  useEffect(() => {
    api.tradesRecent(portfolio, 1000).then(d => {
      setDetails(d.details);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [portfolio]);

  // Phase 6 — fetch the NotesRail envelope (replaces the old bare-array
  // retros list that fed the deleted Review History tab). The rail rows
  // carry id+week_grade+pinned for the form-hydration index, plus
  // sparkline_value + has_content for the rail's own UI.
  //
  // The legacy localStorage key cleanup carries over from Phase 0 to
  // protect against stale "mo-weekly-retros" data resurfacing.
  const refreshRail = useCallback(async () => {
    if (!portfolio) return;
    try {
      const res = await api.weeklyRetroList(portfolio);
      if ("error" in res) {
        // The Phase 6 "0 weeks" regression originated here — a silent
        // return left railItems at its empty-array default and the UI
        // rendered the empty state with no devtools signal. Surfacing
        // the error to console.error gives any future regression of
        // this class an immediate signal in devtools.
        log.error("weekly-retro", "rail fetch failed", res.error);
        return;
      }
      setRailItems(res.weeks);
      setRailYtdStats(res.ytd_stats);
    } catch (err) {
      log.error("weekly-retro", "rail fetch threw", err);
    }
  }, [portfolio]);

  useEffect(() => {
    if (!portfolio) return;
    refreshRail();
    try { localStorage.removeItem("mo-weekly-retros"); } catch { /* incognito */ }
  }, [portfolio, refreshRail]);

  // Phase 5 — fetch the weekly performance metrics whenever the active
  // portfolio or week changes. Fires in parallel with the retros list
  // fetch; the two responses share no data. The tile skeletons render
  // while in-flight (metrics === null && metricsLoading), and any
  // server-side error surfaces inline above the tiles.
  useEffect(() => {
    if (!portfolio) return;
    let cancelled = false;
    setMetricsLoading(true);
    setMetricsError(null);
    api.weeklyMetrics(portfolio, monStr).then(res => {
      if (cancelled) return;
      if ("error" in res) {
        setMetricsError(res.error);
        setMetrics(null);
      } else {
        setMetrics(res);
      }
    }).catch(err => {
      if (cancelled) return;
      setMetricsError(err?.message || "Failed to load metrics");
      setMetrics(null);
    }).finally(() => {
      if (!cancelled) setMetricsLoading(false);
    });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portfolio, weekDate]);

  // Phase 6 — fetch the single retro for the active week on demand.
  // The Phase 0 bulk-list fetch is gone (its endpoint now returns the
  // rail envelope without content fields), so each week-navigation
  // pulls its own full retro. The hydration effect below reads from
  // the `retros` map populated by this fetch.
  useEffect(() => {
    if (!portfolio || retros[monStr]) return;
    let cancelled = false;
    api.weeklyRetroGet(portfolio, monStr).then(res => {
      if (cancelled || !res || "error" in res) return;
      setRetros(prev => prev[monStr] ? prev : { ...prev, [monStr]: res });
    }).catch(() => { /* silent — fresh blank state is fine */ });
    return () => { cancelled = true; };
  }, [portfolio, monStr, retros]);

  // Hydrate per-week local state when the user picks a different week
  // OR when the lazy retro fetch (above) populates retros[monStr] for
  // the first time. The retros[monStr]?.id dep re-fires hydration exactly
  // once per week — when the id appears. Subsequent setRetros calls
  // (post-save updates) keep the id stable, so hydration doesn't
  // re-trigger and clobber in-flight edits — the Phase 0 regression
  // fence still holds.
  useEffect(() => {
    const existing = retros[monStr];
    if (existing) {
      setWeekGrade(existing.week_grade || "");
      setBestDecision(existing.best_decision || "");
      setWorstDecision(existing.worst_decision || "");
      setRuleChange(existing.rule_change || false);
      setRuleChangeText(existing.rule_change_text || "");
      setWeeklyThoughts(existing.weekly_thoughts || "");
      setTickerGrades(existing.ticker_grades || {});
      setExecutionGrade(existing.execution_grade || "");
      setProcessGrade(existing.process_grade || "");
      setPnlGrade(existing.pnl_grade || "");
      setOverallOverride(!!existing.overall_override);
      setReviewedAt(existing.reviewed_at || null);
    } else {
      setWeekGrade(""); setBestDecision(""); setWorstDecision("");
      setRuleChange(false); setRuleChangeText("");
      setWeeklyThoughts(""); setTickerGrades({});
      setExecutionGrade(""); setProcessGrade(""); setPnlGrade("");
      setOverallOverride(false); setReviewedAt(null);
    }
    // Reset dirty flag — hydration is not a user edit. The next render's
    // debounce useEffect sees dirtyRef.current = false and skips firing.
    dirtyRef.current = false;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weekDate, retros[monStr]?.id]);

  const weekTxns = useMemo(() => {
    return details.filter(d => {
      const dt = String(d.date || "").slice(0, 10);
      return dt >= monStr && dt <= sunStr;
    }).sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
  }, [details, monStr, sunStr]);

  const grouped = useMemo(() => {
    const map: Record<string, TradeDetail[]> = {};
    for (const tx of weekTxns) {
      const t = tx.ticker || "Unknown";
      if (!map[t]) map[t] = [];
      map[t].push(tx);
    }
    return Object.entries(map).sort((a, b) => a[0].localeCompare(b[0]));
  }, [weekTxns]);

  const totalTx = weekTxns.length;
  const uniqueTickers = grouped.length;
  const buys = weekTxns.filter(d => String(d.action).toUpperCase() === "BUY");
  const sells = weekTxns.filter(d => String(d.action).toUpperCase() === "SELL");
  const isOveractive = totalTx > 15;
  const gradedTickers = Object.values(ticker_grades).filter(g => g.grade).length;

  const getGrade = (ticker: string): WeeklyRetroTickerGrade =>
    ticker_grades[ticker] || { grade: "", behavior: "", notes: "" };
  const setGradeField = (ticker: string, field: keyof WeeklyRetroTickerGrade, value: string) => {
    dirtyRef.current = true;
    setTickerGrades(prev => ({ ...prev, [ticker]: { ...getGrade(ticker), [field]: value } }));
  };

  // Optimistic save: state already reflects user input; PUT writes through
  // and we merge the authoritative response into the local cache. Errors
  // surface as a non-blocking saveMsg and the local state is preserved so
  // the explicit Save button (or next keystroke) can retry.
  const handleSave = useCallback(async () => {
    const payload: Omit<WeeklyRetro, "id" | "created_at" | "updated_at"> = {
      portfolio,
      week_start: monStr,
      week_grade: week_grade || null,
      best_decision,
      worst_decision,
      rule_change,
      rule_change_text,
      weekly_thoughts,
      ticker_grades,
      // Phase 4.6 fields. Server is authoritative on the derived
      // overall (when overall_override is false AND all 3 axes are
      // set, it recomputes week_grade); the frontend sends both so
      // the round-trip echo reflects whatever the server decided.
      execution_grade: execution_grade || null,
      process_grade: process_grade || null,
      pnl_grade: pnl_grade || null,
      overall_override,
      reviewed_at,
    };
    const result = await api.weeklyRetroUpsert(payload);
    if ("error" in result) {
      setSaveMsg(`Save failed: ${result.error}`);
      setTimeout(() => setSaveMsg(""), 4000);
      throw new Error(result.error);
    }
    setRetros(prev => ({ ...prev, [result.week_start]: result }));
    setSaveMsg("Weekly retro saved!");
    setTimeout(() => setSaveMsg(""), 3000);
    // Phase 6: refresh the rail so the just-saved week's draft dot fills
    // in (or grade letter appears). Fire-and-forget; rail simply stays
    // stale on failure.
    refreshRail();
    return result;
  }, [portfolio, monStr, week_grade, best_decision, worst_decision,
      rule_change, rule_change_text, weekly_thoughts, ticker_grades,
      execution_grade, process_grade, pnl_grade, overall_override,
      reviewed_at, refreshRail]);

  // Debounced auto-save. dirtyRef gates the effect so the initial hydration
  // pass (and every cross-week switch) doesn't fire a wasteful PUT. Mirrors
  // the priceLookup debounce in log-buy.tsx — 800ms idle window. The flag is
  // set by user-driven setters (Save button stays as a "force save now").
  useEffect(() => {
    if (!dirtyRef.current) return;
    const t = setTimeout(() => {
      handleSave().catch(() => { /* surfaced via saveMsg already */ });
    }, 800);
    return () => clearTimeout(t);
  }, [week_grade, best_decision, worst_decision, rule_change, rule_change_text,
      weekly_thoughts, ticker_grades,
      execution_grade, process_grade, pnl_grade, overall_override, reviewed_at,
      handleSave]);

  // Phase 4.6 — Recent Overall trend. Pull the last 4 prior weeks with a
  // non-null week_grade from the rail items already in memory; reverse to
  // oldest→newest for left-to-right reading in the design's strip. Empty
  // array on the user's first ever week — RecentOverallCard renders
  // placeholder dashes.
  const recentOverall = useMemo(() => {
    return railItems
      .filter(it => it.week_start < monStr && it.week_grade)
      .sort((a, b) => b.week_start.localeCompare(a.week_start))
      .slice(0, 4)
      .reverse()
      .map(it => it.week_grade as string);
  }, [railItems, monStr]);

  // Phase 4.6 — adapter so CloseTheWeek can fire one onChange per field
  // without re-rendering the parent on every keystroke for unrelated
  // state. Each entry in the patch dispatches to its own setter +
  // marks dirty.
  const handleCtwChange = useCallback((patch: Partial<CloseTheWeekState>) => {
    dirtyRef.current = true;
    if (patch.week_grade !== undefined) setWeekGrade(patch.week_grade);
    if (patch.execution_grade !== undefined) setExecutionGrade(patch.execution_grade);
    if (patch.process_grade !== undefined) setProcessGrade(patch.process_grade);
    if (patch.pnl_grade !== undefined) setPnlGrade(patch.pnl_grade);
    if (patch.overall_override !== undefined) setOverallOverride(patch.overall_override);
    if (patch.reviewed_at !== undefined) setReviewedAt(patch.reviewed_at);
    if (patch.best_decision !== undefined) setBestDecision(patch.best_decision);
    if (patch.worst_decision !== undefined) setWorstDecision(patch.worst_decision);
    if (patch.rule_change !== undefined) setRuleChange(patch.rule_change);
    if (patch.rule_change_text !== undefined) setRuleChangeText(patch.rule_change_text);
  }, []);

  const ctwState: CloseTheWeekState = {
    week_grade,
    execution_grade,
    process_grade,
    pnl_grade,
    overall_override,
    reviewed_at,
    best_decision,
    worst_decision,
    rule_change,
    rule_change_text,
  };

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  const inputStyle: React.CSSProperties = {
    background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  };

  return (
    <div className="flex" style={{ animation: "slide-up 0.18s ease-out", minHeight: "100%" }}>
      {/* Phase 6 — NotesRail (left side). Hidden below lg via the
          component's own wrapperClass. The rail navigates within the
          page by mutating weekDate; pin toggles call the polymorphic
          /api/pins/toggle endpoint and refresh the rail on success. */}
      <NotesRail
        ref={railRef}
        entityType="weekly_retro"
        items={railItems}
        ytdStats={railYtdStats}
        currentEntityKey={monStr}
        onItemClick={(it) => setWeekDate(it.week_start)}
        onPinToggle={async (entityId, currentlyPinned) => {
          const res = await api.pinsToggle("weekly_retro", entityId);
          if ("error" in res) throw new Error(res.error);
          await refreshRail();
        }}
        onRefresh={refreshRail}
      />

      <div className="flex-1 min-w-0 lg:pl-7">
        <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
          <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Weekly <em className="italic" style={{ color: navColor }}>Retro</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Grade execution · Identify patterns · Refine rules</div>
          {/* Tag bar (Phase 1). entityId is null until the retro has been
              saved at least once — the picker handles the disabled state. */}
          <TagPicker
            entityType="weekly_retro"
            entityId={retros[monStr]?.id ?? null}
            portfolio={portfolio}
            onTagsChanged={() => railRef.current?.refresh()}
          />
        </div>

        {/* Phase 6: Review History tab removed; the NotesRail on the left
            is the canonical way to navigate past retros. The standalone
            "Select Week" date input has also been removed — the rail
            header's calendar icon (Jump to date) is the single date
            picker. The "Reviewing: X → Y" pill stays as a passive
            indicator of which week the form is currently on. */}
        <>
          <div className="flex items-center gap-4 mb-5">
            <div className="px-4 py-2 rounded-[10px] text-[12px] font-medium"
                 style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
              Reviewing: <strong>{monday.toLocaleDateString("en-US", { month: "short", day: "numeric" })}</strong> → <strong>{friday.toLocaleDateString("en-US", { month: "short", day: "numeric" })}</strong>
            </div>
          </div>

          {/* Phase 5 — Weekly Insights gradient tile row. Replaces the
              prior 4-tile activity grid (Total Tickets / Unique Tickers /
              Buys / Sells), which has moved into the Per-Ticker Details
              expander body as <FlightDeck/>. Metrics come from
              /api/analytics/weekly-metrics; tiles render skeletons while
              in-flight and an inline error message on failure. Negative
              values dim + prefix a ↓ glyph — the gradient stays. */}
          {metricsError && (
            <div className="mb-3 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#dc2626", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
              Weekly metrics unavailable: {metricsError}
            </div>
          )}
          {/* Consolidated 3-tile layout. Every tile passes a subtitle
              so the primary numbers vertically align across all three
              (flex-col justify-between needs the same number of children
              in each tile). Previously: 5 tiles with only Win Rate
              carrying a subtitle, causing its primary to shift up while
              the other four sat lower. */}
          <div data-testid="weekly-insights-row" className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5">
            <WeeklyInsightsTile
              label="Weekly Return %"
              value={metrics?.weekly_return_pct ?? null}
              formatType="percent"
              gradient="linear-gradient(135deg, #10b981, #34d399)"
              subtitle={metrics?.weekly_pnl != null
                ? formatCurrency(metrics.weekly_pnl, { showSign: true })
                : undefined}
              loading={metricsLoading && !metrics}
            />
            <WeeklyInsightsTile
              label="YTD %"
              value={metrics?.ytd_pct ?? null}
              formatType="percent"
              gradient="linear-gradient(135deg, #8b5cf6, #a78bfa)"
              subtitle={metrics?.ltd_pct != null
                ? `LTD ${metrics.ltd_pct >= 0 ? "+" : ""}${metrics.ltd_pct.toFixed(2)}%`
                : undefined}
              loading={metricsLoading && !metrics}
            />
            <WeeklyInsightsTile
              label="Win Rate"
              value={metrics ? metrics.win_rate.rate * 100 : null}
              formatType="percent"
              gradient="linear-gradient(135deg, #f97316, #fb923c)"
              subtitle={metrics
                ? (metrics.win_rate.total === 0
                  ? "No closes YTD"
                  : `${metrics.win_rate.wins}W · ${metrics.win_rate.losses}L · ${metrics.win_rate.flat}F of ${metrics.win_rate.total}`)
                : undefined}
              loading={metricsLoading && !metrics}
            />
          </div>

          {/* Per-Ticker Details — collapsible. Migrated to the shared
              <SectionExpander> in the SectionExpander extraction commit;
              previously this was a flat bordered-button + separate body.
              Now it shares the card chrome with Weekly Thoughts (Phase 3)
              and Weekly Snapshot (Phase 4).
              The headerCaption is unconditional (returns the same string
              for both expanded and collapsed) — preserves the Phase 2
              behavior asserted by weekly-retro.test.tsx:300 "Header
              caption stays in sync independent of expand state". The
              prompt suggested hiding it when expanded, but the existing
              test treats always-on as the contract. */}
          <SectionExpander
            title={`Per-Ticker Details (${uniqueTickers})`}
            defaultExpanded={false}
            localStorageKey={TICKETS_EXPANDED_KEY}
            bodyId="per-ticker-body"
            headerCaption={() => `${gradedTickers}/${uniqueTickers} tickers graded`}
          >
              {/* Internal body padding — the card chrome puts the body
                  flush against the header divider; this wrapper restores
                  the 16px breathing room that the pre-refactor
                  marginTop: 12 + the unbordered section gave. */}
              <div style={{ padding: 16 }}>
                {/* Phase 5 — Flight Deck. The original 4-tile activity
                    grid from above the per-ticker section, relocated
                    here. Neutral-surface styling keeps it visually
                    distinct from the gradient performance tiles up top. */}
                <FlightDeck
                  totalTickets={totalTx}
                  uniqueTickers={uniqueTickers}
                  buys={buys.length}
                  sellsTrims={sells.length}
                  isOveractive={isOveractive}
                />
                {/* Progress bar — visual companion to the header caption.
                    Only renders when there are tickers to grade. */}
                {uniqueTickers > 0 && (
                  <div className="mb-4 flex items-center gap-3">
                    <div
                      className="flex-1 h-2 rounded-full overflow-hidden"
                      style={{ background: "var(--bg)" }}
                    >
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${(gradedTickers / uniqueTickers) * 100}%`,
                          background: navColor,
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Ticker cards — original block, untouched apart from the
                    enclosing wrapper. */}
                {grouped.length > 0 ? (
                  <div className="flex flex-col gap-3">
                    {grouped.map(([ticker, txns]) => {
                const isExpanded = expandedTicker === ticker;
                const g = getGrade(ticker);
                const txBuys = txns.filter(t => String(t.action).toUpperCase() === "BUY");
                const txSells = txns.filter(t => String(t.action).toUpperCase() === "SELL");

                return (
                  <div key={ticker} className="rounded-[14px] overflow-hidden"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                    {/* Ticker header */}
                    <button onClick={() => setExpandedTicker(isExpanded ? null : ticker)}
                            className="w-full flex items-center justify-between px-5 py-3 text-left transition-colors hover:brightness-[0.98]">
                      <div className="flex items-center gap-3">
                        <span className="text-[16px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{ticker}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: "color-mix(in oklab, #08a86b 12%, var(--surface))", color: "#16a34a" }}>
                          {txBuys.length}B
                        </span>
                        {txSells.length > 0 && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: "color-mix(in oklab, #e5484d 12%, var(--surface))", color: "#dc2626" }}>
                            {txSells.length}S
                          </span>
                        )}
                        {g.grade && (
                          <span className="text-[10px] px-2 py-0.5 rounded font-bold"
                                style={{ background: `${gradeColor(g.grade)}15`, color: gradeColor(g.grade) }}>
                            {g.grade}
                          </span>
                        )}
                      </div>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2"
                           style={{ transform: isExpanded ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}>
                        <path d="M6 9l6 6 6-6"/>
                      </svg>
                    </button>

                    {isExpanded && (
                      <div style={{ borderTop: "1px solid var(--border)", animation: "slide-up 0.12s ease-out" }}>
                        {/* Transactions (read-only context) */}
                        <div className="px-5 py-3" style={{ background: "var(--bg)" }}>
                          {txns.map((tx, i) => {
                            const isSell = String(tx.action).toUpperCase() === "SELL";
                            const mono = "var(--font-jetbrains), monospace";
                            return (
                              <div key={i} className="flex items-center gap-3 py-1.5 text-[11px]">
                                <span style={{ fontFamily: mono, color: "var(--ink-4)", width: 90 }}>{String(tx.date || "").slice(5, 16)}</span>
                                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold"
                                      style={{ background: isSell ? "color-mix(in oklab, #e5484d 12%, var(--surface))" : "color-mix(in oklab, #08a86b 12%, var(--surface))", color: isSell ? "#dc2626" : "#16a34a" }}>
                                  {tx.action}
                                </span>
                                <span style={{ fontFamily: mono }}>{tx.trx_id || ""}</span>
                                <span style={{ fontFamily: mono }}>{tx.shares} shs</span>
                                <span className="privacy-mask" style={{ fontFamily: mono }}>@ {formatCurrency(parseFloat(String(tx.amount || 0)))}</span>
                                <span style={{ color: "var(--ink-4)" }}>{tx.rule || ""}</span>
                              </div>
                            );
                          })}
                        </div>

                        {/* Ticker-level grading */}
                        <div className="px-5 py-4">
                          <div className="grid grid-cols-3 gap-3">
                            <div>
                              <label className="block text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Grade</label>
                              <select value={g.grade} onChange={e => setGradeField(ticker, "grade", e.target.value)}
                                      className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                                      style={{ ...inputStyle, appearance: "none" as any }}>
                                <option value="">Select...</option>
                                {EXEC_GRADES.map(gr => <option key={gr} value={gr}>{gr}</option>)}
                              </select>
                            </div>
                            <div>
                              <label className="block text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Behavior</label>
                              <select value={g.behavior} onChange={e => setGradeField(ticker, "behavior", e.target.value)}
                                      className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                                      style={{ ...inputStyle, appearance: "none" as any }}>
                                <option value="">Select...</option>
                                {BEHAVIOR_TAGS.map(bt => <option key={bt} value={bt}>{bt}</option>)}
                              </select>
                            </div>
                            <div>
                              <label className="block text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Analysis / Lesson</label>
                              <input type="text" value={g.notes} onChange={e => setGradeField(ticker, "notes", e.target.value)}
                                     placeholder="What did you learn?"
                                     className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                                     style={{ ...inputStyle, fontFamily: "inherit" }} />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
                ) : (
                  <div
                    className="text-center py-8 text-sm"
                    style={{ color: "var(--ink-4)" }}
                  >
                    No trades found for this week.
                  </div>
                )}
              </div>
          </SectionExpander>

          {/* Weekly Thoughts (Phase 3). HTML rich text editor with the
              Phase 0 dirtyRef debounced save pattern. Phase 4.1 added
              inline image paste — retroId + portfolio threaded through
              so the upload endpoint knows which retro to attach to. */}
          <WeeklyThoughts
            value={weekly_thoughts}
            onChange={(next) => { dirtyRef.current = true; setWeeklyThoughts(next); }}
            retroId={retros[monStr]?.id ?? null}
            portfolio={portfolio}
          />

          {/* Weekly Snapshot (Phase 4). Image gallery — drop / paste /
              pick-from-disk images, view in lightbox, two-click delete.
              Third consumer of <SectionExpander>. retroId is null until
              the parent retro has been saved at least once; the component
              shows a disabled drop zone in that state (same idiom as
              TagPicker). */}
          <SectionExpander
            title="Weekly Snapshot"
            showDot
            defaultExpanded={false}
            localStorageKey={SNAPSHOT_EXPANDED_KEY}
            bodyId="weekly-snapshot-body"
            headerCaption={(open) => open ? null : (
              snapshotCount > 0 ? `${snapshotCount} attached` : ""
            )}
          >
            <WeeklySnapshot
              retroId={retros[monStr]?.id ?? null}
              portfolio={portfolio}
              onCountChange={setSnapshotCount}
            />
          </SectionExpander>

          {/* Phase 4.6 — Close the Week. Replaces the inline Weekly
              Summary block + the outer "Save Weekly Retro" button.
              State stays in this parent so the debounced auto-save
              effect (above) still catches every keystroke. */}
          <CloseTheWeek state={ctwState}
                        onChange={handleCtwChange}
                        onSave={() => handleSave().catch(() => { /* saveMsg shown */ })}
                        recentOverall={recentOverall} />

          {saveMsg && (
            <div className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              {saveMsg}
            </div>
          )}
        </>
      </div>
    </div>
  );
}
