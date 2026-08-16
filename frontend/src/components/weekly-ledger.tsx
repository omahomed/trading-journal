"use client";

// Weekly Ledger (migration 069) — per-week transaction review.
//
// One row per BUY + SELL detail that landed Mon–Fri of the selected
// week. Distinct from Weekly Retro (prose/reflection) and Campaign
// Review (aggregates details into a campaign row). The atom here is
// the individual fill / decision, not the campaign.
//
// Data path: GET /api/weekly-ledger?portfolio=X&week_start=YYYY-MM-DD
// Backend returns rows + stats + YTD-avg + page-level free-text note
// in one shot (see api/main.py::get_weekly_ledger).
//
// Interactive surfaces on this page:
//   * Week nav strip — prev/next arrows + Today button
//   * Weekly Notes card — debounced autosave (800ms idle)
//   * Per-row Exit Notes — inline blur-to-save on retro_notes
//   * Per-row Lesson picker — TagPicker with entity_type="trades_details"
//   * Right-click → jump to Trade Journal / Campaign Review
//
// Reusable primitives borrowed from Campaign Review theme: rounded-[14px]
// cards, Fraunces italic-last-word title, JetBrains mono for numeric
// columns, gradient KPI tiles from campaign-detail.

import { useState, useEffect, useMemo, useCallback, useRef, Fragment } from "react";
import { useRouter } from "next/navigation";
import {
  api, getActivePortfolio,
  type WeeklyLedgerResponse, type WeeklyLedgerRow,
} from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { TagPicker } from "./tag-picker";
import { SearchSelect } from "./search-select";

const mono = "var(--font-jetbrains), monospace";

// ── Date helpers (local calendar, no UTC parse) ───────────────────────
function toIso(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function mondayOf(iso: string): string {
  const parts = iso.split("-").map(Number);
  const d = new Date(parts[0], parts[1] - 1, parts[2]);
  const day = d.getDay();                    // Sun=0..Sat=6
  const daysSinceMon = (day + 6) % 7;
  d.setDate(d.getDate() - daysSinceMon);
  return toIso(d);
}

function addDays(iso: string, days: number): string {
  const parts = iso.split("-").map(Number);
  const d = new Date(parts[0], parts[1] - 1, parts[2]);
  d.setDate(d.getDate() + days);
  return toIso(d);
}

function fmtWeekRange(monday: string, friday: string): string {
  const parseIso = (s: string) => {
    const [y, m, d] = s.split("-").map(Number);
    return new Date(y, m - 1, d);
  };
  const mo = parseIso(monday);
  const fr = parseIso(friday);
  const opts: Intl.DateTimeFormatOptions = { month: "short", day: "numeric" };
  const monStr = mo.toLocaleDateString("en-US", opts);
  const friStr = fr.toLocaleDateString("en-US", opts);
  const year = fr.getFullYear();
  return `${monStr} – ${friStr}, ${year}`;
}

// ── Quiet stat card with 5-week sparkline ────────────────────────
// Surface-color card (no gradients), large centered number, subline,
// and a bar-sparkline at the bottom showing the last 4 complete weeks
// + the current week. The current-week bar is drawn in navColor to
// highlight where "now" sits relative to recent history.
//
// The optional `accent` prop tints the primary number for the vs-YTD
// tile — red when the operator is overactive (>15% above avg), green
// when unusually quiet. Every other tile leaves the number ink-neutral.
function QuietStatCard({
  label, value, sub, extraSub, sparkline, navColor, accent, valueTitle,
}: {
  label: string;
  value: string;
  sub: string;
  /** Optional second line under `sub`. Used on the Compliance tile to
   *  break out Buys% · Sells% alongside the overall ratio. Rendered in
   *  the same 11px ink-3 style so it reads as continuation of `sub`. */
  extraSub?: string;
  sparkline: { count: number; is_current: boolean }[];
  navColor: string;
  accent?: "warn" | "good" | null;
  valueTitle?: string;
}) {
  const maxCount = Math.max(1, ...sparkline.map(w => w.count));
  const valueColor = accent === "warn" ? "#e5484d"
                   : accent === "good" ? "#08a86b"
                   : "var(--ink)";

  return (
    <div className="rounded-[14px] px-[16px] pt-[14px] pb-[12px] flex flex-col gap-2 min-h-[124px]"
         style={{
           background: "var(--surface)",
           border: "1px solid var(--border)",
           boxShadow: "var(--card-shadow)",
         }}>
      <div className="text-[10px] font-semibold uppercase tracking-[0.10em]"
           style={{ color: "var(--ink-4)" }}>
        {label}
      </div>
      <div className="flex-1 flex flex-col">
        {/* Value size + weight bumped 2026-08-16: 28px semibold → 36px
            bold. Prior sizing was calibrated for a 4-tile row (~330px
            wide each); with 5 tiles at ~250px wide the primary number
            was reading as ~11% of tile width and felt undersized. */}
        <div className="text-[36px] font-bold tracking-tight leading-none privacy-mask"
             title={valueTitle}
             style={{ fontFamily: mono, color: valueColor }}>
          {value}
        </div>
        <div className="text-[11px] mt-1.5 privacy-mask"
             style={{ color: "var(--ink-3)" }}>
          {sub}
        </div>
        {extraSub && (
          <div className="text-[11px] mt-0.5 privacy-mask"
               style={{ color: "var(--ink-3)" }}>
            {extraSub}
          </div>
        )}
      </div>
      {/* 5-week sparkline. SVG at fixed viewBox — width scales with the
          card. Bars are equal-width; heights normalized to max within the
          strip so a big current week visibly dominates. Current bar drawn
          in navColor at full opacity; prior weeks in ink-4 at 45%. */}
      <div className="flex items-end gap-1 h-[28px] mt-1"
           title={sparkline.map(w => w.count).join(" · ")}>
        {sparkline.map((w, i) => {
          const h = Math.max(2, Math.round((w.count / maxCount) * 28));
          return (
            <div key={i}
                 className="flex-1 rounded-t-[2px]"
                 style={{
                   height: `${h}px`,
                   background: w.is_current ? navColor : "var(--ink-4)",
                   opacity: w.is_current ? 1 : 0.4,
                   minWidth: 6,
                 }}
                 aria-label={`week ${i + 1}: ${w.count}`} />
          );
        })}
      </div>
    </div>
  );
}

// ── Per-row compliance toggle chip. Cycles NULL → Y → N → NULL on
//    click. Optimistic update with revert-on-error. Uses the PATCH
//    endpoint so the change persists per detail row.
function ComplianceChip({ detailId, initial, onChange }: {
  detailId: number;
  // Accept undefined defensively — during mid-deploy the backend may
  // return rows without the compliant field. Normalize to null so
  // the state stays boolean|null everywhere else.
  initial: boolean | null | undefined;
  onChange: (v: boolean | null) => void;
}) {
  const norm = (v: boolean | null | undefined): boolean | null =>
    v === true ? true : v === false ? false : null;
  const [value, setValue] = useState<boolean | null>(norm(initial));
  const [saving, setSaving] = useState(false);
  useEffect(() => { setValue(norm(initial)); }, [initial, detailId]);

  const next = (v: boolean | null): boolean | null =>
    v === null ? true : v === true ? false : null;

  const commit = useCallback(async (target: boolean | null) => {
    const before = value;
    setValue(target);          // optimistic
    setSaving(true);
    try {
      const r = await api.patchTradeDetailCompliant(detailId, target);
      if (r && "error" in r) throw new Error(r.error);
      onChange(target);
    } catch (e) {
      setValue(before);
      log.error("weekly-ledger", "compliant save failed", e);
    } finally {
      setSaving(false);
    }
  }, [detailId, value, onChange]);

  const label = value === true ? "✓" : value === false ? "✗" : "—";
  const bg = value === true
    ? "color-mix(in oklab, #08a86b 18%, var(--surface))"
    : value === false
      ? "color-mix(in oklab, #e5484d 18%, var(--surface))"
      : "var(--bg-2)";
  const fg = value === true ? "#08a86b" : value === false ? "#e5484d" : "var(--ink-4)";

  return (
    <button type="button" onClick={() => commit(next(value))} disabled={saving}
            className="inline-flex items-center justify-center w-[32px] h-[24px] rounded-[6px] text-[13px] font-bold transition-all"
            title={value === true ? "Followed process — click to mark break"
                 : value === false ? "Broke rule — click to reset"
                 : "Ungraded — click to mark followed"}
            style={{ background: bg, color: fg, border: "1px solid var(--border)" }}>
      {label}
    </button>
  );
}

// ── Inline retro_notes editor. Textarea auto-grows; blur triggers save.
function ExitNotesCell({ detailId, initial, action }: {
  detailId: number; initial: string; action: "BUY" | "SELL";
}) {
  const [value, setValue] = useState(initial);
  const [saved, setSaved] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => { setValue(initial); setSaved(initial); }, [initial, detailId]);

  const commit = useCallback(async () => {
    if (value === saved) return;
    setSaving(true);
    setErr("");
    try {
      const r = await api.patchTradeDetailRetroNotes(detailId, value);
      if (r && "error" in r) throw new Error(r.error);
      setSaved(value);
    } catch (e) {
      setErr("save failed");
      log.error("weekly-ledger", "retro_notes save failed", e);
    } finally {
      setSaving(false);
    }
  }, [detailId, value, saved]);

  return (
    <div className="flex flex-col gap-1">
      <textarea
        value={value}
        onChange={e => setValue(e.target.value)}
        onBlur={commit}
        placeholder={action === "SELL" ? "Why did I close this lot?" : "(add note)"}
        rows={1}
        className="w-full text-[11px] rounded-[6px] px-2 py-1 resize-y min-h-[26px]"
        style={{
          background: value === saved ? "var(--bg-2)" : "color-mix(in oklab, #f59f00 8%, var(--surface))",
          border: "1px solid var(--border)",
          color: "var(--ink)",
          fontFamily: "inherit",
        }}
      />
      <div className="text-[9px] flex items-center gap-2" style={{ color: "var(--ink-4)" }}>
        {saving && <span>saving…</span>}
        {err && <span style={{ color: "#e5484d" }}>{err}</span>}
      </div>
    </div>
  );
}

// ── Weekly note card (page-level free text, debounced autosave) ──────
function WeeklyNotesCard({ portfolio, weekStart, initial, navColor }: {
  portfolio: string; weekStart: string; initial: string; navColor: string;
}) {
  const [text, setText] = useState(initial);
  const [saved, setSaved] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  const dirty = useRef(false);

  useEffect(() => { setText(initial); setSaved(initial); dirty.current = false; },
    [initial, portfolio, weekStart]);

  useEffect(() => {
    if (!dirty.current) return;
    const t = setTimeout(async () => {
      setSaving(true);
      setErr("");
      try {
        const r = await api.putWeeklyLedgerNote({
          portfolio, week_start: weekStart, note: text,
        });
        if (r && "error" in r) throw new Error(r.error);
        setSaved(text);
        dirty.current = false;
      } catch (e) {
        setErr("save failed");
        log.error("weekly-ledger", "note save failed", e);
      } finally {
        setSaving(false);
      }
    }, 800);
    return () => clearTimeout(t);
  }, [text, portfolio, weekStart]);

  return (
    <div className="rounded-[14px] overflow-hidden mb-6"
         style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
      <div className="flex items-center justify-between px-[18px] py-3"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Weekly Notes</span>
        </div>
        <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>
          {saving ? "saving…" : (err ? <span style={{ color: "#e5484d" }}>{err}</span> :
            text === saved ? "saved" : "editing")}
        </div>
      </div>
      <textarea
        value={text}
        onChange={e => { dirty.current = true; setText(e.target.value); }}
        placeholder="Anything you noticed this week — themes, mistakes, wins, questions…"
        rows={4}
        className="w-full text-[13px] px-4 py-3 leading-relaxed"
        style={{
          background: "var(--surface)",
          color: "var(--ink)",
          border: "none",
          outline: "none",
          fontFamily: "inherit",
          resize: "vertical",
        }}
      />
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────
export function WeeklyLedger({ navColor, initialWeek }: {
  navColor: string; initialWeek?: string;
}) {
  const router = useRouter();
  const [weekStart, setWeekStart] = useState<string>(() => {
    const seed = initialWeek || toIso(new Date());
    return mondayOf(seed);
  });
  const [data, setData] = useState<WeeklyLedgerResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; row: WeeklyLedgerRow } | null>(null);
  // Filters applied CLIENT-SIDE against the already-fetched week. Backend
  // stats stay week-scoped (they're the "am I overtrading" signal); the
  // table + a small filtered-count badge reflect the current filter.
  const [dayFilter, setDayFilter] = useState<"all" | "mon" | "tue" | "wed" | "thu" | "fri">("all");
  const [tickerFilter, setTickerFilter] = useState<string>("all");
  // Sort state — click any header to sort by that key; click again to
  // reverse. Default: chronological (date asc) — matches the pre-sort
  // implicit order so the initial render is stable.
  type SortKey = "date" | "ticker" | "trx_id" | "action" | "shares"
               | "price" | "amount" | "realized_pl" | "compliant"
               | "buy_rule" | "sell_rule";
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({
    key: "date", dir: "asc",
  });
  const toggleSort = useCallback((key: SortKey) => {
    setSort(prev => prev.key === key
      ? { key, dir: prev.dir === "asc" ? "desc" : "asc" }
      : { key, dir: "asc" });
  }, []);

  const portfolio = getActivePortfolio();

  const loadWeek = useCallback(async (wk: string) => {
    setLoading(true);
    setError(null);
    try {
      const r = await api.weeklyLedger(portfolio, wk);
      if (r && "error" in r) throw new Error(r.error);
      setData(r as WeeklyLedgerResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "load failed");
      log.error("weekly-ledger", "load failed", e);
    } finally {
      setLoading(false);
    }
  }, [portfolio]);

  useEffect(() => { loadWeek(weekStart); }, [loadWeek, weekStart]);

  // Dismiss context menu on any window click or Esc.
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") setCtxMenu(null); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onEsc);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onEsc);
    };
  }, [ctxMenu]);

  const prevWeek = () => setWeekStart(w => mondayOf(addDays(w, -7)));
  const nextWeek = () => setWeekStart(w => mondayOf(addDays(w, 7)));
  const jumpToday = () => setWeekStart(mondayOf(toIso(new Date())));

  const stats = data?.stats;
  const ytd = data?.ytd_avg;
  const allRows = data?.rows ?? [];

  // Reset filters when the week changes so a lingering "TSLA" filter
  // from last week doesn't blank out a fresh week that has no TSLA.
  useEffect(() => {
    setDayFilter("all");
    setTickerFilter("all");
  }, [weekStart]);

  // Ticker options from THIS week's data — sorted alphabetically.
  // Deduped Set → sorted array.
  const tickerOptions = useMemo(
    () => Array.from(new Set(allRows.map(r => r.ticker).filter(Boolean))).sort(),
    [allRows],
  );

  // Client-side filtered rows. Day filter maps M/T/W/Th/F to
  // getDay() 1..5 (parsed via YYYY-MM-DD split — no UTC parse).
  const dayIndex: Record<string, number> = {
    mon: 1, tue: 2, wed: 3, thu: 4, fri: 5,
  };
  const rows = useMemo(() => {
    return allRows.filter(r => {
      if (tickerFilter !== "all" && r.ticker !== tickerFilter) return false;
      if (dayFilter !== "all") {
        if (!r.date) return false;
        const [y, m, d] = r.date.split("-").map(Number);
        const jsDay = new Date(y, m - 1, d).getDay();  // Sun=0..Sat=6
        const target = dayIndex[dayFilter];
        if (jsDay !== target) return false;
      }
      return true;
    });
  }, [allRows, dayFilter, tickerFilter]);

  const filterActive = dayFilter !== "all" || tickerFilter !== "all";

  // Sort AFTER filter. Nulls sort LAST regardless of asc/desc so unset
  // fields (empty Realized on BUY rows, ungraded compliant chips, missing
  // Buy/Sell Rule) always land at the bottom of the sorted view — a
  // sorted table shouldn't hide the interesting rows behind a wall of
  // "—". Compliant is normalized to a number for ordering: true = 2,
  // false = 1, null = null → nulls last.
  const sortedRows = useMemo(() => {
    const arr = [...rows];
    const { key, dir } = sort;
    const mul = dir === "asc" ? 1 : -1;
    const complianceRank = (v: boolean | null): number | null =>
      v === true ? 2 : v === false ? 1 : null;
    arr.sort((a, b) => {
      let av: unknown = a[key];
      let bv: unknown = b[key];
      if (key === "compliant") {
        av = complianceRank(a.compliant);
        bv = complianceRank(b.compliant);
      }
      if (av == null && bv == null) return 0;
      if (av == null) return 1;   // nulls last regardless of direction
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * mul;
      }
      return String(av).localeCompare(String(bv)) * mul;
    });
    return arr;
  }, [rows, sort]);

  // ── YTD delta chip subline for the Transactions KPI ─────────────
  const ytdSub = useMemo(() => {
    if (!stats || !ytd || ytd.avg_transactions == null) {
      return "YTD avg: —";
    }
    const avg = ytd.avg_transactions;
    if (ytd.current_vs_avg_pct == null) return `YTD avg: ${avg.toFixed(1)}`;
    const arrow = ytd.current_vs_avg_pct > 0 ? "↑" : ytd.current_vs_avg_pct < 0 ? "↓" : "→";
    const sign = ytd.current_vs_avg_pct > 0 ? "+" : "";
    return `${sign}${ytd.current_vs_avg_pct.toFixed(0)}% ${arrow} vs YTD avg ${avg.toFixed(1)}`;
  }, [stats, ytd]);

  // vs-YTD accent: red at >15% above YTD avg (overactive), green at
  // >15% below (unusually quiet), null otherwise. Applied ONLY to the
  // vs-YTD tile's number so the rest of the strip stays quiet.
  const ytdAccent: "warn" | "good" | null =
    ytd?.current_vs_avg_pct != null && ytd.current_vs_avg_pct > 15 ? "warn"
    : ytd?.current_vs_avg_pct != null && ytd.current_vs_avg_pct < -15 ? "good"
    : null;
  const ytdTileValue = ytd?.current_vs_avg_pct != null
    ? `${ytd.current_vs_avg_pct > 0 ? "+" : ""}${ytd.current_vs_avg_pct.toFixed(0)}%`
    : "—";
  const sparklineData = data?.recent_weeks ?? [];
  // Compliance sparkline uses a parallel weekly-compliance-% series.
  // Each entry carries compliance_pct (may be null when the week had
  // no graded rows). Map to the same shape the sparkline reads.
  const complianceSparkline = useMemo(
    () => (data?.recent_compliance ?? []).map(w => ({
      count: w.compliance_pct ?? 0,
      is_current: w.is_current,
    })),
    [data],
  );
  const complianceValue = stats?.compliance_pct != null
    ? `${stats.compliance_pct.toFixed(0)}%`
    : "—";
  const complianceSub = stats
    ? `${stats.compliant_count} of ${stats.graded_count} graded${stats.graded_count < stats.total_transactions ? ` · ${stats.total_transactions - stats.graded_count} pending` : ""}`
    : "no data";
  // Buy/Sell breakdown line — entries and exits are separate skills.
  // Each side is dashed out ("—") if no rows are graded on that side
  // yet, so a mid-week check doesn't spuriously read 0%.
  const complianceExtraSub = stats ? (() => {
    const b = stats.buy_graded_count > 0
      ? (stats.buy_compliance_pct != null
          ? `${stats.buy_compliance_pct.toFixed(0)}%` : "—")
      : "—";
    const s = stats.sell_graded_count > 0
      ? (stats.sell_compliance_pct != null
          ? `${stats.sell_compliance_pct.toFixed(0)}%` : "—")
      : "—";
    return `Buys ${b} · Sells ${s}`;
  })() : undefined;
  // Compliance color rule — tightened 2026-08-16:
  //   Green ≥ 80%  (followed process at least 4 of 5 — genuinely good)
  //   Red   < 50%  (broke process more than half — clear alarm)
  //   50–80% neutral (in between, no color) — accent means something.
  const complianceAccent: "warn" | "good" | null =
    stats?.compliance_pct == null ? null
    : stats.compliance_pct < 50 ? "warn"
    : stats.compliance_pct >= 80 ? "good"
    : null;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Weekly <em className="italic" style={{ color: navColor }}>Ledger</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Every buy + sell decision this week — one row per transaction. Not the Retro.
          </div>
        </div>
        <button onClick={() => loadWeek(weekStart)}
                data-testid="wledger-refresh"
                className="px-3 py-2 rounded-[10px] text-[13px]"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
          ⟳ Refresh
        </button>
      </div>

      {error && (
        <div className="px-4 py-3 rounded-[10px] mb-4 text-[13px]"
             style={{
               background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
               border: "1px solid var(--border)",
               color: "#e5484d",
             }}>
          {error}
        </div>
      )}

      {/* Week nav strip */}
      <div className="mb-5 flex items-center gap-3 p-2 rounded-[12px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <button onClick={prevWeek} data-testid="wledger-prev"
                className="px-3 py-1.5 rounded-[8px] text-[13px] font-medium"
                style={{ background: "var(--bg-2)", color: "var(--ink-2)" }}>
          ← Prev
        </button>
        <div className="flex-1 text-center">
          <div className="text-[13px] font-semibold" style={{ fontFamily: mono }}>
            {data ? `Week of ${fmtWeekRange(data.week_start, data.week_end)}` :
             `Week of ${fmtWeekRange(weekStart, addDays(weekStart, 4))}`}
          </div>
          <div className="text-[10px]" style={{ color: "var(--ink-4)" }}>
            {portfolio}
          </div>
        </div>
        <button onClick={jumpToday} data-testid="wledger-today"
                className="px-3 py-1.5 rounded-[8px] text-[13px] font-medium"
                style={{ background: "var(--bg-2)", color: "var(--ink-2)" }}>
          Today
        </button>
        <button onClick={nextWeek} data-testid="wledger-next"
                className="px-3 py-1.5 rounded-[8px] text-[13px] font-medium"
                style={{ background: "var(--bg-2)", color: "var(--ink-2)" }}>
          Next →
        </button>
      </div>

      {/* KPI strip — quiet surface cards, each carries a 5-week
          sparkline. No gradients; only two tiles ever accent their
          number: VS YTD AVG when the operator is >15% off avg, and
          COMPLIANCE when adherence drops below 80% (or rises above
          95%). Reads calm at rest, alerts only when it matters. */}
      <div className="grid grid-cols-5 gap-[14px] mb-6">
        <QuietStatCard label="TRANSACTIONS"
                       value={stats ? String(stats.total_transactions) : "—"}
                       sub={`${stats?.buys ?? 0} buys · ${stats?.sells ?? 0} sells`}
                       sparkline={sparklineData}
                       navColor={navColor} />
        <QuietStatCard label="COMPLIANCE"
                       value={complianceValue}
                       sub={complianceSub}
                       extraSub={complianceExtraSub}
                       sparkline={complianceSparkline}
                       navColor={navColor}
                       accent={complianceAccent} />
        <QuietStatCard label="AVG / DAY"
                       value={stats ? stats.avg_per_day.toFixed(1) : "—"}
                       sub="Mon–Fri denominator"
                       sparkline={sparklineData}
                       navColor={navColor} />
        <QuietStatCard label="UNIQUE TICKERS"
                       value={stats ? String(stats.unique_tickers) : "—"}
                       sub="distinct symbols touched"
                       sparkline={sparklineData}
                       navColor={navColor} />
        <QuietStatCard label="VS YTD AVG"
                       value={ytdTileValue}
                       sub={`YTD avg ${ytd?.avg_transactions != null ? ytd.avg_transactions.toFixed(1) : "—"} · ${ytd?.weeks_counted ?? 0} prior wks`}
                       sparkline={sparklineData}
                       navColor={navColor}
                       accent={ytdAccent}
                       valueTitle={ytdSub} />
      </div>

      {/* Weekly Notes card */}
      {data && (
        <WeeklyNotesCard portfolio={portfolio} weekStart={data.week_start}
                         initial={data.note} navColor={navColor} />
      )}

      {/* Transaction Ledger table */}
      <div className="rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        {/* Filter strip. Client-side filters against the fetched week
            — no re-fetch on toggle. Day chips + Ticker dropdown; both
            reset to "all" on week change. */}
        <div className="flex items-center gap-3 flex-wrap px-[18px] py-3"
             style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-2)" }}>
          <span className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                style={{ color: "var(--ink-4)" }}>Day</span>
          <div className="flex p-0.5 rounded-[8px] gap-0.5"
               style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            {(["all", "mon", "tue", "wed", "thu", "fri"] as const).map(k => (
              <button key={k}
                      onClick={() => setDayFilter(k)}
                      data-testid={`wledger-day-${k}`}
                      className="px-2.5 py-1 rounded text-[11px] font-medium transition-all"
                      style={{
                        background: dayFilter === k ? "var(--bg-2)" : "transparent",
                        color: dayFilter === k ? "var(--ink)" : "var(--ink-4)",
                        fontWeight: dayFilter === k ? 600 : 500,
                      }}>
                {k === "all" ? "All" : k === "thu" ? "Th" : k[0].toUpperCase()}
              </button>
            ))}
          </div>
          <span className="text-[10px] font-semibold uppercase tracking-[0.08em] ml-2"
                style={{ color: "var(--ink-4)" }}>Ticker</span>
          {/* Uses the shared SearchSelect component that Log Buy / Log Sell /
              Position Sizer use — type to filter, pick from the list. Empty
              value means "all tickers"; the "All tickers" option leads the
              list so it's always one keystroke away. */}
          <div style={{ minWidth: 200 }} data-testid="wledger-ticker-filter">
            <SearchSelect value={tickerFilter === "all" ? "" : tickerFilter}
                          onChange={v => setTickerFilter(v || "all")}
                          options={["", ...tickerOptions].map(t => ({
                            value: t, label: t === "" ? "All tickers" : t,
                          }))}
                          placeholder="All tickers" />
          </div>
          {filterActive && (
            <button onClick={() => { setDayFilter("all"); setTickerFilter("all"); }}
                    className="text-[11px] px-2 py-1 rounded-[6px] font-medium"
                    style={{ color: "var(--ink-3)", background: "var(--surface)", border: "1px solid var(--border)" }}>
              × Reset
            </button>
          )}
        </div>
        <div className="flex items-center gap-2 px-[18px] py-3"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Transactions</span>
          <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>
            {loading ? "loading…" : filterActive
              ? `${rows.length} of ${allRows.length} rows`
              : `${rows.length} rows`}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {([
                  { l: "Date", a: "left", k: "date" },
                  { l: "Ticker", a: "left", k: "ticker" },
                  { l: "Trx", a: "left", k: "trx_id" },
                  { l: "Side", a: "left", k: "action" },
                  { l: "Shares", a: "right", k: "shares" },
                  { l: "Price", a: "right", k: "price" },
                  { l: "Amount", a: "right", k: "amount" },
                  { l: "Realized", a: "right", k: "realized_pl" },
                  { l: "OK?", a: "center", k: "compliant" },
                  { l: "Buy Rule", a: "left", k: "buy_rule" },
                  { l: "Sell Rule", a: "left", k: "sell_rule" },
                  { l: "Lesson", a: "left", k: null },       // TagPicker — no sort
                  { l: "Exit Notes", a: "left", k: null },   // free text — no sort
                ] as { l: string; a: "left" | "right" | "center"; k: SortKey | null }[]).map(c => {
                  const active = c.k !== null && sort.key === c.k;
                  return (
                    <th key={c.l}
                        onClick={c.k ? () => toggleSort(c.k as SortKey) : undefined}
                        className={"px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.04em] select-none"
                          + (c.k ? " cursor-pointer" : "")}
                        style={{
                          color: active ? "var(--ink-2)" : "var(--ink-4)",
                          borderBottom: "1px solid var(--border)",
                          textAlign: c.a,
                          whiteSpace: "nowrap",
                        }}>
                      {c.l}{active ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={13} className="px-3 py-8 text-center text-[12px]"
                      style={{ color: "var(--ink-4)" }}>
                    Loading…
                  </td>
                </tr>
              ) : rows.length === 0 ? (
                <tr>
                  <td colSpan={13} className="px-3 py-8 text-center text-[12px]"
                      style={{ color: "var(--ink-4)" }}>
                    {filterActive
                      ? "No transactions match the current filter."
                      : "No transactions this week. Enjoy the quiet."}
                  </td>
                </tr>
              ) : sortedRows.map(r => (
                <Fragment key={r.detail_id}>
                  <tr onContextMenu={e => {
                        e.preventDefault();
                        setCtxMenu({ x: e.clientX, y: e.clientY, row: r });
                      }}
                      style={{ borderBottom: "1px solid var(--border)" }}
                      onMouseEnter={e => e.currentTarget.style.background = "var(--bg-2)"}
                      onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                    <td className="px-3 py-2" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                      {r.date}
                    </td>
                    <td className="px-3 py-2 font-semibold" style={{ fontFamily: mono }}>
                      {r.ticker}
                    </td>
                    <td className="px-3 py-2" style={{ fontFamily: mono, color: "var(--ink-4)" }}>
                      {r.trx_id || "—"}
                    </td>
                    <td className="px-3 py-2">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                            style={{
                              background: r.action === "BUY"
                                ? "color-mix(in oklab, #10b981 14%, var(--surface))"
                                : "color-mix(in oklab, #dc2626 14%, var(--surface))",
                              color: r.action === "BUY" ? "#08a86b" : "#dc2626",
                            }}>
                        {r.action}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
                      {r.shares.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right" style={{ fontFamily: mono }}>
                      {r.price != null ? formatCurrency(r.price, { decimals: 2 }) : "—"}
                    </td>
                    {/* Amount is CASH FLOW, not P&L. Neutral color so
                        a SELL at a loss doesn't read green just because
                        cash came in. Sign carries the direction: BUY
                        negative (cash out), SELL positive (cash in). */}
                    <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
                      {r.amount != null ? formatCurrency(r.amount, { decimals: 0, showSign: true }) : "—"}
                    </td>
                    <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: r.realized_pl == null ? "var(--ink-4)" : r.realized_pl >= 0 ? "#08a86b" : "#e5484d" }}>
                      {r.realized_pl != null ? formatCurrency(r.realized_pl, { decimals: 0, showSign: true }) : "—"}
                    </td>
                    <td className="px-3 py-2 text-center">
                      <ComplianceChip detailId={r.detail_id} initial={r.compliant}
                                      onChange={v => setData(prev => prev ? {
                                        ...prev,
                                        rows: prev.rows.map(row =>
                                          row.detail_id === r.detail_id
                                            ? { ...row, compliant: v } : row),
                                      } : prev)} />
                    </td>
                    <td className="px-3 py-2" style={{ color: "var(--ink-3)" }}>{r.buy_rule || "—"}</td>
                    <td className="px-3 py-2" style={{ color: "var(--ink-3)" }}>{r.sell_rule || "—"}</td>
                    <td className="px-3 py-2" style={{ minWidth: 200 }}>
                      <TagPicker entityType="trades_details" entityId={r.detail_id} portfolio={portfolio} />
                    </td>
                    <td className="px-3 py-2" style={{ minWidth: 200 }}>
                      <ExitNotesCell detailId={r.detail_id} initial={r.retro_notes} action={r.action} />
                    </td>
                  </tr>
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right-click menu */}
      {ctxMenu && (
        <div className="fixed z-50 rounded-[10px] py-1.5 min-w-[220px] overflow-hidden"
             style={{
               left: ctxMenu.x, top: ctxMenu.y,
               background: "var(--surface)",
               border: "1px solid var(--border)",
               boxShadow: "0 8px 24px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.08)",
             }}
             data-testid="wledger-ctx-menu"
             onClick={e => e.stopPropagation()}>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-[0.08em] font-semibold"
               style={{ color: "var(--ink-4)" }}>
            {ctxMenu.row.ticker} · {ctxMenu.row.trade_id}
          </div>
          <button type="button"
                  className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2"
                  style={{ color: "var(--ink)" }}
                  onMouseEnter={e => e.currentTarget.style.background = "var(--bg-2)"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                  onClick={() => {
                    router.push(`/trade-journal?trade_id=${encodeURIComponent(ctxMenu.row.trade_id)}`);
                    setCtxMenu(null);
                  }}>
            <span style={{ color: "var(--ink-4)" }}>📋</span> View in Trade Journal
          </button>
          <button type="button"
                  className="w-full text-left px-3 py-2 text-[12px] font-medium flex items-center gap-2"
                  style={{ color: "var(--ink)" }}
                  onMouseEnter={e => e.currentTarget.style.background = "var(--bg-2)"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                  onClick={() => {
                    router.push(`/campaign-review?trade_id=${encodeURIComponent(ctxMenu.row.trade_id)}`);
                    setCtxMenu(null);
                  }}>
            <span style={{ color: "var(--ink-4)" }}>🔍</span> Open in Campaign Review
          </button>
        </div>
      )}
    </div>
  );
}
