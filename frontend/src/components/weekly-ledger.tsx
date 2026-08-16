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

// ── KPI tile (matches Campaign Review / Risk Manager) ────────────────
function KPITile({ label, value, sub, gradient, extraSub }: {
  label: string; value: string; sub: string; gradient: string;
  extraSub?: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-[14px] p-[14px_16px] text-white flex flex-col justify-between h-[90px] transition-transform duration-150 hover:scale-[1.01]"
         style={{ background: gradient, boxShadow: "var(--kpi-shadow)" }}>
      <div className="absolute -right-5 -top-5 w-[100px] h-[100px] rounded-full"
           style={{ background: "radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%)" }} />
      <div className="relative z-10">
        <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{label}</div>
        <div className="text-[22px] font-semibold tracking-tight mt-0.5 privacy-mask"
             style={{ fontFamily: mono }}>
          {value}
        </div>
      </div>
      <div className="relative z-10 text-[10px] font-medium opacity-80 privacy-mask">
        {sub}
        {extraSub && <div className="opacity-90">{extraSub}</div>}
      </div>
    </div>
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
  const rows = data?.rows ?? [];

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

  const netRealizedGradient = stats
    ? (stats.net_realized >= 0
        ? "linear-gradient(135deg, #10b981, #34d399)"
        : "linear-gradient(135deg, #dc2626, #ef4444)")
    : "linear-gradient(135deg, #64748b, #94a3b8)";

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

      {/* KPI tiles */}
      <div className="grid grid-cols-5 gap-[14px] mb-6">
        <KPITile label="TRANSACTIONS"
                 value={stats ? String(stats.total_transactions) : "—"}
                 sub={`${stats?.buys ?? 0} buys · ${stats?.sells ?? 0} sells`}
                 extraSub={ytdSub}
                 gradient="linear-gradient(135deg, #6366f1, #818cf8)" />
        <KPITile label="AVG / DAY"
                 value={stats ? stats.avg_per_day.toFixed(1) : "—"}
                 sub="denominator = 5 Mon–Fri"
                 gradient="linear-gradient(135deg, #8b5cf6, #a78bfa)" />
        <KPITile label="UNIQUE TICKERS"
                 value={stats ? String(stats.unique_tickers) : "—"}
                 sub="distinct symbols touched"
                 gradient="linear-gradient(135deg, #f59f00, #fbbf24)" />
        <KPITile label="NET REALIZED"
                 value={stats ? formatCurrency(stats.net_realized, { decimals: 0, showSign: true }) : "—"}
                 sub="sells only, dollars"
                 gradient={netRealizedGradient} />
        <KPITile label="ACTIVITY"
                 value={ytd?.current_vs_avg_pct != null
                   ? `${ytd.current_vs_avg_pct > 0 ? "+" : ""}${ytd.current_vs_avg_pct.toFixed(0)}%`
                   : "—"}
                 sub={`vs YTD avg ${ytd?.avg_transactions != null ? ytd.avg_transactions.toFixed(1) : "—"}`}
                 extraSub={ytd?.weeks_counted ? `${ytd.weeks_counted} prior weeks` : undefined}
                 gradient={ytd?.current_vs_avg_pct != null && ytd.current_vs_avg_pct > 15
                   ? "linear-gradient(135deg, #dc2626, #ef4444)"
                   : "linear-gradient(135deg, #64748b, #94a3b8)"} />
      </div>

      {/* Weekly Notes card */}
      {data && (
        <WeeklyNotesCard portfolio={portfolio} weekStart={data.week_start}
                         initial={data.note} navColor={navColor} />
      )}

      {/* Transaction Ledger table */}
      <div className="rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Transactions</span>
          <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>
            {loading ? "loading…" : `${rows.length} rows`}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {[
                  { l: "Date", a: "left" },
                  { l: "Ticker", a: "left" },
                  { l: "Trx", a: "left" },
                  { l: "Side", a: "left" },
                  { l: "Shares", a: "right" },
                  { l: "Price", a: "right" },
                  { l: "Amount", a: "right" },
                  { l: "Buy Rule", a: "left" },
                  { l: "Sell Rule", a: "left" },
                  { l: "Realized", a: "right" },
                  { l: "Lesson", a: "left" },
                  { l: "Exit Notes", a: "left" },
                ].map(c => (
                  <th key={c.l}
                      className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.04em] select-none"
                      style={{
                        color: "var(--ink-4)",
                        borderBottom: "1px solid var(--border)",
                        textAlign: c.a as "left" | "right",
                        whiteSpace: "nowrap",
                      }}>
                    {c.l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={12} className="px-3 py-8 text-center text-[12px]"
                      style={{ color: "var(--ink-4)" }}>
                    Loading…
                  </td>
                </tr>
              ) : rows.length === 0 ? (
                <tr>
                  <td colSpan={12} className="px-3 py-8 text-center text-[12px]"
                      style={{ color: "var(--ink-4)" }}>
                    No transactions this week. Enjoy the quiet.
                  </td>
                </tr>
              ) : rows.map(r => (
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
                    <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: r.amount != null && r.amount >= 0 ? "#08a86b" : "var(--ink-3)" }}>
                      {r.amount != null ? formatCurrency(r.amount, { decimals: 0, showSign: true }) : "—"}
                    </td>
                    <td className="px-3 py-2" style={{ color: "var(--ink-3)" }}>{r.buy_rule || "—"}</td>
                    <td className="px-3 py-2" style={{ color: "var(--ink-3)" }}>{r.sell_rule || "—"}</td>
                    <td className="px-3 py-2 text-right" style={{ fontFamily: mono, color: r.realized_pl == null ? "var(--ink-4)" : r.realized_pl >= 0 ? "#08a86b" : "#e5484d" }}>
                      {r.realized_pl != null ? formatCurrency(r.realized_pl, { decimals: 0, showSign: true }) : "—"}
                    </td>
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
