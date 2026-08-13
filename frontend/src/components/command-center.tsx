"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type CommandCenterRow } from "@/lib/api";
import { classifyDeck, DECK_META, type DeckLevel } from "@/lib/deck-levels";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

/**
 * Cross-portfolio risk at a glance. One row per portfolio the caller
 * owns; sorted worst-drawdown-first so whoever needs attention floats
 * to the top. Deck classification comes from lib/deck-levels — same
 * source of truth Risk Manager reads, so a threshold change updates
 * both pages together.
 *
 * No aggregation. The user manages portfolios independently — the
 * value is the SIDE-BY-SIDE view, not a combined number.
 */
export function CommandCenter({ navColor }: { navColor: string }) {
  const [rows, setRows] = useState<CommandCenterRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);

  const load = (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    api.commandCenter()
      .then((res) => {
        if ("error" in res) {
          setError(res.error);
          setRows([]);
        } else {
          setError(null);
          setRows(res.rows);
          setLastUpdatedAt(new Date());
        }
      })
      .catch((err) => {
        log.error("command-center", "load failed", err);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  };

  useEffect(() => { load(false); }, []);

  // Sort worst-drawdown-first. NULL / no-data portfolios fall to the
  // bottom so the actionable rows stay above the fold.
  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const da = a.drawdown_current_pct;
      const db = b.drawdown_current_pct;
      if (da == null && db == null) return a.portfolio_name.localeCompare(b.portfolio_name);
      if (da == null) return 1;
      if (db == null) return -1;
      // Both signed negatives when in drawdown → smaller (more negative) = worse.
      return da - db;
    });
  }, [rows]);

  const lastUpdatedLabel = lastUpdatedAt
    ? `${lastUpdatedAt.toISOString().slice(0, 10)} ${String(lastUpdatedAt.getHours()).padStart(2, "0")}:${String(lastUpdatedAt.getMinutes()).padStart(2, "0")}`
    : "";

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }} data-testid="command-center-root">
      {/* Page header — Fraunces + italicized last word matches every
          other page under (app)/*. See CLAUDE.md standing rule. */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[22px] md:text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Command <em className="italic" style={{ color: navColor }}>Center</em>
          </h1>
          <div className="text-[12px] md:text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Risk across every portfolio at a glance · sorted worst-drawdown-first
            {lastUpdatedLabel ? ` · as of ${lastUpdatedLabel}` : ""}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button type="button" onClick={() => load(true)} disabled={refreshing}
                  data-testid="command-center-refresh"
                  className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: refreshing ? "var(--ink-4)" : "var(--ink-2)" }}>
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             data-testid="command-center-error"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {error}
        </div>
      )}

      {loading && !lastUpdatedAt ? (
        <div className="rounded-[14px] animate-pulse min-h-[240px]" style={{ background: "var(--bg-2)" }} />
      ) : sorted.length === 0 ? (
        <div className="rounded-[14px] p-10 text-center text-[13px]"
             data-testid="command-center-empty"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
          No portfolios yet. Create one from Settings to see it here.
        </div>
      ) : (
        <div className="rounded-[14px] overflow-hidden"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]" style={{ borderCollapse: "collapse" }}
                   data-testid="command-center-table">
              <thead>
                <tr style={{ background: "var(--surface-2)" }}>
                  {[
                    { l: "Portfolio",  align: "left"  },
                    { l: "NLV",        align: "right" },
                    { l: "Day P&L",    align: "right" },
                    { l: "LTD",        align: "right" },
                    { l: "YTD",        align: "right" },
                    { l: "Exposure",   align: "right" },
                    { l: "Drawdown",   align: "right" },
                    { l: "Deck",       align: "center" },
                  ].map(c => (
                    <th key={c.l}
                        className="px-4 py-3 text-[10px] font-semibold uppercase tracking-[0.04em]"
                        style={{
                          color: "var(--ink-4)",
                          borderBottom: "1px solid var(--border)",
                          textAlign: c.align as "left" | "right" | "center",
                        }}>
                      {c.l}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => <PortfolioRow key={r.portfolio_id} row={r} />)}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function PortfolioRow({ row }: { row: CommandCenterRow }) {
  const deck: DeckLevel = classifyDeck(row.drawdown_current_pct);
  const meta = DECK_META[deck];

  // Day P&L color — green up, red down, neutral if missing.
  const dayColor = row.nlv_delta_dollar == null
    ? "var(--ink-4)"
    : row.nlv_delta_dollar >= 0 ? "#08a86b" : "#e5484d";

  const ltdColor = row.ltd_pct == null
    ? "var(--ink-4)"
    : row.ltd_pct >= 0 ? "#08a86b" : "#e5484d";

  const ytdColor = !row.ytd_available || row.ytd_pct == null
    ? "var(--ink-4)"
    : row.ytd_pct >= 0 ? "#08a86b" : "#e5484d";

  // Drawdown color: same threshold logic as the deck (severity ramps).
  // Uses deck's meta color so the row's severity reads consistently at
  // a glance — the drawdown cell and the deck chip share the ramp.
  const ddColor = deck === "L0" ? "var(--ink-2)" : meta.color;

  const nlvMono: React.CSSProperties = { fontFamily: "var(--font-jetbrains), monospace" };

  return (
    <tr data-testid="cc-row" data-portfolio={row.portfolio_name} data-deck={deck}
        className="transition-colors hover:brightness-[0.98]"
        style={{ borderBottom: "1px solid var(--border)" }}>
      {/* Portfolio name — non-monospace, ink-2 */}
      <td className="px-4 py-3" style={{ color: "var(--ink-2)", fontWeight: 500 }}>
        {row.portfolio_name}
        {!row.journal_available && (
          <span className="ml-2 text-[10px] px-1.5 py-[1px] rounded" style={{
            background: "var(--bg-2)", color: "var(--ink-4)",
          }}>
            no journal
          </span>
        )}
      </td>

      {/* NLV */}
      <td className="px-4 py-3 text-right privacy-mask" style={nlvMono}>
        {row.nlv == null ? "—" : formatCurrency(row.nlv, { decimals: 0 })}
      </td>

      {/* Day P&L: dollar delta on top, percent below */}
      <td className="px-4 py-3 text-right" style={{ ...nlvMono, color: dayColor }}>
        <div className="privacy-mask">
          {row.nlv_delta_dollar == null
            ? "—"
            : formatCurrency(row.nlv_delta_dollar, { decimals: 0, showSign: true })}
        </div>
        {row.nlv_delta_pct != null && (
          <div className="text-[11px] opacity-80">
            {row.nlv_delta_pct >= 0 ? "+" : ""}{row.nlv_delta_pct.toFixed(2)}%
          </div>
        )}
      </td>

      {/* LTD */}
      <td className="px-4 py-3 text-right" style={{ ...nlvMono, color: ltdColor }}>
        <div>
          {row.ltd_pct == null
            ? "—"
            : `${row.ltd_pct >= 0 ? "+" : ""}${row.ltd_pct.toFixed(2)}%`}
        </div>
        {row.ltd_pl_dollar != null && (
          <div className="text-[11px] opacity-80 privacy-mask">
            {formatCurrency(row.ltd_pl_dollar, { decimals: 0, showSign: true })}
          </div>
        )}
      </td>

      {/* YTD */}
      <td className="px-4 py-3 text-right" style={{ ...nlvMono, color: ytdColor }}>
        <div>
          {!row.ytd_available || row.ytd_pct == null
            ? "—"
            : `${row.ytd_pct >= 0 ? "+" : ""}${row.ytd_pct.toFixed(2)}%`}
        </div>
        {row.ytd_available && row.ytd_pl_dollar != null && (
          <div className="text-[11px] opacity-80 privacy-mask">
            {formatCurrency(row.ytd_pl_dollar, { decimals: 0, showSign: true })}
          </div>
        )}
      </td>

      {/* Exposure — % main, N positions below */}
      <td className="px-4 py-3 text-right" style={{ ...nlvMono, color: "var(--ink-2)" }}>
        <div>
          {row.exposure_pct == null ? "—" : `${row.exposure_pct.toFixed(1)}%`}
        </div>
        <div className="text-[11px] opacity-70" style={{ color: "var(--ink-3)" }}>
          {row.open_position_count} pos
        </div>
      </td>

      {/* Drawdown — signed % main, peak sub */}
      <td className="px-4 py-3 text-right" style={{ ...nlvMono, color: ddColor }}>
        <div style={{ fontWeight: deck === "L0" ? 400 : 600 }}>
          {row.drawdown_current_pct == null
            ? "—"
            : `${row.drawdown_current_pct.toFixed(2)}%`}
        </div>
        {row.drawdown_peak_nlv != null && (
          <div className="text-[11px] opacity-70 privacy-mask" style={{ color: "var(--ink-3)" }}>
            peak {formatCurrency(row.drawdown_peak_nlv, { decimals: 0 })}
          </div>
        )}
      </td>

      {/* Deck badge */}
      <td className="px-4 py-3 text-center">
        <span data-testid="cc-deck-chip"
              className="inline-flex flex-col items-center gap-0.5 px-2.5 py-1 rounded-[10px]"
              style={{
                background: `color-mix(in oklab, ${meta.color} 14%, var(--surface))`,
                border: `1px solid color-mix(in oklab, ${meta.color} 30%, var(--border))`,
                color: meta.color,
                minWidth: 78,
              }}>
          <span className="text-[12px] font-semibold leading-none">{meta.label}</span>
          <span className="text-[9.5px] leading-none opacity-90">{meta.sub}</span>
        </span>
      </td>
    </tr>
  );
}
