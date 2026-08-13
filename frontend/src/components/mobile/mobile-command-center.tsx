"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type CommandCenterRow } from "@/lib/api";
import { classifyDeck, DECK_META, type DeckLevel } from "@/lib/deck-levels";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

/**
 * Mobile-native Command Center. Vertical portfolio cards, tap to expand
 * for full metrics. Same source of truth (`/api/command-center`) as the
 * desktop table; the desktop 8-column layout doesn't survive a 390px
 * viewport, so this surface renders each portfolio as its own card.
 *
 * Default view per card:
 *   Portfolio name · Deck badge · Drawdown headline · NLV
 *
 * Tap to expand:
 *   Day P&L · LTD · YTD · Exposure · Peak
 *
 * Sort matches desktop: worst drawdown first, null drawdown last.
 */
export function MobileCommandCenter() {
  const [rows, setRows] = useState<CommandCenterRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

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
        log.error("mobile-command-center", "load failed", err);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  };

  useEffect(() => { load(false); }, []);

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const da = a.drawdown_current_pct;
      const db = b.drawdown_current_pct;
      if (da == null && db == null) return a.portfolio_name.localeCompare(b.portfolio_name);
      if (da == null) return 1;
      if (db == null) return -1;
      return da - db;
    });
  }, [rows]);

  const toggle = (id: number) =>
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const lastUpdatedLabel = lastUpdatedAt
    ? `as of ${lastUpdatedAt.toISOString().slice(11, 16)}`
    : "";

  return (
    <div className="pb-4 flex flex-col gap-3" data-testid="mobile-command-center-root">
      {/* Subtitle strip + Refresh. Page title is provided by MobileShell's
          MobilePageHeader ("Command Center"). */}
      <div className="flex items-center justify-between gap-3 mb-1">
        <div className="text-[12px] leading-snug text-m-text-dim">
          Worst-drawdown-first{lastUpdatedLabel ? ` · ${lastUpdatedLabel}` : ""}
        </div>
        <button type="button" onClick={() => load(true)} disabled={refreshing}
                data-testid="mobile-cc-refresh"
                className="shrink-0 rounded-m-sm px-3 py-2 text-[12px] font-medium"
                style={{
                  background: "var(--m-surface-2)",
                  color: refreshing ? "var(--m-text-faint)" : "var(--m-text-muted)",
                  minHeight: 44,
                }}>
          ⟳ {refreshing ? "…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="px-4 py-3 rounded-m-sm text-[12px]"
             data-testid="mobile-cc-error"
             style={{
               background: "color-mix(in oklab, var(--m-down) 12%, var(--m-surface))",
               border: "1px solid var(--m-warn-border-soft)",
               color: "var(--m-down)",
             }}>
          Failed to load: {error}
        </div>
      )}

      {loading && !lastUpdatedAt ? (
        <>
          {[0, 1, 2, 3].map(i => (
            <div key={i} className="rounded-m-md animate-pulse min-h-[92px]"
                 style={{ background: "var(--m-surface)" }} />
          ))}
        </>
      ) : sorted.length === 0 ? (
        <div className="rounded-m-md p-8 text-center text-[13px]"
             data-testid="mobile-cc-empty"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
               color: "var(--m-text-muted)",
             }}>
          No portfolios yet. Create one from Settings.
        </div>
      ) : (
        sorted.map((row) => (
          <PortfolioCard
            key={row.portfolio_id}
            row={row}
            expanded={expandedIds.has(row.portfolio_id)}
            onToggle={() => toggle(row.portfolio_id)}
          />
        ))
      )}
    </div>
  );
}

function PortfolioCard({
  row,
  expanded,
  onToggle,
}: {
  row: CommandCenterRow;
  expanded: boolean;
  onToggle: () => void;
}) {
  const deck: DeckLevel = classifyDeck(row.drawdown_current_pct);
  const meta = DECK_META[deck];

  const dayColor = row.nlv_delta_dollar == null
    ? "var(--m-text-faint)"
    : row.nlv_delta_dollar >= 0 ? "var(--m-accent)" : "var(--m-down)";

  const ltdColor = row.ltd_pct == null
    ? "var(--m-text-faint)"
    : row.ltd_pct >= 0 ? "var(--m-accent)" : "var(--m-down)";

  const ytdColor = !row.ytd_available || row.ytd_pct == null
    ? "var(--m-text-faint)"
    : row.ytd_pct >= 0 ? "var(--m-accent)" : "var(--m-down)";

  return (
    <button
      type="button"
      onClick={onToggle}
      data-testid="mobile-cc-card"
      data-portfolio={row.portfolio_name}
      data-deck={deck}
      data-expanded={expanded ? "true" : "false"}
      aria-expanded={expanded}
      className="text-left w-full rounded-m-md p-4 transition-colors"
      style={{
        background: "var(--m-surface)",
        border: "0.5px solid var(--m-border)",
        minHeight: 44,
      }}
    >
      {/* Header row — always visible. Portfolio name + deck chip. */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-[15px] font-semibold text-m-text truncate">
            {row.portfolio_name}
          </div>
          <div className="mt-0.5 text-[11px] text-m-text-dim privacy-mask"
               style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
            NLV {row.nlv == null ? "—" : formatCurrency(row.nlv, { decimals: 0 })}
          </div>
        </div>
        <span
          className="inline-flex flex-col items-center px-2.5 py-1 rounded-m-sm shrink-0"
          style={{
            background: `color-mix(in oklab, ${meta.color} 14%, var(--m-surface))`,
            border: `1px solid color-mix(in oklab, ${meta.color} 28%, var(--m-border))`,
            color: meta.color,
            minWidth: 70,
          }}
        >
          <span className="text-[12px] font-semibold leading-none">{meta.label}</span>
          <span className="text-[9.5px] leading-none opacity-90 mt-0.5">{meta.sub}</span>
        </span>
      </div>

      {/* Drawdown headline — the value that drives the sort. */}
      <div className="mt-3 flex items-baseline justify-between gap-3">
        <div className="text-[11px] uppercase tracking-[0.08em] font-semibold text-m-text-dim">
          Drawdown
        </div>
        <div className="text-right">
          <div className="text-[18px] font-semibold leading-none"
               style={{
                 color: deck === "L0" ? "var(--m-text)" : meta.color,
                 fontFamily: "var(--font-jetbrains), monospace",
               }}>
            {row.drawdown_current_pct == null
              ? "—"
              : `${row.drawdown_current_pct.toFixed(2)}%`}
          </div>
          {row.drawdown_peak_nlv != null && (
            <div className="mt-1 text-[10px] text-m-text-faint privacy-mask"
                 style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
              peak {formatCurrency(row.drawdown_peak_nlv, { decimals: 0 })}
            </div>
          )}
        </div>
      </div>

      {/* Expand affordance — tiny chevron rotated when open. */}
      <div className="mt-3 flex items-center justify-center text-m-text-faint text-[10px]"
           aria-hidden>
        <span style={{
          display: "inline-block",
          transform: expanded ? "rotate(180deg)" : "none",
          transition: "transform 150ms",
        }}>▾</span>
        <span className="ml-1.5 uppercase tracking-[0.08em] font-semibold">
          {expanded ? "less" : "more"}
        </span>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t-[0.5px] border-m-border grid grid-cols-2 gap-3">
          <MetricCell label="Day P&L" color={dayColor}
            main={row.nlv_delta_dollar == null
              ? "—"
              : formatCurrency(row.nlv_delta_dollar, { decimals: 0, showSign: true })}
            sub={row.nlv_delta_pct == null
              ? undefined
              : `${row.nlv_delta_pct >= 0 ? "+" : ""}${row.nlv_delta_pct.toFixed(2)}%`} />
          <MetricCell label="Exposure" color="var(--m-text)"
            main={row.exposure_pct == null ? "—" : `${row.exposure_pct.toFixed(1)}%`}
            sub={`${row.open_position_count} pos`} />
          <MetricCell label="LTD" color={ltdColor}
            main={row.ltd_pct == null ? "—" : `${row.ltd_pct >= 0 ? "+" : ""}${row.ltd_pct.toFixed(2)}%`}
            sub={row.ltd_pl_dollar == null ? undefined : formatCurrency(row.ltd_pl_dollar, { decimals: 0, showSign: true })} />
          <MetricCell label="YTD" color={ytdColor}
            main={!row.ytd_available || row.ytd_pct == null
              ? "—"
              : `${row.ytd_pct >= 0 ? "+" : ""}${row.ytd_pct.toFixed(2)}%`}
            sub={row.ytd_available && row.ytd_pl_dollar != null
              ? formatCurrency(row.ytd_pl_dollar, { decimals: 0, showSign: true })
              : undefined} />
        </div>
      )}
    </button>
  );
}

function MetricCell({ label, main, sub, color }: {
  label: string;
  main: string;
  sub?: string;
  color: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
        {label}
      </div>
      <div className="mt-0.5 text-[14px] font-semibold privacy-mask"
           style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
        {main}
      </div>
      {sub && (
        <div className="text-[10px] text-m-text-faint privacy-mask mt-0.5"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {sub}
        </div>
      )}
    </div>
  );
}
