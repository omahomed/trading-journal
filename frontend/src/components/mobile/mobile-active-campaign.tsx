"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { api, getActivePortfolio, type TradePosition, type TradeDetail, type TradeDetailsBundle, type Strategy } from "@/lib/api";
import { computeEnrichedPositions, type EnrichedPosition } from "@/lib/positions";
import { classifySellRuleTier, SELL_RULE_TIER_ORDER, type SellRuleTier } from "@/lib/sell-rule";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { StrategyChip } from "../strategy-chip";

/**
 * Mobile-native Active Campaign Summary. Read-first surface: shows
 * what you're holding at a glance so you know where to look on desktop
 * for action.
 *
 * Header — count + % invested + total P&L.
 * Sort — pos size (default), P&L, ticker, sell-rule tier.
 * Cards — ticker + SR chip + total P&L + pos size (default view);
 *         tap to expand for current price / avg entry / stop / days held.
 *
 * Deliberately no context menu, broker-stop editor, SR8 trim calc, or
 * declare modal — those workflows are edit-heavy and belong on desktop.
 * MobileDesktopOnlyBanner is NOT rendered here because the page IS
 * useful on mobile for the glance case; the desktop chip appears on
 * the More entry only if we later decide to.
 */
export function MobileActiveCampaign({ navColor }: { navColor: string }) {
  const { activePortfolio } = usePortfolio();
  const portfolio = activePortfolio?.name ?? getActivePortfolio();

  const [positions, setPositions] = useState<EnrichedPosition[]>([]);
  const [equity, setEquity] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("pos_size");
  const [expandedTradeIds, setExpandedTradeIds] = useState<Set<string>>(new Set());
  const [strategies, setStrategies] = useState<Strategy[]>([]);

  // O(1) name → color/full-strategy lookup for the per-row strategy chip.
  // Mirrors ACS's strategyByName pattern; keeps color consistent across
  // desktop + mobile surfaces.
  const strategyByName = useMemo(() => {
    const m = new Map<string, Strategy>();
    for (const s of strategies) m.set(s.name, s);
    return m;
  }, [strategies]);

  useEffect(() => {
    api.listStrategies({ active: true, portfolio })
      .then(setStrategies)
      .catch(() => setStrategies([]));
  }, [portfolio]);

  const load = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      const [openTrades, detailsBundle, latest] = await Promise.all([
        api.tradesOpen(portfolio) as Promise<TradePosition[]>,
        api.tradesOpenDetails(portfolio) as Promise<TradeDetailsBundle>,
        api.journalLatest(portfolio).catch(() => null),
      ]);

      const trades = Array.isArray(openTrades) ? openTrades : [];
      const eq = latest && "end_nlv" in (latest as object) && (latest as { end_nlv?: number }).end_nlv
        ? Number((latest as { end_nlv: number }).end_nlv)
        : 0;
      setEquity(eq);

      const tickers = trades.map(t => t.ticker).filter(Boolean);
      let prices: Record<string, number> = {};
      if (tickers.length > 0) {
        try {
          const result = await api.batchPrices(tickers, portfolio);
          if (result && !("error" in result)) prices = result;
        } catch { /* fall back to entry price */ }
      }
      const details: TradeDetail[] = (detailsBundle && "details" in detailsBundle)
        ? detailsBundle.details
        : [];
      setPositions(computeEnrichedPositions(trades, details, eq, prices));
      setError(null);
    } catch (e) {
      log.error("mobile-active-campaign", "load failed", e);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => { void load(false); /* eslint-disable-line react-hooks/exhaustive-deps */ }, [portfolio]);

  const sorted = useMemo(() => sortPositions(positions, sortKey), [positions, sortKey]);

  const headline = useMemo(() => {
    const totalPl = positions.reduce((s, p) => s + (p.overall_pl ?? 0), 0);
    const totalCost = positions.reduce((s, p) => s + (p.total_cost ?? 0), 0);
    const totalMv = positions.reduce((s, p) => s + (p.current_value ?? 0), 0);
    const invested = equity > 0 ? (totalMv / equity) * 100 : 0;
    return { count: positions.length, totalPl, totalCost, totalMv, invested };
  }, [positions, equity]);

  const toggle = (tradeId: string) =>
    setExpandedTradeIds((prev) => {
      const next = new Set(prev);
      if (next.has(tradeId)) next.delete(tradeId);
      else next.add(tradeId);
      return next;
    });

  return (
    <div className="pb-4 flex flex-col gap-3" data-testid="mobile-acs-root">
      {/* Header strip — count + % invested + total P&L. */}
      <div className="rounded-m-md p-4"
           style={{
             background: "var(--m-surface)",
             border: "0.5px solid var(--m-border)",
           }}>
        <div className="grid grid-cols-3 gap-3">
          <HeaderStat label="Open" main={String(headline.count)}
                      sub={headline.count === 1 ? "position" : "positions"} />
          <HeaderStat label="Invested"
                      main={equity > 0 ? `${headline.invested.toFixed(1)}%` : "—"}
                      sub={equity > 0 ? formatCurrency(headline.totalMv, { decimals: 0 }) : "—"} />
          <HeaderStat label="Total P&L"
                      main={formatCurrency(headline.totalPl, { decimals: 0, showSign: true })}
                      sub={equity > 0 && headline.totalCost > 0
                        ? `${((headline.totalPl / headline.totalCost) * 100).toFixed(1)}%`
                        : "—"}
                      color={headline.totalPl >= 0 ? "var(--m-accent)" : "var(--m-down)"} />
        </div>
      </div>

      {/* Sort + Refresh row */}
      <div className="flex items-center justify-between gap-3">
        <label className="flex items-center gap-2 text-[11px] text-m-text-dim">
          <span className="uppercase tracking-[0.06em] font-semibold">Sort</span>
          <select value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                  className="rounded-m-sm px-2 py-1 text-[12px]"
                  style={{
                    background: "var(--m-surface-2)",
                    color: "var(--m-text-muted)",
                    border: "0.5px solid var(--m-border)",
                    minHeight: 32,
                  }}>
            <option value="pos_size">% NAV</option>
            <option value="pl">P&amp;L $</option>
            <option value="return">Return %</option>
            <option value="ticker">Ticker</option>
            <option value="tier">Sell rule</option>
          </select>
        </label>
        <button type="button" onClick={() => void load(true)} disabled={refreshing}
                data-testid="mobile-acs-refresh"
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
             data-testid="mobile-acs-error"
             style={{
               background: "color-mix(in oklab, var(--m-down) 12%, var(--m-surface))",
               border: "1px solid var(--m-warn-border-soft)",
               color: "var(--m-down)",
             }}>
          Failed to load: {error}
        </div>
      )}

      {loading ? (
        <>
          {[0, 1, 2].map(i => (
            <div key={i} className="rounded-m-md animate-pulse min-h-[100px]"
                 style={{ background: "var(--m-surface)" }} />
          ))}
        </>
      ) : sorted.length === 0 ? (
        <div className="rounded-m-md p-8 text-center text-[13px]"
             data-testid="mobile-acs-empty"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
               color: "var(--m-text-muted)",
             }}>
          No open campaigns. Log a buy on desktop to see it here.
        </div>
      ) : (
        sorted.map((p) => (
          <PositionCard
            key={p.trade_id}
            position={p}
            expanded={expandedTradeIds.has(p.trade_id)}
            onToggle={() => toggle(p.trade_id)}
            strategyByName={strategyByName}
          />
        ))
      )}

      {/* Footer nudge — action-heavy workflows live on desktop. */}
      <div className="mt-2 px-4 py-3 rounded-m-sm text-[11px] leading-snug"
           style={{
             background: "color-mix(in oklab, var(--m-warn) 8%, var(--m-surface))",
             color: "var(--m-text-muted)",
           }}>
        Editing stops, declaring SR8, and running trim calcs live on the
        desktop ACS. Mobile is read-first.
      </div>

      {/* Silence unused-import lint until the mobile version gets its own
          navigator; kept in scope so future tap-into-detail links can wire
          up without another import round-trip. */}
      <span className="hidden" data-nav-color={navColor} />
      <Link href="/active-campaign" className="hidden" prefetch={false}>_</Link>
    </div>
  );
}

// ── Sort ────────────────────────────────────────────────────────────

type SortKey = "pos_size" | "pl" | "return" | "ticker" | "tier";

function sortPositions(positions: EnrichedPosition[], key: SortKey): EnrichedPosition[] {
  const arr = [...positions];
  switch (key) {
    case "pos_size":
      return arr.sort((a, b) => (b.pos_size_pct ?? 0) - (a.pos_size_pct ?? 0));
    case "pl":
      return arr.sort((a, b) => (b.overall_pl ?? 0) - (a.overall_pl ?? 0));
    case "return":
      return arr.sort((a, b) => (b.return_pct ?? 0) - (a.return_pct ?? 0));
    case "ticker":
      return arr.sort((a, b) => a.ticker.localeCompare(b.ticker));
    case "tier": {
      // Ladder progression from SELL_RULE_TIER_ORDER: SR1 (0, no floor) →
      // SR11 (1) → SR15 (2) → SR7 (3) → SR8 (4, monster hold). Sorts
      // most-defensive last; null (no tier resolved) sorts to the very
      // end so unclassified rows don't clutter the top.
      const tierIdx = (t: SellRuleTier | null) =>
        t == null ? 999 : SELL_RULE_TIER_ORDER[t];
      return arr.sort((a, b) => tierIdx(getTier(a)) - tierIdx(getTier(b)));
    }
    default:
      return arr;
  }
}

function getTier(p: EnrichedPosition): SellRuleTier | null {
  if (p.sell_rule_tier) return p.sell_rule_tier;
  return classifySellRuleTier(p.b1_return_pct, p.peak_total_pl ?? null, false);
}

// ── UI bits ────────────────────────────────────────────────────────

function HeaderStat({ label, main, sub, color }: {
  label: string;
  main: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
        {label}
      </div>
      <div className="mt-0.5 text-[16px] font-semibold privacy-mask"
           style={{ color: color ?? "var(--m-text)", fontFamily: "var(--font-jetbrains), monospace" }}>
        {main}
      </div>
      {sub && (
        <div className="mt-0.5 text-[10px] text-m-text-faint privacy-mask"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function PositionCard({
  position: p,
  expanded,
  onToggle,
  strategyByName,
}: {
  position: EnrichedPosition;
  expanded: boolean;
  onToggle: () => void;
  strategyByName: Map<string, Strategy>;
}) {
  const tier = getTier(p);
  const tierMeta = tierChipMeta(tier);
  const plColor = p.overall_pl >= 0 ? "var(--m-accent)" : "var(--m-down)";
  const returnColor = p.return_pct >= 0 ? "var(--m-accent)" : "var(--m-down)";

  return (
    <button
      type="button"
      onClick={onToggle}
      data-testid="mobile-acs-card"
      data-ticker={p.ticker}
      data-tier={tier ?? "none"}
      data-expanded={expanded ? "true" : "false"}
      aria-expanded={expanded}
      className="text-left w-full rounded-m-md p-4 transition-colors"
      style={{
        background: "var(--m-surface)",
        border: "0.5px solid var(--m-border)",
        minHeight: 44,
      }}
    >
      {/* Header: ticker + strategy chip + tier chip */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[16px] font-semibold text-m-text"
                  style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
              {p.ticker}
            </span>
            {p.strategy && (
              <StrategyChip
                name={p.strategy}
                color={strategyByName.get(p.strategy)?.color ?? "var(--m-text-faint)"}
                size="sm"
                showName={true}
                variant="filled"
              />
            )}
          </div>
          <div className="mt-0.5 text-[11px] text-m-text-dim">
            {p.shares} sh · {p.rule || "no rule"}
          </div>
        </div>
        {tierMeta && (
          <span className="inline-flex items-center px-2 py-1 rounded-m-sm text-[11px] font-semibold shrink-0"
                style={{
                  background: `color-mix(in oklab, ${tierMeta.color} 14%, var(--m-surface))`,
                  border: `1px solid color-mix(in oklab, ${tierMeta.color} 28%, var(--m-border))`,
                  color: tierMeta.color,
                }}>
            {tierMeta.label}
          </span>
        )}
      </div>

      {/* Primary metrics — P&L, pos size, return */}
      <div className="mt-3 grid grid-cols-3 gap-3">
        <MetricCell label="P&L"
          color={plColor}
          main={formatCurrency(p.overall_pl, { decimals: 0, showSign: true })}
          sub={p.total_cost > 0
            ? `${p.overall_pl >= 0 ? "+" : ""}${((p.overall_pl / p.total_cost) * 100).toFixed(1)}%`
            : undefined} />
        <MetricCell label="% NAV" color="var(--m-text)"
          main={p.pos_size_pct != null ? `${p.pos_size_pct.toFixed(1)}%` : "—"}
          sub={formatCurrency(p.current_value, { decimals: 0 })} />
        <MetricCell label="Return"
          color={returnColor}
          main={p.return_pct != null
            ? `${p.return_pct >= 0 ? "+" : ""}${p.return_pct.toFixed(1)}%`
            : "—"}
          sub={p.b1_return_pct != null
            ? `B1 ${p.b1_return_pct >= 0 ? "+" : ""}${p.b1_return_pct.toFixed(1)}%`
            : undefined} />
      </div>

      {/* Expand affordance */}
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
          <MetricCell label="Current" color="var(--m-text)"
            main={formatCurrency(p.current_price, { decimals: 2 })} />
          <MetricCell label="Avg entry" color="var(--m-text-muted)"
            main={formatCurrency(p.avg_entry, { decimals: 2 })} />
          <MetricCell label="Stop" color="var(--m-text-muted)"
            main={p.avg_stop > 0 ? formatCurrency(p.avg_stop, { decimals: 2 }) : "—"} />
          <MetricCell label="Open risk"
            color={p.open_risk > 0 ? "var(--m-down)" : "var(--m-accent)"}
            main={formatCurrency(p.open_risk, { decimals: 0 })}
            sub={p.open_risk_pct != null ? `${p.open_risk_pct.toFixed(2)}% NAV` : undefined} />
          <MetricCell label="Days held" color="var(--m-text-muted)"
            main={String(p.days_held ?? 0)} />
          <MetricCell label="Strategy" color="var(--m-text-muted)"
            main={p.strategy ?? "—"} />
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
      <div className="text-[9.5px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
        {label}
      </div>
      <div className="mt-0.5 text-[13px] font-semibold privacy-mask"
           style={{ color, fontFamily: "var(--font-jetbrains), monospace" }}>
        {main}
      </div>
      {sub && (
        <div className="mt-0.5 text-[10px] text-m-text-faint privacy-mask"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

// ── Tier chip meta ─────────────────────────────────────────────────

function tierChipMeta(tier: SellRuleTier | null): { label: string; color: string } | null {
  if (!tier) return null;
  switch (tier) {
    case "sr1":  return { label: "SR1",  color: "#e5484d" };
    case "sr7":  return { label: "SR7",  color: "#f59f00" };
    case "sr8":  return { label: "SR8",  color: "#8b5cf6" };
    case "sr11": return { label: "SR11", color: "#0891b2" };
    case "sr15": return { label: "SR15", color: "#08a86b" };
    default:     return null;
  }
}
