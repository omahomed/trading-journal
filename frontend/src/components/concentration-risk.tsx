"use client";

import { useEffect, useMemo, useState, useCallback, Fragment } from "react";
import Link from "next/link";
import { api, type ConcentrationResponse } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { KPITile, TILE_GRADIENTS } from "./campaign-detail";

interface Props { navColor: string; }

// Distinct palette per sector — reused for theme by hashing the name so
// the same sector's ETFs (etc.) hold color across renders. Low-saturation
// so the page stays calm when 6+ buckets stack in the bar chart.
const PALETTE = [
  "#6366f1", "#e5484d", "#08a86b", "#d97706", "#8b5cf6",
  "#0891b2", "#db2777", "#65a30d", "#ea580c", "#0284c7",
  "#a16207", "#7c3aed", "#059669", "#c2410c", "#0d9488",
];

function hashColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0;
  return PALETTE[Math.abs(h) % PALETTE.length];
}

function StackedBar({ buckets }: { buckets: { name: string; weight_pct: number; color: string }[] }) {
  if (buckets.length === 0) return null;
  return (
    <div className="w-full h-6 rounded-md overflow-hidden flex"
         style={{ background: "var(--surface-2)" }}>
      {buckets.map((b) => (
        <div
          key={b.name}
          title={`${b.name}: ${b.weight_pct.toFixed(1)}%`}
          style={{ width: `${b.weight_pct}%`, background: b.color }}
          className="h-full first:rounded-l-md last:rounded-r-md"
        />
      ))}
    </div>
  );
}

function BucketTable({
  buckets, kind, totalMV, unclassifiedMV, onFix,
}: {
  buckets: { name: string; market_value: number; weight_pct: number; positions: string[] }[];
  kind: "sector" | "theme";
  totalMV: number;
  unclassifiedMV: number;
  onFix: () => void;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const unclassifiedPct = totalMV > 0 ? 100 * unclassifiedMV / totalMV : 0;

  const toggle = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  };

  if (buckets.length === 0 && unclassifiedMV === 0) {
    return (
      <div className="text-[13px] p-4 rounded-[14px] text-center"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
        No open positions in this scope.
      </div>
    );
  }

  return (
    <div>
      <StackedBar buckets={[
        ...buckets.map((b) => ({ name: b.name, weight_pct: b.weight_pct, color: hashColor(b.name) })),
        ...(unclassifiedMV > 0 ? [{ name: "⚠ Unclassified", weight_pct: unclassifiedPct, color: "#e5484d" }] : []),
      ]} />
      <div className="mt-3 rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <table className="w-full text-[13px]">
          <thead>
            <tr style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
              <th className="px-3 py-2.5 text-left font-medium" style={{ width: 12 }}></th>
              <th className="px-3 py-2.5 text-left font-medium">{kind === "sector" ? "Sector" : "Theme"}</th>
              <th className="px-3 py-2.5 text-right font-medium">Weight</th>
              <th className="px-3 py-2.5 text-right font-medium">Market Value</th>
              <th className="px-3 py-2.5 text-right font-medium">Positions</th>
            </tr>
          </thead>
          <tbody>
            {buckets.map((b) => (
              <Fragment key={b.name}>
                <tr style={{ borderTop: "1px solid var(--border)" }}
                    className="cursor-pointer hover:bg-[var(--surface-2)]"
                    onClick={() => toggle(b.name)}>
                  <td className="px-3 py-2.5">
                    <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: hashColor(b.name) }} />
                  </td>
                  <td className="px-3 py-2.5 font-medium" style={{ color: "var(--ink-1)" }}>{b.name}</td>
                  <td className="px-3 py-2.5 text-right font-semibold"
                      style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-1)" }}>
                    {b.weight_pct.toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5 text-right"
                      style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
                    {formatCurrency(b.market_value)}
                  </td>
                  <td className="px-3 py-2.5 text-right" style={{ color: "var(--ink-3)" }}>
                    {b.positions.length}{" "}
                    <span style={{ color: "var(--ink-4)" }}>{expanded.has(b.name) ? "▾" : "▸"}</span>
                  </td>
                </tr>
                {expanded.has(b.name) && (
                  <tr style={{ borderTop: "1px solid var(--border)" }}>
                    <td></td>
                    <td colSpan={4} className="px-3 py-2.5 text-[12px]"
                        style={{ background: "var(--surface-2)", color: "var(--ink-3)" }}>
                      {b.positions.join(" · ")}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {unclassifiedMV > 0 && (
              <tr style={{
                borderTop: "1px solid var(--border)",
                background: "color-mix(in oklab, #e5484d 6%, var(--surface))",
              }}>
                <td className="px-3 py-2.5">
                  <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: "#e5484d" }} />
                </td>
                <td className="px-3 py-2.5 font-medium" style={{ color: "#e5484d" }}>⚠ Unclassified</td>
                <td className="px-3 py-2.5 text-right font-semibold"
                    style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#e5484d" }}>
                  {unclassifiedPct.toFixed(1)}%
                </td>
                <td className="px-3 py-2.5 text-right"
                    style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
                  {formatCurrency(unclassifiedMV)}
                </td>
                <td className="px-3 py-2.5 text-right">
                  <button onClick={onFix} className="text-[12px] font-medium" style={{ color: "#e5484d" }}>
                    Fix →
                  </button>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ConcentrationRisk({ navColor }: Props) {
  const { activePortfolio, portfolios } = usePortfolio();
  const [scope, setScope] = useState<string>("active"); // "active" | "all" | portfolio name
  const [data, setData] = useState<ConcentrationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const scopePortfolioName = useMemo(() => {
    if (scope === "all") return "";
    if (scope === "active") return activePortfolio?.name ?? "";
    return scope;
  }, [scope, activePortfolio]);

  const refresh = useCallback(async (opts?: { manual?: boolean }) => {
    if (opts?.manual) setRefreshing(true); else setLoading(true);
    setErr(null);
    try {
      const res = await api.concentration(scopePortfolioName);
      setData(res);
    } catch (e) {
      log.error("concentration-risk", "load failed", e);
      setErr(String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [scopePortfolioName]);

  useEffect(() => { refresh(); }, [refresh]);

  const unclassifiedMV = data?.unclassified.reduce((s, u) => s + u.market_value, 0) ?? 0;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Page header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Concentration <em className="italic" style={{ color: navColor }}>Risk</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Sector + theme rollup of open positions by market value ·{" "}
            <Link href="/sector-mapping" style={{ color: navColor }}>Manage mappings</Link>
          </div>
        </div>
        <div className="flex gap-2 shrink-0 items-center">
          <label className="text-[12px]" style={{ color: "var(--ink-4)" }}>Scope</label>
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value)}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}
          >
            <option value="active">Active ({activePortfolio?.name ?? "—"})</option>
            <option value="all">All portfolios</option>
            {portfolios.map((p) => (
              <option key={p.id} value={p.name}>{p.name}</option>
            ))}
          </select>
          <button
            type="button" onClick={() => refresh({ manual: true })} disabled={refreshing}
            className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors"
            style={{ background: "var(--surface)", border: "1px solid var(--border)",
                     color: refreshing ? "var(--ink-4)" : "var(--ink-2)" }}
          >
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {err && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
                      border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {err}
        </div>
      )}

      {/* KPI strip */}
      {loading && !data ? (
        <div className="grid grid-cols-4 gap-[14px]">
          {[0, 1, 2, 3].map(i => (
            <div key={i} className="rounded-[14px] animate-pulse min-h-[108px]"
                 style={{ background: "var(--bg-2)" }} />
          ))}
        </div>
      ) : data ? (
        <>
          <div className="grid grid-cols-4 gap-[14px]">
            <KPITile
              label="Market Value"
              value={formatCurrency(data.total_market_value, { decimals: 0 })}
              sub={`${data.positions.length} open position${data.positions.length === 1 ? "" : "s"}`}
              gradient={TILE_GRADIENTS.indigo}
            />
            <KPITile
              label="Sectors"
              value={String(data.sectors.length)}
              sub={data.sectors.length === 0 ? "no mapped positions" : `${data.themes.length} theme${data.themes.length === 1 ? "" : "s"}`}
              gradient={TILE_GRADIENTS.blue}
            />
            <KPITile
              label="Top Sector"
              value={data.sectors[0] ? `${data.sectors[0].weight_pct.toFixed(0)}%` : "—"}
              sub={data.sectors[0]?.name ?? "no mapped positions"}
              gradient={
                data.sectors[0] && data.sectors[0].weight_pct >= 50
                  ? TILE_GRADIENTS.red
                  : data.sectors[0] && data.sectors[0].weight_pct >= 30
                    ? TILE_GRADIENTS.orange
                    : TILE_GRADIENTS.green
              }
            />
            <KPITile
              label="Unclassified"
              value={String(data.unclassified.length)}
              sub={data.unclassified.length === 0
                ? "everything mapped"
                : `${(unclassifiedMV / (data.total_market_value || 1) * 100).toFixed(0)}% of MV missing sector`}
              gradient={data.unclassified.length > 0 ? TILE_GRADIENTS.red : TILE_GRADIENTS.green}
            />
          </div>

          {data.positions.length === 0 ? (
            <div className="mt-6 text-[13px] p-8 rounded-[14px] text-center"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
              No open positions in this scope.
            </div>
          ) : (
            <>
              <section className="mt-6">
                <div className="mb-3 text-[13px] font-semibold" style={{ color: "var(--ink-2)" }}>
                  By Sector
                </div>
                <BucketTable
                  buckets={data.sectors}
                  kind="sector"
                  totalMV={data.total_market_value}
                  unclassifiedMV={unclassifiedMV}
                  onFix={() => window.location.assign("/sector-mapping")}
                />
              </section>

              <section className="mt-6">
                <div className="mb-3 text-[13px] font-semibold" style={{ color: "var(--ink-2)" }}>
                  By Theme
                </div>
                <BucketTable
                  buckets={data.themes}
                  kind="theme"
                  totalMV={data.total_market_value}
                  unclassifiedMV={unclassifiedMV}
                  onFix={() => window.location.assign("/sector-mapping")}
                />
              </section>

              {data.unclassified.length > 0 && (
                <section className="mt-6 p-4 rounded-[14px]"
                         style={{ background: "color-mix(in oklab, #e5484d 6%, var(--surface))",
                                  border: "1px solid var(--border)" }}>
                  <div className="text-[13px] font-semibold mb-2" style={{ color: "#e5484d" }}>
                    ⚠ {data.unclassified.length} unclassified position{data.unclassified.length === 1 ? "" : "s"}
                  </div>
                  <div className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
                    Not counted in the sector or theme rollup — percentages ignore these.
                    Fix on Sector Mapping to get a true picture.
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {data.unclassified.map((u) => (
                      <Link
                        key={u.ticker} href="/sector-mapping"
                        className="px-2.5 py-1 rounded-[8px] text-[12px]"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)",
                                 color: "var(--ink-2)", textDecoration: "none" }}
                      >
                        {u.ticker}{" "}
                        <span style={{ color: "var(--ink-4)" }}>({u.weight_pct.toFixed(1)}%)</span>
                      </Link>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </>
      ) : null}
    </div>
  );
}
