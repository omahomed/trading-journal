"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Link from "next/link";
import { api, type ConcentrationResponse } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

interface Props { navColor: string; }

// Distinct palette per sector — reused for theme by hashing the name so
// the same sector's ETFs (etc.) hold color across renders. Colors chosen
// for legibility on both light and dark; low-saturation to keep the page
// calm even when 6+ buckets stack in the bar chart.
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

function StackedBar({ buckets }: { buckets: { name: string; weight_pct: number }[] }) {
  if (buckets.length === 0) return null;
  return (
    <div className="w-full h-6 rounded-md overflow-hidden flex" style={{ background: "var(--surface-2)" }}>
      {buckets.map((b) => (
        <div
          key={b.name}
          title={`${b.name}: ${b.weight_pct.toFixed(1)}%`}
          style={{ width: `${b.weight_pct}%`, background: hashColor(b.name) }}
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
      <div className="text-[13px] p-4 rounded-lg text-center" style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
        No open positions in this scope.
      </div>
    );
  }

  return (
    <div>
      <StackedBar buckets={[
        ...buckets.map((b) => ({ name: b.name, weight_pct: b.weight_pct })),
        ...(unclassifiedMV > 0 ? [{ name: "⚠ Unclassified", weight_pct: unclassifiedPct }] : []),
      ]} />
      <div className="mt-3 overflow-x-auto rounded-lg" style={{ border: "1px solid var(--border)" }}>
        <table className="w-full text-[13px]">
          <thead>
            <tr style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
              <th className="px-3 py-2 text-left font-medium" style={{ width: 12 }}></th>
              <th className="px-3 py-2 text-left font-medium">{kind === "sector" ? "Sector" : "Theme"}</th>
              <th className="px-3 py-2 text-right font-medium">Weight</th>
              <th className="px-3 py-2 text-right font-medium">Market Value</th>
              <th className="px-3 py-2 text-right font-medium">Positions</th>
            </tr>
          </thead>
          <tbody>
            {buckets.map((b) => (
              <>
                <tr key={b.name} style={{ borderTop: "1px solid var(--border)" }} className="cursor-pointer hover:bg-[var(--surface-2)]"
                    onClick={() => toggle(b.name)}>
                  <td className="px-3 py-2"><span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: hashColor(b.name) }} /></td>
                  <td className="px-3 py-2 font-medium" style={{ color: "var(--ink-1)" }}>{b.name}</td>
                  <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-1)" }}>
                    {b.weight_pct.toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
                    {formatCurrency(b.market_value)}
                  </td>
                  <td className="px-3 py-2 text-right" style={{ color: "var(--ink-3)" }}>
                    {b.positions.length}{" "}
                    <span style={{ color: "var(--ink-4)" }}>{expanded.has(b.name) ? "▾" : "▸"}</span>
                  </td>
                </tr>
                {expanded.has(b.name) && (
                  <tr style={{ borderTop: "1px solid var(--border)" }}>
                    <td></td>
                    <td colSpan={4} className="px-3 py-2 text-[12px]" style={{ background: "var(--surface-2)", color: "var(--ink-3)" }}>
                      {b.positions.join(" · ")}
                    </td>
                  </tr>
                )}
              </>
            ))}
            {unclassifiedMV > 0 && (
              <tr style={{ borderTop: "1px solid var(--border)", background: "color-mix(in oklab, #e5484d 6%, var(--surface))" }}>
                <td className="px-3 py-2"><span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: "#e5484d" }} /></td>
                <td className="px-3 py-2 font-medium" style={{ color: "#e5484d" }}>⚠ Unclassified</td>
                <td className="px-3 py-2 text-right font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "#e5484d" }}>
                  {unclassifiedPct.toFixed(1)}%
                </td>
                <td className="px-3 py-2 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-2)" }}>
                  {formatCurrency(unclassifiedMV)}
                </td>
                <td className="px-3 py-2 text-right">
                  <button onClick={onFix} className="text-[12px] font-medium" style={{ color: "#e5484d" }}>Fix →</button>
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

  const scopePortfolioName = useMemo(() => {
    if (scope === "all") return "";
    if (scope === "active") return activePortfolio?.name ?? "";
    return scope;
  }, [scope, activePortfolio]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const res = await api.concentration(scopePortfolioName);
      setData(res);
    } catch (e) {
      log.error("concentration-risk", "load failed", e);
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }, [scopePortfolioName]);

  useEffect(() => { refresh(); }, [refresh]);

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <header className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="w-2 h-6 rounded-sm" style={{ background: navColor }} />
          <div>
            <h1 className="text-2xl font-semibold" style={{ color: "var(--ink-1)" }}>Concentration Risk</h1>
            <p className="mt-1 text-[13px]" style={{ color: "var(--ink-4)" }}>
              Sector + theme rollup of open positions by market value.{" "}
              <Link href="/sector-mapping" style={{ color: navColor }}>Manage mappings →</Link>
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[13px]">
          <label style={{ color: "var(--ink-4)" }}>Scope</label>
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value)}
            className="px-2.5 py-1.5 rounded text-[13px]"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}
          >
            <option value="active">Active ({activePortfolio?.name ?? "—"})</option>
            <option value="all">All portfolios</option>
            {portfolios.map((p) => (
              <option key={p.id} value={p.name}>{p.name}</option>
            ))}
          </select>
        </div>
      </header>

      {err && (
        <div className="mb-4 p-3 rounded-lg text-[13px]" style={{ background: "#fee", color: "#c00", border: "1px solid #fbb" }}>{err}</div>
      )}

      {loading || !data ? (
        <div className="text-[13px]" style={{ color: "var(--ink-4)" }}>Loading…</div>
      ) : (
        <>
          {/* KPI strip */}
          <div className="grid grid-cols-3 gap-3 mb-6">
            <KPICard label="Total Market Value" value={formatCurrency(data.total_market_value)} />
            <KPICard label="Sectors" value={String(data.sectors.length)} />
            <KPICard
              label="Unclassified"
              value={`${data.unclassified.length} pos`}
              tone={data.unclassified.length > 0 ? "warn" : "ok"}
            />
          </div>

          {data.positions.length === 0 ? (
            <div className="text-[13px] p-8 rounded-lg text-center" style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
              No open positions in this scope.
            </div>
          ) : (
            <>
              <section className="mb-8">
                <h2 className="text-[15px] font-semibold mb-3" style={{ color: "var(--ink-2)" }}>By Sector</h2>
                <BucketTable
                  buckets={data.sectors}
                  kind="sector"
                  totalMV={data.total_market_value}
                  unclassifiedMV={data.unclassified.reduce((s, u) => s + u.market_value, 0)}
                  onFix={() => window.location.assign("/sector-mapping")}
                />
              </section>

              <section className="mb-8">
                <h2 className="text-[15px] font-semibold mb-3" style={{ color: "var(--ink-2)" }}>By Theme</h2>
                <BucketTable
                  buckets={data.themes}
                  kind="theme"
                  totalMV={data.total_market_value}
                  unclassifiedMV={data.unclassified.reduce((s, u) => s + u.market_value, 0)}
                  onFix={() => window.location.assign("/sector-mapping")}
                />
              </section>

              {data.unclassified.length > 0 && (
                <section className="mb-4 p-4 rounded-lg" style={{ background: "color-mix(in oklab, #e5484d 6%, var(--surface))", border: "1px dashed #e5484d" }}>
                  <div className="text-[13px] font-semibold mb-2" style={{ color: "#e5484d" }}>
                    ⚠ {data.unclassified.length} unclassified position{data.unclassified.length === 1 ? "" : "s"}
                  </div>
                  <div className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
                    Not counted in the sector or theme rollup. Concentration percentages ignore these — fix on Sector Mapping to get a true picture.
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {data.unclassified.map((u) => (
                      <Link
                        key={u.ticker}
                        href="/sector-mapping"
                        className="px-2.5 py-1 rounded text-[12px]"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}
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
      )}
    </div>
  );
}

function KPICard({ label, value, tone }: { label: string; value: string; tone?: "ok" | "warn" }) {
  const color = tone === "warn" ? "#e5484d" : "var(--ink-1)";
  return (
    <div className="p-4 rounded-lg" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[20px] font-semibold mt-1" style={{ fontFamily: "var(--font-jetbrains), monospace", color }}>{value}</div>
    </div>
  );
}
