"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type Slice, type SliceHolding, type SlicesResponse } from "@/lib/api";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { KPITile, TILE_GRADIENTS } from "./campaign-detail";

interface Props {
  navColor: string;
}

/** Portfolio-level target % of a slice — resolved by walking up the
 *  parent chain and multiplying each `target_pct` step. Root slices'
 *  target_pct is directly their % of portfolio. */
function computeTargetPctOfPortfolio(
  slice: Slice,
  byId: Map<number, Slice>,
): number {
  let pct = slice.target_pct;
  let cursor: Slice | undefined = slice.parent_id
    ? byId.get(slice.parent_id)
    : undefined;
  while (cursor) {
    pct = (pct * cursor.target_pct) / 100;
    cursor = cursor.parent_id ? byId.get(cursor.parent_id) : undefined;
  }
  return pct;
}

const mono = "var(--font-jetbrains), monospace";

function fmtMoney(n: number, precision = 2): string {
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  return `${sign}$${abs.toLocaleString(undefined, {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  })}`;
}

function fmtPct(n: number, digits = 1): string {
  return `${n >= 0 ? "" : ""}${n.toFixed(digits)}%`;
}

/** Small color-rail component that mirrors the M1 screenshots' left-edge
 *  slice-color bar. Derived from the slice's own color or an auto-picked
 *  palette hash if none set. */
function ColorRail({ color }: { color: string }) {
  return (
    <div
      className="w-[3px] self-stretch rounded-[2px]"
      style={{ background: color }}
    />
  );
}

const AUTO_PALETTE = [
  "#3b82f6", "#8b5cf6", "#0891b2", "#10b981", "#f59e0b",
  "#ec4899", "#ef4444", "#14b8a6", "#a855f7", "#f97316",
];

function pickColor(seed: string, override?: string | null): string {
  if (override && override.trim()) return override;
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) | 0;
  return AUTO_PALETTE[Math.abs(h) % AUTO_PALETTE.length];
}

export function Slices({ navColor }: Props) {
  const { activePortfolio } = usePortfolio();
  const [data, setData] = useState<SlicesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // null = at implicit root (showing parent_id=null slices).
  const [focusedSliceId, setFocusedSliceId] = useState<number | null>(null);

  const portfolioName = activePortfolio?.name ?? "";

  useEffect(() => {
    if (!portfolioName) return;
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);
    api.slicesList(portfolioName)
      .then(res => { if (!cancelled) setData(res); })
      .catch(e => {
        if (cancelled) return;
        log.error("slices", "load failed", e);
        setError(String(e));
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [portfolioName]);

  const onManualRefresh = () => {
    if (!portfolioName || refreshing) return;
    setRefreshing(true);
    setError(null);
    api.slicesList(portfolioName)
      .then(res => setData(res))
      .catch(e => { log.error("slices", "refresh failed", e); setError(String(e)); })
      .finally(() => setRefreshing(false));
  };

  // Derived structures — cheap; recompute on every render.
  const byId = useMemo(() => {
    const m = new Map<number, Slice>();
    (data?.slices ?? []).forEach(s => m.set(s.id, s));
    return m;
  }, [data]);

  const childrenByParent = useMemo(() => {
    const m = new Map<number | null, Slice[]>();
    (data?.slices ?? []).forEach(s => {
      const key = s.parent_id;
      if (!m.has(key)) m.set(key, []);
      m.get(key)!.push(s);
    });
    for (const arr of m.values()) {
      arr.sort((a, b) => a.sort_order - b.sort_order || a.name.localeCompare(b.name));
    }
    return m;
  }, [data]);

  const holdingsByLeaf = useMemo(() => {
    const m = new Map<number, SliceHolding[]>();
    (data?.holdings ?? []).forEach(h => {
      if (!m.has(h.slice_id)) m.set(h.slice_id, []);
      m.get(h.slice_id)!.push(h);
    });
    for (const arr of m.values()) arr.sort((a, b) => b.market_value - a.market_value);
    return m;
  }, [data]);

  // Ancestor chain for the breadcrumb — root -> ... -> focused.
  const trail = useMemo(() => {
    if (focusedSliceId == null) return [] as Slice[];
    const path: Slice[] = [];
    let cursor: Slice | undefined = byId.get(focusedSliceId);
    while (cursor) {
      path.unshift(cursor);
      cursor = cursor.parent_id ? byId.get(cursor.parent_id) : undefined;
    }
    return path;
  }, [focusedSliceId, byId]);

  const focused = focusedSliceId ? byId.get(focusedSliceId) : null;
  const focusedChildren = childrenByParent.get(focusedSliceId ?? null) ?? [];
  const focusedHoldings = focusedSliceId ? (holdingsByLeaf.get(focusedSliceId) ?? []) : [];
  const isAtLeaf = focused != null && focusedChildren.length === 0 && focusedHoldings.length > 0;
  const isEmpty = (data?.slices?.length ?? 0) === 0;

  // ─── KPIs ─────────────────────────────────────────────────────────
  const totalMV = data?.total_market_value ?? 0;
  const unassignedCount = data?.unassigned?.length ?? 0;
  const unassignedValue = (data?.unassigned ?? []).reduce((s, u) => s + u.market_value, 0);
  const assignedPct = totalMV > 0 ? ((totalMV - unassignedValue) / totalMV) * 100 : 0;

  // Max drift across leaves — a leaf is any slice with no children.
  const maxDrift = useMemo(() => {
    if (!data) return 0;
    let max = 0;
    for (const s of data.slices) {
      if ((childrenByParent.get(s.id) ?? []).length > 0) continue;
      const targetPct = computeTargetPctOfPortfolio(s, byId);
      const drift = s.subtree_pct - targetPct;
      if (Math.abs(drift) > Math.abs(max)) max = drift;
    }
    return max;
  }, [data, byId, childrenByParent]);

  const rootCount = (childrenByParent.get(null) ?? []).length;
  const leafCount = (data?.slices ?? []).filter(
    s => (childrenByParent.get(s.id) ?? []).length === 0,
  ).length;

  // ─── Row shape helpers ────────────────────────────────────────────
  interface Row {
    key: string;
    kind: "slice" | "holding";
    name: string;
    subLabel?: string;
    color: string;
    value: number;
    actualPct: number;
    targetPct: number;
    onOpen?: () => void;
    rebalanceUsd?: number;
    isChildLeaf?: boolean;
  }

  const sliceRows: Row[] = focusedChildren.map(s => {
    const targetPct = computeTargetPctOfPortfolio(s, byId);
    const targetUsd = (targetPct / 100) * totalMV;
    const hasChildren = (childrenByParent.get(s.id) ?? []).length > 0;
    return {
      key: `slice:${s.id}`,
      kind: "slice",
      name: s.name,
      color: pickColor(String(s.id), s.color),
      value: s.subtree_value,
      actualPct: s.subtree_pct,
      targetPct,
      rebalanceUsd: targetUsd - s.subtree_value,
      onOpen: () => setFocusedSliceId(s.id),
      isChildLeaf: !hasChildren,
    };
  });

  const holdingRows: Row[] = focusedHoldings.map(h => {
    // Leaf slice's portfolio target %, then split by holding's within-leaf target.
    const leaf = byId.get(h.slice_id);
    const leafPortfolioTargetPct = leaf ? computeTargetPctOfPortfolio(leaf, byId) : 0;
    const holdingTargetPct = (leafPortfolioTargetPct * h.target_pct) / 100;
    const holdingTargetUsd = (holdingTargetPct / 100) * totalMV;
    return {
      key: `hold:${h.id}`,
      kind: "holding",
      name: h.ticker,
      subLabel: h.held
        ? `${h.shares} sh @ ${fmtMoney(h.current_price, 2)}`
        : "not currently held",
      color: pickColor(h.ticker),
      value: h.market_value,
      actualPct: h.actual_pct_of_portfolio,
      targetPct: holdingTargetPct,
      rebalanceUsd: h.held ? holdingTargetUsd - h.market_value : 0,
    };
  });

  const rows = focusedSliceId == null
    ? sliceRows                         // root: show top-level slices
    : focusedChildren.length > 0
    ? sliceRows                         // intermediate: show children
    : holdingRows;                      // leaf: show tickers

  // ─── Render ───────────────────────────────────────────────────────
  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <style>{`
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {/* Header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Slice{" "}
            <em className="italic" style={{ color: navColor }}>Allocation</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            M1-Finance-style thematic buckets — target vs. actual with per-slice
            rebalance hints. Every open ticker should land in a leaf.
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onManualRefresh}
            disabled={refreshing}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--ink-2)",
              opacity: refreshing ? 0.6 : 1,
            }}
          >
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
          <button
            disabled
            title="Editing arrives in stage 2 — build slices via the API for now."
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: navColor,
              border: `1px solid ${navColor}`,
              color: "white",
              opacity: 0.4,
              cursor: "not-allowed",
            }}
          >
            Manage Slices
          </button>
        </div>
      </div>

      {/* KPI tiles */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-[14px] mb-[18px]">
        <KPITile
          label={activePortfolio?.name ?? "Portfolio"}
          value={fmtMoney(totalMV, 0)}
          sub="Open positions market value"
          gradient={TILE_GRADIENTS.indigo}
        />
        <KPITile
          label="Assigned"
          value={fmtPct(assignedPct, 1)}
          sub={
            unassignedCount === 0
              ? "All open tickers slotted"
              : `${unassignedCount} unassigned · ${fmtMoney(unassignedValue, 0)}`
          }
          gradient={
            unassignedCount === 0 ? TILE_GRADIENTS.green : TILE_GRADIENTS.orange
          }
        />
        <KPITile
          label="Max Drift"
          value={fmtPct(maxDrift, 1)}
          sub={maxDrift === 0 ? "No leaves configured" : "Largest leaf drift vs target"}
          gradient={
            Math.abs(maxDrift) >= 5 ? TILE_GRADIENTS.red
              : Math.abs(maxDrift) >= 2 ? TILE_GRADIENTS.orange
              : TILE_GRADIENTS.blue
          }
        />
        <KPITile
          label="Slices"
          value={`${leafCount}`}
          sub={`${rootCount} root · ${leafCount} leaves`}
          gradient={TILE_GRADIENTS.pink}
        />
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-[14px] px-4 py-3 rounded-[10px]"
             style={{
               background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
               border: "1px solid var(--border)",
               color: "#e5484d",
             }}>
          {error}
        </div>
      )}

      {/* Unassigned banner */}
      {unassignedCount > 0 && !loading && (
        <div className="mb-[14px] px-4 py-3 rounded-[10px] flex items-center justify-between gap-4"
             style={{
               background: "color-mix(in oklab, #f97316 10%, var(--surface))",
               border: "1px solid var(--border)",
               color: "var(--ink-2)",
             }}>
          <div className="text-[13px]">
            <strong style={{ color: "#f97316" }}>
              {unassignedCount} unassigned holding{unassignedCount === 1 ? "" : "s"}
            </strong>{" "}
            · {fmtMoney(unassignedValue, 0)} unallocated. Editing arrives in
            the next iteration — API is ready today.
          </div>
        </div>
      )}

      {/* Breadcrumb */}
      <div className="mb-[12px] flex items-center gap-2 text-[13px]"
           style={{ color: "var(--ink-2)" }}>
        <button
          onClick={() => setFocusedSliceId(null)}
          className="hover:underline"
          style={{
            color: focusedSliceId == null ? "var(--ink-1)" : "var(--ink-3)",
            fontWeight: focusedSliceId == null ? 600 : 400,
          }}
        >
          All Slices
        </button>
        {trail.map((s, i) => (
          <span key={s.id} className="flex items-center gap-2">
            <span style={{ color: "var(--ink-3)" }}>›</span>
            <button
              onClick={() => setFocusedSliceId(s.id)}
              className="hover:underline"
              style={{
                color: i === trail.length - 1 ? "var(--ink-1)" : "var(--ink-3)",
                fontWeight: i === trail.length - 1 ? 600 : 400,
              }}
            >
              {s.name}
            </button>
          </span>
        ))}
      </div>

      {/* Content */}
      <div
        className="rounded-[14px] overflow-hidden"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          boxShadow: "var(--card-shadow)",
        }}
      >
        {loading && !data ? (
          <div className="p-6 text-[13px]" style={{ color: "var(--ink-3)" }}>
            Loading slice tree…
          </div>
        ) : isEmpty ? (
          <EmptyState
            portfolioName={activePortfolio?.name ?? ""}
            unassignedCount={unassignedCount}
            navColor={navColor}
          />
        ) : (
          <>
            {/* Column headers — match the M1 look */}
            <div className="px-4 py-3 grid items-center gap-3 text-[11px] uppercase tracking-wider"
                 style={{
                   gridTemplateColumns: "auto 1fr 140px 150px 130px 32px",
                   color: "var(--ink-3)",
                   borderBottom: "1px solid var(--border)",
                 }}>
              <div></div>
              <div>Name</div>
              <div className="text-right">Value</div>
              <div className="text-right">Actual / Target</div>
              <div className="text-right">Rebalance</div>
              <div></div>
            </div>
            {rows.length === 0 ? (
              <div className="p-6 text-[13px]" style={{ color: "var(--ink-3)" }}>
                {isAtLeaf
                  ? "This leaf has no assigned tickers yet."
                  : "No children in this slice."}
              </div>
            ) : (
              rows.map(r => <SliceRow key={r.key} row={r} />)
            )}
          </>
        )}
      </div>

      {/* Unassigned list — always visible at the bottom for quick reference */}
      {(data?.unassigned?.length ?? 0) > 0 && (
        <div className="mt-[18px]">
          <div className="text-[13px] font-semibold mb-2"
               style={{ color: "var(--ink-2)" }}>
            Unassigned holdings
          </div>
          <div className="rounded-[14px] overflow-hidden"
               style={{
                 background: "var(--surface)",
                 border: "1px solid var(--border)",
                 boxShadow: "var(--card-shadow)",
               }}>
            {(data?.unassigned ?? []).map(u => (
              <div key={u.ticker}
                   className="px-4 py-3 flex items-center gap-3 text-[13px]"
                   style={{ borderBottom: "1px solid var(--border)" }}>
                <ColorRail color="#94a3b8" />
                <div className="flex-1">
                  <div style={{ fontFamily: mono }}>{u.ticker}</div>
                  <div className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                    {u.shares} sh @ {fmtMoney(u.current_price, 2)}
                  </div>
                </div>
                <div className="text-right" style={{ fontFamily: mono, minWidth: 100 }}>
                  {fmtMoney(u.market_value, 2)}
                </div>
                <div className="text-right text-[11px]"
                     style={{ color: "var(--ink-3)", minWidth: 60 }}>
                  {fmtPct(u.actual_pct_of_portfolio, 1)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SliceRow({ row }: {
  row: {
    key: string; kind: "slice" | "holding";
    name: string; subLabel?: string;
    color: string; value: number;
    actualPct: number; targetPct: number;
    rebalanceUsd?: number; onOpen?: () => void; isChildLeaf?: boolean;
  };
}) {
  const drift = row.actualPct - row.targetPct;
  const rebalance = row.rebalanceUsd ?? 0;

  return (
    <button
      onClick={row.onOpen}
      disabled={!row.onOpen}
      className="w-full px-4 py-3 grid items-center gap-3 text-[13px] text-left transition-colors"
      style={{
        gridTemplateColumns: "auto 1fr 140px 150px 130px 32px",
        borderBottom: "1px solid var(--border)",
        cursor: row.onOpen ? "pointer" : "default",
        background: "transparent",
      }}
      onMouseEnter={e => {
        if (row.onOpen) e.currentTarget.style.background = "var(--hover)";
      }}
      onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
    >
      <ColorRail color={row.color} />
      <div>
        <div style={{
          fontFamily: row.kind === "holding" ? mono : undefined,
          fontWeight: row.kind === "slice" ? 500 : 400,
          color: "var(--ink-1)",
        }}>
          {row.name}
        </div>
        {row.subLabel && (
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-3)" }}>
            {row.subLabel}
          </div>
        )}
      </div>
      <div className="text-right" style={{ fontFamily: mono, color: "var(--ink-1)" }}>
        {fmtMoney(row.value, 2)}
      </div>
      <div className="text-right">
        <div style={{ fontFamily: mono, fontWeight: 600, color: "var(--ink-1)" }}>
          {fmtPct(row.actualPct, 1)}
        </div>
        <div className="text-[11px]" style={{ fontFamily: mono, color: "var(--ink-3)" }}>
          / {fmtPct(row.targetPct, 1)}
          <span className="ml-1" style={{
            color: Math.abs(drift) < 0.5
              ? "var(--ink-3)"
              : drift > 0
              ? "#e5484d"
              : "#08a86b",
          }}>
            ({drift >= 0 ? "+" : ""}{drift.toFixed(1)}pp)
          </span>
        </div>
      </div>
      <div className="text-right">
        {Math.abs(rebalance) < 1 ? (
          <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>on target</span>
        ) : (
          <span
            className="inline-block px-2 py-1 rounded-[8px] text-[12px]"
            style={{
              fontFamily: mono,
              background: rebalance > 0
                ? "color-mix(in oklab, #08a86b 12%, var(--surface))"
                : "color-mix(in oklab, #e5484d 12%, var(--surface))",
              color: rebalance > 0 ? "#08a86b" : "#e5484d",
              border: "1px solid var(--border)",
            }}
          >
            {rebalance > 0 ? "Buy " : "Trim "}
            {fmtMoney(Math.abs(rebalance), 0)}
          </span>
        )}
      </div>
      <div className="text-right" style={{ color: "var(--ink-3)" }}>
        {row.onOpen ? "›" : ""}
      </div>
    </button>
  );
}

function EmptyState({
  portfolioName, unassignedCount, navColor,
}: {
  portfolioName: string; unassignedCount: number; navColor: string;
}) {
  return (
    <div className="p-8 text-[13px]" style={{ color: "var(--ink-2)" }}>
      <div className="text-[16px] font-semibold mb-2" style={{ color: "var(--ink-1)" }}>
        No slices configured for {portfolioName || "this portfolio"}.
      </div>
      <div className="max-w-[560px]" style={{ color: "var(--ink-3)" }}>
        Slices are M1-Finance-style thematic buckets. Group your holdings into
        meaningful categories (e.g., <em>AI Chips</em>, <em>Energy</em>,
        <em> Infrastructure</em>), assign target percentages, then this page
        shows actual vs. target and the exact $ delta to rebalance — you
        execute the trades on your broker of choice.
      </div>
      {unassignedCount > 0 && (
        <div className="mt-4 text-[13px]" style={{ color: "var(--ink-2)" }}>
          Meanwhile, <strong style={{ color: navColor }}>{unassignedCount}</strong> open
          holding{unassignedCount === 1 ? " is" : "s are"} listed below waiting
          to be slotted.
        </div>
      )}
      <div className="mt-4 text-[11px]" style={{ color: "var(--ink-3)" }}>
        Editing UI arrives in the next iteration. API endpoints are live if
        you want to seed a tree via curl / a script today.
      </div>
    </div>
  );
}
