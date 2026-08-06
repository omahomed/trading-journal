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
  const [manageOpen, setManageOpen] = useState(false);
  const [toggling, setToggling] = useState(false);

  const portfolioName = activePortfolio?.name ?? "";

  // Reload from the API — used by initial mount, manual refresh, and every
  // mutation success. Silent mode (no loading spinner) is the default so
  // mid-modal refetches don't flicker the layout underneath.
  const reload = async (opts?: { spinner?: "load" | "refresh" | "none" }) => {
    if (!portfolioName) return;
    const mode = opts?.spinner ?? "none";
    if (mode === "load") setLoading(true);
    if (mode === "refresh") setRefreshing(true);
    setError(null);
    try {
      const res = await api.slicesList(portfolioName);
      setData(res);
    } catch (e) {
      log.error("slices", "load failed", e);
      setError(String(e));
    } finally {
      if (mode === "load") setLoading(false);
      if (mode === "refresh") setRefreshing(false);
    }
  };

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

  const onManualRefresh = () => reload({ spinner: "refresh" });

  const onToggleEnabled = async () => {
    if (!portfolioName || toggling || !data) return;
    const nextEnabled = !data.slices_enabled;
    setToggling(true);
    setError(null);
    try {
      const res = await api.slicesToggle(portfolioName, nextEnabled);
      if ("error" in res || "detail" in res) {
        const detail = "detail" in res ? res.detail : undefined;
        const err = "error" in res ? res.error : undefined;
        throw new Error(String(detail ?? err ?? "toggle failed"));
      }
      await reload();
    } catch (e) {
      log.error("slices", "toggle failed", e);
      setError(String(e));
    } finally {
      setToggling(false);
    }
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

  const disabled = data?.slices_enabled === false;

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
            disabled={refreshing || disabled}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--ink-2)",
              opacity: refreshing || disabled ? 0.6 : 1,
            }}
          >
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
          <button
            onClick={onToggleEnabled}
            disabled={toggling || !data}
            title={
              disabled
                ? "Slices is currently OFF for this portfolio. Click to enable."
                : "Turn Slices off for this portfolio (you can re-enable any time)."
            }
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--ink-2)",
              opacity: toggling ? 0.6 : 1,
            }}
          >
            {toggling
              ? "…"
              : disabled
              ? "Enable for this portfolio"
              : "Disable for this portfolio"}
          </button>
          <button
            onClick={() => setManageOpen(true)}
            disabled={disabled || !data}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: navColor,
              border: `1px solid ${navColor}`,
              color: "white",
              opacity: disabled ? 0.4 : 1,
              cursor: disabled ? "not-allowed" : "pointer",
            }}
          >
            Manage Slices
          </button>
        </div>
      </div>

      {/* Error banner sits above the disabled/enabled split so toggle errors
          show even when the body is a "Slices disabled" state. */}
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

      {disabled ? (
        <DisabledState
          portfolioName={portfolioName}
          navColor={navColor}
          onEnable={onToggleEnabled}
          toggling={toggling}
        />
      ) : <>

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
            · {fmtMoney(unassignedValue, 0)} unallocated.
          </div>
          <button
            onClick={() => setManageOpen(true)}
            className="px-3 py-1.5 rounded-[8px] text-[12px]"
            style={{
              background: navColor,
              border: `1px solid ${navColor}`,
              color: "white",
            }}
          >
            Assign now
          </button>
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

      </>}

      {manageOpen && data && !disabled && (
        <ManageSlicesModal
          data={data}
          navColor={navColor}
          onClose={() => setManageOpen(false)}
          onChange={reload}
        />
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
        Click <strong>Manage Slices</strong> to build your first tree.
      </div>
    </div>
  );
}

function DisabledState({
  portfolioName, navColor, onEnable, toggling,
}: {
  portfolioName: string; navColor: string;
  onEnable: () => void; toggling: boolean;
}) {
  return (
    <div className="rounded-[14px] p-8"
         style={{
           background: "var(--surface)",
           border: "1px solid var(--border)",
           boxShadow: "var(--card-shadow)",
         }}>
      <div className="text-[16px] font-semibold mb-2" style={{ color: "var(--ink-1)" }}>
        Slices is off for {portfolioName || "this portfolio"}.
      </div>
      <div className="text-[13px] max-w-[560px]" style={{ color: "var(--ink-3)" }}>
        This portfolio isn&apos;t using the M1-Finance-style bucket allocation
        model. Nothing about Slices leaks into other pages — no strict-mode
        badges, no assign nudges. If you change your mind, flip it back on
        below and any open positions will land in the unassigned queue.
      </div>
      <button
        onClick={onEnable}
        disabled={toggling}
        className="mt-5 px-3 py-2 rounded-[10px] text-[13px]"
        style={{
          background: navColor,
          border: `1px solid ${navColor}`,
          color: "white",
          opacity: toggling ? 0.6 : 1,
        }}
      >
        {toggling ? "Enabling…" : `Enable Slices for ${portfolioName}`}
      </button>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// Manage Slices Modal
// ══════════════════════════════════════════════════════════════════
//
// Single-column outline editor. Every action commits immediately so
// there's no "Save" ambiguity — Close just closes. Actions:
//
//   * Add root slice — inline row with name + target %
//   * Add sub-slice — same inline form, indented under parent
//   * Rename slice — click name → inline text input
//   * Retarget slice — click target % → inline number input
//   * Delete slice — confirm → cascade or 409 with actionable error
//   * Assign ticker — dropdown of leaf slices next to each unassigned row
//   * Retarget holding — click % → inline number input
//   * Move / unassign holding — Move dropdown reassigns; × removes
//
// Business rules enforced by the backend (bubbled up as errors):
//   * Leaf-only invariant: can't add a sub-slice to a slice that holds
//     tickers, and vice versa
//   * Delete-with-holdings blocked (409) — unassign first
//
function ManageSlicesModal({
  data, navColor, onClose, onChange,
}: {
  data: SlicesResponse;
  navColor: string;
  onClose: () => void;
  onChange: () => Promise<void>;
}) {
  const [busy, setBusy] = useState<string | null>(null);   // action key, for spinner + disable
  const [error, setError] = useState<string | null>(null);
  const [addingUnder, setAddingUnder] = useState<number | null | "root" | null>(null);
  const [newSliceName, setNewSliceName] = useState("");
  const [newSliceTarget, setNewSliceTarget] = useState("");

  const doWithBusy = async (key: string, fn: () => Promise<void>) => {
    if (busy) return;
    setBusy(key);
    setError(null);
    try {
      await fn();
      await onChange();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };

  const detailFromResp = (
    resp: { detail?: string } | { error?: string; detail?: string },
  ): string => {
    if ("detail" in resp && resp.detail) return resp.detail;
    if ("error" in resp && resp.error) return resp.error;
    return "unknown error";
  };

  // Build lookup structures.
  const slicesById = useMemo(() => {
    const m = new Map<number, Slice>();
    data.slices.forEach(s => m.set(s.id, s));
    return m;
  }, [data.slices]);

  const childrenByParent = useMemo(() => {
    const m = new Map<number | null, Slice[]>();
    data.slices.forEach(s => {
      const key = s.parent_id;
      if (!m.has(key)) m.set(key, []);
      m.get(key)!.push(s);
    });
    for (const arr of m.values()) {
      arr.sort((a, b) => a.sort_order - b.sort_order || a.name.localeCompare(b.name));
    }
    return m;
  }, [data.slices]);

  const holdingsBySlice = useMemo(() => {
    const m = new Map<number, SliceHolding[]>();
    data.holdings.forEach(h => {
      if (!m.has(h.slice_id)) m.set(h.slice_id, []);
      m.get(h.slice_id)!.push(h);
    });
    for (const arr of m.values()) arr.sort((a, b) => a.ticker.localeCompare(b.ticker));
    return m;
  }, [data.holdings]);

  // Every leaf slice (used as the target set for the unassigned dropdown
  // + the ticker Move dropdown). A leaf here means "no child slices" —
  // holdings status doesn't matter (an empty slice can also receive).
  const leafSlices = useMemo(() => {
    return data.slices.filter(s => !(childrenByParent.get(s.id) ?? []).length);
  }, [data.slices, childrenByParent]);

  const leafBreadcrumb = (s: Slice): string => {
    const parts: string[] = [s.name];
    let cursor: Slice | undefined = s.parent_id ? slicesById.get(s.parent_id) : undefined;
    while (cursor) {
      parts.unshift(cursor.name);
      cursor = cursor.parent_id ? slicesById.get(cursor.parent_id) : undefined;
    }
    return parts.join(" › ");
  };

  const addSlice = async (parentId: number | null) => {
    const name = newSliceName.trim();
    const targetPct = parseFloat(newSliceTarget) || 0;
    if (!name) { setError("Name is required"); return; }
    await doWithBusy(`add:${parentId ?? "root"}`, async () => {
      const res = await api.slicesCreate({
        portfolio: data.portfolio,
        parent_id: parentId,
        name,
        target_pct: targetPct,
      });
      if ("detail" in res) throw new Error(detailFromResp(res));
      setNewSliceName("");
      setNewSliceTarget("");
      setAddingUnder(null);
    });
  };

  const renameSlice = async (s: Slice, newName: string) => {
    if (newName.trim() === s.name) return;
    await doWithBusy(`rename:${s.id}`, async () => {
      const res = await api.slicesUpdate(s.id, { name: newName.trim() });
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  const retargetSlice = async (s: Slice, newPct: number) => {
    if (newPct === s.target_pct) return;
    await doWithBusy(`retarget:${s.id}`, async () => {
      const res = await api.slicesUpdate(s.id, { target_pct: newPct });
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  const deleteSlice = async (s: Slice) => {
    if (!confirm(`Delete slice "${s.name}"? Descendant slices will be removed too. Tickers assigned to any leaf under this slice must be unassigned first.`)) return;
    await doWithBusy(`delete:${s.id}`, async () => {
      const res = await api.slicesDelete(s.id);
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  const assignTicker = async (ticker: string, sliceId: number, targetPct = 0) => {
    await doWithBusy(`assign:${ticker}`, async () => {
      const res = await api.slicesAssignHolding(sliceId, { ticker, target_pct: targetPct });
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  const removeHolding = async (h: SliceHolding) => {
    if (!confirm(`Unassign ${h.ticker} from its slice?`)) return;
    await doWithBusy(`remove:${h.id}`, async () => {
      const res = await api.slicesRemoveHolding(h.id);
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  const retargetHolding = async (h: SliceHolding, newPct: number) => {
    if (newPct === h.target_pct) return;
    await doWithBusy(`retarget-h:${h.id}`, async () => {
      // Re-assign to the same slice — endpoint is upsert-by-(portfolio,ticker).
      const res = await api.slicesAssignHolding(h.slice_id, {
        ticker: h.ticker, target_pct: newPct,
      });
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  const moveHolding = async (h: SliceHolding, targetSliceId: number) => {
    if (targetSliceId === h.slice_id) return;
    await doWithBusy(`move:${h.id}`, async () => {
      const res = await api.slicesAssignHolding(targetSliceId, {
        ticker: h.ticker, target_pct: h.target_pct,
      });
      if ("detail" in res) throw new Error(detailFromResp(res));
    });
  };

  // Render one slice row + recursively its children + its holdings.
  const renderSlice = (s: Slice, depth: number): React.ReactNode => {
    const children = childrenByParent.get(s.id) ?? [];
    const holdings = holdingsBySlice.get(s.id) ?? [];
    const isLeaf = children.length === 0;
    const canAddChild = isLeaf ? holdings.length === 0 : true;
    return (
      <div key={s.id}>
        <SliceEditRow
          slice={s}
          depth={depth}
          color={pickColor(String(s.id), s.color)}
          onRename={newName => renameSlice(s, newName)}
          onRetarget={newPct => retargetSlice(s, newPct)}
          onDelete={() => deleteSlice(s)}
          onAddChild={canAddChild ? () => {
            setAddingUnder(s.id);
            setNewSliceName("");
            setNewSliceTarget("");
          } : undefined}
          busy={busy}
        />
        {addingUnder === s.id && (
          <AddSliceRow
            depth={depth + 1}
            name={newSliceName}
            target={newSliceTarget}
            setName={setNewSliceName}
            setTarget={setNewSliceTarget}
            onSubmit={() => addSlice(s.id)}
            onCancel={() => setAddingUnder(null)}
            busy={busy === `add:${s.id}`}
          />
        )}
        {children.map(c => renderSlice(c, depth + 1))}
        {holdings.map(h => (
          <HoldingEditRow
            key={h.id}
            holding={h}
            depth={depth + 1}
            leafSlices={leafSlices}
            leafBreadcrumb={leafBreadcrumb}
            onRetarget={newPct => retargetHolding(h, newPct)}
            onMove={targetId => moveHolding(h, targetId)}
            onRemove={() => removeHolding(h)}
            busy={busy}
          />
        ))}
      </div>
    );
  };

  const roots = childrenByParent.get(null) ?? [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.5)" }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="rounded-[14px] w-full max-w-[880px] max-h-[90vh] flex flex-col"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.35)",
        }}
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="px-5 py-4 flex items-end justify-between gap-4"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="font-normal text-[22px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Manage <em className="italic" style={{ color: navColor }}>Slices</em>
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-[20px] leading-none px-2 py-1 rounded-[8px]"
            style={{
              background: "transparent",
              color: "var(--ink-3)",
              border: "1px solid var(--border)",
            }}
          >
            ×
          </button>
        </div>

        {/* Body — scrollable */}
        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <div className="mb-3 px-3 py-2 rounded-[10px] text-[12px]"
                 style={{
                   background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
                   border: "1px solid var(--border)",
                   color: "#e5484d",
                 }}>
              {error}
            </div>
          )}

          <div className="text-[11px] uppercase tracking-wider mb-2"
               style={{ color: "var(--ink-3)" }}>
            Slice tree — {data.portfolio}
          </div>
          {roots.length === 0 && addingUnder !== "root" && (
            <div className="text-[13px] py-4" style={{ color: "var(--ink-3)" }}>
              No slices yet. Add the first root slice to get started.
            </div>
          )}
          {roots.map(r => renderSlice(r, 0))}
          {addingUnder === "root" && (
            <AddSliceRow
              depth={0}
              name={newSliceName}
              target={newSliceTarget}
              setName={setNewSliceName}
              setTarget={setNewSliceTarget}
              onSubmit={() => addSlice(null)}
              onCancel={() => setAddingUnder(null)}
              busy={busy === "add:root"}
            />
          )}
          {addingUnder !== "root" && (
            <button
              onClick={() => {
                setAddingUnder("root");
                setNewSliceName("");
                setNewSliceTarget("");
              }}
              className="mt-3 px-3 py-1.5 rounded-[8px] text-[12px]"
              style={{
                background: "var(--surface)",
                border: `1px dashed ${navColor}`,
                color: navColor,
              }}
            >
              + Add root slice
            </button>
          )}

          {/* Unassigned tickers */}
          {data.unassigned.length > 0 && (
            <>
              <div className="text-[11px] uppercase tracking-wider mt-6 mb-2"
                   style={{ color: "var(--ink-3)" }}>
                Unassigned tickers ({data.unassigned.length})
              </div>
              {leafSlices.length === 0 && (
                <div className="text-[12px] px-3 py-2 rounded-[8px] mb-2"
                     style={{
                       background: "color-mix(in oklab, #f97316 8%, var(--surface))",
                       border: "1px solid var(--border)",
                       color: "var(--ink-2)",
                     }}>
                  Create at least one slice above before assigning tickers.
                </div>
              )}
              {data.unassigned.map(u => (
                <UnassignedRow
                  key={u.ticker}
                  ticker={u.ticker}
                  marketValue={u.market_value}
                  leafSlices={leafSlices}
                  leafBreadcrumb={leafBreadcrumb}
                  onAssign={sliceId => assignTicker(u.ticker, sliceId)}
                  busy={busy === `assign:${u.ticker}`}
                />
              ))}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 flex justify-end gap-2"
             style={{ borderTop: "1px solid var(--border)" }}>
          <button
            onClick={onClose}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--ink-2)",
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Modal row helpers ────────────────────────────────────────────

function SliceEditRow({
  slice, depth, color, onRename, onRetarget, onDelete, onAddChild, busy,
}: {
  slice: Slice; depth: number; color: string;
  onRename: (v: string) => Promise<void>;
  onRetarget: (v: number) => Promise<void>;
  onDelete: () => Promise<void>;
  onAddChild?: () => void;
  busy: string | null;
}) {
  const [editingName, setEditingName] = useState(false);
  const [editingTarget, setEditingTarget] = useState(false);
  const [nameDraft, setNameDraft] = useState(slice.name);
  const [targetDraft, setTargetDraft] = useState(String(slice.target_pct));
  const isDeleting = busy === `delete:${slice.id}`;
  return (
    <div className="grid items-center gap-2 py-2 text-[13px]"
         style={{
           gridTemplateColumns: "auto 1fr 90px 130px",
           paddingLeft: depth * 20,
           borderBottom: "1px solid var(--border)",
         }}>
      <div className="w-[3px] h-[22px] rounded-[2px]" style={{ background: color }} />
      {editingName ? (
        <input
          autoFocus
          value={nameDraft}
          onChange={e => setNameDraft(e.target.value)}
          onBlur={async () => {
            setEditingName(false);
            await onRename(nameDraft);
          }}
          onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") { setNameDraft(slice.name); setEditingName(false); } }}
          className="px-2 py-1 rounded-[6px] text-[13px]"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--ink-1)",
          }}
        />
      ) : (
        <button
          onClick={() => { setNameDraft(slice.name); setEditingName(true); }}
          className="text-left hover:underline"
          style={{ color: "var(--ink-1)", fontWeight: 500, background: "transparent" }}
        >
          {slice.name}
        </button>
      )}
      {editingTarget ? (
        <input
          autoFocus
          type="number"
          step="0.1"
          min="0"
          max="100"
          value={targetDraft}
          onChange={e => setTargetDraft(e.target.value)}
          onBlur={async () => {
            setEditingTarget(false);
            const n = parseFloat(targetDraft);
            if (Number.isFinite(n)) await onRetarget(n);
          }}
          onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") { setTargetDraft(String(slice.target_pct)); setEditingTarget(false); } }}
          className="px-2 py-1 rounded-[6px] text-[13px] text-right"
          style={{
            fontFamily: mono,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--ink-1)",
          }}
        />
      ) : (
        <button
          onClick={() => { setTargetDraft(String(slice.target_pct)); setEditingTarget(true); }}
          className="text-right hover:underline"
          style={{
            fontFamily: mono,
            color: "var(--ink-2)",
            background: "transparent",
          }}
        >
          {slice.target_pct}%
        </button>
      )}
      <div className="flex gap-1 justify-end text-[11px]">
        {onAddChild && (
          <button
            onClick={onAddChild}
            className="px-2 py-1 rounded-[6px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--ink-2)",
            }}
          >
            + child
          </button>
        )}
        <button
          onClick={onDelete}
          disabled={isDeleting}
          className="px-2 py-1 rounded-[6px]"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "#e5484d",
            opacity: isDeleting ? 0.5 : 1,
          }}
        >
          {isDeleting ? "…" : "Delete"}
        </button>
      </div>
    </div>
  );
}

function AddSliceRow({
  depth, name, target, setName, setTarget, onSubmit, onCancel, busy,
}: {
  depth: number; name: string; target: string;
  setName: (v: string) => void; setTarget: (v: string) => void;
  onSubmit: () => void; onCancel: () => void; busy: boolean;
}) {
  return (
    <div className="grid items-center gap-2 py-2 text-[13px]"
         style={{
           gridTemplateColumns: "auto 1fr 90px 130px",
           paddingLeft: depth * 20,
           borderBottom: "1px dashed var(--border)",
         }}>
      <div className="w-[3px] h-[22px] rounded-[2px]" style={{ background: "var(--border)" }} />
      <input
        autoFocus
        placeholder="New slice name"
        value={name}
        onChange={e => setName(e.target.value)}
        onKeyDown={e => { if (e.key === "Enter") onSubmit(); if (e.key === "Escape") onCancel(); }}
        className="px-2 py-1 rounded-[6px] text-[13px]"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          color: "var(--ink-1)",
        }}
      />
      <input
        type="number"
        step="0.1"
        min="0"
        max="100"
        placeholder="0"
        value={target}
        onChange={e => setTarget(e.target.value)}
        onKeyDown={e => { if (e.key === "Enter") onSubmit(); if (e.key === "Escape") onCancel(); }}
        className="px-2 py-1 rounded-[6px] text-[13px] text-right"
        style={{
          fontFamily: mono,
          background: "var(--surface)",
          border: "1px solid var(--border)",
          color: "var(--ink-1)",
        }}
      />
      <div className="flex gap-1 justify-end text-[11px]">
        <button
          onClick={onSubmit}
          disabled={busy}
          className="px-2 py-1 rounded-[6px]"
          style={{
            background: "#08a86b",
            border: "1px solid #08a86b",
            color: "white",
            opacity: busy ? 0.6 : 1,
          }}
        >
          {busy ? "…" : "Add"}
        </button>
        <button
          onClick={onCancel}
          className="px-2 py-1 rounded-[6px]"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--ink-3)",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

function HoldingEditRow({
  holding, depth, leafSlices, leafBreadcrumb,
  onRetarget, onMove, onRemove, busy,
}: {
  holding: SliceHolding; depth: number;
  leafSlices: Slice[];
  leafBreadcrumb: (s: Slice) => string;
  onRetarget: (v: number) => Promise<void>;
  onMove: (targetSliceId: number) => Promise<void>;
  onRemove: () => Promise<void>;
  busy: string | null;
}) {
  const [editingTarget, setEditingTarget] = useState(false);
  const [targetDraft, setTargetDraft] = useState(String(holding.target_pct));
  const isRemoving = busy === `remove:${holding.id}`;
  return (
    <div className="grid items-center gap-2 py-1.5 text-[12px]"
         style={{
           gridTemplateColumns: "auto 1fr 90px 130px",
           paddingLeft: depth * 20 + 12,
           borderBottom: "1px solid var(--border)",
         }}>
      <div style={{ color: "var(--ink-3)" }}>•</div>
      <div style={{ fontFamily: mono, color: "var(--ink-2)" }}>
        {holding.ticker}
        {!holding.held && (
          <span className="ml-2 text-[10px]" style={{ color: "#f97316" }}>
            not held
          </span>
        )}
      </div>
      {editingTarget ? (
        <input
          autoFocus
          type="number"
          step="0.1"
          min="0"
          max="100"
          value={targetDraft}
          onChange={e => setTargetDraft(e.target.value)}
          onBlur={async () => {
            setEditingTarget(false);
            const n = parseFloat(targetDraft);
            if (Number.isFinite(n)) await onRetarget(n);
          }}
          onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") { setTargetDraft(String(holding.target_pct)); setEditingTarget(false); } }}
          className="px-2 py-1 rounded-[6px] text-right"
          style={{
            fontFamily: mono,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--ink-1)",
          }}
        />
      ) : (
        <button
          onClick={() => { setTargetDraft(String(holding.target_pct)); setEditingTarget(true); }}
          className="text-right hover:underline"
          style={{
            fontFamily: mono,
            color: "var(--ink-3)",
            background: "transparent",
          }}
        >
          {holding.target_pct}%
        </button>
      )}
      <div className="flex gap-1 justify-end">
        <select
          value={holding.slice_id}
          onChange={e => onMove(Number(e.target.value))}
          className="px-1 py-1 rounded-[6px] text-[11px]"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--ink-2)",
            maxWidth: 90,
          }}
        >
          {leafSlices.map(s => (
            <option key={s.id} value={s.id}>{leafBreadcrumb(s)}</option>
          ))}
        </select>
        <button
          onClick={onRemove}
          disabled={isRemoving}
          className="px-2 py-1 rounded-[6px] text-[11px]"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "#e5484d",
            opacity: isRemoving ? 0.5 : 1,
          }}
        >
          ×
        </button>
      </div>
    </div>
  );
}

function UnassignedRow({
  ticker, marketValue, leafSlices, leafBreadcrumb, onAssign, busy,
}: {
  ticker: string; marketValue: number;
  leafSlices: Slice[];
  leafBreadcrumb: (s: Slice) => string;
  onAssign: (sliceId: number) => Promise<void>;
  busy: boolean;
}) {
  const [pending, setPending] = useState<number | "">("");
  return (
    <div className="grid items-center gap-2 py-2 text-[13px]"
         style={{
           gridTemplateColumns: "auto 1fr auto",
           borderBottom: "1px solid var(--border)",
         }}>
      <div className="w-[3px] h-[22px] rounded-[2px]" style={{ background: "#94a3b8" }} />
      <div>
        <div style={{ fontFamily: mono, color: "var(--ink-1)" }}>{ticker}</div>
        <div className="text-[11px]" style={{ color: "var(--ink-3)" }}>
          {fmtMoney(marketValue, 2)}
        </div>
      </div>
      <div className="flex items-center gap-1">
        <select
          value={pending}
          onChange={e => {
            const v = e.target.value ? Number(e.target.value) : "";
            setPending(v);
            if (v !== "") onAssign(v as number).then(() => setPending(""));
          }}
          disabled={busy || leafSlices.length === 0}
          className="px-2 py-1 rounded-[6px] text-[12px]"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--ink-2)",
            minWidth: 160,
          }}
        >
          <option value="">Assign to…</option>
          {leafSlices.map(s => (
            <option key={s.id} value={s.id}>{leafBreadcrumb(s)}</option>
          ))}
        </select>
        {busy && <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>…</span>}
      </div>
    </div>
  );
}
