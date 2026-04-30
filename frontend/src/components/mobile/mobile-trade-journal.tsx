"use client";

import { Search, ListOrdered, TriangleAlert } from "lucide-react";

/**
 * Phase 1 stub of the mobile Trade Journal. Translates
 * `docs/mobile/anchors/trade-journal-v3.html` to React with hardcoded
 * mock positions. The search input, sort icon, filter chips, and the
 * position-card click targets are visual stubs only — TODO comments
 * mark Phase 4 wiring. Content-only: tape pill, page header, and
 * bottom nav come from `AdaptiveShell` → `MobileShell`.
 */

// ── Mock data ─────────────────────────────────────────────────────
//
// Discriminated union by `kind` because each card state has its own
// footer composition. Phase 4 will reshape this into a real domain
// model fed by the API; the field names below are deliberately close
// to what the API already returns so the migration stays mechanical.
type StockOriginal = {
  kind: "ORIGINAL";
  ticker: string;
  pnlPct: number; // signed: +12.4 or -2.1
  notional: number; // dollars
  shares: number;
  pnlUsd: number; // signed dollars
  dayNum: number;
  stopPrice: number;
  stopPct: number; // signed, e.g. -3.5
};

type StockReady = Omit<StockOriginal, "kind"> & {
  kind: "READY";
  lastAddNum: number; // 1 = A1, 2 = A2
  lastAddPct: number; // ≥ +5.0 by definition of READY
};

type StockAdded = Omit<StockOriginal, "kind"> & {
  kind: "ADDED";
  lastAddNum: number;
  lastAddPct: number; // < +5.0 (else it would be READY)
};

type StockAtRisk = {
  kind: "AT_RISK";
  ticker: string;
  pnlPct: number;
  notional: number;
  shares: number;
  pnlUsd: number;
  dayNum: number;
  stopDistancePct: number; // unsigned, e.g. 0.8 reads as "0.8% away"
};

type OptionsCall = {
  kind: "OPTIONS_CALL";
  ticker: string;
  pnlPct: number;
  notional: number;
  contracts: number;
  pnlUsd: number;
  dayNum: number;
  strike: string; // e.g. "$5C"
  expiry: string; // e.g. "May 16"
};

export type MobileTradeJournalPosition =
  | StockOriginal
  | StockReady
  | StockAdded
  | StockAtRisk
  | OptionsCall;

const MOCK_HOLDINGS = {
  totalUsd: 891234,
  todayPnlUsd: 8420,
  todayPnlPct: 0.95,
  openCount: 23,
  readyCount: 3,
  losersCount: 4,
  atRiskCount: 1,
} as const;

const MOCK_POSITIONS: readonly MobileTradeJournalPosition[] = [
  {
    kind: "ORIGINAL",
    ticker: "NVDA",
    pnlPct: 12.4,
    notional: 139125,
    shares: 750,
    pnlUsd: 15340,
    dayNum: 12,
    stopPrice: 172.69,
    stopPct: -3.5,
  },
  {
    kind: "READY",
    ticker: "ARM",
    pnlPct: 18.7,
    notional: 62540,
    shares: 425,
    pnlUsd: 9840,
    dayNum: 9,
    stopPrice: 129.0,
    stopPct: -4.2,
    lastAddNum: 2,
    lastAddPct: 6.4,
  },
  {
    kind: "ADDED",
    ticker: "HOOD",
    pnlPct: 8.2,
    notional: 67200,
    shares: 1200,
    pnlUsd: 5100,
    dayNum: 7,
    stopPrice: 52.4,
    stopPct: -2.1,
    lastAddNum: 1,
    lastAddPct: 2.8,
  },
  {
    kind: "AT_RISK",
    ticker: "ASML",
    pnlPct: -2.1,
    notional: 78400,
    shares: 200,
    pnlUsd: -1680,
    dayNum: 8,
    stopDistancePct: 0.8,
  },
  {
    kind: "ORIGINAL",
    ticker: "ANET",
    pnlPct: 6.7,
    notional: 92400,
    shares: 280,
    pnlUsd: 5820,
    dayNum: 9,
    stopPrice: 315.2,
    stopPct: -2.4,
  },
  {
    kind: "OPTIONS_CALL",
    ticker: "LUMN",
    pnlPct: 24.0,
    notional: 8400,
    contracts: 5,
    pnlUsd: 1620,
    dayNum: 8,
    strike: "$5C",
    expiry: "May 16",
  },
];

const FILTERS = [
  { label: "All", count: MOCK_HOLDINGS.openCount, active: true, tone: "primary" as const },
  { label: "Ready", count: MOCK_HOLDINGS.readyCount, tone: "accent" as const },
  { label: "Winners", count: 18, tone: "neutral" as const },
  { label: "Losers", count: MOCK_HOLDINGS.losersCount, tone: "neutral" as const },
  { label: "At risk", count: MOCK_HOLDINGS.atRiskCount, tone: "warn" as const },
  { label: "Options", count: 3, tone: "neutral" as const },
];

const fmtUsd = (n: number) =>
  (n < 0 ? "−$" : "$") + Math.abs(n).toLocaleString("en-US");
const fmtPctSigned = (n: number) =>
  (n >= 0 ? "+" : "−") + Math.abs(n).toFixed(1) + "%";
const fmtPctSignedSmall = (n: number) =>
  (n >= 0 ? "+" : "−") + Math.abs(n).toFixed(1) + "%";

export function MobileTradeJournal() {
  return (
    <div className="-mx-5 flex flex-col gap-3 pt-2">
      {/* Search bar — non-functional in Phase 1 */}
      <div className="mx-5">
        <div className="flex items-center gap-2.5 rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
          <Search size={15} strokeWidth={1.5} className="text-m-text-dim" aria-hidden="true" />
          <input
            type="text"
            placeholder="Search ticker"
            aria-label="Search ticker"
            // TODO Phase 4: filter MOCK_POSITIONS by ticker substring
            onChange={() => {}}
            className="flex-1 bg-transparent text-sm text-m-text placeholder:text-m-text-dim focus:outline-none"
          />
          <button
            type="button"
            aria-label="Sort"
            // TODO Phase 4: open sort sheet
            onClick={() => {}}
            className="text-m-text-dim"
          >
            <ListOrdered size={14} strokeWidth={1.4} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Holdings summary */}
      <div className="mx-5">
        <div className="mb-1 text-[11px] font-medium text-m-text-dim">Holdings</div>
        <div className="font-m-num text-[32px] font-medium tabular-nums tracking-[-0.02em] text-m-text">
          {fmtUsd(MOCK_HOLDINGS.totalUsd)}
        </div>
        <div className="mb-2 mt-1 flex items-baseline gap-2">
          <span className="font-m-num text-sm font-medium tabular-nums text-m-accent">
            {fmtUsd(MOCK_HOLDINGS.todayPnlUsd).replace("$", "+$")}
          </span>
          <span className="font-m-num text-[13px] tabular-nums text-m-accent">
            +{MOCK_HOLDINGS.todayPnlPct.toFixed(2)}%
          </span>
          <span className="text-xs text-m-text-dim">today</span>
        </div>
        <div className="flex gap-3 text-xs text-m-text-dim">
          <SummaryCount n={MOCK_HOLDINGS.openCount} label="open" tone="text" />
          <SummaryCount n={MOCK_HOLDINGS.readyCount} label="ready" tone="accent" />
          <SummaryCount n={MOCK_HOLDINGS.losersCount} label="losers" tone="down" />
          <SummaryCount n={MOCK_HOLDINGS.atRiskCount} label="at risk" tone="warn" />
        </div>
      </div>

      {/* Filter chips — horizontal scroll */}
      <div className="overflow-x-auto whitespace-nowrap px-4">
        {FILTERS.map((f) => (
          <FilterChip key={f.label} {...f} />
        ))}
      </div>

      {/* Position cards */}
      <div className="mx-5 flex flex-col gap-2">
        {MOCK_POSITIONS.map((p) => (
          <PositionCard key={p.ticker} position={p} />
        ))}
        <div className="py-3 text-center text-xs text-m-text-dim">
          18 more positions · scroll
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────

function SummaryCount({
  n,
  label,
  tone,
}: {
  n: number;
  label: string;
  tone: "text" | "accent" | "down" | "warn";
}) {
  const numClass =
    tone === "accent" ? "text-m-accent"
      : tone === "down" ? "text-m-down"
      : tone === "warn" ? "text-m-warn"
      : "text-m-text";
  return (
    <span>
      <span className={`font-medium ${numClass}`}>{n}</span> {label}
    </span>
  );
}

function FilterChip({
  label,
  count,
  active,
  tone,
}: {
  label: string;
  count: number;
  active?: boolean;
  tone: "primary" | "accent" | "warn" | "neutral";
}) {
  const baseLayout = "mr-1.5 inline-block rounded-m-pill px-[14px] py-1.5 text-xs";
  let className: string;
  if (active) {
    className = `${baseLayout} bg-m-accent font-medium text-m-accent-text-on`;
  } else if (tone === "accent") {
    className = `${baseLayout} border-[0.5px] border-m-accent-border bg-m-accent-tint font-medium text-m-accent`;
  } else if (tone === "warn") {
    className = `${baseLayout} border-[0.5px] border-m-warn-border-soft bg-m-warn-tint font-medium text-m-warn`;
  } else {
    className = `${baseLayout} border-[0.5px] border-m-border bg-m-surface text-m-text-muted`;
  }
  return (
    <button
      type="button"
      onClick={() => {
        /* TODO Phase 4: switch active filter */
      }}
      aria-pressed={active}
      className={className}
    >
      {label} · {count}
    </button>
  );
}

function PositionCard({ position }: { position: MobileTradeJournalPosition }) {
  const isAtRisk = position.kind === "AT_RISK";
  const isReady = position.kind === "READY";
  const cardClass = isAtRisk
    ? "rounded-m-lg border-[0.5px] border-m-warn-border bg-m-warn-tint-soft px-4 py-3"
    : isReady
    ? "rounded-m-lg border-[0.5px] border-m-accent-border-strong bg-m-surface px-4 py-3"
    : "rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-4 py-3";

  const pnlIsPositive = position.pnlPct >= 0;
  const pnlPctClass = pnlIsPositive ? "text-m-accent" : "text-m-down";
  const pnlUsdClass = pnlIsPositive ? "text-m-accent" : "text-m-down";

  return (
    <button
      type="button"
      onClick={() => {
        /* TODO Phase 4: navigate to position detail */
      }}
      className={`text-left ${cardClass}`}
    >
      {/* Row 1: ticker (+ optional badge) | pnl % */}
      <div className="mb-1 flex items-baseline justify-between">
        <div className="flex items-baseline gap-1.5">
          <span className="text-[17px] font-medium tracking-[-0.01em] text-m-text">
            {position.ticker}
          </span>
          {position.kind === "READY" && (
            <span className="rounded-[4px] bg-m-accent px-1.5 py-px text-[10px] font-semibold tracking-wider text-m-accent-text-on">
              READY
            </span>
          )}
          {position.kind === "OPTIONS_CALL" && (
            <span className="rounded-[4px] bg-m-purple-tint px-1.5 py-px text-[10px] font-medium tracking-wide text-m-purple-text">
              CALLS
            </span>
          )}
        </div>
        <span className={`font-m-num text-sm font-medium tabular-nums ${pnlPctClass}`}>
          {fmtPctSigned(position.pnlPct)}
        </span>
      </div>

      {/* Row 2: notional · units | pnl $ */}
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="font-m-num text-[13px] tabular-nums text-m-text-muted">
          {fmtUsd(position.notional)} ·{" "}
          {position.kind === "OPTIONS_CALL"
            ? `${position.contracts} contracts`
            : `${position.shares.toLocaleString("en-US")} sh`}
        </span>
        <span className={`font-m-num text-[13px] font-medium tabular-nums ${pnlUsdClass}`}>
          {fmtUsd(position.pnlUsd).replace("$", pnlIsPositive ? "+$" : "$")}
        </span>
      </div>

      {/* Row 3: state-specific footer */}
      <PositionFooter position={position} />
    </button>
  );
}

function PositionFooter({ position }: { position: MobileTradeJournalPosition }) {
  switch (position.kind) {
    case "ORIGINAL":
      return (
        <div className="flex items-baseline justify-between">
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            Day {position.dayNum} · B1 only
          </span>
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            stop {fmtUsd(position.stopPrice)} · {fmtPctSignedSmall(position.stopPct)}
          </span>
        </div>
      );
    case "READY":
      return (
        <div className="flex items-baseline justify-between">
          <span className="font-m-num text-[11px] font-medium tabular-nums text-m-accent">
            A{position.lastAddNum} last add {fmtPctSignedSmall(position.lastAddPct)}
          </span>
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            stop {fmtUsd(position.stopPrice)} · {fmtPctSignedSmall(position.stopPct)}
          </span>
        </div>
      );
    case "ADDED":
      return (
        <div className="flex items-baseline justify-between">
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            A{position.lastAddNum} last add {fmtPctSignedSmall(position.lastAddPct)}
          </span>
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            stop {fmtUsd(position.stopPrice)} · {fmtPctSignedSmall(position.stopPct)}
          </span>
        </div>
      );
    case "AT_RISK":
      return (
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-[11px] font-medium text-m-warn">
            <TriangleAlert size={11} strokeWidth={1.2} aria-hidden="true" />
            Stop only −{position.stopDistancePct.toFixed(1)}% away
          </span>
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            Day {position.dayNum}
          </span>
        </div>
      );
    case "OPTIONS_CALL":
      return (
        <div className="flex items-baseline justify-between">
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            {position.strike} · exp {position.expiry}
          </span>
          <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">
            Day {position.dayNum}
          </span>
        </div>
      );
  }
}
