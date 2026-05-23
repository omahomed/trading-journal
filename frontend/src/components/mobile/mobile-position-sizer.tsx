"use client";

import { ChevronDown, Lock } from "lucide-react";
import { formatCurrency } from "@/lib/format";
import { usePortfolio } from "@/lib/portfolio-context";

/**
 * Phase 1 stub of the mobile Position Sizer screen. Translates
 * `docs/mobile/anchors/position-sizer-v6.html` to React with hardcoded
 * NVDA mock data. All interactivity (mode chip, picker tiles, CTA) is
 * a visual stub — TODO comments mark the phase that wires real
 * behavior. Content-only: the surrounding tape pill, page header, and
 * bottom nav are provided by `AdaptiveShell` → `MobileShell`.
 *
 * Phase 2 step 1: reads the active portfolio from the shared context
 * and surfaces its name in the body subtitle. Underlying numeric data
 * remains mock — Phase 2 later steps wire real backend data.
 */

const MOCK = {
  ticker: "NVDA",
  ma: { e8: 182.10, e21: 173.40, s50: 162.50 },
  entry: 185.50,
  nlv: 487704,
  keyMa: 173.40,
  bufferPct: 1.0,
  atrPct: 5.0,
  mode: "Volatility",
  picker: {
    mode: { name: "Offense", subValue: "1.00%" },
    profile: { name: "Tight", subValue: "1.0×" },
    size: { name: "Full", subValue: "10%" },
  },
  audit: {
    shares: 750,
    notional: 139125,
    pctOfNlv: 28.5,
    totalRisk: 4875,
    atrStop: 172.69,
    target2r: 9750,
  },
} as const;

export function MobilePositionSizer() {
  const { activePortfolio } = usePortfolio();
  return (
    <div className="flex flex-col gap-2.5 pt-2">
      {activePortfolio && (
        <div className="text-[11px] text-m-text-dim">
          Sizing for <span className="text-m-text-muted">{activePortfolio.name}</span>
        </div>
      )}
      {/* Mode chip — opens grouped sheet (Entry / Position management / Options) */}
      <button
        type="button"
        onClick={() => {
          /* TODO Phase 2: open mode sheet */
        }}
        className="flex items-center justify-between rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-[11px] font-medium text-m-text-dim">Mode</span>
          <span className="text-sm font-medium text-m-text">{MOCK.mode}</span>
        </div>
        <ChevronDown size={14} strokeWidth={1.5} className="text-m-text-dim" aria-hidden="true" />
      </button>

      {/* Ticker card — symbol + 8E/21E/50S MA stack */}
      <div className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-[18px] py-[14px]">
        <div className="font-m-num text-[28px] font-medium tracking-[-0.02em] text-m-text">{MOCK.ticker}</div>
        <div className="mt-1.5 flex gap-4 font-m-num text-xs tabular-nums text-m-text-dim">
          <MaCell label="8E" value={MOCK.ma.e8} />
          <MaCell label="21E" value={MOCK.ma.e21} />
          <MaCell label="50S" value={MOCK.ma.s50} />
        </div>
      </div>

      {/* 2x2 input grid — Entry / NLV (locked) / Key MA / Buffer */}
      <div className="grid grid-cols-2 gap-2">
        <FieldCell label="Entry" value={formatCurrency(MOCK.entry, { decimals: 0 })} />
        <FieldCell
          label="NLV"
          labelIcon={<Lock size={9} strokeWidth={1} className="text-m-text-dim" aria-hidden="true" />}
          value={formatCurrency(MOCK.nlv, { decimals: 0 })}
        />
        <FieldCell label="Key MA" value={formatCurrency(MOCK.keyMa, { decimals: 0 })} />
        <FieldCell label="Buffer" value={`${MOCK.bufferPct.toFixed(2)}%`} />
      </div>

      {/* ATR row */}
      <div className="flex items-baseline justify-between rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
        <span className="text-[11px] font-medium text-m-text-dim">ATR % (21D)</span>
        <span className="font-m-num text-xl font-medium tabular-nums tracking-[-0.01em] text-m-text">
          {MOCK.atrPct.toFixed(1)}%
        </span>
      </div>

      {/* Three picker tiles — Mode / Profile / Size, each opens its own sheet */}
      <div className="grid grid-cols-3 gap-2">
        <PickerTile label="Mode" valueText={MOCK.picker.mode.name} subValue={MOCK.picker.mode.subValue} accent selected />
        <PickerTile label="Profile" valueText={MOCK.picker.profile.name} subValue={MOCK.picker.profile.subValue} />
        <PickerTile label="Size" valueText={MOCK.picker.size.name} subValue={MOCK.picker.size.subValue} />
      </div>

      {/* Audit result — live computed (no Run Audit button) */}
      <div className="rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 pt-4 pb-3.5">
        <div className="flex items-baseline justify-between">
          <span className="text-[11px] font-medium text-m-text-dim">Audit result</span>
          <span className="text-[11px] font-medium text-m-warn">{MOCK.audit.pctOfNlv.toFixed(1)}% of NLV</span>
        </div>
        <div className="mt-1.5 flex items-baseline gap-2">
          <span className="font-m-num text-[38px] font-medium tabular-nums tracking-[-0.03em] text-m-text">
            {MOCK.audit.shares}
          </span>
          <span className="text-[15px] text-m-text-muted">shares · {formatCurrency(MOCK.audit.notional, { decimals: 0 })}</span>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2.5 border-t-[0.5px] border-m-border pt-3">
          <Stat label="Total risk" value={formatCurrency(MOCK.audit.totalRisk, { decimals: 0 })} tone="down" />
          <Stat label="ATR stop" value={formatCurrency(MOCK.audit.atrStop, { decimals: 0 })} />
          <Stat label="2R target" value={formatCurrency(MOCK.audit.target2r, { decimals: 0 })} tone="up" />
        </div>
      </div>

      {/* CTA */}
      <button
        type="button"
        onClick={() => {
          /* TODO Phase 3: hand off to log-buy with prefilled values */
        }}
        className="block w-full rounded-m-lg bg-m-accent px-4 py-4 text-center text-[15px] font-medium tracking-[-0.01em] text-m-accent-text-on"
      >
        Log buy →
      </button>
    </div>
  );
}

function MaCell({ label, value }: { label: string; value: number }) {
  return (
    <span>
      {label} <span className="font-medium text-m-text">{value.toFixed(2)}</span>
    </span>
  );
}

function FieldCell({
  label,
  labelIcon,
  value,
}: {
  label: string;
  labelIcon?: React.ReactNode;
  value: string;
}) {
  return (
    <div className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
      <div className="mb-0.5 flex items-center gap-1 text-[10px] font-medium text-m-text-dim">
        {label}
        {labelIcon}
      </div>
      <div className="font-m-num text-lg font-medium tabular-nums text-m-text">{value}</div>
    </div>
  );
}

function PickerTile({
  label,
  valueText,
  subValue,
  accent = false,
  selected = false,
}: {
  label: string;
  valueText: string;
  subValue: string;
  accent?: boolean;
  selected?: boolean;
}) {
  const borderClass = selected ? "border-m-accent-border-soft" : "border-m-border";
  const valueToneClass = accent ? "text-m-accent" : "text-m-text";
  return (
    <button
      type="button"
      onClick={() => {
        /* TODO Phase 2: open picker sheet for `label` */
      }}
      className={`rounded-m-md border-[0.5px] ${borderClass} bg-m-surface px-3 py-[10px] text-left`}
    >
      <div className="mb-1 text-[10px] font-medium text-m-text-dim">{label}</div>
      <div className="flex items-baseline justify-between">
        <span className={`text-sm font-medium ${valueToneClass}`}>{valueText}</span>
        <ChevronDown size={10} strokeWidth={1.5} className="text-m-text-dim" aria-hidden="true" />
      </div>
      <div className="mt-px font-m-num text-[11px] tabular-nums text-m-text-muted">{subValue}</div>
    </button>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "up" | "down";
}) {
  const valueClass =
    tone === "down" ? "text-m-down" : tone === "up" ? "text-m-accent" : "text-m-text";
  return (
    <div>
      <div className="mb-0.5 text-[10px] text-m-text-dim">{label}</div>
      <div className={`font-m-num text-[13px] font-medium tabular-nums ${valueClass}`}>{value}</div>
    </div>
  );
}
