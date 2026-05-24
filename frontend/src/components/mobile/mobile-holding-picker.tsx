"use client";

import { type TradePosition } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { Check } from "lucide-react";
import { MobileSelectSheet } from "./mobile-select-sheet";

type Props = {
  /** Open trades sourced from api.tradesOpen — same shape the
   *  desktop sizer's <select> at L685-744 consumes. */
  holdings: readonly TradePosition[];
  /** Currently-selected trade_id, or null when none picked. */
  selectedTradeId: string | null;
  /** Fires when the user picks a row in the sheet. Parent is
   *  responsible for the downstream priceLookup → entry/ATR
   *  auto-fill (matches the desktop pattern at L692-701). */
  onSelect: (holding: TradePosition) => void;
  /** Override the empty-state copy with a portfolio-specific
   *  hint when no open trades are present. */
  portfolioName?: string;
};

/**
 * Mobile holding picker — trigger tile + bottom sheet listing the
 * active portfolio's open trades. Built on MobileSelectSheet to
 * stay visually consistent with the Mode / Profile / Size pickers.
 *
 * Used by Scale-In (PR1), Pyramid (PR2), and Trim (PR3). Each
 * consumer hooks `onSelect` to (a) update its own `selectedHolding`
 * state and (b) fire `api.priceLookup(h.ticker)` to auto-fill the
 * entry + ATR cells — same desktop pattern at L692-701.
 *
 * Empty state: the trigger renders disabled with a "No open
 * trades" subtitle. The sheet would never open in that case, but
 * for accessibility the disabled button keeps its tab-stop.
 */
export function MobileHoldingPicker({
  holdings,
  selectedTradeId,
  onSelect,
  portfolioName,
}: Props) {
  const selected = holdings.find((h) => h.trade_id === selectedTradeId) ?? null;
  const empty = holdings.length === 0;

  if (empty) {
    return (
      <button
        type="button"
        disabled
        aria-label="Holding picker — no open trades"
        className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-[10px] text-left opacity-60"
      >
        <div className="mb-1 text-[10px] font-medium text-m-text-dim">Holding</div>
        <div className="text-sm font-medium text-m-text-dim">No open trades</div>
        {portfolioName && (
          <div className="mt-px text-[11px] text-m-text-faint">in {portfolioName}</div>
        )}
      </button>
    );
  }

  const triggerValue = selected ? selected.ticker : "Select holding…";
  const triggerSub = selected
    ? `${selected.shares} sh @ ${formatCurrency(Number(selected.avg_entry ?? 0))}`
    : "";

  return (
    <MobileSelectSheet
      triggerLabel="Holding"
      triggerValue={triggerValue}
      triggerSubValue={triggerSub}
      triggerAccent={Boolean(selected)}
      triggerSelected={Boolean(selected)}
      sheetTitle="Select holding"
    >
      {(close) => (
        <div className="flex flex-col">
          {holdings.map((h) => {
            const isActive = h.trade_id === selectedTradeId;
            return (
              <button
                key={h.trade_id}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  onSelect(h);
                  close();
                }}
                className="flex min-h-[52px] items-center justify-between border-b-[0.5px] border-m-border px-5 py-3.5 text-left last:border-b-0"
              >
                <span className="flex min-w-0 flex-1 flex-col pr-3">
                  <span className={`truncate text-base ${isActive ? "font-medium" : ""} text-m-text`}>
                    {h.ticker}
                  </span>
                  <span className="font-m-num text-[12px] tabular-nums text-m-text-dim">
                    {h.shares} sh @ {formatCurrency(Number(h.avg_entry ?? 0))}
                    {h.trade_id ? ` · ${h.trade_id}` : ""}
                  </span>
                </span>
                {isActive && (
                  <Check size={20} strokeWidth={2} className="text-m-accent" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </MobileSelectSheet>
  );
}
