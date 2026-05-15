"use client";

export interface FlightDeckProps {
  totalTickets: number;
  uniqueTickers: number;
  buys: number;
  sellsTrims: number;
  // Phase 4.5 carryover — overactive (>15 tx) flags the Total Tickets value red.
  isOveractive?: boolean;
}

// Phase 5 — neutral-surface activity tiles relocated from the top of the
// Weekly Retro page to the body of the Per-Ticker Details expander.
// Visual style matches the design's PerTickerExpander body and the prior
// 4-tile grid that lived above the per-ticker section (var(--surface) +
// 1px border) — distinct from the gradient performance tiles up top.
export function FlightDeck({
  totalTickets, uniqueTickers, buys, sellsTrims, isOveractive = false,
}: FlightDeckProps) {
  const tiles = [
    { k: "Total Tickets",  v: totalTickets, alert: isOveractive },
    { k: "Unique Tickers", v: uniqueTickers, alert: false },
    { k: "Buys",           v: buys, alert: false },
    { k: "Sells / Trims",  v: sellsTrims, alert: false },
  ];
  return (
    <div data-testid="flight-deck" className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
      {tiles.map(t => (
        <div
          key={t.k}
          className="p-3 rounded-[12px]"
          style={{ border: "1px solid var(--border)", background: "var(--surface)" }}
        >
          <div
            className="text-[10px] uppercase tracking-[0.08em] font-semibold"
            style={{ color: "var(--ink-4)" }}
          >
            {t.k}
          </div>
          <div
            className="text-[22px] font-semibold mt-0.5"
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              color: t.alert ? "#e5484d" : "var(--ink)",
            }}
          >
            {t.v}
          </div>
        </div>
      ))}
    </div>
  );
}
