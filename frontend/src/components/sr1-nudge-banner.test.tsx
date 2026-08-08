import { render, screen, fireEvent } from "@testing-library/react";
import { describe, test, expect, vi } from "vitest";
import { SR1NudgeBanner } from "./sr1-nudge-banner";
import type { EnrichedPosition } from "@/lib/positions";

/** Minimal EnrichedPosition builder — every field defaults to something
 *  benign so tests only spell out the axes they care about (peak, B1
 *  entry, ATR%, broker stop). */
function pos(overrides: Partial<EnrichedPosition>): EnrichedPosition {
  return {
    trade_id: "T1",
    ticker: "TEST",
    shares: 100,
    avg_entry: 100,
    total_cost: 10_000,
    realized_pl: 0,
    rule: "",
    buy_notes: "",
    risk_budget: 0,
    open_date: "2026-01-01",
    days_held: 1,
    avg_stop: 0,
    risk_dollars: 0,
    signed_risk: 0,
    risk_pct: 0,
    current_price: 100,
    current_value: 10_000,
    unrealized_pl: 0,
    overall_pl: 0,
    return_pct: 0,
    pos_size_pct: 0,
    is_option: false,
    multiplier: 1,
    pyramid_pct: 0,
    risk_status: "Free Roll",
    projected_pl: 0,
    projected_pct: 0,
    realized_bank: 0,
    expiration: null,
    manual_price: null,
    grade: null,
    strategy: null,
    b1_return_pct: 0,
    b1_max_return_pct: 0,
    sell_rule_tier: "sr1",
    ...overrides,
  };
}

describe("SR1NudgeBanner", () => {
  test("renders nothing when no positions need a nudge", () => {
    // No positions.
    const { container: c1 } = render(<SR1NudgeBanner positions={[]} />);
    expect(c1.querySelector('[data-testid="sr1-nudge-banner"]')).toBeNull();

    // Position at target — nudge cleared.
    const { container: c2 } = render(
      <SR1NudgeBanner positions={[pos({
        b1_max_return_pct: 5, b1_entry_price: 100, atr21_entry_pct: 8,
        broker_stop_price: 94,
      })]} />
    );
    expect(c2.querySelector('[data-testid="sr1-nudge-banner"]')).toBeNull();
  });

  test("renders one chip per SR1 position needing a stop", () => {
    render(
      <SR1NudgeBanner positions={[
        // AAPL: peak 5%, B1=100, ATR=8, no stop → target 94, needs nudge
        pos({ trade_id: "T1", ticker: "AAPL",
              b1_max_return_pct: 5, b1_entry_price: 100, atr21_entry_pct: 8,
              broker_stop_price: null }),
        // MSFT: peak 8%, B1=200, ATR=6, stop 190 → target 191, still needs
        pos({ trade_id: "T2", ticker: "MSFT",
              b1_max_return_pct: 8, b1_entry_price: 200, atr21_entry_pct: 6,
              broker_stop_price: 190 }),
        // NVDA: peak 25% — out of SR1 band, no nudge
        pos({ trade_id: "T3", ticker: "NVDA",
              b1_max_return_pct: 25, b1_entry_price: 300, atr21_entry_pct: 10,
              broker_stop_price: null }),
      ]} />
    );

    expect(screen.getByTestId("sr1-nudge-banner")).toBeTruthy();
    expect(screen.getByText(/2 positions/i)).toBeInTheDocument();
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("MSFT")).toBeInTheDocument();
    expect(screen.queryByText("NVDA")).toBeNull();
  });

  test("ticker chips are clickable when onTickerClick is provided", () => {
    const onClick = vi.fn();
    render(
      <SR1NudgeBanner
        positions={[
          pos({ ticker: "AAPL",
                b1_max_return_pct: 5, b1_entry_price: 100, atr21_entry_pct: 8,
                broker_stop_price: null }),
        ]}
        onTickerClick={onClick}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /AAPL/i }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  test("target is computed as B1 × (1 − 0.75 × ATR%)", () => {
    // Locks the display value: B1=176.21, ATR%=4.5
    // → target = 176.21 × (1 − 0.03375) = 170.26
    render(
      <SR1NudgeBanner positions={[
        pos({ ticker: "DELL",
              b1_max_return_pct: 5, b1_entry_price: 176.21, atr21_entry_pct: 4.5,
              broker_stop_price: null }),
      ]} />
    );
    // Chip displays "→ $170.26" (formatCurrency rounds to 2 dp).
    expect(screen.getByText(/170\.26/)).toBeInTheDocument();
  });
});
