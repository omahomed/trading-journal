import { render, screen, fireEvent } from "@testing-library/react";
import { describe, test, expect, beforeEach } from "vitest";
import { SR8TrimCalculator } from "./sr8-trim-calculator";
import type { EnrichedPosition } from "@/lib/positions";

// Minimal localStorage shim under jsdom (mirrors active-campaign.test.tsx).
if (typeof window !== "undefined" && !(window as any).localStorage?.getItem) {
  const _store = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => _store.get(k) ?? null,
      setItem: (k: string, v: string) => { _store.set(k, String(v)); },
      removeItem: (k: string) => { _store.delete(k); },
      clear: () => { _store.clear(); },
      key: (i: number) => Array.from(_store.keys())[i] ?? null,
      get length() { return _store.size; },
    },
  });
}

function makePosition(overrides: Partial<EnrichedPosition> = {}): EnrichedPosition {
  return {
    trade_id: "T1",
    ticker: "COHR",
    shares: 1500,
    current_price: 100,
    b1_return_pct: 60,
    b1_max_return_pct: 70,
    sell_rule_tier: "sr8",
    // The rest of the EnrichedPosition fields aren't read by the
    // calculator but TypeScript requires them. Fill with safe defaults.
    avg_entry: 60,
    total_cost: 90_000,
    realized_pl: 0,
    rule: "",
    buy_notes: "",
    risk_budget: 0,
    open_date: "2026-01-01",
    days_held: 100,
    avg_stop: 50,
    risk_dollars: 0,
    signed_risk: 0,
    risk_pct: 0,
    current_value: 150_000,
    unrealized_pl: 0,
    overall_pl: 0,
    return_pct: 60,
    pos_size_pct: 25,
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
    ...overrides,
  };
}

beforeEach(() => {
  window.localStorage.clear();
});

describe("SR8TrimCalculator", () => {
  test("renders empty state when no positions", () => {
    render(<SR8TrimCalculator positions={[]} />);
    expect(screen.getByText(/No positions currently classified as SR8/i)).toBeDefined();
  });

  test("prompts for NAV before computing", () => {
    render(<SR8TrimCalculator positions={[makePosition()]} />);
    expect(screen.getByText(/Enter NAV to compute trim/i)).toBeDefined();
  });

  test("computes trim when NAV is entered", () => {
    render(<SR8TrimCalculator positions={[makePosition()]} />);
    const navInput = screen.getByPlaceholderText("$612,636") as HTMLInputElement;
    fireEvent.change(navInput, { target: { value: "600000" } });
    // SR7 default rule, cushion 60% (gt50) → trim ADDS = 600
    expect(screen.getByTestId("trim-shares").textContent).toBe("600");
  });

  test("NAV formatted with $ and commas still parses", () => {
    render(<SR8TrimCalculator positions={[makePosition()]} />);
    const navInput = screen.getByPlaceholderText("$612,636") as HTMLInputElement;
    fireEvent.change(navInput, { target: { value: "$600,000" } });
    expect(screen.getByTestId("trim-shares").textContent).toBe("600");
  });

  test("'Core floor binds' badge appears when intended > capped", () => {
    // 1000 sh, core 900 → ADDS = 100. SR2 trims 25% = 250 intended; capped to 100.
    const p = makePosition({ totalShares: 1000 } as any);
    p.shares = 1000;
    render(<SR8TrimCalculator positions={[p]} />);
    fireEvent.change(screen.getByPlaceholderText("$612,636"), { target: { value: "600000" } });
    // Switch rule dropdown to SR2 (default is SR7).
    const ruleSelects = screen.getAllByRole("combobox");
    // Two selects: position + rule. The rule one is the second.
    fireEvent.change(ruleSelects[1], { target: { value: "sr2" } });
    expect(screen.getByTestId("core-floor-binds-badge")).toBeDefined();
    expect(screen.getByTestId("trim-shares").textContent).toBe("100");
  });

  test("preselectedTradeId locks position to a readonly chip", () => {
    const pA = makePosition({ trade_id: "T1", ticker: "COHR" });
    const pB = makePosition({ trade_id: "T2", ticker: "NVDA" });
    render(<SR8TrimCalculator positions={[pA, pB]} preselectedTradeId="T2" />);
    // No <select> for position (only the rule select remains).
    const selects = screen.getAllByRole("combobox");
    expect(selects.length).toBe(1);
    // Ticker shown via the readonly chip.
    expect(screen.getByText(/NVDA \(T2\)/)).toBeDefined();
  });

  test("NAV persists in localStorage across re-mounts", () => {
    const { unmount } = render(<SR8TrimCalculator positions={[makePosition()]} />);
    fireEvent.change(screen.getByPlaceholderText("$612,636"), { target: { value: "$750,000" } });
    unmount();
    render(<SR8TrimCalculator positions={[makePosition()]} />);
    const navInput = screen.getByPlaceholderText("$612,636") as HTMLInputElement;
    expect(navInput.value).toBe("$750,000");
  });

  test("shows cushion-tier badge when SR7 is selected", () => {
    render(<SR8TrimCalculator positions={[makePosition()]} />);
    fireEvent.change(screen.getByPlaceholderText("$612,636"), { target: { value: "600000" } });
    // SR7 is the default selection — cushion 60% → 'gt50' tier.
    expect(screen.getByText(/>50% cushion/i)).toBeDefined();
  });

  test("SR13 trims entire position", () => {
    render(<SR8TrimCalculator positions={[makePosition()]} />);
    fireEvent.change(screen.getByPlaceholderText("$612,636"), { target: { value: "600000" } });
    const ruleSelects = screen.getAllByRole("combobox");
    fireEvent.change(ruleSelects[1], { target: { value: "sr13" } });
    expect(screen.getByTestId("trim-shares").textContent).toBe("1,500");
    expect(screen.getByText(/Closed/i)).toBeDefined();
  });
});
