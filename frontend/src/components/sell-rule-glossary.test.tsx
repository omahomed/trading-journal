import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, beforeEach } from "vitest";

// jsdom localStorage shim — matches the section-expander.test.tsx
// shim so the lazy initializer doesn't throw on first call.
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
    writable: true,
  });
}

import { SellRuleGlossary } from "./sell-rule-glossary";
import { SELL_RULES, RULE_HIERARCHY } from "@/lib/trade-rules";

const GLOSSARY_KEY = "mo-log-sell-glossary-expanded";

describe("SellRuleGlossary", () => {
  beforeEach(() => {
    try { localStorage.clear(); } catch { /* shim */ }
  });

  test("defaults to collapsed (body not in DOM, aria-expanded=false)", () => {
    render(<SellRuleGlossary />);
    const btn = screen.getByRole("button", { name: /Sell rule reference/i });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    // No rule code visible while collapsed.
    expect(screen.queryByText("sr1")).not.toBeInTheDocument();
    expect(screen.queryByText("Rule Interaction Hierarchy")).not.toBeInTheDocument();
  });

  test("header caption reflects collapsed/expanded state", () => {
    render(<SellRuleGlossary />);
    const btn = screen.getByRole("button", { name: /Sell rule reference/i });
    expect(btn).toHaveTextContent(/Show 13 rules/);
    act(() => { fireEvent.click(btn); });
    expect(btn).toHaveTextContent(/Hide/);
  });

  test("expanded: renders all 13 rule codes", () => {
    render(<SellRuleGlossary />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /Sell rule reference/i })); });

    for (const rule of SELL_RULES) {
      // Each rule's code appears in the document.
      expect(screen.getByText(rule.code)).toBeInTheDocument();
    }
  });

  test("expanded: each rule's oneLiner is rendered", () => {
    render(<SellRuleGlossary />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /Sell rule reference/i })); });

    // Spot-check a few rules across the list.
    expect(screen.getByText(/Initial stop on every new position/i)).toBeInTheDocument();
    expect(screen.getByText(/Cut after 8 weeks of no meaningful movement/i)).toBeInTheDocument();
    expect(screen.getByText(/Exit on structural shifts/i)).toBeInTheDocument();
  });

  test("expanded: sr7 mechanics markdown table renders as HTML <table>", () => {
    render(<SellRuleGlossary />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /Sell rule reference/i })); });

    // remark-gfm should turn the pipe-table into a real <table>. Pick a
    // cell value unique to the sr7 table.
    expect(screen.getByText("Up <25% from entry")).toBeInTheDocument();
    // And it should live inside a table element.
    const cell = screen.getByText("Up <25% from entry");
    expect(cell.closest("table")).not.toBeNull();
  });

  test("expanded: sr8 weekly MO RS table renders the three signals", () => {
    render(<SellRuleGlossary />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /Sell rule reference/i })); });

    // The Quick/Quicksand/Grateful Dead row labels render as table cells.
    expect(screen.getAllByText(/Quick/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Quicksand/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Grateful Dead/).length).toBeGreaterThan(0);
  });

  test("expanded: hierarchy table renders header + all 6 entries", () => {
    render(<SellRuleGlossary />);
    act(() => { fireEvent.click(screen.getByRole("button", { name: /Sell rule reference/i })); });

    expect(screen.getByText("Rule Interaction Hierarchy")).toBeInTheDocument();
    // Each conflict label is a unique row identifier.
    for (const entry of RULE_HIERARCHY) {
      expect(screen.getByText(entry.conflict)).toBeInTheDocument();
    }
  });

  test("collapsed state persists via the documented localStorage key", () => {
    render(<SellRuleGlossary />);
    const btn = screen.getByRole("button", { name: /Sell rule reference/i });
    // Default is collapsed, no key written yet.
    expect(localStorage.getItem(GLOSSARY_KEY)).toBeNull();

    act(() => { fireEvent.click(btn); });
    expect(localStorage.getItem(GLOSSARY_KEY)).toBe("true");

    act(() => { fireEvent.click(btn); });
    expect(localStorage.getItem(GLOSSARY_KEY)).toBe("false");
  });
});
