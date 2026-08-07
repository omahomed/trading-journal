import { render, screen } from "@testing-library/react";
import { describe, test, expect } from "vitest";
import { SellRuleBadge } from "./sell-rule-badge";
import { SELL_RULES, SELL_RULE_FAMILIES } from "@/lib/trade-rules";

// Post-family-stripe: the SR label lives in an inner <span>; the
// data-tier / data-family / title / background all live on the
// wrapping element. Tests query the wrapper via `[data-tier=...]`.

function wrapper(container: HTMLElement, tier: string): HTMLElement {
  const el = container.querySelector<HTMLElement>(`[data-tier="${tier}"]`);
  if (!el) throw new Error(`no wrapper found for tier ${tier}`);
  return el;
}

describe("SellRuleBadge", () => {
  test("renders em-dash for null tier", () => {
    render(<SellRuleBadge tier={null} />);
    expect(screen.getByText("—")).toBeDefined();
  });

  test.each([
    ["sr1", "SR1"],
    ["sr11", "SR11"],
    ["sr8", "SR8"],
  ] as const)("renders %s label as %s", (tier, label) => {
    render(<SellRuleBadge tier={tier} />);
    expect(screen.getByText(label)).toBeDefined();
  });

  test("title attribute includes rule description and oneLiner", () => {
    const { container } = render(<SellRuleBadge tier="sr8" />);
    const badge = wrapper(container, "sr8");
    const title = badge.getAttribute("title") || "";
    const rule = SELL_RULES.find((r) => r.code === "sr8")!;
    expect(title).toContain("SR8");
    expect(title).toContain(rule.description);
    expect(title).toContain(rule.oneLiner);
  });

  test("data-tier attribute set on the wrapper per tier", () => {
    const { container, rerender } = render(<SellRuleBadge tier="sr1" />);
    expect(wrapper(container, "sr1").getAttribute("data-tier")).toBe("sr1");

    rerender(<SellRuleBadge tier="sr11" />);
    expect(wrapper(container, "sr11").getAttribute("data-tier")).toBe("sr11");

    rerender(<SellRuleBadge tier="sr8" />);
    expect(wrapper(container, "sr8").getAttribute("data-tier")).toBe("sr8");
  });

  test("inline background color differs per tier", () => {
    const { container, rerender } = render(<SellRuleBadge tier="sr1" />);
    const bgSr1 = wrapper(container, "sr1").getAttribute("style") || "";
    rerender(<SellRuleBadge tier="sr11" />);
    const bgSr11 = wrapper(container, "sr11").getAttribute("style") || "";
    rerender(<SellRuleBadge tier="sr8" />);
    const bgSr8 = wrapper(container, "sr8").getAttribute("style") || "";
    expect(bgSr1).not.toEqual(bgSr11);
    expect(bgSr11).not.toEqual(bgSr8);
    expect(bgSr1).not.toEqual(bgSr8);
  });

  // Family-stripe additions (2026-08-07). Every tier that has a
  // matching rule in SELL_RULES gets a colored stripe on the left
  // edge; the color comes from SELL_RULE_FAMILIES for that rule's
  // family. Locks the "at-a-glance grouping" contract Option B ships.

  test("data-family attribute reflects the rule's family", () => {
    const { container, rerender } = render(<SellRuleBadge tier="sr1" />);
    expect(wrapper(container, "sr1").getAttribute("data-family")).toBe("defense");

    rerender(<SellRuleBadge tier="sr8" />);
    expect(wrapper(container, "sr8").getAttribute("data-family")).toBe("monster");

    rerender(<SellRuleBadge tier="sr11" />);
    expect(wrapper(container, "sr11").getAttribute("data-family")).toBe("floor");

    rerender(<SellRuleBadge tier="sr15" />);
    expect(wrapper(container, "sr15").getAttribute("data-family")).toBe("floor");
  });

  test("renders a family-colored stripe alongside the label", () => {
    const { container } = render(<SellRuleBadge tier="sr8" />);
    const stripe = container.querySelector<HTMLElement>('[data-testid="family-stripe"]');
    expect(stripe).not.toBeNull();
    const monster = SELL_RULE_FAMILIES.find((f) => f.key === "monster")!;
    // JSDOM normalizes hex → rgb() in the style string, so match on the
    // rgb triple derived from monster.color rather than the raw hex.
    const styleStr = (stripe!.getAttribute("style") || "").toLowerCase();
    const r = parseInt(monster.color.slice(1, 3), 16);
    const g = parseInt(monster.color.slice(3, 5), 16);
    const b = parseInt(monster.color.slice(5, 7), 16);
    expect(styleStr).toContain(`rgb(${r}, ${g}, ${b})`);
  });

  test("title mentions the family name for quick grouping context", () => {
    const { container } = render(<SellRuleBadge tier="sr7" />);
    const title = wrapper(container, "sr7").getAttribute("title") || "";
    expect(title).toContain("Trend break");
  });
});
