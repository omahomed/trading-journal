import { render, screen } from "@testing-library/react";
import { describe, test, expect } from "vitest";
import { SellRuleBadge } from "./sell-rule-badge";
import { SELL_RULES } from "@/lib/trade-rules";

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
    render(<SellRuleBadge tier="sr8" />);
    const badge = screen.getByText("SR8");
    const title = badge.getAttribute("title") || "";
    const rule = SELL_RULES.find((r) => r.code === "sr8")!;
    expect(title).toContain("SR8");
    expect(title).toContain(rule.description);
    expect(title).toContain(rule.oneLiner);
  });

  test("data-tier attribute set per tier (color-variant probe)", () => {
    const { rerender } = render(<SellRuleBadge tier="sr1" />);
    expect(screen.getByText("SR1").getAttribute("data-tier")).toBe("sr1");

    rerender(<SellRuleBadge tier="sr11" />);
    expect(screen.getByText("SR11").getAttribute("data-tier")).toBe("sr11");

    rerender(<SellRuleBadge tier="sr8" />);
    expect(screen.getByText("SR8").getAttribute("data-tier")).toBe("sr8");
  });

  test("inline background color differs per tier", () => {
    const { rerender } = render(<SellRuleBadge tier="sr1" />);
    const bgSr1 = screen.getByText("SR1").getAttribute("style") || "";
    rerender(<SellRuleBadge tier="sr11" />);
    const bgSr11 = screen.getByText("SR11").getAttribute("style") || "";
    rerender(<SellRuleBadge tier="sr8" />);
    const bgSr8 = screen.getByText("SR8").getAttribute("style") || "";
    expect(bgSr1).not.toEqual(bgSr11);
    expect(bgSr11).not.toEqual(bgSr8);
    expect(bgSr1).not.toEqual(bgSr8);
  });
});
