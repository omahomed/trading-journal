import { describe, test, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("next/navigation", () => ({
  usePathname: () => "/dashboard",
}));

import { MobileBottomNav } from "./mobile-bottom-nav";

describe("MobileBottomNav — Phase 2 4-destination layout", () => {
  test("renders exactly 4 destinations: Dashboard, Sizer, Journal, More", () => {
    render(<MobileBottomNav />);
    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(4);
    const labels = links.map((l) => l.getAttribute("aria-label"));
    expect(labels).toEqual(["Dashboard", "Sizer", "Journal", "More"]);
  });

  test("Cycle destination is removed", () => {
    render(<MobileBottomNav />);
    expect(screen.queryByRole("link", { name: "Cycle" })).not.toBeInTheDocument();
  });

  test("active destination matches current pathname (/dashboard)", () => {
    render(<MobileBottomNav />);
    const dashboard = screen.getByRole("link", { name: "Dashboard" });
    expect(dashboard).toHaveAttribute("aria-current", "page");
    const sizer = screen.getByRole("link", { name: "Sizer" });
    expect(sizer).not.toHaveAttribute("aria-current");
  });
});
