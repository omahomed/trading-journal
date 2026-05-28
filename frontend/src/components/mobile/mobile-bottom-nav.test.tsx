import { describe, test, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

let mockPathname = "/dashboard";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

import { MobileBottomNav } from "./mobile-bottom-nav";

function setPathname(p: string) {
  mockPathname = p;
}

describe("MobileBottomNav — Phase 2 T2-6 5-destination layout", () => {
  test("renders exactly 5 destinations in order: Dashboard, Sizer, Journal, Daily, More", () => {
    setPathname("/dashboard");
    render(<MobileBottomNav />);
    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(5);
    const labels = links.map((l) => l.getAttribute("aria-label"));
    expect(labels).toEqual(["Dashboard", "Sizer", "Journal", "Daily", "More"]);
  });

  test("Cycle destination is still removed (regression)", () => {
    setPathname("/dashboard");
    render(<MobileBottomNav />);
    expect(screen.queryByRole("link", { name: "Cycle" })).not.toBeInTheDocument();
  });

  test("active destination matches current pathname (/dashboard)", () => {
    setPathname("/dashboard");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Dashboard" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("link", { name: "Daily" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  test("Daily tab href is /daily-journal", () => {
    setPathname("/dashboard");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).toHaveAttribute(
      "href",
      "/daily-journal",
    );
  });
});

describe("MobileBottomNav — Daily tab active across daily-workflow routes", () => {
  test("Daily tab is active on /daily-journal", () => {
    setPathname("/daily-journal");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  test("Daily tab is active on /daily-routine", () => {
    setPathname("/daily-routine");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  test("Daily tab is active on /weekly-retro", () => {
    setPathname("/weekly-retro");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  test("Daily tab is active on /daily-report (queried date)", () => {
    // The Daily Report detail view is reached via /daily-report?date=...
    // — usePathname() returns just "/daily-report".
    setPathname("/daily-report");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  test("Daily tab is NOT active on /trade-journal (sibling tab)", () => {
    setPathname("/trade-journal");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).not.toHaveAttribute(
      "aria-current",
    );
    expect(screen.getByRole("link", { name: "Journal" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  test("Daily tab is NOT active on /dashboard", () => {
    setPathname("/dashboard");
    render(<MobileBottomNav />);
    expect(screen.getByRole("link", { name: "Daily" })).not.toHaveAttribute(
      "aria-current",
    );
  });
});
