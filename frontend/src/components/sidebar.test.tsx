import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { describe, test, expect, vi, beforeEach } from "vitest";

class ResizeObserverStub {
  observe(): void { /* noop */ }
  unobserve(): void { /* noop */ }
  disconnect(): void { /* noop */ }
}
(globalThis as any).ResizeObserver = ResizeObserverStub;

// JSDOM ships with a localStorage stub that throws on most methods. Replace
// with an in-memory impl so PortfolioProvider's localStorage read on mount
// doesn't crash. Sidebar doesn't read localStorage directly, but
// usePortfolio()'s setActive path touches window.localStorage if invoked.
const _lsStore = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (k: string) => _lsStore.get(k) ?? null,
    setItem: (k: string, v: string) => { _lsStore.set(k, v); },
    removeItem: (k: string) => { _lsStore.delete(k); },
    clear: () => { _lsStore.clear(); },
    key: (i: number) => Array.from(_lsStore.keys())[i] ?? null,
    get length() { return _lsStore.size; },
  },
});

// next/link renders a plain <a>; next/navigation's usePathname needs a
// per-test value so the active-state assertions can target specific routes.
let _mockPathname = "/dashboard";
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(), replace: vi.fn(), refresh: vi.fn(),
    back: vi.fn(), forward: vi.fn(), prefetch: vi.fn(),
  }),
  usePathname: () => _mockPathname,
}));

vi.mock("@/lib/api", () => ({
  api: {
    pinnedRoutesList: vi.fn(),
    pinnedRoutesToggle: vi.fn(),
    listPortfolios: vi.fn(),
  },
  setActivePortfolio: vi.fn(),
}));

// Stub the portfolio context. The real provider fires an api.listPortfolios
// call and triggers a window.location.reload on setActive — neither is in
// scope for these sidebar tests, so we shortcut with a fixed return value.
vi.mock("@/lib/portfolio-context", () => ({
  usePortfolio: () => ({
    portfolios: [{ id: 1, name: "CanSlim" }],
    activePortfolio: { id: 1, name: "CanSlim" },
    loading: false,
    error: null,
    refetch: vi.fn(),
    setActive: vi.fn(),
  }),
}));


import { api } from "@/lib/api";
import { Sidebar } from "./sidebar";


function mockPins(routes: { route_path: string; pinned_at: string }[]) {
  vi.mocked(api.pinnedRoutesList).mockResolvedValue({ routes });
}


describe("Sidebar — Pinned routes (Commit 2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _mockPathname = "/dashboard";
    vi.mocked(api.pinnedRoutesToggle).mockResolvedValue({ pinned: true });
  });

  test("renders nav without Pinned section when no pins exist", async () => {
    mockPins([]);
    render(<Sidebar />);

    // Wait for the pinnedRoutesList fetch to settle. The section should
    // never appear when the fetched list is empty.
    await waitFor(() => expect(api.pinnedRoutesList).toHaveBeenCalled());
    expect(screen.queryByTestId("sidebar-pinned-section")).not.toBeInTheDocument();
    // Regular NAV still renders — Dashboard is a known group.
    expect(screen.getByText("Dashboards")).toBeInTheDocument();
  });

  test("renders Pinned section above NAV when pins exist", async () => {
    mockPins([
      { route_path: "/log-buy", pinned_at: "2026-05-10T09:00:00Z" },
      { route_path: "/active-campaign", pinned_at: "2026-05-11T09:00:00Z" },
    ]);
    render(<Sidebar />);

    const pinnedSection = await screen.findByTestId("sidebar-pinned-section");
    expect(pinnedSection).toBeInTheDocument();
    expect(pinnedSection.textContent).toContain("Pinned");
    // Count badge surfaces 2.
    expect(pinnedSection.textContent).toContain("2");

    // The pinned section renders BEFORE the first NAV group header
    // ("Dashboards"). Compare DOM order.
    const dashboardsHeader = screen.getByText("Dashboards");
    const sectionRect = pinnedSection.compareDocumentPosition(dashboardsHeader);
    // DOCUMENT_POSITION_FOLLOWING = 4 → dashboardsHeader follows the section.
    expect(sectionRect & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  test("pin button hidden on inactive item by default, revealed on hover via group-hover class", async () => {
    mockPins([]);
    render(<Sidebar />);
    await waitFor(() => expect(api.pinnedRoutesList).toHaveBeenCalled());

    // Find any regular nav item's pin button. Group items are inside
    // collapsible group panels — Dashboards opens by default because
    // activePage=/dashboard puts it in the open-groups set.
    const buttons = await screen.findAllByTestId("sidebar-pin-btn");
    expect(buttons.length).toBeGreaterThan(0);

    // Default (unpinned) pin button uses the group-hover class to reveal on
    // hover. Verify the class is on the element and that aria-label says "Pin".
    const unpinnedBtn = buttons.find(b => b.getAttribute("data-pinned") === "false");
    expect(unpinnedBtn).toBeDefined();
    expect(unpinnedBtn!.className).toContain("group-hover/navitem:opacity-100");
    expect(unpinnedBtn!.className).toContain("opacity-0");
    expect(unpinnedBtn!.getAttribute("aria-label")).toBe("Pin");
  });

  test("pin button always visible (opacity-100) on a pinned nav item", async () => {
    mockPins([{ route_path: "/log-buy", pinned_at: "2026-05-10T09:00:00Z" }]);
    render(<Sidebar />);
    await screen.findByTestId("sidebar-pinned-section");

    // Find the pin button inside the pinned-section's /log-buy entry. The
    // pinned section's items have data-pinned="true" on the Link, and so
    // do the matching items in the regular NAV.
    const pinnedButtons = (await screen.findAllByTestId("sidebar-pin-btn"))
      .filter(b => b.getAttribute("data-pinned") === "true");
    // There are TWO data-pinned="true" buttons: one in the pinned section
    // (the dedicated row) and one in the regular NAV group's /log-buy entry.
    // Both must show opacity-100 (always-visible) and have aria-label="Unpin".
    expect(pinnedButtons.length).toBe(2);
    for (const btn of pinnedButtons) {
      expect(btn.className).toContain("opacity-100");
      expect(btn.className).not.toContain("opacity-0");
      expect(btn.getAttribute("aria-label")).toBe("Unpin");
    }
  });

  test("clicking pin sends toggle request and optimistically flips state", async () => {
    mockPins([]);
    render(<Sidebar />);
    await waitFor(() => expect(api.pinnedRoutesList).toHaveBeenCalled());

    // Pick the unpinned button for /log-buy inside the regular Trading Ops
    // group. The group needs to be open — it expands on demand via altKey,
    // but Trading Ops contains /log-buy and Dashboards is auto-opened on
    // mount. Click the Trading Ops header to open it first.
    await act(async () => {
      fireEvent.click(screen.getByText("Trading Ops"));
    });
    // The /log-buy item is now rendered. Find its pin button.
    const logBuyLink = screen.getAllByTestId("sidebar-nav-item")
      .find(el => el.getAttribute("data-href") === "/log-buy");
    expect(logBuyLink).toBeDefined();
    const pinBtn = logBuyLink!.querySelector('[data-testid="sidebar-pin-btn"]') as HTMLElement;
    expect(pinBtn).toBeTruthy();
    expect(pinBtn.getAttribute("data-pinned")).toBe("false");

    await act(async () => { fireEvent.click(pinBtn); });

    // Optimistic flip: pinned section appears immediately (without
    // waiting for the toggle round-trip).
    await waitFor(() => {
      expect(screen.getByTestId("sidebar-pinned-section")).toBeInTheDocument();
    });
    // Toggle endpoint called with the correct route_path.
    expect(api.pinnedRoutesToggle).toHaveBeenCalledWith("/log-buy");
  });

  test("active state applies to BOTH the pinned-section entry AND the regular group entry when pathname matches", async () => {
    _mockPathname = "/log-buy";
    mockPins([{ route_path: "/log-buy", pinned_at: "2026-05-10T09:00:00Z" }]);
    render(<Sidebar />);
    await screen.findByTestId("sidebar-pinned-section");

    // Both Link instances with href=/log-buy should have data-active="true".
    const logBuyLinks = screen.getAllByTestId("sidebar-nav-item")
      .filter(el => el.getAttribute("data-href") === "/log-buy");
    expect(logBuyLinks.length).toBe(2);  // pinned section + Trading Ops
    for (const link of logBuyLinks) {
      expect(link.getAttribute("data-active")).toBe("true");
    }
  });

  test("Pinned section NOT rendered in rail mode (collapsed sidebar)", async () => {
    mockPins([{ route_path: "/log-buy", pinned_at: "2026-05-10T09:00:00Z" }]);
    render(<Sidebar rail={true} />);
    await waitFor(() => expect(api.pinnedRoutesList).toHaveBeenCalled());

    // Pinned section hidden in rail mode by design — the rail collapses
    // nav to colored dots per group and a "Pinned" group has no natural
    // single-color representation.
    expect(screen.queryByTestId("sidebar-pinned-section")).not.toBeInTheDocument();
    // Per-item pin buttons also gone — the rail-mode render is a dot per
    // group, no item-level Link wrappers.
    expect(screen.queryByTestId("sidebar-pin-btn")).not.toBeInTheDocument();
  });
});
