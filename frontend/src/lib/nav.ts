// Navigation structure — single source of truth for sidebar + command palette
export interface NavItem {
  id: string;
  label: string;
  /** URL path for this page. Sub-tab items omit it and use parentPage + tab. */
  href?: string;
  /** If set, this is a sub-page: navigate to parentPage and pass this tab key */
  parentPage?: string;
  tab?: string;
}

export interface NavGroup {
  id: string;
  label: string;
  color: string;
  softColor: string;
  items: NavItem[];
}

// Top-level items — first-class nav links that render ABOVE the Pinned
// section in the sidebar. Not part of any NavGroup; they're the sidebar's
// "hub" landing surfaces. Add sparingly — every entry here pays for itself
// in permanent sidebar real estate. Item ids must be unique across
// TOP_LEVEL_ITEMS and NAV combined.
export const TOP_LEVEL_ITEMS: NavItem[] = [
  { id: "cmdcenter", label: "Command Center", href: "/command-center" },
];

// Accent color for top-level items — same slot in the visual language
// getGroupForHref() returns for grouped items. Slate reads as "hub /
// operations HQ" and doesn't collide with any existing group hue.
export const TOP_LEVEL_COLOR = "#334155";
export const TOP_LEVEL_SOFT = "#e2e8f0";

export const NAV: NavGroup[] = [
  {
    id: "dashboards", label: "Dashboards", color: "#6366f1", softColor: "#eef0ff",
    items: [
      { id: "dashboard", label: "Dashboard", href: "/dashboard" },
      { id: "overview", label: "Trading Overview", href: "/overview" },
      { id: "realized", label: "Realized Equity", href: "/realized-equity" },
    ],
  },
  {
    id: "ops", label: "Trading Ops", color: "#08a86b", softColor: "#e6f8ef",
    items: [
      { id: "campaign", label: "Campaign Summary", href: "/active-campaign" },
      { id: "campaign-detail", label: "Campaign Detail", href: "/campaign-detail" },
      { id: "import", label: "Import Trades", href: "/import-trades" },
      { id: "logbuy", label: "Log Buy", href: "/log-buy" },
      { id: "logsell", label: "Log Sell", href: "/log-sell" },
      { id: "sizer", label: "Position Sizer", href: "/position-sizer" },
      { id: "sizer:volatility", label: "New Entry", parentPage: "sizer", tab: "volatility" },
      { id: "sizer:scalein", label: "Scale In Sizer", parentPage: "sizer", tab: "scalein" },
      { id: "sizer:pyramid", label: "Pyramid Sizer", parentPage: "sizer", tab: "pyramid" },
      { id: "sizer:trim", label: "Trim / Sell Down", parentPage: "sizer", tab: "trim" },
      { id: "sizer:options", label: "Options Sizer", parentPage: "sizer", tab: "options" },
      { id: "journal", label: "Trade Journal", href: "/trade-journal" },
      { id: "manager", label: "Trade Manager", href: "/trade-manager" },
      { id: "manager:stops", label: "Stop Loss Adjustment", parentPage: "manager", tab: "stops" },
      { id: "manager:edit", label: "Edit Transaction", parentPage: "manager", tab: "edit" },
      { id: "manager:delete", label: "Delete Trade", parentPage: "manager", tab: "delete" },
      { id: "manager:export", label: "Export Trades", parentPage: "manager", tab: "export" },
    ],
  },
  {
    id: "risk", label: "Risk Management", color: "#e5484d", softColor: "#fdecec",
    items: [
      { id: "earnings", label: "Earnings Planner", href: "/earnings" },
      { id: "heat", label: "Portfolio Heat", href: "/portfolio-heat" },
      { id: "riskmgr", label: "Risk Manager", href: "/risk-manager" },
      { id: "sr8mon", label: "SR8 Monitor", href: "/sr8-monitor" },
    ],
  },
  {
    id: "allocation", label: "Allocation", color: "#0891b2", softColor: "#e0f5f9",
    items: [
      { id: "slices", label: "Slices", href: "/slices" },
      { id: "conc", label: "Concentration Risk", href: "/concentration-risk" },
      { id: "sectormap", label: "Sector Mapping", href: "/sector-mapping" },
    ],
  },
  {
    id: "daily", label: "Daily Workflow", color: "#f59f00", softColor: "#fff4dd",
    items: [
      { id: "djournal", label: "Daily Journal", href: "/daily-journal" },
      { id: "jlog", label: "Journal Log", href: "/journal-log" },
      { id: "nlventry", label: "NLV Entry", href: "/nlv-entry" },
      { id: "retro", label: "Weekly Retro", href: "/weekly-retro" },
    ],
  },
  {
    id: "market", label: "Market Intel", color: "#8b5cf6", softColor: "#f1ecfe",
    items: [
      { id: "cycle", label: "M Factor", href: "/m-factor" },
      { id: "rally", label: "Rally Context", href: "/rally-context" },
      { id: "tcmeth", label: "Trend Cycle Methodology", href: "/trend-cycle-methodology" },
    ],
  },
  {
    id: "ai", label: "AI", color: "#0ea5a4", softColor: "#e0f5f4",
    items: [
      { id: "coach", label: "AI Coach", href: "/ai-coach" },
    ],
  },
  {
    id: "deep", label: "Deep Dive", color: "#0d6efd", softColor: "#e7f0ff",
    items: [
      // Renamed to "Edge Report" — route stays /analytics for backward
      // compat (existing bookmarks + deep links keep working).
      { id: "analytics", label: "Edge Report", href: "/analytics" },
      { id: "analytics:scenarios", label: "Setup Scorecard", parentPage: "analytics", tab: "scenarios" },
      { id: "analytics:buyrules", label: "Buy Rules Analysis", parentPage: "analytics", tab: "buyrules" },
      { id: "analytics:sellrules", label: "Sell Rules Analysis", parentPage: "analytics", tab: "sellrules" },
      { id: "analytics:drawdown", label: "Drawdown Analysis", parentPage: "analytics", tab: "drawdown" },
      { id: "analytics:review", label: "Trade Review", parentPage: "analytics", tab: "review" },
      { id: "analytics:campaigns", label: "All Campaigns", parentPage: "analytics", tab: "campaigns" },
      { id: "analytics:add-effectiveness", label: "Add effectiveness", parentPage: "analytics", tab: "add-effectiveness" },
      { id: "heatmap", label: "Performance Heat Map", href: "/performance-heatmap" },
      { id: "campaign-review", label: "Campaign Review", href: "/campaign-review" },
      { id: "trend-cycle-review", label: "Trend Cycle Review", href: "/trend-cycle-review" },
      { id: "trader-mindset", label: "Trader Mindset", href: "/trader-mindset" },
      { id: "period", label: "Period Review", href: "/period-review" },
      { id: "period:weekly", label: "Weekly Review", parentPage: "period", tab: "weekly" },
      { id: "period:monthly", label: "Monthly Review", parentPage: "period", tab: "monthly" },
      { id: "period:annual", label: "Annual & CAGR", parentPage: "period", tab: "annual" },
    ],
  },
  {
    id: "admin", label: "Account", color: "#0f1524", softColor: "#eceef3",
    items: [
      { id: "settings", label: "Settings", href: "/settings" },
      { id: "admin", label: "Admin", href: "/admin" },
    ],
  },
];

// Synthetic group returned by getGroupForHref for top-level items so
// page components can pull a navColor uniformly (the components don't
// need to know whether the current page is grouped or top-level).
const TOP_LEVEL_GROUP: NavGroup = {
  id: "top", label: "Overview",
  color: TOP_LEVEL_COLOR, softColor: TOP_LEVEL_SOFT,
  items: TOP_LEVEL_ITEMS,
};

// Flatten for command palette
export function getAllPages() {
  const top = TOP_LEVEL_ITEMS.map((i) => ({
    ...i, group: TOP_LEVEL_GROUP.label, color: TOP_LEVEL_COLOR,
  }));
  const grouped = NAV.flatMap((g) =>
    g.items.map((i) => ({ ...i, group: g.label, color: g.color }))
  );
  return [...top, ...grouped];
}

// Find which group a page belongs to (by id OR by href)
export function getGroupForPage(pageId: string): NavGroup | undefined {
  if (TOP_LEVEL_ITEMS.some((i) => i.id === pageId)) return TOP_LEVEL_GROUP;
  return NAV.find((g) => g.items.some((i) => i.id === pageId));
}

// Given a pathname like "/log-buy", find the nav item (and therefore group).
export function getGroupForHref(href: string): NavGroup | undefined {
  // Trim trailing slash and query string for matching.
  const clean = href.split("?")[0].replace(/\/$/, "") || "/";
  if (TOP_LEVEL_ITEMS.some((i) => i.href === clean)) return TOP_LEVEL_GROUP;
  return NAV.find((g) => g.items.some((i) => i.href === clean));
}

// Find the nav item matching a pathname (top-level page, not sub-tab).
export function getNavItemForHref(href: string): NavItem | undefined {
  const clean = href.split("?")[0].replace(/\/$/, "") || "/";
  const topHit = TOP_LEVEL_ITEMS.find((i) => i.href === clean);
  if (topHit) return topHit;
  for (const g of NAV) {
    const hit = g.items.find((i) => i.href === clean);
    if (hit) return hit;
  }
  return undefined;
}

// Look up the URL for a nav id (used by cross-page router.push).
export function hrefForId(id: string): string | undefined {
  const topHit = TOP_LEVEL_ITEMS.find((i) => i.id === id);
  if (topHit?.href) return topHit.href;
  for (const g of NAV) {
    const hit = g.items.find((i) => i.id === id);
    if (hit?.href) return hit.href;
  }
  return undefined;
}
