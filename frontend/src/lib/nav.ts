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

export const NAV: NavGroup[] = [
  {
    id: "dashboards", label: "Dashboards", color: "#6366f1", softColor: "#eef0ff",
    items: [
      { id: "dashboard", label: "Dashboard", href: "/dashboard" },
      { id: "overview", label: "Trading Overview", href: "/overview" },
    ],
  },
  {
    id: "ops", label: "Trading Ops", color: "#08a86b", softColor: "#e6f8ef",
    items: [
      { id: "campaign", label: "Active Campaign Summary", href: "/active-campaign" },
      { id: "import", label: "Import Trades", href: "/import-trades" },
      { id: "logbuy", label: "Log Buy", href: "/log-buy" },
      { id: "logsell", label: "Log Sell", href: "/log-sell" },
      { id: "sizer", label: "Position Sizer", href: "/position-sizer" },
      { id: "sizer:normal", label: "Normal Sizer", parentPage: "sizer", tab: "normal" },
      { id: "sizer:volatility", label: "Volatility Sizer", parentPage: "sizer", tab: "volatility" },
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
    ],
  },
  {
    id: "daily", label: "Daily Workflow", color: "#f59f00", softColor: "#fff4dd",
    items: [
      { id: "djournal", label: "Daily Journal", href: "/daily-journal" },
      { id: "report", label: "Daily Report", href: "/daily-report" },
      { id: "routine", label: "Daily Routine", href: "/daily-routine" },
      { id: "retro", label: "Weekly Retro", href: "/weekly-retro" },
    ],
  },
  {
    id: "market", label: "Market Intel", color: "#8b5cf6", softColor: "#f1ecfe",
    items: [
      { id: "cycle", label: "M Factor", href: "/m-factor" },
      { id: "rally", label: "Rally Context", href: "/rally-context" },
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
      { id: "analytics", label: "Analytics", href: "/analytics" },
      { id: "analytics:buyrules", label: "Buy Rules Analysis", parentPage: "analytics", tab: "buyrules" },
      { id: "analytics:sellrules", label: "Sell Rules Analysis", parentPage: "analytics", tab: "sellrules" },
      { id: "analytics:drawdown", label: "Drawdown Analysis", parentPage: "analytics", tab: "drawdown" },
      { id: "analytics:review", label: "Trade Review", parentPage: "analytics", tab: "review" },
      { id: "analytics:campaigns", label: "All Campaigns", parentPage: "analytics", tab: "campaigns" },
      { id: "heatmap", label: "Performance Heat Map", href: "/performance-heatmap" },
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

// Flatten for command palette
export function getAllPages() {
  return NAV.flatMap((g) =>
    g.items.map((i) => ({ ...i, group: g.label, color: g.color }))
  );
}

// Find which group a page belongs to (by id OR by href)
export function getGroupForPage(pageId: string): NavGroup | undefined {
  return NAV.find((g) => g.items.some((i) => i.id === pageId));
}

// Given a pathname like "/log-buy", find the nav item (and therefore group).
export function getGroupForHref(href: string): NavGroup | undefined {
  // Trim trailing slash and query string for matching.
  const clean = href.split("?")[0].replace(/\/$/, "") || "/";
  return NAV.find((g) => g.items.some((i) => i.href === clean));
}

// Find the nav item matching a pathname (top-level page, not sub-tab).
export function getNavItemForHref(href: string): NavItem | undefined {
  const clean = href.split("?")[0].replace(/\/$/, "") || "/";
  for (const g of NAV) {
    const hit = g.items.find((i) => i.href === clean);
    if (hit) return hit;
  }
  return undefined;
}

// Look up the URL for a nav id (used by cross-page router.push).
export function hrefForId(id: string): string | undefined {
  for (const g of NAV) {
    const hit = g.items.find((i) => i.id === id);
    if (hit?.href) return hit.href;
  }
  return undefined;
}
