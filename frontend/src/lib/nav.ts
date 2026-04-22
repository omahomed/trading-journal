// Navigation structure — single source of truth for sidebar + command palette
export interface NavItem {
  id: string;
  label: string;
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
      { id: "dashboard", label: "Dashboard" },
      { id: "overview", label: "Trading Overview" },
    ],
  },
  {
    id: "ops", label: "Trading Ops", color: "#08a86b", softColor: "#e6f8ef",
    items: [
      { id: "campaign", label: "Active Campaign Summary" },
      { id: "import", label: "Import Trades" },
      { id: "logbuy", label: "Log Buy" },
      { id: "logsell", label: "Log Sell" },
      { id: "sizer", label: "Position Sizer" },
      { id: "sizer:normal", label: "Normal Sizer", parentPage: "sizer", tab: "normal" },
      { id: "sizer:volatility", label: "Volatility Sizer", parentPage: "sizer", tab: "volatility" },
      { id: "sizer:scalein", label: "Scale In Sizer", parentPage: "sizer", tab: "scalein" },
      { id: "sizer:pyramid", label: "Pyramid Sizer", parentPage: "sizer", tab: "pyramid" },
      { id: "sizer:trim", label: "Trim / Sell Down", parentPage: "sizer", tab: "trim" },
      { id: "sizer:options", label: "Options Sizer", parentPage: "sizer", tab: "options" },
      { id: "journal", label: "Trade Journal" },
      { id: "manager", label: "Trade Manager" },
      { id: "manager:stops", label: "Stop Loss Adjustment", parentPage: "manager", tab: "stops" },
      { id: "manager:edit", label: "Edit Transaction", parentPage: "manager", tab: "edit" },
      { id: "manager:delete", label: "Delete Trade", parentPage: "manager", tab: "delete" },
      { id: "manager:export", label: "Export Trades", parentPage: "manager", tab: "export" },
    ],
  },
  {
    id: "risk", label: "Risk Management", color: "#e5484d", softColor: "#fdecec",
    items: [
      { id: "earnings", label: "Earnings Planner" },
      { id: "heat", label: "Portfolio Heat" },
      { id: "riskmgr", label: "Risk Manager" },
    ],
  },
  {
    id: "daily", label: "Daily Workflow", color: "#f59f00", softColor: "#fff4dd",
    items: [
      { id: "djournal", label: "Daily Journal" },
      { id: "report", label: "Daily Report" },
      { id: "routine", label: "Daily Routine" },
      { id: "retro", label: "Weekly Retro" },
    ],
  },
  {
    id: "market", label: "Market Intel", color: "#8b5cf6", softColor: "#f1ecfe",
    items: [
      { id: "mfactor", label: "M Factor" },
      { id: "cycle", label: "Market Cycle Tracker" },
      { id: "rally", label: "Rally Context" },
    ],
  },
  {
    id: "ai", label: "AI", color: "#0ea5a4", softColor: "#e0f5f4",
    items: [
      { id: "coach", label: "AI Coach" },
    ],
  },
  {
    id: "deep", label: "Deep Dive", color: "#0d6efd", softColor: "#e7f0ff",
    items: [
      { id: "analytics", label: "Analytics" },
      { id: "analytics:buyrules", label: "Buy Rules Analysis", parentPage: "analytics", tab: "buyrules" },
      { id: "analytics:sellrules", label: "Sell Rules Analysis", parentPage: "analytics", tab: "sellrules" },
      { id: "analytics:drawdown", label: "Drawdown Analysis", parentPage: "analytics", tab: "drawdown" },
      { id: "analytics:review", label: "Trade Review", parentPage: "analytics", tab: "review" },
      { id: "analytics:campaigns", label: "All Campaigns", parentPage: "analytics", tab: "campaigns" },
      { id: "heatmap", label: "Performance Heat Map" },
      { id: "period", label: "Period Review" },
      { id: "period:weekly", label: "Weekly Review", parentPage: "period", tab: "weekly" },
      { id: "period:monthly", label: "Monthly Review", parentPage: "period", tab: "monthly" },
      { id: "period:annual", label: "Annual & CAGR", parentPage: "period", tab: "annual" },
    ],
  },
  {
    id: "admin", label: "Account", color: "#0f1524", softColor: "#eceef3",
    items: [
      { id: "settings", label: "Settings" },
      { id: "admin", label: "Admin" },
    ],
  },
];

// Flatten for command palette
export function getAllPages() {
  return NAV.flatMap((g) =>
    g.items.map((i) => ({ ...i, group: g.label, color: g.color }))
  );
}

// Find which group a page belongs to
export function getGroupForPage(pageId: string): NavGroup | undefined {
  return NAV.find((g) => g.items.some((i) => i.id === pageId));
}
