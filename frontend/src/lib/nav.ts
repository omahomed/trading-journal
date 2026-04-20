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
      { id: "heatmap", label: "Performance Heat Map" },
      { id: "period", label: "Period Review" },
    ],
  },
  {
    id: "admin", label: "Admin", color: "#0f1524", softColor: "#eceef3",
    items: [
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
