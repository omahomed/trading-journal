// API client for the FastAPI backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// Types
export interface JournalEntry {
  day: string;
  end_nlv: number;
  beg_nlv: number;
  daily_dollar_change: number;
  daily_pct_change: number;
  pct_invested: number;
  spy: number;
  nasdaq: number;
  portfolio_heat: number;
  score: number;
  market_window: string;
  [key: string]: any;
}

export interface JournalHistoryPoint {
  day: string;
  end_nlv: number;
  daily_pct_change: number;
  portfolio_ltd: number;
  spy_ltd: number;
  ndx_ltd: number;
  pct_invested: number;
  portfolio_heat: number;
  [key: string]: any;
}

export interface TradePosition {
  trade_id: string;
  ticker: string;
  status: string;
  shares: number;
  avg_entry: number;
  total_cost: number;
  realized_pl: number;
  rule: string;
  [key: string]: any;
}

export interface TradeDetail {
  trade_id: string;
  ticker: string;
  action: string;
  date: string;
  shares: number;
  amount: number;
  value: number;
  rule: string;
  [key: string]: any;
}

// API functions
export const api = {
  // Journal
  journalLatest: (portfolio = "CanSlim") =>
    fetchJSON<JournalEntry>(`/api/journal/latest?portfolio=${portfolio}`),

  journalHistory: (portfolio = "CanSlim", days = 365) =>
    fetchJSON<JournalHistoryPoint[]>(`/api/journal/history?portfolio=${portfolio}&days=${days}`),

  // Trades
  tradesOpen: (portfolio = "CanSlim") =>
    fetchJSON<TradePosition[]>(`/api/trades/open?portfolio=${portfolio}`),

  tradesClosed: (portfolio = "CanSlim", limit = 50) =>
    fetchJSON<TradePosition[]>(`/api/trades/closed?portfolio=${portfolio}&limit=${limit}`),

  tradeDetails: (tradeId: string, portfolio = "CanSlim") =>
    fetchJSON<TradeDetail[]>(`/api/trades/details/${tradeId}?portfolio=${portfolio}`),

  tradesOpenDetails: (portfolio = "CanSlim") =>
    fetchJSON<TradeDetail[]>(`/api/trades/open/details?portfolio=${portfolio}`),

  tradesRecent: (portfolio = "CanSlim", limit = 20) =>
    fetchJSON<TradeDetail[]>(`/api/trades/recent?portfolio=${portfolio}&limit=${limit}`),

  // Trade lessons
  getTradeLessons: (portfolio = "CanSlim") =>
    fetchJSON<{ lessons: Record<string, { note: string; category: string }> }>(`/api/trades/lessons?portfolio=${portfolio}`),

  saveTradeLessons: (entry: { portfolio: string; trade_id: string; note: string; category: string }) =>
    fetch(`${API_BASE}/api/trades/lessons`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    }).then(r => r.json()) as Promise<{ status: string }>,

  // Journal write
  journalEdit: (entry: Record<string, any>) =>
    fetch(`${API_BASE}/api/journal/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    }).then(r => r.json()) as Promise<{ status: string; id?: number; detail?: string }>,

  // Prices
  priceLookup: (ticker: string) =>
    fetchJSON<{ ticker: string; price: number; atr: number; atr_pct: number }>(`/api/prices/lookup?ticker=${encodeURIComponent(ticker)}`),

  batchPrices: (tickers: string[]) =>
    fetchJSON<Record<string, number>>(`/api/prices/batch?tickers=${encodeURIComponent(tickers.join(","))}`),

  // Market
  rallyPrefix: () => fetchJSON<{ prefix: string; day_num?: number; state?: string }>(`/api/market/rally-prefix`),
  mfactor: () => fetchJSON<any>(`/api/market/mfactor`),

  // Config
  config: (key: string) => fetchJSON<{ key: string; value: any }>(`/api/config/${key}`),

  // Admin — config
  setConfig: (key: string, value: any, opts?: { value_type?: string; category?: string; description?: string }) =>
    fetch(`${API_BASE}/api/config/${key}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value, ...opts, user: "admin" }),
    }).then(r => r.json()) as Promise<{ status: string; detail?: string }>,

  // Admin — events
  events: (scope = "CanSlim") => fetchJSON<any[]>(`/api/events?scope=${scope}`),
  addEvent: (body: Record<string, any>) =>
    fetch(`${API_BASE}/api/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json()),
  updateEvent: (id: number, body: Record<string, any>) =>
    fetch(`${API_BASE}/api/events/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json()),
  deleteEvent: (id: number) =>
    fetch(`${API_BASE}/api/events/${id}`, { method: "DELETE" }).then(r => r.json()),

  // Admin — audit
  audit: (limit = 100, actionFilter?: string) =>
    fetchJSON<any[]>(`/api/audit?limit=${limit}${actionFilter ? `&action_filter=${actionFilter}` : ""}`),

  // Admin — cleanup
  cleanupMarketsurge: (dryRun = true) =>
    fetch(`${API_BASE}/api/admin/cleanup-marketsurge`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dry_run: dryRun }),
    }).then(r => r.json()),

  // AI Coach
  coachChat: (message: string, preset?: string, portfolio = "CanSlim") => {
    return fetch(`${API_BASE}/api/coach/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, preset, portfolio }),
    });
  },

  // Health
  health: () => fetchJSON<{ status: string; timestamp: string }>(`/api/health`),
};
