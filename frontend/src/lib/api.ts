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

  tradesRecent: (portfolio = "CanSlim", limit = 20) =>
    fetchJSON<TradeDetail[]>(`/api/trades/recent?portfolio=${portfolio}&limit=${limit}`),

  // Market
  mfactor: () => fetchJSON<any>(`/api/market/mfactor`),

  // Config
  config: (key: string) => fetchJSON<{ key: string; value: any }>(`/api/config/${key}`),

  // Health
  health: () => fetchJSON<{ status: string; timestamp: string }>(`/api/health`),
};
