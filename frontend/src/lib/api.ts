// API client for the FastAPI backend
// NEXT_PUBLIC_API_URL is baked in at build time. If not set, detect environment.
// Production fallback points at the canonical Railway backend (lucky-adaptation
// → web service). Set NEXT_PUBLIC_API_URL on Vercel to override.
import { getSession } from "next-auth/react";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL
  || (typeof window !== "undefined" && window.location.hostname !== "localhost"
      ? "https://web-production-ad135.up.railway.app"
      : "http://localhost:8000");

// fetchWithAuth: drop-in replacement for fetch() that attaches the Bearer token
// from the current next-auth session. Cached in-memory by next-auth so the
// /api/auth/session round-trip is only paid once per few minutes.
export async function fetchWithAuth(input: string, init?: RequestInit): Promise<Response> {
  const session = await getSession();
  const headers = new Headers(init?.headers);
  if (session?.apiToken) headers.set("Authorization", `Bearer ${session.apiToken}`);
  return fetch(input, { ...init, headers });
}

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetchWithAuth(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// Active portfolio — single global value that PortfolioProvider keeps in sync
// with the user's selection. API defaults read this via getActivePortfolio()
// so handlers don't have to thread a prop through every component. Empty string
// until the provider has fetched + set it; callers that fire before that point
// will pass "" to the backend and get a "portfolio not found" error (which is
// correct — we should never fire API calls before the onboarding gate clears).
let _activePortfolio = "";
export function setActivePortfolio(name: string): void {
  _activePortfolio = name;
}
export function getActivePortfolio(): string {
  return _activePortfolio;
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
  grade?: number | null;
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

export interface Portfolio {
  id: number;
  name: string;
  starting_capital: number | null;
  reset_date: string | null;
  created_at: string;
  cash_balance: number;
}

export interface PortfolioInput {
  name?: string;
  starting_capital?: number | null;
  reset_date?: string | null;
}

export interface CashTransaction {
  id: number;
  portfolio_id: number;
  date: string;
  amount: number;
  source: "deposit" | "withdraw" | "buy" | "sell" | "reconcile";
  trade_detail_id: number | null;
  note: string | null;
  created_at?: string;
}

export type CashAction = "deposit" | "withdraw" | "reconcile";

// API functions
export const api = {
  // Journal
  journalLatest: (portfolio = getActivePortfolio()) =>
    fetchJSON<JournalEntry>(`/api/journal/latest?portfolio=${portfolio}`),

  journalHistory: (portfolio = getActivePortfolio(), days = 365) =>
    fetchJSON<JournalHistoryPoint[]>(`/api/journal/history?portfolio=${portfolio}&days=${days}`),

  // Trades
  tradesOpen: (portfolio = getActivePortfolio()) =>
    fetchJSON<TradePosition[]>(`/api/trades/open?portfolio=${portfolio}`),

  tradesClosed: (portfolio = getActivePortfolio(), limit = 50) =>
    fetchJSON<TradePosition[]>(`/api/trades/closed?portfolio=${portfolio}&limit=${limit}`),

  tradeDetails: (tradeId: string, portfolio = getActivePortfolio()) =>
    fetchJSON<TradeDetail[]>(`/api/trades/details/${tradeId}?portfolio=${portfolio}`),

  tradesOpenDetails: (portfolio = getActivePortfolio()) =>
    fetchJSON<TradeDetail[]>(`/api/trades/open/details?portfolio=${portfolio}`),

  tradesRecent: (portfolio = getActivePortfolio(), limit = 20) =>
    fetchJSON<TradeDetail[]>(`/api/trades/recent?portfolio=${portfolio}&limit=${limit}`),

  setTradeGrade: (body: { portfolio?: string; trade_id: string; grade: number | null }) =>
    fetchWithAuth(`${API_BASE}/api/trades/grade`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{ status?: string; error?: string; trade_id?: string; grade?: number | null }>,

  // Trade lessons
  getTradeLessons: (portfolio = getActivePortfolio()) =>
    fetchJSON<{ lessons: Record<string, { note: string; category: string }> }>(`/api/trades/lessons?portfolio=${portfolio}`),

  saveTradeLessons: (entry: { portfolio: string; trade_id: string; note: string; category: string }) =>
    fetchWithAuth(`${API_BASE}/api/trades/lessons`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    }).then(r => r.json()) as Promise<{ status: string }>,

  // Journal write
  journalEdit: (entry: Record<string, any>) =>
    fetchWithAuth(`${API_BASE}/api/journal/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    }).then(r => r.json()) as Promise<{ status: string; id?: number; detail?: string }>,

  journalDelete: (day: string, portfolio = getActivePortfolio()) =>
    fetchWithAuth(`${API_BASE}/api/journal/delete?portfolio=${encodeURIComponent(portfolio)}&day=${encodeURIComponent(day)}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status: string; id?: number; detail?: string }>,

  journalBackfillMetrics: (body: { portfolio?: string; start_date?: string; end_date?: string; force?: boolean }) =>
    fetchWithAuth(`${API_BASE}/api/journal/backfill-metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{ status: string; checked?: number; updated?: number; errors?: string[]; detail?: string }>,

  // Prices
  priceLookup: (ticker: string) =>
    fetchJSON<{ ticker: string; price: number; atr: number; atr_pct: number }>(`/api/prices/lookup?ticker=${encodeURIComponent(ticker)}`),

  batchPrices: (tickers: string[]) =>
    fetchJSON<Record<string, number>>(`/api/prices/batch?tickers=${encodeURIComponent(tickers.join(","))}`),

  chartOhlcv: (ticker: string, start?: string, end?: string, period?: string, interval?: string) => {
    const params = new URLSearchParams();
    if (start) params.set("start", start);
    if (end) params.set("end", end);
    if (period) params.set("period", period);
    if (interval) params.set("interval", interval);
    return fetchJSON<{ ticker: string; candles: { time: number; open: number; high: number; low: number; close: number; volume: number }[] }>(
      `/api/charts/ohlcv/${encodeURIComponent(ticker)}?${params.toString()}`
    );
  },

  // Market
  rallyPrefix: () => fetchJSON<{ prefix: string; day_num?: number; state?: string }>(`/api/market/rally-prefix`),

  nlvShadow: (portfolio = "CanSlim") =>
    fetchJSON<{
      portfolio: string;
      as_of: string;
      prior_day?: string;
      yesterday_end_nlv?: number;
      yesterday_cash?: number;
      today_cash_change?: number;
      today_trade_flow?: number;
      today_cash?: number;
      today_holdings_value?: number;
      computed_nlv?: number;
      manual_nlv?: number | null;
      diff?: number | null;
      diff_pct?: number | null;
      missing_prices?: string[];
      error?: string;
    }>(`/api/nlv/shadow-today?portfolio=${encodeURIComponent(portfolio)}`),
  mfactor: () => fetchJSON<any>(`/api/market/mfactor`),

  // Config
  config: (key: string) => fetchJSON<{ key: string; value: any }>(`/api/config/${key}`),

  // Admin — config
  setConfig: (key: string, value: any, opts?: { value_type?: string; category?: string; description?: string }) =>
    fetchWithAuth(`${API_BASE}/api/config/${key}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value, ...opts, user: "admin" }),
    }).then(r => r.json()) as Promise<{ status: string; detail?: string }>,

  // Admin — events
  events: (scope = "CanSlim") => fetchJSON<any[]>(`/api/events?scope=${scope}`),
  addEvent: (body: Record<string, any>) =>
    fetchWithAuth(`${API_BASE}/api/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json()),
  updateEvent: (id: number, body: Record<string, any>) =>
    fetchWithAuth(`${API_BASE}/api/events/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json()),
  deleteEvent: (id: number) =>
    fetchWithAuth(`${API_BASE}/api/events/${id}`, { method: "DELETE" }).then(r => r.json()),

  // Admin — audit
  audit: (limit = 100, actionFilter?: string) =>
    fetchJSON<any[]>(`/api/audit?limit=${limit}${actionFilter ? `&action_filter=${actionFilter}` : ""}`),

  // Admin — cleanup
  cleanupMarketsurge: (dryRun = true) =>
    fetchWithAuth(`${API_BASE}/api/admin/cleanup-marketsurge`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dry_run: dryRun }),
    }).then(r => r.json()),

  // AI Coach
  coachChat: (message: string, preset?: string, portfolio = getActivePortfolio()) => {
    return fetchWithAuth(`${API_BASE}/api/coach/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, preset, portfolio }),
    });
  },

  // Trade writes
  nextTradeId: (portfolio = getActivePortfolio(), date = "") =>
    fetchJSON<{ trade_id: string }>(`/api/trades/next-id?portfolio=${portfolio}&date=${date}`),

  importTrades: () =>
    fetchWithAuth(`${API_BASE}/api/trades/import`, { method: "POST" }).then(r => r.json()) as Promise<{ status?: string; error?: string; trades?: any[]; count?: number; message?: string; debug?: Record<string, any> }>,

  logBuy: (body: Record<string, any>) =>
    fetchWithAuth(`${API_BASE}/api/trades/buy`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{ status?: string; error?: string; trx_id?: string }>,

  logSell: (body: Record<string, any>) =>
    fetchWithAuth(`${API_BASE}/api/trades/sell`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{ status?: string; error?: string; trx_id?: string; realized_pl?: number; remaining_shares?: number; is_closed?: boolean }>,

  deleteTransactionsByDate: (date: string, portfolio = getActivePortfolio()) =>
    fetchWithAuth(`${API_BASE}/api/trades/delete-transactions-by-date?date=${encodeURIComponent(date)}&portfolio=${encodeURIComponent(portfolio)}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status?: string; error?: string; deleted?: number; trade_ids?: string[] }>,

  editTransaction: (body: { detail_id: number; trade_id: string; ticker: string; action: string; date: string; shares: number; amount: number; value: number; rule: string; notes: string; stop_loss: number; trx_id: string; portfolio?: string }) =>
    fetchWithAuth(`${API_BASE}/api/trades/edit-transaction`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio: getActivePortfolio(), ...body }),
    }).then(r => r.json()) as Promise<{ status?: string; error?: string }>,

  deleteTrade: (tradeId: string, portfolio = getActivePortfolio()) =>
    fetchWithAuth(`${API_BASE}/api/trades/delete?trade_id=${encodeURIComponent(tradeId)}&portfolio=${portfolio}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status?: string; error?: string }>,

  // Fundamentals
  tradeFundamentals: (tradeId: string, portfolio = getActivePortfolio()) =>
    fetchJSON<any[]>(`/api/fundamentals/${tradeId}?portfolio=${portfolio}`),

  deleteFundamentals: (tradeId: string, portfolio = getActivePortfolio()) =>
    fetchWithAuth(`${API_BASE}/api/fundamentals/${encodeURIComponent(tradeId)}?portfolio=${encodeURIComponent(portfolio)}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status?: string; error?: string; deleted?: number }>,

  // R2 Images
  tradeImages: (tradeId: string, portfolio = getActivePortfolio()) =>
    fetchJSON<any[]>(`/api/images/${tradeId}?portfolio=${portfolio}`),

  uploadEodSnapshot: (blob: Blob, day: string, snapshotType: string, portfolio = getActivePortfolio()) => {
    const form = new FormData();
    form.append("file", blob, `${snapshotType}_${day}.png`);
    form.append("portfolio", portfolio);
    form.append("day", day);
    form.append("snapshot_type", snapshotType);
    return fetchWithAuth(`${API_BASE}/api/snapshots/upload`, { method: "POST", body: form })
      .then(r => r.json()) as Promise<{ status?: string; image_id?: number; error?: string }>;
  },

  listEodSnapshots: (day: string, portfolio = getActivePortfolio()) =>
    fetchJSON<{ image_url?: string; view_url?: string; image_type?: string; file_name?: string; uploaded_at?: string; id?: number }[]>(
      `/api/snapshots/${day}?portfolio=${portfolio}`
    ),

  uploadImage: (file: File, portfolio: string, tradeId: string, ticker: string, imageType: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("portfolio", portfolio);
    form.append("trade_id", tradeId);
    form.append("ticker", ticker);
    form.append("image_type", imageType);
    return fetchWithAuth(`${API_BASE}/api/images/upload`, { method: "POST", body: form }).then(r => r.json());
  },

  deleteImage: (imageId: number) =>
    fetchWithAuth(`${API_BASE}/api/images/${imageId}`, { method: "DELETE" }).then(r => r.json()),

  r2Status: () => fetchJSON<{ available: boolean }>(`/api/r2/status`),

  // Health
  health: () => fetchJSON<{ status: string; timestamp: string }>(`/api/health`),

  // Portfolios — multi-tenant CRUD. listPortfolios is called at app load by
  // PortfolioProvider; others are used by the onboarding screen and Settings.
  listPortfolios: () => fetchJSON<Portfolio[]>(`/api/portfolios`),

  createPortfolio: (body: { name: string; starting_capital?: number | null; reset_date?: string | null }) =>
    fetchWithAuth(`${API_BASE}/api/portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<Portfolio | { error: string }>,

  updatePortfolio: (id: number, body: PortfolioInput) =>
    fetchWithAuth(`${API_BASE}/api/portfolios/${id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<Portfolio | { error: string }>,

  deletePortfolio: (id: number) =>
    fetchWithAuth(`${API_BASE}/api/portfolios/${id}`, { method: "DELETE" })
      .then(r => r.json()) as Promise<{ status?: string; error?: string }>,

  portfolioNlv: (id: number) =>
    fetchJSON<PortfolioNlv>(`/api/portfolios/${id}/nlv`),

  portfolioReturns: (id: number) =>
    fetchJSON<PortfolioReturns>(`/api/portfolios/${id}/returns`),

  // Cash transactions — deposits, withdrawals, reconcile. Buy/sell rows
  // are emitted automatically by the trade logging backend; the UI never
  // creates those directly.
  listCashTransactions: (portfolioId: number, limit = 50) =>
    fetchJSON<CashTransaction[]>(`/api/portfolios/${portfolioId}/cash-transactions?limit=${limit}`),

  createCashTransaction: (portfolioId: number, body: {
    source: CashAction;
    amount: number;
    date?: string | null;
    note?: string | null;
  }) =>
    fetchWithAuth(`${API_BASE}/api/portfolios/${portfolioId}/cash-transactions`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<CashTransaction | { status: string; delta: number; message: string } | { error: string }>,
};

// Live-derived NLV snapshot (cash + Σ positions at live price)
export interface PortfolioNlv {
  cash: number;
  market_value: number;
  nlv: number;
  positions: Array<{
    ticker: string;
    shares: number;
    avg_entry: number;
    current_price: number | null;
    market_value: number;
    unrealized_pl: number;
    price_unavailable?: boolean;
  }>;
  as_of: string;
}

// LTD / YTD returns derived from NLV + cash-transactions ledger.
// ytd_available=false when the portfolio started before the current year
// and no start-of-year NLV snapshot exists yet (pre-Phase-4).
export interface PortfolioReturns {
  nlv: number;
  net_contributions: number;
  ltd_pl: number;
  ltd_pct: number;
  ytd_pl: number | null;
  ytd_pct: number | null;
  ytd_available: boolean;
  as_of: string;
}
