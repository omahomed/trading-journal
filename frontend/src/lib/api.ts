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
  // market_window (V10 vocabulary) was removed from this interface when
  // the M Factor page was deleted; the column still exists in the
  // trading_journal table for historical preservation, and CSV export
  // still emits a "Window" column via untyped (h as any).market_window
  // access. The [key: string]: any below covers that legacy access.
  [key: string]: any;
}

// Weekly retro (Migration 025 — Phase 0). Persisted server-side via
// /api/weekly-retros. Both top-level fields and the nested ticker_grades
// shape mirror the DB columns one-for-one.
export interface WeeklyRetroTickerGrade {
  grade: string;
  behavior: string;
  notes: string;
}

// Phase 6 — NotesRail (left-rail navigator). Entity-agnostic shape so
// the same component mounts on Daily Report in Phase 7. The list endpoint
// returns one row per week from inception to "now"; synthetic empty
// rows (id: null, has_content: false) cover Mondays without a retro so
// the sparkline grid is continuous.
export type NotesRailEntityType = "weekly_retro" | "daily_journal";

export interface NotesRailItemTag {
  name: string;
  /** Matches the TAG_PALETTE keys: rose | amber | emerald | sky | violet. */
  color: string;
}

export interface NotesRailItem {
  id: number | null;            // null on synthetic empty rows
  key: string;                  // week_start ISO; stable id for activeKey
  week_start: string;
  week_end: string;
  year: number;
  month: number;                // 1..12
  title: string;                // "May 11 – May 15"
  has_content: boolean;         // false → draft-dot styling
  pinned: boolean;
  sparkline_value: number | null;   // weekly_return_pct
  week_grade: string | null;
  // Phase 6 design-fidelity additions (rail per-row subtitle line + chips).
  // weekly_pnl: NLV-delta for the week (matches Weekly P&L tile source).
  // trades_count: count of trade_details transactions for the week (matches
  // Flight Deck Total Tickets source).
  weekly_pnl: number | null;
  trades_count: number;
  tags: NotesRailItemTag[];         // attached tags (Phase 1 polymorphic system)
  // Phase 4.6 — drives the rail's tri-state dot (empty/draft/reviewed).
  // null on synthetic empty rows and on retros where the user hasn't
  // checked "Mark week as reviewed".
  reviewed_at: string | null;
}

export interface NotesRailYtdStats {
  total_weeks: number;
  weeks_graded: number;
  avg_grade: string | null;     // letter; null when weeks_graded == 0
  weeks_pinned: number;
}

export interface NotesRailListResponse {
  weeks: NotesRailItem[];
  ytd_stats: NotesRailYtdStats;
}

// Phase 7 — daily report rail envelope. Same item shape as the weekly
// envelope (NotesRailItem), but the top-level key is "days" (the rail
// component branches on entityType for date-label copy). YTD stats reuse
// the weekly field names ("weeks_*") to keep NotesRailYtdStats compat.
export interface DailyJournalListResponse {
  days: NotesRailItem[];
  ytd_stats: NotesRailYtdStats;
}

// Phase 5 — performance metrics powering the Weekly Retro top-tile row.
// Live-computed server-side by /api/analytics/weekly-metrics; matches the
// shape returned by nlv_service.weekly_metrics.
export interface WeeklyMetricsWinRate {
  rate: number;     // wins / (wins + losses + flat). 0 when total == 0.
  wins: number;
  losses: number;
  flat: number;
  total: number;
}

export interface WeeklyMetrics {
  weekly_pnl: number;
  weekly_return_pct: number;
  ytd_pct: number;
  ltd_pct: number;
  win_rate: WeeklyMetricsWinRate;
  week_start: string;
  week_end: string;
  as_of: string;
}

export interface WeeklyRetro {
  id: number;
  portfolio: string;
  week_start: string;
  week_grade: string | null;
  // Phase 4.6 — 3-axis grading. Axes are nullable on legacy rows (pre-
  // Phase 4.6) and on rows where the user hasn't graded every axis yet.
  // overall_override == true means the user chose a week_grade that
  // differs from the derived average; backend trusts the client value.
  // reviewed_at is the persisted "Mark week as reviewed" timestamp;
  // null means not reviewed yet.
  execution_grade: string | null;
  process_grade: string | null;
  pnl_grade: string | null;
  overall_override: boolean;
  reviewed_at: string | null;
  best_decision: string;
  worst_decision: string;
  rule_change: boolean;
  rule_change_text: string;
  // Phase 3: HTML body of the Weekly Thoughts editor (DOMPurify-sanitized
  // before send). Backend column is NOT NULL DEFAULT '' so this is never
  // null on a row read from the API.
  weekly_thoughts: string;
  ticker_grades: Record<string, WeeklyRetroTickerGrade>;
  created_at: string;
  updated_at: string;
}

// Phase 4 — Weekly Retro Snapshots (Migration 028). Image attachments on
// weekly retros. storage_ref is the R2 object key; view_url is the public
// CDN URL pre-composed by the server (so the frontend never has to know
// R2_PUBLIC_URL). caption / sort_order are pre-provisioned for Phase
// 4-followup features (captions, drag-reorder); v1 always sends defaults.
export interface SnapshotRow {
  id: number;
  weekly_retro_id: number;
  storage_ref: string;
  view_url: string;
  file_name: string | null;
  mime_type: string | null;
  file_size_bytes: number | null;
  width: number | null;
  height: number | null;
  sort_order: number;
  caption: string;
  created_at: string;
}

// Phase 7 — Daily Journal Captures (Migration 031). Mirrors SnapshotRow
// byte-for-byte except the FK is named daily_journal_id. The shared
// <SnapshotGallery> consumes both via an entityType prop and a union row
// type via DailyJournalCaptureRow | SnapshotRow.
export interface DailyJournalCaptureRow {
  id: number;
  daily_journal_id: number;
  storage_ref: string;
  view_url: string;
  file_name: string | null;
  mime_type: string | null;
  file_size_bytes: number | null;
  width: number | null;
  height: number | null;
  sort_order: number;
  caption: string;
  created_at: string;
}

// Tags — Phase 1 (Migration 026). Portfolio-scoped, polymorphic
// (entity_type ∈ {weekly_retro, daily_journal, trades_summary}). Color is a
// closed-palette key matching TAG_PALETTE in src/lib/tag-palette.ts; see
// that file's header for the lockstep contract with the backend.
export interface Tag {
  id: number;
  portfolio: string;
  name: string;
  color: string;
  created_at: string;
  updated_at: string;
}

export interface TagAssignment {
  id: number;
  tag_id: number;
  tag_name: string;
  tag_color: string;
  entity_type: "weekly_retro" | "daily_journal" | "trades_summary";
  entity_id: number;
  created_at: string;
}

export interface JournalHistoryPoint {
  // Phase 7 — journal row PK. Present on every row from the backend after
  // migration 031; null only on synthetic / pre-migration data the
  // frontend should not normally see. TagPicker, NotesRail, and the
  // SnapshotGallery on the daily report page bind to this as entity_id.
  id: number | null;
  day: string;
  end_nlv: number;
  daily_pct_change: number;
  portfolio_ltd: number;
  spy_ltd: number;
  ndx_ltd: number;
  pct_invested: number;
  portfolio_heat: number;
  // Phase 7 — rich-text body of the Daily Thoughts editor (migration 031).
  // Backend column is NOT NULL DEFAULT '' so the value is never null on
  // post-migration rows. Marked optional for the migration-window case
  // where the column might be absent from the SELECT (pre-031 DBs).
  daily_thoughts?: string;
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
  // Migration 016: equity options carry instrument_type='OPTION' + multiplier=100
  // so the UI can format dollar amounts as notional (×100) instead of premium.
  // Optional because legacy rows pre-migration may not have them populated.
  instrument_type?: "STOCK" | "OPTION";
  multiplier?: number;
  // Migration 019. Tags the campaign with a strategy from the strategies
  // lookup table. Optional only because legacy un-migrated rows might lack
  // the column — post-019 every row has a value (defaults to 'CanSlim').
  strategy?: string;
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
  instrument_type?: "STOCK" | "OPTION";
  multiplier?: number;
  [key: string]: any;
}

// One persisted BUY × SELL pairing from the lot_closures table (migration
// 017). Returned alongside details by /api/trades/open/details and
// /api/trades/recent so the trade-journal frontend can render per-row
// realized P&L without re-walking LIFO client-side.
export interface LotClosure {
  trade_id: string;
  buy_trx_id: string;
  sell_trx_id: string;
  shares: number;
  buy_price: number;
  sell_price: number;
  multiplier: number;
  realized_pl: number;
  closed_at: string;
}

// Wrapper response shape for the two trade-detail endpoints. lot_closures
// is empty for trades that haven't had their closures backfilled (the 6
// deferred trades) — frontend silently falls back to its own LIFO walk
// in that case.
export interface TradeDetailsBundle {
  details: TradeDetail[];
  lot_closures: LotClosure[];
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

// Strategy lookup row from the `strategies` table (Migration 019). Each
// trades_summary row carries a `strategy` text key referencing this table.
// `color` is a hex string used to distinguish strategies in the UI.
export interface Strategy {
  name: string;
  description: string | null;
  color: string;
  is_active: boolean;
  created_at: string | null;
}

// Drift-scan response (Phase 2 Commit 8). One entry per check, plus a
// summary tile. Severity is decided by the runner — a check that errored
// (e.g. statement timeout) is bucketed as "error" regardless of its
// declared severity, so the summary count can't be silently misleading.
export interface DriftScanCheckResult {
  check_id: string;
  description: string;
  severity: "warning" | "error";
  violation_count: number;
  // samples have a check-specific column shape — first three are always
  // (trade_id, ticker, portfolio) when present, then check-specific extras.
  samples: Array<Record<string, string | number | null>>;
  remediation: string;
  duration_ms: number;
  error: string | null;
}

export interface DriftScanResponse {
  scanned_at: string;
  portfolio_filter: string | null;
  check_filter: string | null;
  sample_limit: number;
  checks: DriftScanCheckResult[];
  summary: {
    total_checks: number;
    passed: number;
    warnings: number;
    errors: number;
  };
}

// API functions
export const api = {
  // Journal
  journalLatest: (portfolio = getActivePortfolio(), before = "") => {
    const qs = new URLSearchParams({ portfolio });
    if (before) qs.set("before", before);
    return fetchJSON<JournalEntry>(`/api/journal/latest?${qs.toString()}`);
  },

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
    fetchJSON<TradeDetailsBundle>(`/api/trades/open/details?portfolio=${portfolio}`),

  tradesRecent: (portfolio = getActivePortfolio(), limit = 20) =>
    fetchJSON<TradeDetailsBundle>(`/api/trades/recent?portfolio=${portfolio}&limit=${limit}`),

  updateTradeStops: (body: { portfolio?: string; trade_id: string; new_stop: number }) =>
    fetchWithAuth(`${API_BASE}/api/trades/update-stops`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{
      status?: string;
      error?: string;
      trade_id?: string;
      updated_lots?: number;
      be_applied?: boolean;
      current_price?: number;
    }>,

  flagBeRule: (body: { portfolio?: string; trade_id: string; flagged: boolean }) =>
    fetchWithAuth(`${API_BASE}/api/trades/flag-be`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{
      status?: string;
      error?: string;
      trade_id?: string;
      flagged?: boolean;
      updated?: number;
    }>,

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

  journalRestampMct: (day: string, portfolio = getActivePortfolio()) =>
    fetchWithAuth(`${API_BASE}/api/journal/restamp-mct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio, day }),
    }).then(r => r.json()) as Promise<{ status: string; market_cycle?: string; mct_display_day_num?: number | null; detail?: string }>,

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

  // Weekly retros (Migration 025 — Phase 0). Server-side replacement for the
  // old "mo-weekly-retros" localStorage key. The GET endpoint returns
  // { error: "not_found" } for a missing week — callers treat that as a
  // fresh blank retro, not an error UI state.
  weeklyRetroGet: (portfolio: string, weekStart: string) =>
    fetchJSON<WeeklyRetro | { error: string }>(
      `/api/weekly-retros?portfolio=${encodeURIComponent(portfolio)}&week_start=${encodeURIComponent(weekStart)}`
    ),

  // Phase 6 — wrapped envelope for the NotesRail. Shape changed from a
  // bare array (used by the now-removed Review History tab) to
  // {weeks, ytd_stats}. Coordinated cutover; no surviving consumer of
  // the old shape.
  weeklyRetroList: (portfolio: string) =>
    fetchJSON<NotesRailListResponse | { error: string }>(
      `/api/weekly-retros/list?portfolio=${encodeURIComponent(portfolio)}`
    ),

  // Phase 6 — polymorphic pin toggle. Idempotent server-side; the
  // response is the NEW pinned state. UI should call this with the
  // CURRENT state's inverse and trust the response (optimistic update
  // with rollback on rejection).
  pinsToggle: (entityType: "weekly_retro" | "daily_journal", entityId: number) =>
    fetchWithAuth(`${API_BASE}/api/pins/toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entity_type: entityType, entity_id: entityId }),
    }).then(r => r.json()) as Promise<{ pinned: boolean } | { error: string }>,

  weeklyRetroUpsert: (
    payload: Omit<WeeklyRetro, "id" | "created_at" | "updated_at">,
  ) =>
    fetchWithAuth(`${API_BASE}/api/weekly-retros`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(r => r.json()) as Promise<WeeklyRetro | { error: string }>,

  // Phase 5 — performance tiles. Live-computed (no snapshot columns).
  // weekStart is the Monday of the requested week (YYYY-MM-DD); the
  // backend echoes Friday in `week_end`.
  weeklyMetrics: (portfolio: string, weekStart: string) =>
    fetchJSON<WeeklyMetrics | { error: string }>(
      `/api/analytics/weekly-metrics?portfolio=${encodeURIComponent(portfolio)}&week_start=${encodeURIComponent(weekStart)}`
    ),

  weeklyRetroDelete: (retroId: number) =>
    fetchWithAuth(`${API_BASE}/api/weekly-retros/${retroId}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status: string; id: number } | { error: string }>,

  // Phase 4 — Weekly Retro Snapshots (Migration 028). Image attachments
  // on weekly retros. Bytes live in R2; metadata + view_url come from the
  // server. Browser fetches bytes directly from view_url (public R2 CDN).
  uploadWeeklyRetroSnapshot: (
    retroId: number,
    file: File,
    portfolio: string = getActivePortfolio(),
  ) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("portfolio", portfolio);
    return fetchWithAuth(
      `${API_BASE}/api/weekly-retros/${retroId}/snapshots`,
      { method: "POST", body: form },
    ).then(r => r.json()) as Promise<SnapshotRow | { error: string; detail?: any }>;
  },

  listWeeklyRetroSnapshots: (retroId: number, portfolio: string = getActivePortfolio()) =>
    fetchJSON<SnapshotRow[] | { error: string }>(
      `/api/weekly-retros/${retroId}/snapshots?portfolio=${encodeURIComponent(portfolio)}`,
    ),

  deleteWeeklyRetroSnapshot: (snapshotId: number) =>
    fetchWithAuth(
      `${API_BASE}/api/weekly-retros/snapshots/${snapshotId}`,
      { method: "DELETE" },
    ).then(r => r.json()) as Promise<{ deleted: true; id: number } | { error: string }>,

  // Phase 4.1 — inline image paste into Weekly Thoughts editor. Returns
  // a public R2 URL the editor swaps into <img src> in place of the
  // ephemeral blob URL. No DB row created server-side — the editor's
  // HTML is the source of truth for which inline images exist.
  uploadWeeklyThoughtsImage: (
    retroId: number,
    file: File,
    portfolio: string = getActivePortfolio(),
  ) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("portfolio", portfolio);
    return fetchWithAuth(
      `${API_BASE}/api/weekly-retros/${retroId}/thoughts-images`,
      { method: "POST", body: form },
    ).then(r => r.json()) as Promise<{ view_url: string } | { error: string; detail?: any }>;
  },

  // Phase 7 — Daily Journal Captures (Migration 031). Mirrors the weekly
  // retro snapshot endpoints. The shared <SnapshotGallery> branches on
  // entityType to pick the right method here.
  dailyJournalList: (portfolio: string) =>
    fetchJSON<DailyJournalListResponse | { error: string }>(
      `/api/daily-journals/list?portfolio=${encodeURIComponent(portfolio)}`
    ),

  uploadDailyJournalCapture: (
    journalId: number,
    file: File,
    portfolio: string = getActivePortfolio(),
  ) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("portfolio", portfolio);
    return fetchWithAuth(
      `${API_BASE}/api/daily-journals/${journalId}/captures`,
      { method: "POST", body: form },
    ).then(r => r.json()) as Promise<DailyJournalCaptureRow | { error: string; detail?: any }>;
  },

  listDailyJournalCaptures: (journalId: number, portfolio: string = getActivePortfolio()) =>
    fetchJSON<DailyJournalCaptureRow[] | { error: string }>(
      `/api/daily-journals/${journalId}/captures?portfolio=${encodeURIComponent(portfolio)}`,
    ),

  deleteDailyJournalCapture: (captureId: number) =>
    fetchWithAuth(
      `${API_BASE}/api/daily-journals/captures/${captureId}`,
      { method: "DELETE" },
    ).then(r => r.json()) as Promise<{ deleted: true; id: number } | { error: string }>,

  // Phase 7 — inline image paste into Daily Thoughts editor. Same shape
  // as uploadWeeklyThoughtsImage; the shared <ThoughtsEditor> branches
  // on entityType to pick which method to invoke.
  uploadDailyThoughtsImage: (
    journalId: number,
    file: File,
    portfolio: string = getActivePortfolio(),
  ) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("portfolio", portfolio);
    return fetchWithAuth(
      `${API_BASE}/api/daily-journals/${journalId}/thoughts-images`,
      { method: "POST", body: form },
    ).then(r => r.json()) as Promise<{ view_url: string } | { error: string; detail?: any }>;
  },

  // Tag system (Migration 026 — Phase 1). Polymorphic: assignments use
  // entity_type to discriminate weekly_retro / daily_journal / trades_summary.
  // Phase 1 mounts on weekly_retro only; later phases reuse these endpoints.
  listTags: (portfolio: string) =>
    fetchJSON<Tag[]>(`/api/tags?portfolio=${encodeURIComponent(portfolio)}`),

  createTag: (payload: { portfolio: string; name: string; color: string }) =>
    fetchWithAuth(`${API_BASE}/api/tags`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(r => r.json()) as Promise<Tag | { error: string }>,

  updateTag: (id: number, payload: { name?: string; color?: string }) =>
    fetchWithAuth(`${API_BASE}/api/tags/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(r => r.json()) as Promise<Tag | { error: string }>,

  deleteTag: (id: number) =>
    fetchWithAuth(`${API_BASE}/api/tags/${id}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status: string; id: number } | { error: string }>,

  listTagAssignments: (query: {
    entity_type: "weekly_retro" | "daily_journal" | "trades_summary";
    entity_id: number;
  }) =>
    fetchJSON<TagAssignment[]>(
      `/api/tags/assignments?entity_type=${encodeURIComponent(query.entity_type)}&entity_id=${query.entity_id}`,
    ),

  createTagAssignment: (payload: {
    tag_id: number;
    entity_type: "weekly_retro" | "daily_journal" | "trades_summary";
    entity_id: number;
  }) =>
    fetchWithAuth(`${API_BASE}/api/tags/assignments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(r => r.json()) as Promise<TagAssignment | { error: string }>,

  deleteTagAssignment: (id: number) =>
    fetchWithAuth(`${API_BASE}/api/tags/assignments/${id}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status: string; id: number } | { error: string }>,

  // IBKR — broker NAV pull for Daily Routine auto-fill. The endpoint always
  // returns 200 OK; success/error is read from the body, never the HTTP code.
  ibkrNavForDate: (date: string) =>
    fetchJSON<
      | { success: true; nav: number; cash_balance: number; position_value: number;
          report_date: string; currency: string; account: string; source: string }
      | { success: false; error: string; message: string }
    >(`/api/ibkr/nav-for-date?date=${encodeURIComponent(date)}`),

  // Prices
  priceLookup: (ticker: string) =>
    fetchJSON<{ ticker: string; price: number; atr: number; atr_pct: number }>(`/api/prices/lookup?ticker=${encodeURIComponent(ticker)}`),

  setManualPrice: (body: { portfolio: string; trade_id: string; manual_price: number | null }) =>
    fetchWithAuth(`${API_BASE}/api/trades/manual-price`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<{ status: "ok"; trade_id: string; manual_price: number | null; manual_price_set_at: string | null } | { error: string }>,

  batchPrices: (tickers: string[], portfolio?: string, date?: string) => {
    const qs = new URLSearchParams({ tickers: tickers.join(",") });
    if (portfolio) qs.set("portfolio", portfolio);
    if (date) qs.set("date", date);
    return fetchJSON<Record<string, number>>(`/api/prices/batch?${qs.toString()}`);
  },

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

  // Market — `as_of_date` is optional. Pass it on Daily Routine so the
  // prefix reflects the routine's date rather than the latest ingested
  // market_data bar (which can lag by a trading day around market open /
  // overnight cron windows).
  rallyPrefix: (as_of_date?: string) => fetchJSON<{
    prefix: string;
    day_num?: number;
    state?: "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";
    cap_at_100?: boolean;
    drawdown_pct?: number;
    power_trend_on_since?: string | null;
    ftd_date?: string | null;
    reference_high?: number;
    reference_high_date?: string | null;
    cycle_start_date?: string | null;
    day_num_projected?: boolean;
    day_num_projection_offset?: number;
  }>(`/api/market/rally-prefix${as_of_date ? `?as_of_date=${encodeURIComponent(as_of_date)}` : ""}`),

  marketSignals: (days = 30, signal_type?: string) => {
    const params = new URLSearchParams({ days: String(days) });
    if (signal_type) params.set("signal_type", signal_type);
    return fetchJSON<{
      signals: Array<{
        trade_date: string;
        signal_type: string;
        signal_label: string;
        exposure_before: number | null;
        exposure_after: number | null;
        state_before: string | null;
        state_after: string | null;
        meta: Record<string, unknown>;
      }>;
    }>(`/api/market/signals?${params.toString()}`);
  },

  mctStateByDateRange: (start_date: string, end_date: string) =>
    fetchJSON<{
      states: Array<{
        trade_date: string;
        state: "POWERTREND" | "UPTREND" | "RALLY MODE" | "CORRECTION";
        exposure_ceiling: number;
        cap_at_100: boolean;
        cycle_day: number;
        // display_day_num is what the journal MCT State badge appends as
        // "D{N}" — POWERTREND counts from STEP_8 firing, UPTREND/RALLY
        // MODE count from cycle STEP_0, CORRECTION renders no suffix.
        display_day_num: number | null;
        in_correction: boolean;
        correction_active: boolean;
        power_trend: boolean;
      }>;
    }>(`/api/journal/mct-state-by-date-range?start_date=${start_date}&end_date=${end_date}`),

  // api.mfactor() / /api/market/mfactor was the V10 MA-stack snapshot
  // that fed Position Sizer + Log Buy's sizing-mode picker. Both surfaces
  // now derive sizing mode from V11 MCT state via @/lib/sizing-mode.

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

  rebuildMctSignals: () =>
    fetchWithAuth(`${API_BASE}/api/admin/rebuild-mct-signals`, {
      method: "POST",
    }).then(r => r.json()) as Promise<{
      deleted?: number;
      inserted?: number;
      events_emitted?: number;
      first_signal_date?: string | null;
      last_signal_date?: string | null;
      bars_processed?: number;
      error?: string;
    }>,

  // Phase 2 Commit 8 — drift-scan admin endpoint. Founder-gated server-side;
  // a non-founder gets { error: "forbidden_not_admin" } at HTTP 200. Each
  // check is read-only SQL; the response shape is documented inline below.
  runDriftScan: (opts: { portfolio?: string; checkId?: string; limitSamples?: number } = {}) => {
    const qs = new URLSearchParams();
    if (opts.portfolio) qs.set("portfolio", opts.portfolio);
    if (opts.checkId) qs.set("check_id", opts.checkId);
    if (opts.limitSamples != null) qs.set("limit_samples", String(opts.limitSamples));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return fetchWithAuth(`${API_BASE}/api/admin/drift-scan${suffix}`).then(r => r.json()) as Promise<
      | DriftScanResponse
      | { error: string }
    >;
  },

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

  // Exercise an open option: closes ALL held contracts on the option trade
  // and either scales into an existing OPEN stock trade for the underlying
  // or opens a new stock trade. Backend writes are atomic — see
  // api/main.py:exercise_option for the contract.
  exerciseOption: (body: { trade_id: string; date: string; notes?: string; portfolio?: string }) =>
    fetchWithAuth(`${API_BASE}/api/trades/exercise-option`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio: getActivePortfolio(), ...body }),
    }).then(r => r.json()) as Promise<{
      status?: "ok"; error?: string;
      option_trade_id?: string; stock_trade_id?: string;
      stock_was_new?: boolean;
      contracts_exercised?: number; shares_acquired?: number;
      stock_entry_price?: number;
    }>,

  deleteTransactionsByDate: (date: string, portfolio = getActivePortfolio()) =>
    fetchWithAuth(`${API_BASE}/api/trades/delete-transactions-by-date?date=${encodeURIComponent(date)}&portfolio=${encodeURIComponent(portfolio)}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status?: string; error?: string; deleted?: number; trade_ids?: string[] }>,

  editTransaction: (body: { detail_id: number; trade_id: string; ticker: string; action: string; date: string; shares: number; amount: number; value: number; rule: string; notes: string; stop_loss: number; trx_id: string; portfolio?: string }) =>
    fetchWithAuth(`${API_BASE}/api/trades/edit-transaction`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio: getActivePortfolio(), ...body }),
    }).then(r => r.json()) as Promise<{ status?: string; error?: string }>,

  deleteTransaction: (detail_id: number, trade_id: string, ticker: string, portfolio = getActivePortfolio()) => {
    const qs = new URLSearchParams({
      detail_id: String(detail_id),
      trade_id: trade_id,
      ticker: ticker,
      portfolio: portfolio,
    });
    return fetchWithAuth(`${API_BASE}/api/trades/transaction?${qs.toString()}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status?: string; error?: string }>;
  },

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

  // Strategies — small global lookup table (Migration 019). Log Buy fetches
  // the active list to populate its Strategy dropdown; Phase 2 admin UI
  // fetches with active=false to show disabled strategies for editing.
  listStrategies: ({ active = true }: { active?: boolean } = {}) =>
    fetchJSON<Strategy[]>(`/api/strategies?active=${active}`),

  // Phase 2 — strategies CRUD. Founder-gated server-side; non-founder
  // calls return { error: "forbidden_not_admin" }.
  createStrategy: (body: { name: string; color: string; description?: string; is_active?: boolean }) =>
    fetchWithAuth(`${API_BASE}/api/strategies`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<Strategy | { error: string }>,

  updateStrategy: (name: string, body: { description?: string; color?: string; is_active?: boolean }) =>
    fetchWithAuth(`${API_BASE}/api/strategies/${encodeURIComponent(name)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<Strategy | { error: string }>,

  // Phase 2 — retroactive tagging (per-trade and bulk). Not founder-gated.
  setTradeStrategy: (tradeId: string, body: { strategy: string; portfolio?: string }) =>
    fetchWithAuth(`${API_BASE}/api/trades/${encodeURIComponent(tradeId)}/strategy`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio: getActivePortfolio(), ...body }),
    }).then(r => r.json()) as Promise<{ ok?: true; trade_id?: string; strategy?: string; error?: string }>,

  bulkSetStrategy: (body: { trade_ids: string[]; strategy: string; portfolio?: string }) =>
    fetchWithAuth(`${API_BASE}/api/trades/bulk-strategy`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio: getActivePortfolio(), ...body }),
    }).then(r => r.json()) as Promise<{ ok?: true; updated?: number; failed?: string[]; strategy?: string; error?: string }>,

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

  portfolioTwrReturns: (id: number) =>
    fetchJSON<PortfolioTwrReturns>(`/api/portfolios/${id}/twr-returns`),

  // Aggregated dashboard read view. Single source of truth: every
  // journal-derived field comes from the latest saved trading_journal row.
  // See nlv_service.dashboard_metrics for the field-level contract.
  dashboardMetrics: (id: number) =>
    fetchJSON<DashboardMetrics>(`/api/portfolios/${id}/dashboard-metrics`),

  // Cash transactions — deposits, withdrawals, reconcile. Buy/sell rows
  // are emitted automatically by the trade logging backend; the UI never
  // creates those directly.
  listCashTransactions: (portfolioId: number, limit = 50, excludeTradeRows = false) =>
    fetchJSON<CashTransaction[]>(`/api/portfolios/${portfolioId}/cash-transactions?limit=${limit}&exclude_trade_rows=${excludeTradeRows}`),

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

  updateCashTransaction: (portfolioId: number, txId: number, body: {
    amount?: number;
    date?: string | null;
    note?: string | null;
  }) =>
    fetchWithAuth(`${API_BASE}/api/portfolios/${portfolioId}/cash-transactions/${txId}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()) as Promise<CashTransaction | { error: string }>,

  deleteCashTransaction: (portfolioId: number, txId: number) =>
    fetchWithAuth(`${API_BASE}/api/portfolios/${portfolioId}/cash-transactions/${txId}`, {
      method: "DELETE",
    }).then(r => r.json()) as Promise<{ status: string } | { error: string }>,
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

// Time-weighted LTD / YTD chained from daily journal returns.
// Answers 'what compound return did the strategy produce' independent
// of when capital was added/withdrawn — the headline LTD tile reads from
// here, not from PortfolioReturns.ltd_pct (which is the snapshot ratio).
export interface PortfolioTwrReturns {
  twr_ltd_pct: number;
  twr_ytd_pct: number | null;
  twr_ytd_available: boolean;
  as_of: string;
}

// Aggregated dashboard read view — see nlv_service.dashboard_metrics for
// the field-level contract. journal-derived fields are nullable (frontend
// renders "—" or empty state when journal_available is false).
export interface DashboardMetrics {
  journal_available: boolean;
  as_of_date: string | null;
  nlv: number | null;
  nlv_delta_dollar: number | null;
  nlv_delta_pct: number | null;
  total_holdings: number | null;
  exposure_pct: number | null;
  cash: number | null;
  drawdown_current_pct: number | null;
  drawdown_peak_nlv: number | null;
  drawdown_peak_date: string | null;
  ltd_pct: number | null;
  ltd_pl_dollar: number | null;
  ytd_pct: number | null;
  ytd_pl_dollar: number | null;
  ytd_available: boolean;
  as_of: string;
}
