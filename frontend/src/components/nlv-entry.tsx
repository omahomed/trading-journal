"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { api, type RecurringCashEvent } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { autoTickByPrefix, SYSTEM_ITEM_PREFIXES } from "@/lib/routine-autotick";

// IBKR Flex auto-fill is dormant: the upstream Flex Query has been returning
// "request error (1001) — statement could not be generated" intermittently,
// so the auto-pull is more annoying than helpful. Manual entry only until
// the upstream issue is resolved. Flip this back to true to re-enable the
// effect + the warning banner. NB: the multi-portfolio redesign keeps the
// per-card IBKR scaffolding so re-enabling drops in cleanly; the effect
// would have to be retargeted per portfolio (the Flex Query is account-
// scoped, so each portfolio needs its own pull URL or filter).
const IBKR_AUTOFILL_ENABLED = false;

// Scorecard helpers (REPORT_CATEGORIES / letterGrade / gradeToScore /
// scoreColor) moved to @/lib/scorecard when the Journal-item mini-form
// took over capture. Deleted from here in the same trim.

// ─────────────────────────────────────────────────────────────────────────────
// Multi-portfolio card state. Each card mirrors the per-portfolio fields
// that previously lived as singletons on the component (portNlv, portHold,
// etc.). end_nlv + total_holdings are required (0 valid, empty blocks save);
// cash_change defaults to "0"; actions is auto-populated from per-portfolio
// trade activity.
// ─────────────────────────────────────────────────────────────────────────────
type IbkrSource = "manual" | "ibkr_auto" | "ibkr_override";

interface PortfolioCardState {
  name: string;
  id: number;
  end_nlv: string;
  total_holdings: string;
  cash_change: string;
  actions: string;
  prev_end_nlv: number;          // loaded from journalLatest(before=entryDate)
  nlv_source: IbkrSource;
  holdings_source: IbkrSource;
  errors: { end_nlv?: string; total_holdings?: string };
  // Per-field touched flags gate inline error rendering. A field is
  // "touched" once the user has blurred it (or once a save is attempted,
  // which marks all fields touched at once). Required-but-empty errors
  // don't render until the user has had a chance to interact with the
  // field — prevents the "every input shows red on mount" footgun.
  touched: { end_nlv: boolean; total_holdings: boolean };
}

function emptyCard(p: { id: number; name: string }): PortfolioCardState {
  return {
    name: p.name,
    id: p.id,
    end_nlv: "",
    total_holdings: "",
    cash_change: "0",
    actions: "",
    prev_end_nlv: 0,
    nlv_source: "manual",
    holdings_source: "manual",
    errors: {},
    touched: { end_nlv: false, total_holdings: false },
  };
}

function validateCard(p: PortfolioCardState): PortfolioCardState["errors"] {
  const errors: PortfolioCardState["errors"] = {};
  if (p.end_nlv.trim() === "") errors.end_nlv = "Required";
  else if (isNaN(parseFloat(p.end_nlv))) errors.end_nlv = "Must be a number";
  if (p.total_holdings.trim() === "") errors.total_holdings = "Required";
  else if (isNaN(parseFloat(p.total_holdings))) errors.total_holdings = "Must be a number";
  return errors;
}

function deriveCardMetrics(p: PortfolioCardState) {
  const nlv = parseFloat(p.end_nlv) || 0;
  const hold = parseFloat(p.total_holdings) || 0;
  const cash = parseFloat(p.cash_change) || 0;
  // App convention: divisor is the post-deposit baseline. Matches
  // daily-journal.tsx pre-redesign and the journal importer's
  // compute_derived per the snapshot-fix commits.
  const adjustedBeg = p.prev_end_nlv + cash;
  const daily_dollar_change = p.prev_end_nlv > 0 ? nlv - adjustedBeg : 0;
  const daily_pct_change = adjustedBeg > 0 ? (daily_dollar_change / adjustedBeg) * 100 : 0;
  const pct_invested = nlv > 0 ? (hold / nlv) * 100 : 0;
  return { daily_dollar_change, daily_pct_change, pct_invested, nlv, cash };
}

// ─────────────────────────────────────────────────────────────────────────────
// Small presentational helpers
// ─────────────────────────────────────────────────────────────────────────────

function Field({ label, error, children }: { label: string; error?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
        {label}
      </label>
      {children}
      {error && (
        <p className="text-[11px] mt-1 font-medium" role="alert" style={{ color: "#dc2626" }}>
          {error}
        </p>
      )}
    </div>
  );
}

const inputCls = "w-full h-[38px] px-3 rounded-[10px] text-[13px] outline-none";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};
const inputErrorStyle: React.CSSProperties = {
  ...inputStyle,
  border: "1px solid #dc2626",
};

// ─────────────────────────────────────────────────────────────────────────────
// PortfolioCard — per-portfolio entry tile. Owns no fetch logic; receives
// state + onChange from the parent component. Renders inline validation
// errors via the Field wrapper.
// ─────────────────────────────────────────────────────────────────────────────

function PortfolioCard({
  card,
  onChange,
  accentColor,
  recurringEvent,
  onRecurringPost,
  onRecurringSkip,
  onOpenManage,
}: {
  card: PortfolioCardState;
  onChange: (patch: Partial<PortfolioCardState>) => void;
  accentColor: string;
  /** null when no config exists for this portfolio yet. */
  recurringEvent: RecurringCashEvent | null;
  /** Fires the deposit + bumps card.cash_change. Returns the actually-
   *  posted amount so the caller can flash a confirmation. */
  onRecurringPost: (id: number, amount: number) => Promise<number | null>;
  onRecurringSkip: (id: number) => Promise<void>;
  /** Opens the Manage-recurring modal, either editing this portfolio's
   *  existing event or creating a fresh one when recurringEvent is null. */
  onOpenManage: () => void;
}) {
  const m = deriveCardMetrics(card);

  // Recurring-deposit reminder — inline editable amount so the user can
  // override this cycle without touching the config. Local state seeds
  // from computed_amount; resets whenever the event id / amount changes.
  const [recurringAmount, setRecurringAmount] = useState<string>(
    recurringEvent ? String(recurringEvent.computed_amount) : "",
  );
  const [recurringPosting, setRecurringPosting] = useState(false);
  const [recurringMsg, setRecurringMsg] = useState<string>("");
  useEffect(() => {
    if (recurringEvent) setRecurringAmount(String(recurringEvent.computed_amount));
  }, [recurringEvent?.id, recurringEvent?.computed_amount]);

  // Live Portfolio Heat preview — fires once per card mount. The backend
  // recomputes against the latest saved end_nlv, so this is
  // "yesterday's-NLV heat with today's positions/prices." Good enough for
  // a pre-save glance at risk exposure; the exact stamp still happens on
  // save via _compute_portfolio_heat, which uses today's typed NLV.
  const [previewHeat, setPreviewHeat] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.portfolioHeatPreview(card.name)
      .then(r => { if (!cancelled) setPreviewHeat(r.heat); })
      .catch(err => log.error("daily-journal", `heat preview fetch failed for ${card.name}`, err));
    return () => { cancelled = true; };
  }, [card.name]);

  return (
    <div
      className="rounded-[14px] overflow-hidden"
      style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}
      data-testid={`portfolio-card-${card.name}`}
    >
      <div className="flex items-center gap-2 px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: accentColor }} />
        <span className="text-[13px] font-semibold">{card.name}</span>
      </div>
      <div className="p-4 flex flex-col gap-3">
        {/* Recurring-deposit reminder (Migration 059). Renders only when
            this portfolio has an active config whose next_due_date has
            arrived. Post writes a cash_transactions row AND bumps this
            card's cash_change so the TWR-relevant journal row picks up
            the deposit on save. Amount input is pre-filled but editable
            (one-off override; config stays intact). */}
        {recurringEvent?.is_due && (
          <div className="rounded-[10px] p-3 flex flex-col gap-2"
               style={{ background: "color-mix(in oklab, " + accentColor + " 10%, var(--surface))",
                        border: "1px solid color-mix(in oklab, " + accentColor + " 30%, var(--border))" }}
               data-testid={`recurring-reminder-${card.name}`}>
            <div className="flex items-baseline justify-between gap-2">
              <div className="text-[12px] font-semibold" style={{ color: "var(--ink-1)" }}>
                {recurringEvent.note || "Recurring deposit"} · due {recurringEvent.next_due_date}
              </div>
              <button type="button" onClick={onOpenManage}
                      className="text-[10px] underline"
                      style={{ color: "var(--ink-4)" }}>
                Manage
              </button>
            </div>
            <div className="text-[10px]" style={{ color: "var(--ink-3)" }}>
              Base ${recurringEvent.base_amount.toFixed(2)} × {recurringEvent.percent.toFixed(0)}% ={" "}
              <b>${recurringEvent.computed_amount.toFixed(2)}</b>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>$</span>
              <input
                type="number"
                step="1"
                value={recurringAmount}
                onChange={e => setRecurringAmount(e.target.value)}
                disabled={recurringPosting}
                className="flex-1 h-[30px] px-2 rounded-[6px] text-[12px] outline-none"
                style={{ background: "var(--surface)", border: "1px solid var(--border)",
                         color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" }}
                aria-label={`Recurring deposit amount for ${card.name}`}
              />
              <button type="button"
                      disabled={recurringPosting}
                      onClick={async () => {
                        setRecurringPosting(true);
                        setRecurringMsg("");
                        try {
                          await onRecurringSkip(recurringEvent.id);
                          setRecurringMsg("Skipped");
                        } catch (e) {
                          setRecurringMsg(`Skip failed: ${e}`);
                        } finally {
                          setRecurringPosting(false);
                        }
                      }}
                      className="px-2.5 py-1 rounded-[6px] text-[11px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)",
                               color: "var(--ink-3)" }}>
                Skip
              </button>
              <button type="button"
                      disabled={recurringPosting}
                      onClick={async () => {
                        setRecurringPosting(true);
                        setRecurringMsg("");
                        try {
                          const amt = parseFloat(recurringAmount || "0") || 0;
                          const posted = await onRecurringPost(recurringEvent.id, amt);
                          if (posted !== null) setRecurringMsg(`Posted $${posted.toFixed(2)}`);
                        } catch (e) {
                          setRecurringMsg(`Post failed: ${e}`);
                        } finally {
                          setRecurringPosting(false);
                        }
                      }}
                      className="px-2.5 py-1 rounded-[6px] text-[11px] font-semibold"
                      style={{ background: accentColor, color: "white",
                               opacity: recurringPosting ? 0.5 : 1 }}>
                {recurringPosting ? "…" : "Post deposit"}
              </button>
            </div>
            {recurringMsg && (
              <div className="text-[10px]"
                   style={{ color: recurringMsg.startsWith("Posted") ? "#0a8f5e"
                                   : recurringMsg === "Skipped" ? "var(--ink-4)"
                                   : "#b23a2b" }}>
                {recurringMsg}
              </div>
            )}
          </div>
        )}
        <Field label="Closing NLV*" error={card.touched.end_nlv ? card.errors.end_nlv : undefined}>
          <input
            type="number"
            value={card.end_nlv}
            onChange={(e) => onChange({ end_nlv: e.target.value, errors: { ...card.errors, end_nlv: undefined } })}
            onBlur={() => onChange({
              touched: { ...card.touched, end_nlv: true },
              errors: validateCard({ ...card }),
            })}
            step="100"
            placeholder="0.00"
            className={inputCls}
            style={card.touched.end_nlv && card.errors.end_nlv ? inputErrorStyle : inputStyle}
            aria-label={`Closing NLV for ${card.name}`}
            data-testid={`nlv-input-${card.name}`}
          />
        </Field>
        <Field label="Total Holdings*" error={card.touched.total_holdings ? card.errors.total_holdings : undefined}>
          <input
            type="number"
            value={card.total_holdings}
            onChange={(e) => onChange({ total_holdings: e.target.value, errors: { ...card.errors, total_holdings: undefined } })}
            onBlur={() => onChange({
              touched: { ...card.touched, total_holdings: true },
              errors: validateCard({ ...card }),
            })}
            step="100"
            placeholder="0.00"
            className={inputCls}
            style={card.touched.total_holdings && card.errors.total_holdings ? inputErrorStyle : inputStyle}
            aria-label={`Total Holdings for ${card.name}`}
            data-testid={`holdings-input-${card.name}`}
          />
        </Field>
        <Field label="Cash +/-">
          <input
            type="number"
            value={card.cash_change}
            onChange={(e) => onChange({ cash_change: e.target.value })}
            step="100"
            className={inputCls}
            style={inputStyle}
            aria-label={`Cash flow for ${card.name}`}
          />
        </Field>
        <Field label="Actions">
          <input
            type="text"
            value={card.actions}
            onChange={(e) => onChange({ actions: e.target.value })}
            placeholder="BUY: NVDA"
            className={inputCls}
            style={{ ...inputStyle, fontFamily: "inherit" }}
            aria-label={`Actions for ${card.name}`}
          />
        </Field>
        {/* Quiet always-visible link — the entry point to Manage-recurring
            when the reminder isn't currently rendered. When not-due (or
            no config exists) surfaces a one-line status + Manage. When due
            it's redundant with the reminder card above, so hide. */}
        {!recurringEvent?.is_due && (
          <button type="button" onClick={onOpenManage}
                  className="self-start text-[10px] underline"
                  style={{ color: "var(--ink-4)" }}
                  data-testid={`recurring-manage-link-${card.name}`}>
            {recurringEvent
              ? `Recurring: next $${recurringEvent.computed_amount.toFixed(0)} on ${recurringEvent.next_due_date} · Manage`
              : "Add recurring deposit"}
          </button>
        )}
        {m.nlv > 0 && (
          <div className="grid grid-cols-2 gap-2 mt-1">
            {[
              { k: "Prev NLV", v: formatCurrency(card.prev_end_nlv, { decimals: 0 }) },
              { k: "Daily $", v: formatCurrency(m.daily_dollar_change, { showSign: true, decimals: 0 }), c: m.daily_dollar_change >= 0 ? "#08a86b" : "#e5484d" },
              { k: "Daily %", v: `${m.daily_pct_change >= 0 ? "+" : ""}${m.daily_pct_change.toFixed(2)}%`, c: m.daily_pct_change >= 0 ? "#08a86b" : "#e5484d" },
              { k: "% Invested", v: `${m.pct_invested.toFixed(1)}%` },
              // Portfolio Heat preview. Amber >20% (target ceiling), red >30%
              // as an unmistakable "too hot to save without a look" nudge.
              // Uses the current daily % swing threshold from Portfolio Heat.
              { k: "Heat (auto)", v: previewHeat === null ? "…" : `${previewHeat.toFixed(2)}%`,
                c: previewHeat === null ? undefined
                   : previewHeat > 30 ? "#e5484d"
                   : previewHeat > 20 ? "#f59f00"
                   : "#08a86b" },
            ].map((s) => (
              <div
                key={s.k}
                className={`p-2 rounded-[8px] ${s.k === "Heat (auto)" ? "col-span-2" : ""}`}
                style={{ border: "1px solid var(--border)" }}
              >
                <div className="text-[8px] uppercase tracking-[0.06em] font-semibold" style={{ color: "var(--ink-4)" }}>{s.k}</div>
                <div
                  className="text-[13px] font-semibold mt-0.5 privacy-mask"
                  style={{ fontFamily: "var(--font-jetbrains), monospace", color: (s as { c?: string }).c || "var(--ink)" }}
                >
                  {s.v}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// NLVEntry — multi-portfolio entry view. One Market block + N portfolio
// cards + one Report Card block. Saves via /api/journal/batch-edit
// atomically across all portfolios.
// ─────────────────────────────────────────────────────────────────────────────

type SaveError =
  | { kind: "conflict"; conflicting_portfolios: string[] }
  | { kind: "error"; detail: string };

export function NLVEntry({ navColor }: { navColor: string }) {
  const { portfolios } = usePortfolio();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState<string>("");
  const [saveError, setSaveError] = useState<SaveError | null>(null);

  // Shared singletons.
  const [spyClose, setSpyClose] = useState("");
  const [ndxClose, setNdxClose] = useState("");
  const [marketNotes, setMarketNotes] = useState("");
  // "Fetch official close" state — button on the header row that hits
  // /api/journal/refresh-index-closes for the current entryDate. Fixes
  // stored intraday captures without the user manually retyping.
  const [refreshingIdx, setRefreshingIdx] = useState(false);
  const [refreshIdxMsg, setRefreshIdxMsg] = useState<string>("");
  const [entryDate, setEntryDate] = useState(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  });
  // Scorecard (plan/stops/sized/fomo + grade notes) was retired from
  // this form 2026-07-26 when the ScorecardMiniForm on the Journal
  // checklist item became the primary capture point. Both paths write
  // to the same trading_journal fields (score, highlights, mistakes);
  // removing the duplicate here prevents the two entry points from
  // clobbering each other.
  const [forceOverwrite, setForceOverwrite] = useState(false);
  // Tracks whether the user has clicked Save yet. Combined with per-card
  // `touched` flags to gate the validation-summary banner — first paint
  // shouldn't surface "fix N errors" before the user has interacted.
  const [submitAttempted, setSubmitAttempted] = useState(false);

  // Per-card state, derived from the portfolios context. Initialized empty
  // and populated by the per-entryDate effect below.
  const [cards, setCards] = useState<PortfolioCardState[]>([]);

  // Recurring-cash configs keyed by portfolio name. Loaded once per
  // portfolios-list change. Card renders the reminder / Manage link off
  // the entry for its portfolio (null when none configured).
  const [recurringByPortfolio, setRecurringByPortfolio] =
    useState<Record<string, RecurringCashEvent | null>>({});
  const [manageModalPortfolio, setManageModalPortfolio] = useState<string | null>(null);

  const reloadRecurringFor = useCallback(async (portfolioName: string) => {
    try {
      const res = await api.recurringCashList(portfolioName);
      if ("events" in res) {
        setRecurringByPortfolio(prev => ({
          ...prev,
          [portfolioName]: res.events[0] || null,
        }));
      }
    } catch (err) {
      log.error("nlv-entry", `recurring-cash reload failed for ${portfolioName}`, err);
    }
  }, []);

  // Shared IBKR loading flag (dormant while IBKR_AUTOFILL_ENABLED=false).
  const [nlvLoading, setNlvLoading] = useState(true);
  const [ibkrError, setIbkrError] = useState<string>("");

  // Rebuild cards whenever the portfolio list or the entry date changes.
  // We:
  //   1. Build a fresh emptyCard() per portfolio
  //   2. Concurrently fetch prior-day end_nlv per portfolio (drives Daily %
  //      computation)
  //   3. Concurrently fetch today's trade details per portfolio (drives the
  //      Actions auto-fill string)
  //   4. Fetch SPY/NDX close once (shared)
  // The single Promise.all keeps the loading flag honest and lets failures
  // degrade gracefully per portfolio (one failure doesn't tank the whole
  // page).
  useEffect(() => {
    if (!portfolios.length) return;

    let cancelled = false;
    setLoading(true);
    setSaveOk("");
    setSaveError(null);
    // A new date is a fresh validation context — clear the submit flag so
    // the summary banner doesn't carry over from a prior save attempt on
    // a different day.
    setSubmitAttempted(false);

    const today = new Date();
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
    const isPastDate = entryDate < todayStr;

    const pricesPromise = api.batchPrices(["SPY", "^IXIC"], undefined, isPastDate ? entryDate : undefined).catch((err) => {
      log.debug.devOnly("daily-journal", "batchPrices pre-fill missing (expected)", err);
      return {} as Record<string, number>;
    });

    const perPortfolioPromises = portfolios.map((p) =>
      Promise.all([
        api.journalLatest(p.name, entryDate).catch((err) => {
          log.debug.devOnly("daily-journal", `journalLatest pre-fill missing for ${p.name}`, err);
          return { end_nlv: 0 };
        }),
        api.tradesRecent(p.name, 1000).catch((err) => {
          log.debug.devOnly("daily-journal", `tradesRecent pre-fill missing for ${p.name}`, err);
          return { details: [], lot_closures: [] };
        }),
      ]).then(([latest, trades]) => ({ p, latest, trades }))
    );

    Promise.all([pricesPromise, ...perPortfolioPromises]).then((results) => {
      if (cancelled) return;

      const prices = results[0] as Record<string, number>;
      if (prices["SPY"]) setSpyClose(prices["SPY"].toFixed(2));
      if (prices["^IXIC"]) setNdxClose(prices["^IXIC"].toFixed(2));

      const built: PortfolioCardState[] = [];
      for (let i = 1; i < results.length; i++) {
        const { p, latest, trades } = results[i] as { p: { id: number; name: string }; latest: { end_nlv?: number }; trades: { details?: { date?: string; action?: string; ticker?: string }[] } };
        const card = emptyCard(p);
        card.prev_end_nlv = parseFloat(String(latest.end_nlv || 0)) || 0;
        card.actions = buildActionsString(trades.details || [], entryDate);
        built.push(card);
      }
      setCards(built);
      setLoading(false);
      setNlvLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [portfolios, entryDate]);

  // IBKR auto-fill — dormant by config. Kept as a no-op skeleton so the
  // surrounding UI machinery (warning banner, loading flag) stays consistent
  // with the prior implementation if/when the flag flips back on.
  useEffect(() => {
    if (!IBKR_AUTOFILL_ENABLED) {
      setNlvLoading(false);
      setIbkrError("");
      return;
    }
    // When re-enabled: fire api.ibkrNavForDate per portfolio, update each
    // card's end_nlv + total_holdings + sources independently. Today's IBKR
    // Flex Query is account-scoped (one user → one account), so this would
    // need a per-portfolio account mapping or a switch to a portfolio-aware
    // IBKR endpoint. Out of scope for the Phase B redesign.
  }, [entryDate]);

  // Rally prefix — same behavior as pre-redesign (shared across all
  // portfolios since the prefix is market-state-driven, not portfolio-
  // scoped).
  useEffect(() => {
    let cancelled = false;
    api.rallyPrefix(entryDate).catch((err) => {
      log.debug.devOnly("daily-journal", "rallyPrefix pre-fill missing (expected)", err);
      return { prefix: "" };
    }).then((rally) => {
      if (cancelled) return;
      const prefix = (rally as { prefix?: string }).prefix || "";
      if (prefix) setMarketNotes(prefix);
    });
    return () => { cancelled = true; };
  }, [entryDate]);

  const updateCard = (name: string, patch: Partial<PortfolioCardState>) => {
    setCards((prev) => prev.map((c) => (c.name === name ? { ...c, ...patch } : c)));
    // Any input edit clears the save banners so the user sees their input
    // was registered.
    if (saveOk) setSaveOk("");
    if (saveError) setSaveError(null);
  };

  // Load recurring configs for every portfolio (parallel). Re-runs when
  // the portfolios list changes; date changes don't refetch since the
  // config isn't date-scoped (is_due is computed against today).
  useEffect(() => {
    if (!portfolios.length) return;
    let cancelled = false;
    Promise.all(portfolios.map(p =>
      api.recurringCashList(p.name).catch(err => {
        log.debug.devOnly("nlv-entry", `recurringCashList failed for ${p.name}`, err);
        return { events: [] as RecurringCashEvent[] };
      }).then(r => [p.name, ("events" in r ? r.events[0] : null) || null] as const)
    )).then(pairs => {
      if (cancelled) return;
      setRecurringByPortfolio(Object.fromEntries(pairs));
    });
    return () => { cancelled = true; };
  }, [portfolios]);

  // Post → writes cash_transactions + bumps this card's cash_change so
  // the journal row (TWR consumer) picks up the deposit on save. Returns
  // the posted amount for the reminder's inline confirmation.
  const handleRecurringPost = useCallback(async (portfolioName: string, id: number, amount: number) => {
    const res = await api.recurringCashPost(id, { amount });
    if ("error" in res) throw new Error(res.error);
    // Bump the card's cash_change by the posted amount (preserves any
    // manual entry the user might already have made).
    setCards(prev => prev.map(c => {
      if (c.name !== portfolioName) return c;
      const cur = parseFloat(c.cash_change || "0") || 0;
      return { ...c, cash_change: String(cur + amount) };
    }));
    setRecurringByPortfolio(prev => ({ ...prev, [portfolioName]: res.event }));
    return amount;
  }, []);

  const handleRecurringSkip = useCallback(async (portfolioName: string, id: number) => {
    const res = await api.recurringCashSkip(id);
    if ("error" in res) throw new Error(res.error);
    setRecurringByPortfolio(prev => ({ ...prev, [portfolioName]: res.event }));
  }, []);

  // Aggregate validation across all cards. Memoized so the disabled-state
  // calculation doesn't re-run validateCard on every render of children.
  const validationSummary = useMemo(() => {
    const errs: { name: string; field: string; message: string }[] = [];
    for (const c of cards) {
      const cardErrs = validateCard(c);
      for (const [field, message] of Object.entries(cardErrs)) {
        if (message) errs.push({ name: c.name, field, message });
      }
    }
    return errs;
  }, [cards]);

  const hasErrors = validationSummary.length > 0;

  async function handleSave() {
    setSaving(true);
    setSaveOk("");
    setSaveError(null);
    setSubmitAttempted(true);

    // Defensive: re-run validation on submit even though the button is
    // disabled when hasErrors. Mark all fields touched so any errors that
    // were silent (user never blurred the field) light up the red borders
    // and inline messages. Mirror errors back onto cards.
    const validated = cards.map((c) => ({
      ...c,
      touched: { end_nlv: true, total_holdings: true },
      errors: validateCard(c),
    }));
    setCards(validated);
    const stillHasErrors = validated.some((c) => Object.keys(c.errors).length > 0);
    if (stillHasErrors) {
      setSaving(false);
      return;
    }

    const payload = {
      day: entryDate,
      shared: {
        spy: parseFloat(spyClose) || 0,
        nasdaq: parseFloat(ndxClose) || 0,
        market_notes: marketNotes,
        // score / highlights / mistakes intentionally omitted — the
        // Journal checklist item's ScorecardMiniForm owns those
        // fields now. Backend PATCH keeps existing values when a
        // key is missing from `shared`, so today's grade is
        // preserved when NLV Entry saves after the mini-form.
        nlv_source: "manual",
        holdings_source: "manual",
      },
      portfolios: validated.map((c) => {
        const m = deriveCardMetrics(c);
        return {
          portfolio: c.name,
          end_nlv: parseFloat(c.end_nlv),
          total_holdings: parseFloat(c.total_holdings),
          cash_change: parseFloat(c.cash_change) || 0,
          actions: c.actions,
          pct_invested: m.pct_invested,
          daily_dollar_change: m.daily_dollar_change,
          daily_pct_change: m.daily_pct_change,
        };
      }),
      force_overwrite: forceOverwrite,
    };

    try {
      const r = await api.journalBatchEdit(payload);
      if (r.status === "exists") {
        setSaveError({
          kind: "conflict",
          conflicting_portfolios: r.conflicting_portfolios || [],
        });
      } else if (r.status === "ok") {
        setSaveOk(`Saved ${r.rows_written ?? validated.length} portfolios`);
        // Autotick "Equity routine" in the checklist — fire and forget,
        // does not block the save's OK state.
        void autoTickByPrefix(SYSTEM_ITEM_PREFIXES.equityRoutine);
      } else {
        // 422 validation (shouldn't reach here client-side; defensive), 404,
        // 500 surface their detail.
        setSaveError({
          kind: "error",
          detail: r.detail || `Save failed (${r.status})`,
        });
      }
    } catch (e) {
      setSaveError({
        kind: "error",
        detail: e instanceof Error ? e.message : String(e),
      });
    }
    setSaving(false);
  }

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  const cardAccents = ["#6366f1", "#08a86b", "#f59f00", "#a855f7", "#06b6d4"];

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          NLV <em className="italic" style={{ color: navColor }}>Entry</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Master Blotter · All Portfolios · End-of-Day
        </div>
      </div>

      {ibkrError && !nlvLoading && (
        <div className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]" role="alert" data-testid="ibkr-warning-banner"
             style={{
               background: "color-mix(in oklab, #f59f00 10%, var(--surface))",
               color: "#b45309",
               border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))",
             }}>
          ⚠ Could not auto-fill NLV from IBKR — please enter manually. Reason: {ibkrError}
        </div>
      )}

      {/* Market — shared inputs */}
      <div className="rounded-[14px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Market</span>
        </div>
        <div className="p-4 grid grid-cols-1 md:grid-cols-4 gap-3">
          <Field label="Date">
            <input type="date" value={entryDate} onChange={(e) => setEntryDate(e.target.value)} className={inputCls} style={inputStyle} aria-label="Entry date" />
          </Field>
          <Field label="SPY Close">
            <input type="number" value={spyClose} onChange={(e) => setSpyClose(e.target.value)} step="0.01" className={inputCls} style={inputStyle} />
          </Field>
          <Field label="Nasdaq Close">
            <input type="number" value={ndxClose} onChange={(e) => setNdxClose(e.target.value)} step="0.01" className={inputCls} style={inputStyle} />
          </Field>
          <Field label="Market Notes">
            <input type="text" value={marketNotes} onChange={(e) => setMarketNotes(e.target.value)}
                   placeholder="Day 14 UPTREND: ..." className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
          </Field>
        </div>

        {/* Intraday-capture warning + "Fetch official close" button.
            The SPY/NDX values pre-fill from a live batchPrices call. When
            entryDate is today AND the market is still open, those are
            INTRADAY values, not the day's close — persisting them silently
            distorted the historical %-change series for months. This strip
            makes the risk visible and gives a one-click fix. */}
        {(() => {
          const now = new Date();
          const todayStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
          const isToday = entryDate === todayStr;
          // Rough US Eastern market-hours check. 9:30am-4:15pm ET.
          const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
          const marketOpen = (et.getHours() > 9 || (et.getHours() === 9 && et.getMinutes() >= 30))
                          && (et.getHours() < 16 || (et.getHours() === 16 && et.getMinutes() < 15));
          const showWarning = isToday && marketOpen;
          const runRefresh = async () => {
            setRefreshingIdx(true);
            setRefreshIdxMsg("");
            try {
              const res = await api.refreshIndexCloses(entryDate);
              if (res.error) {
                setRefreshIdxMsg(`Failed: ${res.error}`);
              } else {
                const changes = res.updates.length;
                if (res.spy_official) setSpyClose(res.spy_official.toFixed(2));
                if (res.ixic_official) setNdxClose(res.ixic_official.toFixed(2));
                setRefreshIdxMsg(
                  changes === 0
                    ? "Already at official close."
                    : `Corrected ${changes} row${changes === 1 ? "" : "s"} across portfolios.`,
                );
              }
            } catch (e) {
              setRefreshIdxMsg(`Failed: ${e}`);
            } finally {
              setRefreshingIdx(false);
            }
          };
          if (!showWarning && !refreshIdxMsg) {
            return (
              <div className="px-4 pb-3 flex items-center justify-end gap-3 text-[12px]"
                   style={{ color: "var(--ink-4)" }}>
                <button type="button" onClick={runRefresh} disabled={refreshingIdx}
                        className="underline hover:no-underline">
                  {refreshingIdx ? "Fetching…" : "Fetch official close"}
                </button>
              </div>
            );
          }
          return (
            <div className="mx-4 mb-4 px-3 py-2.5 rounded-[8px] flex items-start gap-2 text-[12px]"
                 style={{ background: "color-mix(in oklab, #d97706 8%, var(--surface))",
                          border: "1px solid color-mix(in oklab, #d97706 25%, var(--border))",
                          color: "#92400e" }}>
              <span className="text-[14px] leading-none pt-0.5">⚠</span>
              <div className="flex-1">
                {showWarning && (
                  <div>
                    <strong>Market is still open</strong> — the pre-filled SPY / Nasdaq values are
                    intraday, not today&apos;s close. Save now if you want a mid-session snapshot;
                    otherwise click <em>Fetch official close</em> after 4:15pm ET (or run the
                    nightly backend job, which does the same thing).
                  </div>
                )}
                {refreshIdxMsg && (
                  <div className={showWarning ? "mt-1.5" : ""}
                       style={{ color: refreshIdxMsg.startsWith("Failed") ? "#b23a2b" : "#0a8f5e" }}>
                    {refreshIdxMsg}
                  </div>
                )}
              </div>
              <button type="button" onClick={runRefresh} disabled={refreshingIdx}
                      className="shrink-0 text-[11px] font-medium px-2.5 py-1 rounded-[6px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)",
                               color: "var(--ink-2)" }}>
                {refreshingIdx ? "Fetching…" : "Fetch official close"}
              </button>
            </div>
          );
        })()}
      </div>

      {/* Portfolio cards — N side-by-side on desktop, stack on mobile */}
      <div
        className="grid gap-4 mb-4"
        style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}
        data-testid="portfolio-grid"
      >
        {cards.map((card, i) => (
          <PortfolioCard
            key={card.name}
            card={card}
            onChange={(patch) => updateCard(card.name, patch)}
            accentColor={cardAccents[i % cardAccents.length]}
            recurringEvent={recurringByPortfolio[card.name] || null}
            onRecurringPost={(id, amount) => handleRecurringPost(card.name, id, amount)}
            onRecurringSkip={(id) => handleRecurringSkip(card.name, id)}
            onOpenManage={() => setManageModalPortfolio(card.name)}
          />
        ))}
      </div>

      {/* Report Card retired 2026-07-26 — captured via the
          ScorecardMiniForm on the Daily Journal page's Journal
          checklist item. */}

      {/* Submit area */}
      <label className="flex items-center gap-2 mb-4 cursor-pointer text-[12px]" style={{ color: "var(--ink-3)" }}>
        <input
          type="checkbox"
          checked={forceOverwrite}
          onChange={(e) => setForceOverwrite(e.target.checked)}
          className="rounded"
          data-testid="force-overwrite-checkbox"
        />
        Force Overwrite Existing Entry
      </label>

      {/* Validation summary — shown only after the user has interacted.
          Two gates:
            (a) submitAttempted: the user has clicked Save at least once
            (b) any blurred-while-empty field exists
          Either of these means the user has surfaced their intent to
          fill out the form; we can confidently nag them about gaps.
          Without these gates, the banner would appear on initial paint
          before the user has typed anything — a hostile first impression. */}
      {hasErrors && (submitAttempted || cards.some((c) =>
        (c.touched.end_nlv && c.errors.end_nlv) ||
        (c.touched.total_holdings && c.errors.total_holdings)
      )) && (
        <div
          className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
          role="alert"
          data-testid="validation-summary"
          style={{
            background: "color-mix(in oklab, #f59f00 10%, var(--surface))",
            color: "#b45309",
            border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))",
          }}
        >
          <div className="font-semibold mb-1">
            Fix {validationSummary.length} {validationSummary.length === 1 ? "error" : "errors"} before saving:
          </div>
          <ul className="list-disc pl-5">
            {validationSummary.map((e, idx) => (
              <li key={`${e.name}-${e.field}-${idx}`}>
                {e.name}: {e.field === "end_nlv" ? "Closing NLV" : "Total Holdings"} {e.message.toLowerCase()}
              </li>
            ))}
          </ul>
        </div>
      )}

      {saveError?.kind === "conflict" && (
        <div
          className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
          role="alert"
          data-testid="conflict-banner"
          style={{
            background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
            color: "#dc2626",
            border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
          }}
        >
          Rows already exist for {saveError.conflicting_portfolios.join(", ")}. Check
          <strong> Force Overwrite Existing Entry </strong> above to replace them.
        </div>
      )}

      {saveError?.kind === "error" && (
        <div
          className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
          role="alert"
          data-testid="save-error-banner"
          style={{
            background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
            color: "#dc2626",
            border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
          }}
        >
          Error: {saveError.detail}
        </div>
      )}

      {saveOk && (
        <div
          className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
          role="status"
          data-testid="save-ok-banner"
          style={{
            background: "color-mix(in oklab, #08a86b 10%, var(--surface))",
            color: "#16a34a",
            border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))",
          }}
        >
          {saveOk}
        </div>
      )}

      <button
        onClick={handleSave}
        disabled={saving || hasErrors || cards.length === 0}
        className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
        style={{ background: "#6366f1" }}
        data-testid="save-button"
      >
        {saving ? "Saving..." : "Save NLV Entry"}
      </button>
      {manageModalPortfolio && (
        <ManageRecurringModal
          portfolio={manageModalPortfolio}
          event={recurringByPortfolio[manageModalPortfolio] || null}
          onClose={() => setManageModalPortfolio(null)}
          onSaved={async () => { await reloadRecurringFor(manageModalPortfolio); }}
          navColor={navColor}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ManageRecurringModal — create/edit/pause/delete the recurring config for a
// portfolio. Opened from the reminder card's "Manage" button or from the
// always-visible "Add / Manage recurring" footer link on each PortfolioCard.
// Single form; the backend accepts partial updates on PUT so unchanged
// fields pass through.
// ─────────────────────────────────────────────────────────────────────────────

function ManageRecurringModal({
  portfolio,
  event,
  onClose,
  onSaved,
  navColor,
}: {
  portfolio: string;
  event: RecurringCashEvent | null;
  onClose: () => void;
  onSaved: () => Promise<void>;
  navColor: string;
}) {
  const [baseAmount, setBaseAmount] = useState(
    event ? String(event.base_amount) : "",
  );
  const [percent, setPercent] = useState(
    event ? String(event.percent) : "100",
  );
  const [cadenceDays, setCadenceDays] = useState(
    event ? String(event.cadence_days) : "14",
  );
  // For create: next fire is the anchor. For edit: showing next_due_date
  // and letting the user reseed the cycle is more useful than exposing
  // the historic anchor separately.
  const [nextDueDate, setNextDueDate] = useState(
    event?.next_due_date || event?.anchor_date || "",
  );
  const [note, setNote] = useState(event?.note || "");
  const [active, setActive] = useState(event?.active ?? true);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");

  const base = parseFloat(baseAmount || "0") || 0;
  const pct = parseFloat(percent || "0") || 0;
  const computed = Math.round(base * pct) / 100;

  const save = async () => {
    setSaving(true); setErr("");
    try {
      if (event) {
        const res = await api.recurringCashUpdate(event.id, {
          base_amount: base,
          percent: pct,
          cadence_days: parseInt(cadenceDays, 10) || 14,
          next_due_date: nextDueDate || undefined,
          note,
          active,
        });
        if ("error" in res) throw new Error(res.error);
      } else {
        if (!nextDueDate) { setErr("Anchor date is required."); setSaving(false); return; }
        const res = await api.recurringCashCreate({
          portfolio,
          anchor_date: nextDueDate,
          base_amount: base,
          percent: pct,
          cadence_days: parseInt(cadenceDays, 10) || 14,
          note,
          active,
        });
        if ("error" in res) throw new Error(res.error);
      }
      await onSaved();
      onClose();
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally { setSaving(false); }
  };

  const remove = async () => {
    if (!event) return;
    if (!confirm(`Delete the recurring deposit config for ${portfolio}?`)) return;
    setSaving(true); setErr("");
    try {
      const res = await api.recurringCashDelete(event.id);
      if ("error" in res) throw new Error(res.error);
      await onSaved();
      onClose();
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally { setSaving(false); }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.4)" }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="rounded-[14px] p-6 max-w-md w-full"
           style={{ background: "var(--surface)", border: "1px solid var(--border)",
                    boxShadow: "0 20px 60px rgba(0,0,0,0.2)" }}
           data-testid="recurring-manage-modal">
        <div className="flex items-baseline justify-between mb-4">
          <h3 className="font-normal text-[22px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: "var(--ink-1)" }}>
            {event ? "Edit recurring" : "Add recurring"}{" "}
            <em className="italic" style={{ color: navColor }}>{portfolio}</em>
          </h3>
          <button onClick={onClose} className="text-[24px] leading-none"
                  style={{ color: "var(--ink-4)" }}>×</button>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-3">
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold block mb-1"
                  style={{ color: "var(--ink-4)" }}>Base amount</span>
            <input type="number" step="1" value={baseAmount}
                   onChange={e => setBaseAmount(e.target.value)}
                   className="w-full h-[36px] px-3 rounded-[8px] text-[13px] outline-none"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </label>
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold block mb-1"
                  style={{ color: "var(--ink-4)" }}>Percent</span>
            <input type="number" step="1" value={percent}
                   onChange={e => setPercent(e.target.value)}
                   className="w-full h-[36px] px-3 rounded-[8px] text-[13px] outline-none"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </label>
        </div>
        <div className="text-[11px] mb-3" style={{ color: "var(--ink-3)" }}>
          Post amount each cycle: <b>${computed.toFixed(2)}</b> ({percent || 0}% × ${baseAmount || 0})
        </div>

        <div className="grid grid-cols-2 gap-3 mb-3">
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold block mb-1"
                  style={{ color: "var(--ink-4)" }}>Cadence (days)</span>
            <input type="number" step="1" value={cadenceDays}
                   onChange={e => setCadenceDays(e.target.value)}
                   className="w-full h-[36px] px-3 rounded-[8px] text-[13px] outline-none"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </label>
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold block mb-1"
                  style={{ color: "var(--ink-4)" }}>{event ? "Next due" : "First due (anchor)"}</span>
            <input type="date" value={nextDueDate}
                   onChange={e => setNextDueDate(e.target.value)}
                   className="w-full h-[36px] px-3 rounded-[8px] text-[13px] outline-none"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </label>
        </div>

        <label className="block mb-3">
          <span className="text-[10px] uppercase tracking-[0.10em] font-semibold block mb-1"
                style={{ color: "var(--ink-4)" }}>Note</span>
          <input type="text" value={note} onChange={e => setNote(e.target.value)}
                 placeholder="457B bi-weekly contribution"
                 className="w-full h-[36px] px-3 rounded-[8px] text-[13px] outline-none"
                 style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
        </label>

        <label className="flex items-center gap-2 mb-4 cursor-pointer text-[12px]"
               style={{ color: "var(--ink-2)" }}>
          <input type="checkbox" checked={active}
                 onChange={e => setActive(e.target.checked)} />
          <span>Active — reminder card renders when due</span>
        </label>

        {err && <div className="mb-3 text-[12px]" style={{ color: "#e5484d" }}>{err}</div>}

        <div className="flex justify-between items-center gap-2">
          <div>
            {event && (
              <button onClick={remove} disabled={saving}
                      className="px-3 py-2 rounded-[10px] text-[13px]"
                      style={{ background: "transparent", border: "1px solid var(--border)", color: "#e5484d" }}>
                Delete
              </button>
            )}
          </div>
          <div className="flex gap-2">
            <button onClick={onClose} disabled={saving}
                    className="px-3 py-2 rounded-[10px] text-[13px]"
                    style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
              Cancel
            </button>
            <button onClick={save} disabled={saving || base < 0}
                    className="px-3 py-2 rounded-[10px] text-[13px] font-medium"
                    style={{ background: navColor, color: "white", opacity: saving ? 0.5 : 1 }}>
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function buildActionsString(details: { date?: string; action?: string; ticker?: string }[], day: string): string {
  const grouped: Record<string, string[]> = {};
  for (const d of details) {
    const dDate = String(d.date || "").slice(0, 10);
    if (dDate !== day) continue;
    const action = String(d.action || "").toUpperCase();
    const ticker = String(d.ticker || "").trim();
    if (!action || !ticker) continue;
    if (!grouped[action]) grouped[action] = [];
    if (!grouped[action].includes(ticker)) grouped[action].push(ticker);
  }
  const parts: string[] = [];
  for (const label of ["SELL", "BUY"]) {
    if (grouped[label]) parts.push(`${label}: ${grouped[label].join(", ")}`);
  }
  for (const label of Object.keys(grouped)) {
    if (label !== "SELL" && label !== "BUY") parts.push(`${label}: ${grouped[label].join(", ")}`);
  }
  return parts.join(" | ");
}
