"use client";

import { useState, useEffect, useMemo } from "react";
import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { formatCurrency } from "@/lib/format";
import { gradeColor } from "@/lib/grade-helpers";
import { log } from "@/lib/log";

// IBKR Flex auto-fill is dormant: the upstream Flex Query has been returning
// "request error (1001) — statement could not be generated" intermittently,
// so the auto-pull is more annoying than helpful. Manual entry only until
// the upstream issue is resolved. Flip this back to true to re-enable the
// effect + the warning banner. NB: the multi-portfolio redesign keeps the
// per-card IBKR scaffolding so re-enabling drops in cleanly; the effect
// would have to be retargeted per portfolio (the Flex Query is account-
// scoped, so each portfolio needs its own pull URL or filter).
const IBKR_AUTOFILL_ENABLED = false;

const REPORT_CATEGORIES = [
  { key: "plan", label: "Followed plan" },
  { key: "stops", label: "Respected stops" },
  { key: "sized", label: "Sized correctly" },
  { key: "fomo", label: "No FOMO entries" },
];

function letterGrade(total: number, max: number): string {
  const pct = (total / max) * 100;
  if (pct >= 100) return "A+";
  if (pct >= 93) return "A";
  if (pct >= 87) return "A-";
  if (pct >= 83) return "B+";
  if (pct >= 77) return "B";
  if (pct >= 70) return "B-";
  if (pct >= 67) return "C+";
  if (pct >= 60) return "C";
  if (pct >= 53) return "C-";
  if (pct >= 47) return "D";
  return "F";
}
function gradeToScore(g: string) {
  return g.startsWith("A") ? 5 : g.startsWith("B") ? 4 : g.startsWith("C") ? 3 : g.startsWith("D") ? 2 : 1;
}
function scoreColor(v: number) {
  return v >= 4 ? "#08a86b" : v >= 3 ? "#f59f00" : "#e5484d";
}

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
  // daily-routine.tsx pre-redesign at line 258-259 and the journal importer's
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
}: {
  card: PortfolioCardState;
  onChange: (patch: Partial<PortfolioCardState>) => void;
  accentColor: string;
}) {
  const m = deriveCardMetrics(card);

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
        {m.nlv > 0 && (
          <div className="grid grid-cols-2 gap-2 mt-1">
            {[
              { k: "Prev NLV", v: formatCurrency(card.prev_end_nlv, { decimals: 0 }) },
              { k: "Daily $", v: formatCurrency(m.daily_dollar_change, { showSign: true, decimals: 0 }), c: m.daily_dollar_change >= 0 ? "#08a86b" : "#e5484d" },
              { k: "Daily %", v: `${m.daily_pct_change >= 0 ? "+" : ""}${m.daily_pct_change.toFixed(2)}%`, c: m.daily_pct_change >= 0 ? "#08a86b" : "#e5484d" },
              { k: "% Invested", v: `${m.pct_invested.toFixed(1)}%` },
            ].map((s) => (
              <div key={s.k} className="p-2 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
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
// DailyRoutine — multi-portfolio entry view. One Market block + N portfolio
// cards + one Report Card block. Saves via /api/journal/batch-edit
// atomically across all portfolios.
// ─────────────────────────────────────────────────────────────────────────────

type SaveError =
  | { kind: "conflict"; conflicting_portfolios: string[] }
  | { kind: "error"; detail: string };

export function DailyRoutine({ navColor }: { navColor: string }) {
  const { portfolios } = usePortfolio();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState<string>("");
  const [saveError, setSaveError] = useState<SaveError | null>(null);

  // Shared singletons.
  const [spyClose, setSpyClose] = useState("");
  const [ndxClose, setNdxClose] = useState("");
  const [marketNotes, setMarketNotes] = useState("");
  const [entryDate, setEntryDate] = useState(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  });
  const [scores, setScores] = useState<Record<string, number>>({ plan: 5, stops: 5, sized: 5, fomo: 5 });
  const [gradeNotes, setGradeNotes] = useState("");
  const [forceOverwrite, setForceOverwrite] = useState(false);
  // Tracks whether the user has clicked Save yet. Combined with per-card
  // `touched` flags to gate the validation-summary banner — first paint
  // shouldn't surface "fix N errors" before the user has interacted.
  const [submitAttempted, setSubmitAttempted] = useState(false);

  // Per-card state, derived from the portfolios context. Initialized empty
  // and populated by the per-entryDate effect below.
  const [cards, setCards] = useState<PortfolioCardState[]>([]);

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
      log.debug.devOnly("daily-routine", "batchPrices pre-fill missing (expected)", err);
      return {} as Record<string, number>;
    });

    const perPortfolioPromises = portfolios.map((p) =>
      Promise.all([
        api.journalLatest(p.name, entryDate).catch((err) => {
          log.debug.devOnly("daily-routine", `journalLatest pre-fill missing for ${p.name}`, err);
          return { end_nlv: 0 };
        }),
        api.tradesRecent(p.name, 1000).catch((err) => {
          log.debug.devOnly("daily-routine", `tradesRecent pre-fill missing for ${p.name}`, err);
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
      log.debug.devOnly("daily-routine", "rallyPrefix pre-fill missing (expected)", err);
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

  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const grade = letterGrade(totalScore, REPORT_CATEGORIES.length * 5);
  const overallScore = gradeToScore(grade);

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
        score: overallScore,
        highlights: JSON.stringify(scores),
        mistakes: gradeNotes,
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
          Daily <em className="italic" style={{ color: navColor }}>Routine</em>
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
          />
        ))}
      </div>

      {/* Report Card — shared */}
      <div className="rounded-[14px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center justify-between px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#08a86b" }} />
            <span className="text-[13px] font-semibold">Report Card</span>
          </div>
          <span className="text-[28px] font-semibold" style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: gradeColor(grade), lineHeight: 1 }}>
            {grade}
          </span>
        </div>
        <div className="divide-y" style={{ borderColor: "var(--border)" }}>
          {REPORT_CATEGORIES.map((cat) => (
            <div key={cat.key} className="flex items-center justify-between px-4 py-3">
              <span className="text-[12px] font-medium">{cat.label}</span>
              <div className="flex items-center gap-2.5">
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={scores[cat.key]}
                  onChange={(e) => setScores({ ...scores, [cat.key]: parseInt(e.target.value) })}
                  className="w-[80px] h-1 rounded-full appearance-none cursor-pointer"
                  style={{ accentColor: scoreColor(scores[cat.key]) }}
                />
                <span className="text-[11px] font-semibold w-[28px] text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: scoreColor(scores[cat.key]) }}>
                  {scores[cat.key]}/5
                </span>
              </div>
            </div>
          ))}
        </div>
        <div className="px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <Field label="Grade Notes">
            <input
              type="text"
              value={gradeNotes}
              onChange={(e) => setGradeNotes(e.target.value)}
              placeholder="Optional..."
              className={inputCls}
              style={{ ...inputStyle, fontFamily: "inherit" }}
            />
          </Field>
        </div>
      </div>

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
        {saving ? "Saving..." : "Save Daily Routine"}
      </button>
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
