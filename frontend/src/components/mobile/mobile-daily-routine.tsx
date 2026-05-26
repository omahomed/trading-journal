"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, Edit3 } from "lucide-react";
import { api, type JournalEntry } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { gradeColor } from "@/lib/grade-helpers";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { MobileScoreSelector } from "./mobile-score-selector";
import { MobileToggleSwitch } from "./mobile-toggle-switch";
import { NumberFieldCell, TextFieldCell } from "./mobile-form-fields";

/**
 * Mobile Daily Routine — Phase 2 Step 5. Ports the desktop multi-
 * portfolio EOD save (daily-routine.tsx) to a touch-friendly layout
 * with three UX wins over the desktop flow:
 *
 *   1. Pre-load existing entry — single `journalLatest(name, day+1)`
 *      call per portfolio. If `response.day === entryDate`, the entry
 *      already exists for the target date; pre-fill the form, surface
 *      a "Editing existing entry" banner, and auto-enable Force
 *      Overwrite. Eliminates the desktop conflict-then-re-fill loop.
 *
 *   2. localStorage autosave — 500ms-debounced writes keyed per-date
 *      (`mo-daily-routine-draft-{YYYY-MM-DD}`). Restored on mount
 *      before fetches resolve so the user's most recent intent always
 *      wins over backend defaults. Cleared on save success; preserved
 *      on save failure so transient errors don't lose data.
 *
 *   3. MobileScoreSelector chip rows — replaces the desktop range
 *      slider (touch-hostile on small viewports) with a 1-5 tap-
 *      target chip group, tier-tinted for at-a-glance score reading.
 *
 * Math + payload shape mirror desktop daily-routine.tsx field-for-
 * field. Desktop file is intentionally untouched.
 */

// ── Constants ─────────────────────────────────────────────────────

const REPORT_CATEGORIES = [
  { key: "plan", label: "Followed plan" },
  { key: "stops", label: "Respected stops" },
  { key: "sized", label: "Sized correctly" },
  { key: "fomo", label: "No FOMO entries" },
] as const;

const DRAFT_KEY_PREFIX = "mo-daily-routine-draft-";
const AUTOSAVE_DEBOUNCE_MS = 500;

// ── Helpers (mirrored from desktop daily-routine.tsx) ─────────────

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
  return g.startsWith("A")
    ? 5
    : g.startsWith("B")
      ? 4
      : g.startsWith("C")
        ? 3
        : g.startsWith("D")
          ? 2
          : 1;
}

function todayStr(): string {
  const n = new Date();
  return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
}

function addOneDay(d: string): string {
  // YYYY-MM-DD → YYYY-MM-DD (next calendar day, local TZ). Used for
  // journalLatest's `before` param (which is strictly-before-date)
  // so the response covers entries on-or-before entryDate.
  const [y, m, day] = d.split("-").map(Number);
  const dt = new Date(y, m - 1, day);
  dt.setDate(dt.getDate() + 1);
  return `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`;
}

function buildActionsString(
  details: { date?: string; action?: string; ticker?: string }[],
  day: string,
): string {
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
    if (label !== "SELL" && label !== "BUY") {
      parts.push(`${label}: ${grouped[label].join(", ")}`);
    }
  }
  return parts.join(" | ");
}

// ── Types ─────────────────────────────────────────────────────────

interface PortfolioCardState {
  name: string;
  id: number;
  end_nlv: string;
  total_holdings: string;
  cash_change: string;
  actions: string;
  prev_end_nlv: number;
  errors: { end_nlv?: string; total_holdings?: string };
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
  const adjustedBeg = p.prev_end_nlv + cash;
  const daily_dollar_change = p.prev_end_nlv > 0 ? nlv - adjustedBeg : 0;
  const daily_pct_change = adjustedBeg > 0 ? (daily_dollar_change / adjustedBeg) * 100 : 0;
  const pct_invested = nlv > 0 ? (hold / nlv) * 100 : 0;
  return { daily_dollar_change, daily_pct_change, pct_invested, nlv, cash };
}

type SaveError =
  | { kind: "conflict"; conflicting_portfolios: string[] }
  | { kind: "error"; detail: string };

type DraftPayload = {
  entryDate: string;
  spyClose: string;
  ndxClose: string;
  marketNotes: string;
  scores: Record<string, number>;
  gradeNotes: string;
  forceOverwrite: boolean;
  cards: Array<{
    name: string;
    end_nlv: string;
    total_holdings: string;
    cash_change: string;
    actions: string;
  }>;
};

// ── Autosave helpers ──────────────────────────────────────────────

function loadDraft(date: string): DraftPayload | null {
  try {
    const raw = window.localStorage.getItem(DRAFT_KEY_PREFIX + date);
    if (!raw) return null;
    return JSON.parse(raw) as DraftPayload;
  } catch {
    return null;
  }
}

function saveDraft(date: string, payload: DraftPayload): void {
  try {
    window.localStorage.setItem(DRAFT_KEY_PREFIX + date, JSON.stringify(payload));
  } catch {
    /* quota / private mode — autosave is best-effort */
  }
}

function clearDraft(date: string): void {
  try {
    window.localStorage.removeItem(DRAFT_KEY_PREFIX + date);
  } catch {
    /* ignore */
  }
}

// ── Main component ────────────────────────────────────────────────

export function MobileDailyRoutine() {
  const { portfolios } = usePortfolio();

  const [entryDate, setEntryDate] = useState(todayStr);
  const [spyClose, setSpyClose] = useState("");
  const [ndxClose, setNdxClose] = useState("");
  const [marketNotes, setMarketNotes] = useState("");
  const [scores, setScores] = useState<Record<string, number>>({
    plan: 5,
    stops: 5,
    sized: 5,
    fomo: 5,
  });
  const [gradeNotes, setGradeNotes] = useState("");
  const [forceOverwrite, setForceOverwrite] = useState(false);
  const [cards, setCards] = useState<PortfolioCardState[]>([]);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState("");
  const [saveError, setSaveError] = useState<SaveError | null>(null);
  const [submitAttempted, setSubmitAttempted] = useState(false);

  // Pre-load tracking — which portfolio names already have a journal
  // entry for entryDate. Drives the banner + force-overwrite auto-on.
  const [preloadedPortfolios, setPreloadedPortfolios] = useState<string[]>([]);
  const [preloadedAt, setPreloadedAt] = useState<string | null>(null);

  // Draft autosave state. `lastAutosaveAt` ticks each successful write
  // so the indicator can show "Autosaved Ns ago"; `restoredFromDraft`
  // flags whether the mount restore consumed a saved draft (drives
  // the "Restored unsaved draft" subtitle).
  const [lastAutosaveAt, setLastAutosaveAt] = useState<number | null>(null);
  const [restoredFromDraft, setRestoredFromDraft] = useState(false);
  const autosaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Skip the very first autosave fire (the initial mount render
  // would otherwise blast a write of the empty form).
  const skipNextAutosaveRef = useRef(true);

  // ── Mount restore from localStorage ────────────────────────────
  // Synchronous restore via lazy state init isn't enough here
  // because the entryDate that keys the draft is also state. Use a
  // dedicated useEffect that runs once with the initial entryDate.
  useEffect(() => {
    const draft = loadDraft(entryDate);
    if (!draft) return;
    setSpyClose(draft.spyClose);
    setNdxClose(draft.ndxClose);
    setMarketNotes(draft.marketNotes);
    setScores(draft.scores);
    setGradeNotes(draft.gradeNotes);
    setForceOverwrite(draft.forceOverwrite);
    // Cards repopulate via the fetch effect (which seeds prev_end_nlv);
    // draft values for editable fields are merged in there.
    setRestoredFromDraft(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Per-entryDate fetches ──────────────────────────────────────
  useEffect(() => {
    if (!portfolios.length) return;
    let cancelled = false;
    setLoading(true);
    setSaveOk("");
    setSaveError(null);
    setSubmitAttempted(false);
    setPreloadedPortfolios([]);
    setPreloadedAt(null);

    const today = todayStr();
    const isPastDate = entryDate < today;
    const beforeNext = addOneDay(entryDate);

    // Per-portfolio: journalLatest with before=entryDate+1 returns the
    // most-recent entry on-or-before entryDate. If response.day ===
    // entryDate, an entry exists for the target date → pre-load.
    // Otherwise the response IS the prior day's entry → use its
    // end_nlv as prev_end_nlv.
    const perPortfolioPromises = portfolios.map((p) =>
      Promise.all([
        api.journalLatest(p.name, beforeNext).catch((err) => {
          log.debug.devOnly?.("mobile-daily-routine", `journalLatest fetch failed for ${p.name}`, err);
          return null as JournalEntry | null;
        }),
        api.tradesRecent(p.name, 1000).catch((err) => {
          log.debug.devOnly?.("mobile-daily-routine", `tradesRecent fetch failed for ${p.name}`, err);
          return { details: [], lot_closures: [] };
        }),
      ]).then(([latest, trades]) => ({ p, latest, trades })),
    );

    const pricesPromise = api
      .batchPrices(["SPY", "^IXIC"], undefined, isPastDate ? entryDate : undefined)
      .catch((err) => {
        log.debug.devOnly?.("mobile-daily-routine", "batchPrices pre-fill missing (expected)", err);
        return {} as Record<string, number>;
      });

    Promise.all([pricesPromise, ...perPortfolioPromises]).then((results) => {
      if (cancelled) return;

      const prices = results[0] as Record<string, number>;
      const draft = loadDraft(entryDate);

      // Shared fields: draft wins over fetched defaults. SPY/NDX only
      // pre-fill if the draft didn't already have them.
      if (!draft || draft.spyClose === "") {
        if (prices["SPY"]) setSpyClose(prices["SPY"].toFixed(2));
      }
      if (!draft || draft.ndxClose === "") {
        if (prices["^IXIC"]) setNdxClose(prices["^IXIC"].toFixed(2));
      }

      // Per-portfolio card build. Order matches portfolios array.
      const matchingPreloaded: string[] = [];
      let preloadTimestamp: string | null = null;
      const built: PortfolioCardState[] = [];
      for (let i = 1; i < results.length; i++) {
        const { p, latest, trades } = results[i] as {
          p: { id: number; name: string };
          latest: JournalEntry | null;
          trades: { details?: { date?: string; ticker?: string; action?: string }[] };
        };
        const card = emptyCard(p);
        const hasExistingEntry = latest != null && latest.day === entryDate;
        if (hasExistingEntry) {
          // Pre-load form from existing entry. prev_end_nlv is the
          // entry's beg_nlv (prior day's end by definition).
          card.prev_end_nlv = parseFloat(String(latest.beg_nlv || 0)) || 0;
          card.end_nlv = String(latest.end_nlv ?? "");
          card.total_holdings = String((latest as Record<string, unknown>).total_holdings ?? "");
          card.cash_change = String(
            (latest as Record<string, unknown>).cash_change ?? "0",
          );
          card.actions = String((latest as Record<string, unknown>).actions ?? "");
          matchingPreloaded.push(p.name);
          if (!preloadTimestamp) preloadTimestamp = entryDate;
        } else {
          // No entry for entryDate; latest is prior day. Seed prev_end_nlv.
          card.prev_end_nlv = parseFloat(String(latest?.end_nlv || 0)) || 0;
          card.actions = buildActionsString(trades.details || [], entryDate);
        }

        // Draft wins per-field for editable values.
        const draftCard = draft?.cards.find((c) => c.name === p.name);
        if (draftCard) {
          if (draftCard.end_nlv !== "") card.end_nlv = draftCard.end_nlv;
          if (draftCard.total_holdings !== "") card.total_holdings = draftCard.total_holdings;
          if (draftCard.cash_change !== "") card.cash_change = draftCard.cash_change;
          if (draftCard.actions !== "") card.actions = draftCard.actions;
        }
        built.push(card);
      }
      setCards(built);

      // Pre-load banner + auto-on Force Overwrite when ANY portfolio
      // already has an entry for entryDate. Subtitle calls out which
      // ones (CSV) so the user knows what they're about to overwrite.
      if (matchingPreloaded.length > 0) {
        setPreloadedPortfolios(matchingPreloaded);
        setPreloadedAt(preloadTimestamp);
        // Honor an explicit user toggle off if they've already touched
        // it via draft restore; otherwise auto-enable.
        if (!draft) setForceOverwrite(true);
      }

      // Shared pre-load: if any portfolio's entry has populated shared
      // fields, prefer them when draft hasn't already restored.
      if (!draft) {
        const sharedSource = (results.slice(1) as Array<{ latest: JournalEntry | null }>).find(
          (r) => r.latest && r.latest.day === entryDate,
        )?.latest;
        if (sharedSource) {
          if (sharedSource.spy) setSpyClose(String(sharedSource.spy));
          if (sharedSource.nasdaq) setNdxClose(String(sharedSource.nasdaq));
          const mn = (sharedSource as Record<string, unknown>).market_notes;
          if (typeof mn === "string" && mn) setMarketNotes(mn);
          const gn = (sharedSource as Record<string, unknown>).mistakes;
          if (typeof gn === "string" && gn) setGradeNotes(gn);
          const highlights = (sharedSource as Record<string, unknown>).highlights;
          if (typeof highlights === "string") {
            try {
              const parsed = JSON.parse(highlights);
              if (parsed && typeof parsed === "object") {
                setScores((prev) => ({ ...prev, ...(parsed as Record<string, number>) }));
              }
            } catch {
              /* ignore malformed highlights */
            }
          }
        }
      }

      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portfolios, entryDate]);

  // ── rallyPrefix seed ───────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    api
      .rallyPrefix(entryDate)
      .catch((err) => {
        log.debug.devOnly?.("mobile-daily-routine", "rallyPrefix pre-fill missing (expected)", err);
        return { prefix: "" };
      })
      .then((rally) => {
        if (cancelled) return;
        const prefix = (rally as { prefix?: string }).prefix || "";
        if (prefix && !marketNotes) setMarketNotes(prefix);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entryDate]);

  // ── Autosave ───────────────────────────────────────────────────
  useEffect(() => {
    if (skipNextAutosaveRef.current) {
      skipNextAutosaveRef.current = false;
      return;
    }
    if (autosaveTimerRef.current) clearTimeout(autosaveTimerRef.current);
    autosaveTimerRef.current = setTimeout(() => {
      const payload: DraftPayload = {
        entryDate,
        spyClose,
        ndxClose,
        marketNotes,
        scores,
        gradeNotes,
        forceOverwrite,
        cards: cards.map((c) => ({
          name: c.name,
          end_nlv: c.end_nlv,
          total_holdings: c.total_holdings,
          cash_change: c.cash_change,
          actions: c.actions,
        })),
      };
      saveDraft(entryDate, payload);
      setLastAutosaveAt(Date.now());
    }, AUTOSAVE_DEBOUNCE_MS);
    return () => {
      if (autosaveTimerRef.current) clearTimeout(autosaveTimerRef.current);
    };
  }, [entryDate, spyClose, ndxClose, marketNotes, scores, gradeNotes, forceOverwrite, cards]);

  // Reset the autosave skip flag whenever entryDate changes — the new
  // date may have its own draft, and the next field-touch should
  // write to the new key.
  useEffect(() => {
    skipNextAutosaveRef.current = true;
  }, [entryDate]);

  const updateCard = useCallback((name: string, patch: Partial<PortfolioCardState>) => {
    setCards((prev) => prev.map((c) => (c.name === name ? { ...c, ...patch } : c)));
    setSaveOk("");
    setSaveError(null);
  }, []);

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
  const isExistingEntry = preloadedPortfolios.length > 0;

  // ── Submit ─────────────────────────────────────────────────────
  async function handleSave() {
    setSaving(true);
    setSaveOk("");
    setSaveError(null);
    setSubmitAttempted(true);

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
        // Per directive: do NOT auto-enable the toggle on conflict.
        // The user has to explicitly consent to overwrite.
        setSaveError({
          kind: "conflict",
          conflicting_portfolios: r.conflicting_portfolios || [],
        });
      } else if (r.status === "ok") {
        setSaveOk(`Saved ${r.rows_written ?? validated.length} portfolios`);
        clearDraft(entryDate);
        setRestoredFromDraft(false);
        setLastAutosaveAt(null);
      } else {
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

  // ── Render ─────────────────────────────────────────────────────

  const portfolioNamesCsv = portfolios.map((p) => p.name).join(" · ");
  const friendlyDate = (() => {
    const [y, m, d] = entryDate.split("-").map(Number);
    if (!y || !m || !d) return entryDate;
    return new Date(y, m - 1, d).toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  })();

  if (loading && cards.length === 0) {
    return (
      <div className="flex flex-col gap-3 pt-2">
        <div className="h-9 animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-32 animate-pulse rounded-m-lg bg-m-surface-2" />
        <div className="h-48 animate-pulse rounded-m-lg bg-m-surface-2" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 pt-2 pb-[120px]">
      <Header friendlyDate={friendlyDate} portfolioNamesCsv={portfolioNamesCsv} />

      {isExistingEntry && (
        <PreloadBanner
          portfolios={preloadedPortfolios}
          restoredFromDraft={restoredFromDraft}
        />
      )}

      {restoredFromDraft && !isExistingEntry && (
        <div
          role="status"
          data-testid="restored-draft-banner"
          className="rounded-m-md border-[0.5px] border-m-accent-border bg-m-accent-tint px-[14px] py-2.5 text-[12px] text-m-text"
        >
          Restored your in-progress draft for this date.
        </div>
      )}

      <MarketSection
        entryDate={entryDate}
        onEntryDate={setEntryDate}
        spyClose={spyClose}
        onSpy={setSpyClose}
        ndxClose={ndxClose}
        onNdx={setNdxClose}
        marketNotes={marketNotes}
        onMarketNotes={setMarketNotes}
      />

      <section>
        <div className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Portfolios
        </div>
        <div className="flex flex-col gap-3">
          {cards.map((card) => (
            <PortfolioCard
              key={card.name}
              card={card}
              onChange={(patch) => updateCard(card.name, patch)}
            />
          ))}
        </div>
      </section>

      <ReportCardSection
        grade={grade}
        totalScore={totalScore}
        scores={scores}
        onScore={(k, v) => setScores((prev) => ({ ...prev, [k]: v }))}
        gradeNotes={gradeNotes}
        onGradeNotes={setGradeNotes}
      />

      <MobileToggleSwitch
        id="dr-force-overwrite"
        checked={forceOverwrite}
        onChange={setForceOverwrite}
        label="Force overwrite existing entry"
        description={
          isExistingEntry
            ? "Auto-enabled because an entry already exists for this date."
            : "Replace today's saved entry on submit."
        }
      />

      {hasErrors &&
        (submitAttempted ||
          cards.some(
            (c) =>
              (c.touched.end_nlv && c.errors.end_nlv) ||
              (c.touched.total_holdings && c.errors.total_holdings),
          )) && (
          <ValidationSummary errors={validationSummary} />
        )}

      {saveError?.kind === "conflict" && (
        <div
          role="alert"
          data-testid="conflict-banner"
          className="rounded-m-md border-[0.5px] border-m-down-border bg-m-down-tint px-[14px] py-3 text-[12px] text-m-text"
          style={{
            background: "color-mix(in oklab, var(--m-down) 10%, var(--m-surface))",
            borderColor: "var(--m-down)",
          }}
        >
          <strong>Entry exists</strong> for {saveError.conflicting_portfolios.join(", ")}.
          Enable <em>Force overwrite</em> above and save again.
        </div>
      )}

      {saveError?.kind === "error" && (
        <div
          role="alert"
          data-testid="save-error-banner"
          className="rounded-m-md px-[14px] py-3 text-[12px] text-m-text"
          style={{
            background: "color-mix(in oklab, var(--m-down) 10%, var(--m-surface))",
            border: "0.5px solid var(--m-down)",
          }}
        >
          Error: {saveError.detail}
        </div>
      )}

      {saveOk && (
        <div
          role="status"
          data-testid="save-ok-banner"
          className="rounded-m-md border-[0.5px] border-m-accent-border bg-m-accent-tint px-[14px] py-3 text-[12px] font-medium text-m-accent"
        >
          {saveOk}
        </div>
      )}

      <StickyBottomSaveBar
        onSave={handleSave}
        disabled={saving || hasErrors || cards.length === 0}
        saving={saving}
        lastAutosaveAt={lastAutosaveAt}
      />
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────

function Header({
  friendlyDate,
  portfolioNamesCsv,
}: {
  friendlyDate: string;
  portfolioNamesCsv: string;
}) {
  // Wordmark dropped — the shell's MobilePageHeader owns the page
  // title. This is the per-page metadata strip only.
  return (
    <div className="pt-1 text-[12px] text-m-text-dim">
      {friendlyDate}
      {portfolioNamesCsv && <> · {portfolioNamesCsv}</>}
    </div>
  );
}

function PreloadBanner({
  portfolios,
  restoredFromDraft,
}: {
  portfolios: string[];
  restoredFromDraft: boolean;
}) {
  return (
    <div
      role="status"
      data-testid="preload-banner"
      className="flex items-start gap-2.5 rounded-m-md border-[0.5px] border-m-accent-border bg-m-accent-tint px-[14px] py-3"
      style={{ borderLeftWidth: 3, borderLeftColor: "var(--m-accent)" }}
    >
      <Edit3 size={14} strokeWidth={1.6} className="mt-0.5 shrink-0 text-m-accent" aria-hidden="true" />
      <div className="min-w-0 flex-1 text-[12px] text-m-text">
        <div className="font-medium">Editing existing entry</div>
        <div className="mt-0.5 text-m-text-dim">
          {portfolios.join(", ")} already saved for this date.
          {restoredFromDraft ? " Local draft also restored." : ""} Force
          overwrite auto-enabled below.
        </div>
      </div>
    </div>
  );
}

function MarketSection({
  entryDate,
  onEntryDate,
  spyClose,
  onSpy,
  ndxClose,
  onNdx,
  marketNotes,
  onMarketNotes,
}: {
  entryDate: string;
  onEntryDate: (v: string) => void;
  spyClose: string;
  onSpy: (v: string) => void;
  ndxClose: string;
  onNdx: (v: string) => void;
  marketNotes: string;
  onMarketNotes: (v: string) => void;
}) {
  return (
    <section className="flex flex-col gap-2">
      <div className="px-1 text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
        Market
      </div>
      <label className="block rounded-m-md border-[0.5px] border-m-border bg-m-surface px-[14px] py-[10px]">
        <span className="mb-0.5 block text-[10px] font-medium text-m-text-dim">Date</span>
        <input
          type="date"
          value={entryDate}
          onChange={(e) => onEntryDate(e.target.value)}
          aria-label="Entry date"
          className="w-full bg-transparent font-m-num text-base tabular-nums text-m-text focus:outline-none"
        />
      </label>
      <div className="grid grid-cols-2 gap-2">
        <NumberFieldCell
          label="SPY Close"
          value={spyClose}
          onChange={onSpy}
          ariaLabel="SPY close"
          placeholder="0.00"
        />
        <NumberFieldCell
          label="Nasdaq Close"
          value={ndxClose}
          onChange={onNdx}
          ariaLabel="Nasdaq close"
          placeholder="0.00"
        />
      </div>
      <TextFieldCell
        label="Market Notes"
        value={marketNotes}
        onChange={onMarketNotes}
        ariaLabel="Market notes"
        placeholder="Day 14 UPTREND: …"
        multiline
        rows={2}
      />
    </section>
  );
}

function PortfolioCard({
  card,
  onChange,
}: {
  card: PortfolioCardState;
  onChange: (patch: Partial<PortfolioCardState>) => void;
}) {
  const m = deriveCardMetrics(card);
  const dailyClass =
    m.daily_dollar_change > 0
      ? "text-m-accent"
      : m.daily_dollar_change < 0
        ? "text-m-down"
        : "text-m-text-dim";

  return (
    <div
      data-testid={`portfolio-card-${card.name}`}
      className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface"
    >
      <div className="flex items-baseline justify-between border-b-[0.5px] border-m-border px-4 py-2.5">
        <span className="text-[13px] font-semibold text-m-text">{card.name}</span>
        {m.nlv > 0 && (
          <span className={`font-m-num text-[12px] font-medium tabular-nums ${dailyClass}`}>
            {m.daily_pct_change >= 0 ? "+" : ""}
            {m.daily_pct_change.toFixed(2)}%
          </span>
        )}
      </div>
      {m.nlv > 0 && (
        <div className="grid grid-cols-4 gap-1 border-b-[0.5px] border-m-border px-4 py-2.5">
          <DerivedTile label="Prev NLV" value={formatCurrency(card.prev_end_nlv, { decimals: 0 })} privacy />
          <DerivedTile
            label="Daily $"
            value={formatCurrency(m.daily_dollar_change, {
              showSign: true,
              signGlyph: "unicode",
              decimals: 0,
            })}
            valueClass={dailyClass}
            privacy
          />
          <DerivedTile
            label="Daily %"
            value={`${m.daily_pct_change >= 0 ? "+" : ""}${m.daily_pct_change.toFixed(2)}%`}
            valueClass={dailyClass}
          />
          <DerivedTile label="% Invested" value={`${m.pct_invested.toFixed(1)}%`} />
        </div>
      )}
      <div className="flex flex-col gap-2 p-4">
        <NumberFieldCell
          label="Closing NLV *"
          value={card.end_nlv}
          onChange={(v) =>
            onChange({ end_nlv: v, errors: { ...card.errors, end_nlv: undefined } })
          }
          onBlur={() =>
            onChange({
              touched: { ...card.touched, end_nlv: true },
              errors: validateCard({ ...card }),
            })
          }
          ariaLabel={`Closing NLV for ${card.name}`}
          placeholder="0"
          hasError={card.touched.end_nlv && Boolean(card.errors.end_nlv)}
        />
        {card.touched.end_nlv && card.errors.end_nlv && (
          <div className="-mt-1 px-2 text-[11px] font-medium text-m-down">{card.errors.end_nlv}</div>
        )}
        <NumberFieldCell
          label="Total Holdings *"
          value={card.total_holdings}
          onChange={(v) =>
            onChange({
              total_holdings: v,
              errors: { ...card.errors, total_holdings: undefined },
            })
          }
          onBlur={() =>
            onChange({
              touched: { ...card.touched, total_holdings: true },
              errors: validateCard({ ...card }),
            })
          }
          ariaLabel={`Total Holdings for ${card.name}`}
          placeholder="0"
          hasError={card.touched.total_holdings && Boolean(card.errors.total_holdings)}
        />
        {card.touched.total_holdings && card.errors.total_holdings && (
          <div className="-mt-1 px-2 text-[11px] font-medium text-m-down">
            {card.errors.total_holdings}
          </div>
        )}
        <NumberFieldCell
          label="Cash +/−"
          value={card.cash_change}
          onChange={(v) => onChange({ cash_change: v })}
          ariaLabel={`Cash flow for ${card.name}`}
          placeholder="0"
        />
        <TextFieldCell
          label="Actions"
          value={card.actions}
          onChange={(v) => onChange({ actions: v })}
          ariaLabel={`Actions for ${card.name}`}
          placeholder="BUY: NVDA"
        />
      </div>
    </div>
  );
}

function DerivedTile({
  label,
  value,
  valueClass,
  privacy,
}: {
  label: string;
  value: string;
  valueClass?: string;
  privacy?: boolean;
}) {
  return (
    <div className="rounded-m-sm bg-m-surface-2 px-2 py-1">
      <div className="text-[9px] font-medium uppercase tracking-[0.06em] text-m-text-dim">
        {label}
      </div>
      <div
        className={`mt-0.5 font-m-num text-[11px] font-semibold tabular-nums ${
          valueClass ?? "text-m-text"
        } ${privacy ? "privacy-mask" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function ReportCardSection({
  grade,
  totalScore,
  scores,
  onScore,
  gradeNotes,
  onGradeNotes,
}: {
  grade: string;
  totalScore: number;
  scores: Record<string, number>;
  onScore: (k: string, v: number) => void;
  gradeNotes: string;
  onGradeNotes: (v: string) => void;
}) {
  return (
    <section className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface px-4 pb-4 pt-3">
      <div className="mb-3 flex flex-col items-center gap-0.5">
        <div className="text-[10px] font-semibold uppercase tracking-[0.10em] text-m-text-dim">
          Report Card
        </div>
        <div
          className="font-medium tabular-nums"
          style={{
            fontFamily: "var(--font-fraunces), Georgia, serif",
            fontSize: 44,
            lineHeight: 1,
            color: gradeColor(grade),
          }}
          data-testid="report-grade"
        >
          {grade}
        </div>
        <div className="font-m-num text-[11px] tabular-nums text-m-text-dim">
          {totalScore} / {REPORT_CATEGORIES.length * 5}
        </div>
      </div>
      <div className="flex flex-col gap-3">
        {REPORT_CATEGORIES.map((cat) => (
          <MobileScoreSelector
            key={cat.key}
            label={cat.label}
            value={scores[cat.key] ?? 5}
            onChange={(v) => onScore(cat.key, v)}
          />
        ))}
      </div>
      <div className="mt-3">
        <TextFieldCell
          label="Grade Notes"
          value={gradeNotes}
          onChange={onGradeNotes}
          ariaLabel="Grade notes"
          placeholder="Optional…"
          multiline
          rows={2}
        />
      </div>
    </section>
  );
}

function ValidationSummary({
  errors,
}: {
  errors: { name: string; field: string; message: string }[];
}) {
  return (
    <div
      role="alert"
      data-testid="validation-summary"
      className="rounded-m-md px-[14px] py-3 text-[12px] text-m-text"
      style={{
        background: "color-mix(in oklab, var(--m-warn) 10%, var(--m-surface))",
        border: "0.5px solid var(--m-warn-border, var(--m-warn))",
      }}
    >
      <div className="mb-1 font-semibold">
        Fix {errors.length} {errors.length === 1 ? "error" : "errors"} before saving:
      </div>
      <ul className="list-disc pl-5">
        {errors.map((e, i) => (
          <li key={`${e.name}-${e.field}-${i}`}>
            {e.name}: {e.field === "end_nlv" ? "Closing NLV" : "Total Holdings"}{" "}
            {e.message.toLowerCase()}
          </li>
        ))}
      </ul>
    </div>
  );
}

function StickyBottomSaveBar({
  onSave,
  disabled,
  saving,
  lastAutosaveAt,
}: {
  onSave: () => void;
  disabled: boolean;
  saving: boolean;
  lastAutosaveAt: number | null;
}) {
  // Re-render the "Ns ago" indicator each second so it stays current
  // without coupling to the autosave write itself.
  const [, tick] = useState(0);
  useEffect(() => {
    if (lastAutosaveAt == null) return;
    const id = setInterval(() => tick((v) => v + 1), 1000);
    return () => clearInterval(id);
  }, [lastAutosaveAt]);

  const autosaveLabel = (() => {
    if (lastAutosaveAt == null) return "Not saved yet";
    const secs = Math.max(0, Math.floor((Date.now() - lastAutosaveAt) / 1000));
    if (secs < 5) return "Autosaved just now";
    if (secs < 60) return `Autosaved ${secs}s ago`;
    const mins = Math.floor(secs / 60);
    return `Autosaved ${mins}m ago`;
  })();

  return (
    <div
      className="fixed inset-x-0 bottom-0 z-30 border-t-[0.5px] border-m-border bg-m-bg px-5 pb-5 pt-3"
      style={{ paddingBottom: "max(1.25rem, env(safe-area-inset-bottom))" }}
    >
      <button
        type="button"
        onClick={onSave}
        disabled={disabled}
        data-testid="save-button"
        className="flex h-12 w-full items-center justify-center gap-2 rounded-m-lg bg-m-accent text-[15px] font-semibold text-m-accent-text-on disabled:opacity-50"
      >
        {saving ? "Saving…" : (
          <>
            <Check size={16} strokeWidth={2} aria-hidden="true" />
            Save daily routine
          </>
        )}
      </button>
      <div
        className="mt-1.5 text-center text-[11px] text-m-text-dim"
        data-testid="autosave-indicator"
      >
        {autosaveLabel}
      </div>
    </div>
  );
}
