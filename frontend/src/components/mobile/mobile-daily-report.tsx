"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Plus,
} from "lucide-react";
import {
  api,
  getActivePortfolio,
  type DailyJournalCaptureRow,
  type JournalHistoryPoint,
  type Tag,
  type TagAssignment,
  type TradeDetail,
  type TradePosition,
} from "@/lib/api";
import { formatCurrency, setFocusModeActive } from "@/lib/format";
import { log } from "@/lib/log";
import { usePortfolio } from "@/lib/portfolio-context";
import { ImageLightbox, type LightboxImage } from "@/components/image-lightbox";
import { TAG_PALETTE, type TagTone } from "@/lib/tag-palette";
import { MobileImageUpload, type ImageUploadRow } from "./mobile-image-upload";

/**
 * Mobile Daily Report — Phase 2 T2-4 (core).
 *
 * First real consumer of the T2-3 MobileImageUpload primitive. Ports
 * the desktop daily-report-card.tsx (789 LOC, multi-section detail
 * view) to a touch-friendly read-mostly surface with one editable
 * text field (Daily Recap / lowlights).
 *
 * Scope-locked decisions (see audit report):
 *   - Daily Recap (`lowlights`) is the only editable text field.
 *   - Daily Thoughts renders read-only HTML (rich editor port = T2-4b).
 *   - Drawdown formula + thresholds match daily-report-card.tsx exactly.
 *   - Focus mode lifts-and-shifts from desktop's `mo-focus-mode`
 *     localStorage + `setFocusModeActive` module singleton; toggle UI
 *     lives in MobilePageHeader rightSlot.
 *   - Back nav via MobilePageHeader leftSlot (new primitive slot).
 *   - Captures + EOD merged into one chronological gallery via the
 *     T2-3 primitive's new `nonDeletable` + `renderBadge` props.
 *   - Tags: read-only chips + "+ Add" → MobileSelectSheet (cap 10).
 *   - Performance Comparison: 3 vertical rows (Portfolio / SPY /
 *     NASDAQ) showing Daily % + YTD %.
 */

type Props = { initialDate?: string };

type EodRow = {
  id?: number;
  image_url?: string;
  view_url?: string;
  image_type?: string;
  file_name?: string;
  uploaded_at?: string;
};

// Merged row shape for the unified captures+EOD gallery. The discriminator
// (`_kind`) drives nonDeletable / renderBadge / lightbox routing.
type MergedRow = ImageUploadRow & {
  _kind: "capture" | "eod";
  _sortKey: string; // ISO timestamp for chronological sort
};

const DRAFT_KEY_PREFIX = "mo-daily-report-recap-draft-";
const AUTOSAVE_DEBOUNCE_MS = 500;
const SAVED_INDICATOR_VISIBLE_MS = 3000;
const FOCUS_MODE_LS_KEY = "mo-focus-mode";

// ── Pure helpers ────────────────────────────────────────────────────

function fmtDateHeader(iso: string): string {
  // "2026-05-25" → "Mon, May 25"
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function fmtDateLong(iso: string): string {
  // "2026-05-25" → "May 25, 2026"
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

function pctClass(v: number): string {
  if (v > 0) return "text-m-accent";
  if (v < 0) return "text-m-down";
  return "text-m-text-dim";
}

function signedPct(v: number, decimals = 2): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}%`;
}

// Score → letter grade (mirrors daily-report-card.tsx:577).
function gradeLabel(score: number): string {
  if (score >= 5) return "A+";
  if (score >= 4) return "A";
  if (score >= 3) return "B";
  if (score >= 2) return "C";
  if (score > 0) return "D";
  return "";
}

function gradeColor(score: number): string {
  if (score >= 4) return "var(--m-accent)";
  if (score >= 3) return "var(--m-warn)";
  if (score > 0) return "var(--m-down)";
  return "var(--m-text-dim)";
}

// Drawdown color tiers — mirrors daily-report-card.tsx:336-337 exactly.
function drawdownTierColor(pct: number): string {
  if (pct >= -7.5) return "var(--m-accent)";
  if (pct >= -12.5) return "var(--m-warn)";
  return "var(--m-down)";
}

function drawdownTierMessage(pct: number): string {
  if (pct >= -7.5) return "GREEN LIGHT";
  if (pct >= -12.5) return "CAUTION";
  if (pct >= -15) return "MAX 30% INVESTED";
  return "GO TO CASH";
}

// Chained YTD% from a series of daily_pct_change values.
function chainPct(rows: number[]): number {
  if (rows.length === 0) return 0;
  let mult = 1;
  for (const r of rows) mult *= 1 + (r || 0) / 100;
  return (mult - 1) * 100;
}

// ── Main component ─────────────────────────────────────────────────

export function MobileDailyReport({ initialDate }: Props) {
  const router = useRouter();
  const { activePortfolio } = usePortfolio();
  const portfolio = activePortfolio?.name ?? getActivePortfolio();
  const date = useMemo(() => {
    if (initialDate && /^\d{4}-\d{2}-\d{2}$/.test(initialDate)) return initialDate;
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  }, [initialDate]);

  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [captures, setCaptures] = useState<DailyJournalCaptureRow[]>([]);
  const [eodSnapshots, setEodSnapshots] = useState<EodRow[]>([]);
  const [tagAssignments, setTagAssignments] = useState<TagAssignment[]>([]);
  const [tradesDetails, setTradesDetails] = useState<TradeDetail[]>([]);
  const [tradesClosed, setTradesClosed] = useState<TradePosition[]>([]);
  const [availableTags, setAvailableTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(true);

  const journalRow = useMemo(
    () => history.find((h) => String(h.day).slice(0, 10) === date) ?? null,
    [history, date],
  );
  const prevDayRow = useMemo(() => {
    if (!journalRow) return null;
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const idx = sorted.findIndex((h) => String(h.day).slice(0, 10) === date);
    return idx > 0 ? sorted[idx - 1] : null;
  }, [history, date, journalRow]);

  // ── Mount fetch ───────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.journalHistory(portfolio, 0).catch((err) => {
        log.error("mobile-daily-report", "journalHistory failed", err);
        return [] as JournalHistoryPoint[];
      }),
      api.listEodSnapshots(date, portfolio).catch((err) => {
        log.error("mobile-daily-report", "listEodSnapshots failed", err);
        return [] as EodRow[];
      }),
      api.tradesRecent(portfolio, 500).catch((err) => {
        log.error("mobile-daily-report", "tradesRecent failed", err);
        return { details: [], lot_closures: [] };
      }),
      api.tradesClosed(portfolio, 500).catch((err) => {
        log.error("mobile-daily-report", "tradesClosed failed", err);
        return [] as TradePosition[];
      }),
      api.listTags(portfolio).catch((err) => {
        log.error("mobile-daily-report", "listTags failed", err);
        return [] as Tag[];
      }),
    ]).then(([hist, eod, recent, closed, tags]) => {
      if (cancelled) return;
      setHistory(Array.isArray(hist) ? hist : []);
      setEodSnapshots(Array.isArray(eod) ? eod : []);
      setTradesDetails(Array.isArray(recent?.details) ? recent.details : []);
      setTradesClosed(Array.isArray(closed) ? closed : []);
      setAvailableTags(Array.isArray(tags) ? tags : []);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [portfolio, date]);

  // Captures + tag assignments depend on journalRow.id — separate fetch
  // so the dependency chain stays clean.
  useEffect(() => {
    const journalId = journalRow?.id;
    if (!journalId) {
      setCaptures([]);
      setTagAssignments([]);
      return;
    }
    let cancelled = false;
    Promise.all([
      api.listDailyJournalCaptures(journalId, portfolio).catch(() => []),
      api.listTagAssignments({ entity_type: "daily_journal", entity_id: journalId }).catch(() => []),
    ]).then(([caps, assigns]) => {
      if (cancelled) return;
      setCaptures(Array.isArray(caps) ? (caps as DailyJournalCaptureRow[]) : []);
      setTagAssignments(Array.isArray(assigns) ? (assigns as TagAssignment[]) : []);
    });
    return () => {
      cancelled = true;
    };
  }, [journalRow?.id, portfolio]);

  if (loading) {
    return <LoadingSkeleton date={date} portfolio={portfolio} onBack={() => router.back()} />;
  }

  if (!journalRow) {
    return <DisabledState date={date} portfolio={portfolio} onBack={() => router.back()} />;
  }

  return (
    <LoadedReport
      date={date}
      portfolio={portfolio}
      journalRow={journalRow}
      prevDayRow={prevDayRow}
      history={history}
      captures={captures}
      setCaptures={setCaptures}
      eodSnapshots={eodSnapshots}
      tagAssignments={tagAssignments}
      setTagAssignments={setTagAssignments}
      availableTags={availableTags}
      tradesDetails={tradesDetails}
      tradesClosed={tradesClosed}
      onBack={() => {
        if (typeof window !== "undefined" && window.history.length > 1) {
          router.back();
        } else {
          router.push("/daily-journal");
        }
      }}
    />
  );
}

// ── Loaded report (split out so the disabled/loading paths stay tight) ──

function LoadedReport({
  date,
  portfolio,
  journalRow,
  prevDayRow,
  history,
  captures,
  setCaptures,
  eodSnapshots,
  tagAssignments,
  setTagAssignments,
  availableTags,
  tradesDetails,
  tradesClosed,
  onBack,
}: {
  date: string;
  portfolio: string;
  journalRow: JournalHistoryPoint;
  prevDayRow: JournalHistoryPoint | null;
  history: JournalHistoryPoint[];
  captures: DailyJournalCaptureRow[];
  setCaptures: (rows: DailyJournalCaptureRow[]) => void;
  eodSnapshots: EodRow[];
  tagAssignments: TagAssignment[];
  setTagAssignments: (a: TagAssignment[]) => void;
  availableTags: Tag[];
  tradesDetails: TradeDetail[];
  tradesClosed: TradePosition[];
  onBack: () => void;
}) {
  const journalId = journalRow.id ?? null;

  // ── Metrics row ─────────────────────────────────────────────────
  const endNlv = Number(journalRow.end_nlv ?? 0) || 0;
  const dailyPct = Number(journalRow.daily_pct_change ?? 0) || 0;
  const score = Number(journalRow.score ?? 0) || 0;
  const grade = gradeLabel(score);
  const dailyDollar = prevDayRow
    ? endNlv - (Number(prevDayRow.end_nlv ?? 0) || 0)
    : null;

  // ── Drawdown (mirrors daily-report-card.tsx:320-327) ─────────────
  const ddPct = useMemo(() => {
    if (history.length === 0) return 0;
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const upTo = sorted.filter((h) => String(h.day).slice(0, 10) <= date);
    if (upTo.length === 0) return 0;
    const peak = Math.max(...upTo.map((h) => Number(h.end_nlv ?? 0) || 0));
    const curr = Number(upTo[upTo.length - 1].end_nlv ?? 0) || 0;
    return peak > 0 ? ((curr - peak) / peak) * 100 : 0;
  }, [history, date]);

  const ddDollar = useMemo(() => {
    if (history.length === 0) return 0;
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const upTo = sorted.filter((h) => String(h.day).slice(0, 10) <= date);
    if (upTo.length === 0) return 0;
    const peak = Math.max(...upTo.map((h) => Number(h.end_nlv ?? 0) || 0));
    return endNlv - peak;
  }, [history, date, endNlv]);

  // ── Performance Comparison (3 rows: Portfolio / SPY / NASDAQ) ────
  // Desktop sources: spy_daily_pct / ndx_daily_pct derived from raw
  // spy / nasdaq price columns (daily-report-card.tsx:341-342). YTD
  // uses first-of-year vs current row for SPY/NDX, chained portfolio
  // daily_pct_change for portfolio (daily-report-card.tsx:310-316).
  const perf = useMemo(() => {
    const year = date.slice(0, 4);
    const sorted = [...history].sort((a, b) => String(a.day).localeCompare(String(b.day)));
    const idxToday = sorted.findIndex((h) => String(h.day).slice(0, 10) === date);
    const today = idxToday >= 0 ? sorted[idxToday] : null;
    const prev = idxToday > 0 ? sorted[idxToday - 1] : null;
    const ytdRows = sorted.filter(
      (h) => String(h.day).slice(0, 4) === year && String(h.day).slice(0, 10) <= date,
    );
    const portfolioDaily = Number(today?.daily_pct_change ?? 0) || 0;
    const portfolioYtd = chainPct(ytdRows.map((h) => Number(h.daily_pct_change ?? 0) || 0));

    const spyToday = Number(today?.spy ?? 0) || 0;
    const spyPrev = Number(prev?.spy ?? 0) || 0;
    const spyJan1 = Number(ytdRows[0]?.spy ?? 0) || 0;
    const spyDaily = spyPrev > 0 ? ((spyToday - spyPrev) / spyPrev) * 100 : 0;
    const spyYtd = spyJan1 > 0 ? ((spyToday - spyJan1) / spyJan1) * 100 : 0;

    const ndxToday = Number(today?.nasdaq ?? 0) || 0;
    const ndxPrev = Number(prev?.nasdaq ?? 0) || 0;
    const ndxJan1 = Number(ytdRows[0]?.nasdaq ?? 0) || 0;
    const ndxDaily = ndxPrev > 0 ? ((ndxToday - ndxPrev) / ndxPrev) * 100 : 0;
    const ndxYtd = ndxJan1 > 0 ? ((ndxToday - ndxJan1) / ndxJan1) * 100 : 0;

    return { portfolioDaily, portfolioYtd, spyDaily, spyYtd, ndxDaily, ndxYtd };
  }, [history, date]);

  // ── Positions Opened / Closed (filtered by date) ─────────────────
  const positionsOpened = useMemo(
    () =>
      tradesDetails.filter(
        (d) =>
          String(d.date ?? "").slice(0, 10) === date &&
          String(d.action ?? "").toUpperCase() === "BUY",
      ),
    [tradesDetails, date],
  );
  const positionsClosed = useMemo(
    () =>
      tradesClosed.filter(
        (t) => String((t as Record<string, unknown>).closed_date ?? "").slice(0, 10) === date,
      ),
    [tradesClosed, date],
  );

  // ── Daily Recap (lowlights) edit ─────────────────────────────────
  const initialRecap = String((journalRow as Record<string, unknown>).lowlights ?? "");
  const [recap, setRecap] = useState(initialRecap);
  const [recapSavedAt, setRecapSavedAt] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [savedFlashVisible, setSavedFlashVisible] = useState(false);
  const draftKey = `${DRAFT_KEY_PREFIX}${date}-${portfolio}`;
  const autosaveTimerRef = useRef<number | null>(null);

  // Hydrate from localStorage draft if present (overrides server value
  // when user has unsaved changes). Runs once on mount.
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (hydratedRef.current) return;
    hydratedRef.current = true;
    try {
      const draft = window.localStorage.getItem(draftKey);
      if (draft != null && draft !== initialRecap) {
        setRecap(draft);
      }
    } catch {
      // localStorage may be unavailable (SSR / private mode); fall back
      // to server value.
    }
  }, [draftKey, initialRecap]);

  // Debounced localStorage autosave.
  useEffect(() => {
    if (!hydratedRef.current) return;
    if (autosaveTimerRef.current) {
      window.clearTimeout(autosaveTimerRef.current);
    }
    autosaveTimerRef.current = window.setTimeout(() => {
      try {
        window.localStorage.setItem(draftKey, recap);
      } catch {
        // best-effort
      }
    }, AUTOSAVE_DEBOUNCE_MS);
    return () => {
      if (autosaveTimerRef.current) {
        window.clearTimeout(autosaveTimerRef.current);
      }
    };
  }, [recap, draftKey]);

  // Fade the "Saved Ns ago" indicator after the visible window. State
  // persists; only visibility flips.
  useEffect(() => {
    if (!recapSavedAt) return;
    setSavedFlashVisible(true);
    const id = window.setTimeout(() => setSavedFlashVisible(false), SAVED_INDICATOR_VISIBLE_MS);
    return () => window.clearTimeout(id);
  }, [recapSavedAt]);

  const recapDirty = recap !== initialRecap;

  const handleRecapSave = useCallback(async () => {
    if (!recapDirty || isSaving) return;
    setIsSaving(true);
    try {
      const res = await api.journalEdit({ portfolio, day: date, lowlights: recap });
      if (res && typeof res === "object" && "status" in res && res.status !== "ok") {
        log.error("mobile-daily-report", "journalEdit non-ok", res);
        return;
      }
      try {
        window.localStorage.removeItem(draftKey);
      } catch {
        // best-effort
      }
      // Mutate the in-memory journalRow.lowlights so dirty check resets.
      (journalRow as Record<string, unknown>).lowlights = recap;
      setRecapSavedAt(Date.now());
    } catch (err) {
      log.error("mobile-daily-report", "journalEdit threw", err);
    } finally {
      setIsSaving(false);
    }
  }, [recapDirty, isSaving, portfolio, date, recap, draftKey, journalRow]);

  // ── Focus mode hydration + toggle ────────────────────────────────
  const [focusOn, setFocusOn] = useState(false);
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(FOCUS_MODE_LS_KEY) === "on";
      setFocusOn(saved);
      setFocusModeActive(saved);
      if (saved) {
        document.body.classList.add("privacy");
      }
    } catch {
      // SSR / private mode — focus mode stays off
    }
  }, []);

  const toggleFocus = useCallback(() => {
    setFocusOn((prev) => {
      const next = !prev;
      setFocusModeActive(next);
      try {
        window.localStorage.setItem(FOCUS_MODE_LS_KEY, next ? "on" : "off");
        if (next) document.body.classList.add("privacy");
        else document.body.classList.remove("privacy");
      } catch {
        // best-effort
      }
      return next;
    });
  }, []);

  // ── Captures + EOD merged gallery ────────────────────────────────
  const mergedRows: MergedRow[] = useMemo(() => {
    const capRows: MergedRow[] = captures.map((c) => ({
      id: c.id,
      view_url: c.view_url,
      file_name: c.file_name,
      _kind: "capture" as const,
      _sortKey: c.created_at ?? "",
    }));
    const eodRows: MergedRow[] = eodSnapshots
      .filter((e) => e.view_url || e.image_url)
      .map((e, i) => ({
        // Negate id and offset by a constant to namespace EOD ids away
        // from capture ids. EOD never goes through delete (nonDeletable
        // predicate suppresses the X button) so collision math is moot
        // but keeps the React key stable.
        id: -1000 - (e.id ?? i),
        view_url: e.view_url ?? e.image_url ?? "",
        file_name: e.file_name ?? e.image_type ?? "EOD snapshot",
        _kind: "eod" as const,
        _sortKey: e.uploaded_at ?? "",
      }));
    return [...capRows, ...eodRows].sort((a, b) => b._sortKey.localeCompare(a._sortKey));
  }, [captures, eodSnapshots]);

  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const lightboxImages: LightboxImage[] = useMemo(
    () =>
      mergedRows.map((r) => ({
        url: r.view_url,
        alt: r.file_name ?? "Image",
      })),
    [mergedRows],
  );

  const handleUpload = useCallback(
    async (file: File) => {
      if (!journalId) {
        return { error: "Save the journal entry first to add images." };
      }
      const res = await api.uploadDailyJournalCapture(journalId, file, portfolio);
      if (res && typeof res === "object" && "error" in res) return res;
      // Refresh captures from server to pick up the new row in canonical shape.
      const fresh = await api.listDailyJournalCaptures(journalId, portfolio).catch(() => []);
      if (Array.isArray(fresh)) {
        setCaptures(fresh as DailyJournalCaptureRow[]);
      }
      return res as DailyJournalCaptureRow;
    },
    [journalId, portfolio, setCaptures],
  );

  const handleDelete = useCallback(
    async (id: number) => {
      // Only captures get here — nonDeletable predicate blocks EOD ids.
      const res = await api.deleteDailyJournalCapture(id);
      if (res && typeof res === "object" && "deleted" in res) {
        setCaptures(captures.filter((c) => c.id !== id));
      }
      return res;
    },
    [captures, setCaptures],
  );

  // ── Tags ─────────────────────────────────────────────────────────
  const assignedTagIds = useMemo(
    () => new Set(tagAssignments.map((a) => a.tag_id)),
    [tagAssignments],
  );
  const addableTags = useMemo(
    () => availableTags.filter((t) => !assignedTagIds.has(t.id)),
    [availableTags, assignedTagIds],
  );

  const handleAddTag = useCallback(
    async (tag: Tag) => {
      if (!journalId) return;
      const res = await api.createTagAssignment({
        tag_id: tag.id,
        entity_type: "daily_journal",
        entity_id: journalId,
      });
      if (res && typeof res === "object" && !("error" in res)) {
        const fresh = await api
          .listTagAssignments({ entity_type: "daily_journal", entity_id: journalId })
          .catch(() => []);
        if (Array.isArray(fresh)) {
          setTagAssignments(fresh as TagAssignment[]);
        }
      }
    },
    [journalId, setTagAssignments],
  );

  // ── Collapsible sections ─────────────────────────────────────────
  const [expandPerf, setExpandPerf] = useState(true);
  const [expandOpened, setExpandOpened] = useState(positionsOpened.length > 0);
  const [expandClosed, setExpandClosed] = useState(positionsClosed.length > 0);

  // Default expand follows row counts when data first lands.
  useEffect(() => {
    setExpandOpened(positionsOpened.length > 0);
  }, [positionsOpened.length]);
  useEffect(() => {
    setExpandClosed(positionsClosed.length > 0);
  }, [positionsClosed.length]);

  return (
    <div className="flex flex-col gap-3 pt-2">
      <ReportHeader
        date={date}
        portfolio={portfolio}
        marketCycle={String(journalRow.market_cycle ?? "")}
        dayNum={Number(journalRow.mct_display_day_num ?? 0) || null}
        focusOn={focusOn}
        onToggleFocus={toggleFocus}
        onBack={onBack}
      />

      <MetricsRow
        endNlv={endNlv}
        dailyPct={dailyPct}
        dailyDollar={dailyDollar}
        grade={grade}
        score={score}
      />

      <DrawdownTile ddPct={ddPct} ddDollar={ddDollar} />

      <PerformanceSection
        expanded={expandPerf}
        onToggle={() => setExpandPerf((v) => !v)}
        portfolioDaily={perf.portfolioDaily}
        portfolioYtd={perf.portfolioYtd}
        spyDaily={perf.spyDaily}
        spyYtd={perf.spyYtd}
        ndxDaily={perf.ndxDaily}
        ndxYtd={perf.ndxYtd}
      />

      <PositionsOpenedSection
        expanded={expandOpened}
        onToggle={() => setExpandOpened((v) => !v)}
        rows={positionsOpened}
      />

      <PositionsClosedSection
        expanded={expandClosed}
        onToggle={() => setExpandClosed((v) => !v)}
        rows={positionsClosed}
      />

      <DailyRecapSection
        value={recap}
        onChange={setRecap}
        dirty={recapDirty}
        isSaving={isSaving}
        savedAt={recapSavedAt}
        flashVisible={savedFlashVisible}
        onSave={handleRecapSave}
      />

      <DailyThoughtsSection html={String(journalRow.daily_thoughts ?? "")} />

      <CapturesGallerySection
        merged={mergedRows}
        captures={captures}
        eodCount={eodSnapshots.length}
        journalId={journalId}
        onUpload={handleUpload}
        onDelete={handleDelete}
        onThumbnailTap={(_row, idx) => setLightboxIdx(idx)}
      />

      <TagsSection
        assignments={tagAssignments}
        addableTags={addableTags}
        atCap={tagAssignments.length >= 10}
        disabled={!journalId}
        onAddTag={handleAddTag}
      />

      <ImageLightbox
        images={lightboxImages}
        activeIndex={lightboxIdx}
        onClose={() => setLightboxIdx(null)}
        onNavigate={(idx) => setLightboxIdx(idx)}
        ariaLabel="Daily Report image"
      />
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────

function ReportHeader({
  date,
  portfolio,
  marketCycle,
  dayNum,
  focusOn,
  onToggleFocus,
  onBack,
}: {
  date: string;
  portfolio: string;
  marketCycle: string;
  dayNum: number | null;
  focusOn: boolean;
  onToggleFocus: () => void;
  onBack: () => void;
}) {
  const meta = [portfolio, dayNum ? `D${dayNum}` : null, marketCycle || null]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="flex flex-col gap-1 px-5 pt-2">
      <div className="flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={onBack}
          aria-label="Back to Daily Journal"
          data-testid="report-back-button"
          className="-ml-1 flex h-9 w-9 items-center justify-center rounded-m-pill text-m-text-dim active:opacity-80"
        >
          <ChevronLeft size={22} strokeWidth={1.6} aria-hidden="true" />
        </button>
        <h1
          data-testid="report-title"
          className="min-w-0 flex-1 truncate text-center text-[20px] font-medium text-m-text"
        >
          {fmtDateHeader(date)}
        </h1>
        <button
          type="button"
          onClick={onToggleFocus}
          aria-label={focusOn ? "Disable focus mode" : "Enable focus mode"}
          aria-pressed={focusOn}
          data-testid="focus-mode-toggle"
          className={`flex h-9 w-9 items-center justify-center rounded-m-pill border-[0.5px] ${
            focusOn
              ? "border-m-accent-border bg-m-accent-tint text-m-accent"
              : "border-m-border bg-m-surface text-m-text-dim"
          } active:opacity-80`}
        >
          {focusOn ? (
            <EyeOff size={16} strokeWidth={1.6} aria-hidden="true" />
          ) : (
            <Eye size={16} strokeWidth={1.6} aria-hidden="true" />
          )}
        </button>
      </div>
      {meta && (
        <div className="text-center text-[11px] text-m-text-dim">{meta}</div>
      )}
    </div>
  );
}

function MetricsRow({
  endNlv,
  dailyPct,
  dailyDollar,
  grade,
  score,
}: {
  endNlv: number;
  dailyPct: number;
  dailyDollar: number | null;
  grade: string;
  score: number;
}) {
  return (
    <div className="grid grid-cols-3 gap-2 px-5">
      <MetricTile label="NLV">
        <span className="font-m-num text-[17px] font-medium tabular-nums text-m-text privacy-mask">
          {formatCurrency(endNlv, { decimals: 0 })}
        </span>
      </MetricTile>

      <MetricTile label="DAILY">
        <span
          className={`font-m-num text-[17px] font-medium tabular-nums ${pctClass(dailyPct)}`}
        >
          {signedPct(dailyPct)}
        </span>
        {dailyDollar != null && (
          <span
            className={`mt-0.5 font-m-num text-[11px] tabular-nums privacy-mask ${pctClass(dailyDollar)}`}
          >
            {dailyDollar >= 0 ? "+" : "−"}
            {formatCurrency(Math.abs(dailyDollar), { decimals: 0 })}
          </span>
        )}
      </MetricTile>

      <MetricTile label="GRADE">
        {grade ? (
          <span
            data-testid="metric-grade-value"
            className="font-m-num text-[17px] font-semibold tabular-nums"
            style={{ color: gradeColor(score) }}
          >
            {grade}
          </span>
        ) : (
          <span className="font-m-num text-[14px] tabular-nums text-m-text-faint">—</span>
        )}
      </MetricTile>
    </div>
  );
}

function MetricTile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-start rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2.5">
      <span className="mb-0.5 text-[10px] font-medium text-m-text-dim">{label}</span>
      {children}
    </div>
  );
}

function DrawdownTile({ ddPct, ddDollar }: { ddPct: number; ddDollar: number }) {
  const color = drawdownTierColor(ddPct);
  const message = drawdownTierMessage(ddPct);
  // Bar fill = abs(ddPct) / 15 capped at 100%. -15% is the deepest threshold.
  const fillPct = Math.min(Math.abs(ddPct) / 15, 1) * 100;

  return (
    <div
      data-testid="drawdown-tile"
      className="mx-5 flex flex-col gap-2 rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2.5"
    >
      <div className="flex items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="text-[10px] font-medium text-m-text-dim">DRAWDOWN</span>
          <span
            data-testid="drawdown-pct"
            className="font-m-num text-[15px] font-semibold tabular-nums"
            style={{ color }}
          >
            {ddPct.toFixed(2)}%
          </span>
        </div>
        <span
          data-testid="drawdown-dollar"
          className="font-m-num text-[11px] tabular-nums text-m-text-dim privacy-mask"
        >
          {ddDollar < 0 ? "−" : ""}
          {formatCurrency(Math.abs(ddDollar), { decimals: 0 })}
        </span>
      </div>
      {/* Progress bar */}
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-m-surface-2">
        <div
          data-testid="drawdown-bar-fill"
          className="absolute left-0 top-0 h-full rounded-full"
          style={{ width: `${fillPct}%`, background: color }}
        />
      </div>
      <div className="flex items-baseline justify-between">
        <span
          data-testid="drawdown-message"
          className="text-[10px] font-medium tracking-[0.06em]"
          style={{ color }}
        >
          {message}
        </span>
        <span className="text-[10px] text-m-text-faint">warn at −7.5%</span>
      </div>
    </div>
  );
}

function CollapsibleHeader({
  label,
  count,
  expanded,
  onToggle,
}: {
  label: string;
  count?: number | null;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={expanded}
      className="flex w-full items-center justify-between gap-2 px-5 py-1.5 text-left"
    >
      <span className="flex items-baseline gap-2 text-[11px] font-semibold tracking-[0.06em] text-m-text-dim">
        {label}
        {count != null && (
          <span className="font-m-num text-[10px] tabular-nums text-m-text-faint">
            · {count}
          </span>
        )}
      </span>
      <ChevronDown
        size={14}
        strokeWidth={1.6}
        aria-hidden="true"
        className={`text-m-text-dim transition-transform ${expanded ? "" : "-rotate-90"}`}
      />
    </button>
  );
}

function PerformanceSection({
  expanded,
  onToggle,
  portfolioDaily,
  portfolioYtd,
  spyDaily,
  spyYtd,
  ndxDaily,
  ndxYtd,
}: {
  expanded: boolean;
  onToggle: () => void;
  portfolioDaily: number;
  portfolioYtd: number;
  spyDaily: number;
  spyYtd: number;
  ndxDaily: number;
  ndxYtd: number;
}) {
  return (
    <section data-testid="perf-section">
      <CollapsibleHeader label="PERFORMANCE" expanded={expanded} onToggle={onToggle} />
      {expanded && (
        <div className="mx-5 overflow-hidden rounded-m-md border-[0.5px] border-m-border bg-m-surface">
          <div className="grid grid-cols-[80px_1fr_1fr] gap-1 border-b-[0.5px] border-m-border px-3 py-1.5 text-[9px] font-medium uppercase tracking-[0.06em] text-m-text-faint">
            <span>Index</span>
            <span className="text-right">Daily</span>
            <span className="text-right">YTD</span>
          </div>
          <PerfRow label="Portfolio" daily={portfolioDaily} ytd={portfolioYtd} />
          <PerfRow label="SPY" daily={spyDaily} ytd={spyYtd} />
          <PerfRow label="NASDAQ" daily={ndxDaily} ytd={ndxYtd} last />
        </div>
      )}
    </section>
  );
}

function PerfRow({
  label,
  daily,
  ytd,
  last,
}: {
  label: string;
  daily: number;
  ytd: number;
  last?: boolean;
}) {
  return (
    <div
      data-testid={`perf-row-${label.toLowerCase()}`}
      className={`grid grid-cols-[80px_1fr_1fr] items-baseline gap-1 px-3 py-2 ${
        last ? "" : "border-b-[0.5px] border-m-border"
      }`}
    >
      <span className="text-[12px] font-medium text-m-text">{label}</span>
      <span
        data-testid={`perf-${label.toLowerCase()}-daily`}
        className={`text-right font-m-num text-[13px] tabular-nums ${pctClass(daily)}`}
      >
        {signedPct(daily)}
      </span>
      <span
        data-testid={`perf-${label.toLowerCase()}-ytd`}
        className={`text-right font-m-num text-[13px] tabular-nums ${pctClass(ytd)}`}
      >
        {signedPct(ytd)}
      </span>
    </div>
  );
}

function PositionsOpenedSection({
  expanded,
  onToggle,
  rows,
}: {
  expanded: boolean;
  onToggle: () => void;
  rows: TradeDetail[];
}) {
  return (
    <section data-testid="opened-section">
      <CollapsibleHeader
        label="POSITIONS OPENED"
        count={rows.length}
        expanded={expanded}
        onToggle={onToggle}
      />
      {expanded && (
        <div className="mx-5 flex flex-col gap-2">
          {rows.length === 0 ? (
            <div
              data-testid="opened-empty"
              className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-3 text-center text-[12px] text-m-text-faint"
            >
              No positions opened.
            </div>
          ) : (
            rows.map((r) => (
              <div
                key={`${r.trade_id}-${r.date}`}
                data-testid={`opened-row-${r.trade_id}`}
                className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-m-num text-[13px] font-medium tabular-nums text-m-text">
                    {r.ticker}
                  </span>
                  <span className="font-m-num text-[11px] tabular-nums text-m-text-dim privacy-mask">
                    {r.shares} sh @ {formatCurrency(Number(r.amount ?? 0), { decimals: 2 })}
                  </span>
                </div>
                <div className="mt-0.5 text-[11px] text-m-text-faint">
                  {r.rule || "—"}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
}

function PositionsClosedSection({
  expanded,
  onToggle,
  rows,
}: {
  expanded: boolean;
  onToggle: () => void;
  rows: TradePosition[];
}) {
  return (
    <section data-testid="closed-section">
      <CollapsibleHeader
        label="POSITIONS CLOSED"
        count={rows.length}
        expanded={expanded}
        onToggle={onToggle}
      />
      {expanded && (
        <div className="mx-5 flex flex-col gap-2">
          {rows.length === 0 ? (
            <div
              data-testid="closed-empty"
              className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-3 text-center text-[12px] text-m-text-faint"
            >
              No positions closed.
            </div>
          ) : (
            rows.map((r) => {
              const pl = Number(r.realized_pl ?? 0) || 0;
              const pct = Number((r as Record<string, unknown>).return_pct ?? 0) || 0;
              return (
                <div
                  key={r.trade_id}
                  data-testid={`closed-row-${r.trade_id}`}
                  className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-m-num text-[13px] font-medium tabular-nums text-m-text">
                      {r.ticker}
                    </span>
                    <span
                      className={`font-m-num text-[13px] font-semibold tabular-nums privacy-mask ${pctClass(pl)}`}
                    >
                      {pl >= 0 ? "+" : "−"}
                      {formatCurrency(Math.abs(pl), { decimals: 0 })}
                    </span>
                  </div>
                  <div className="mt-0.5 flex items-baseline justify-between gap-2 text-[11px]">
                    <span className="text-m-text-faint">{r.rule || "—"}</span>
                    <span
                      className={`font-m-num tabular-nums ${pctClass(pct)}`}
                    >
                      {signedPct(pct)}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </section>
  );
}

function DailyRecapSection({
  value,
  onChange,
  dirty,
  isSaving,
  savedAt,
  flashVisible,
  onSave,
}: {
  value: string;
  onChange: (v: string) => void;
  dirty: boolean;
  isSaving: boolean;
  savedAt: number | null;
  flashVisible: boolean;
  onSave: () => void;
}) {
  const buttonLabel = isSaving ? "Saving…" : "Save";
  return (
    <section data-testid="recap-section" className="mx-5">
      <div className="mb-1.5 text-[11px] font-semibold tracking-[0.06em] text-m-text-dim">
        DAILY RECAP
      </div>
      <label className="block rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2.5">
        <textarea
          data-testid="recap-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label="Daily recap"
          placeholder="What happened today? (lowlights, mistakes, observations)"
          rows={4}
          className="block w-full resize-none bg-transparent text-sm text-m-text placeholder:text-m-text-faint focus:outline-none"
        />
      </label>
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <span
          data-testid="recap-saved-indicator"
          className={`text-[11px] text-m-text-faint transition-opacity ${
            flashVisible && savedAt ? "opacity-100" : "opacity-0"
          }`}
          aria-live="polite"
        >
          {savedAt ? `Saved ${Math.max(1, Math.round((Date.now() - savedAt) / 1000))}s ago ✓` : ""}
        </span>
        <button
          type="button"
          onClick={onSave}
          disabled={!dirty || isSaving}
          data-testid="recap-save-button"
          className="rounded-m-pill bg-m-accent px-4 py-1.5 text-[12px] font-medium text-m-accent-text-on disabled:opacity-40"
        >
          {buttonLabel}
        </button>
      </div>
    </section>
  );
}

function DailyThoughtsSection({ html }: { html: string }) {
  const hasContent = html && html.trim().length > 0;
  return (
    <section data-testid="thoughts-section" className="mx-5">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-[11px] font-semibold tracking-[0.06em] text-m-text-dim">
          DAILY THOUGHTS
        </span>
        <span className="text-[10px] italic text-m-purple">Edit in T2-4b →</span>
      </div>
      {hasContent ? (
        // Backend sanitizes via DOMPurify on the save side (ThoughtsEditor
        // pipeline); we render the stored HTML as-is here. T2-4b will add
        // an edit affordance with its own sanitizer if/when the editor
        // approach is chosen.
        <div
          data-testid="thoughts-html"
          className="prose-mobile rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-2.5 text-[13px] leading-snug text-m-text"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <div
          data-testid="thoughts-empty"
          className="rounded-m-md border-[0.5px] border-m-border bg-m-surface px-3 py-3 text-[12px] italic text-m-text-faint"
        >
          No thoughts written yet · Edit on desktop
        </div>
      )}
    </section>
  );
}

function CapturesGallerySection({
  merged,
  captures,
  eodCount,
  journalId,
  onUpload,
  onDelete,
  onThumbnailTap,
}: {
  merged: MergedRow[];
  captures: DailyJournalCaptureRow[];
  eodCount: number;
  journalId: number | null;
  onUpload: (file: File) => Promise<{ id: number; view_url: string; file_name: string | null } | { error: string }>;
  onDelete: (id: number) => Promise<{ deleted: true; id?: number } | { error: string }>;
  onThumbnailTap: (row: MergedRow, idx: number) => void;
}) {
  const subtitle = useMemo(() => {
    const parts: string[] = [];
    if (captures.length > 0) parts.push(`${captures.length} upload${captures.length === 1 ? "" : "s"}`);
    if (eodCount > 0) parts.push(`${eodCount} EOD`);
    return parts.join(" · ");
  }, [captures.length, eodCount]);

  return (
    <section data-testid="captures-section" className="mx-5">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-[11px] font-semibold tracking-[0.06em] text-m-text-dim">
          CAPTURES & EOD
        </span>
        {subtitle && (
          <span className="font-m-num text-[10px] tabular-nums text-m-text-faint">
            {subtitle}
          </span>
        )}
      </div>
      <MobileImageUpload<MergedRow>
        rows={merged}
        onUpload={(file) =>
          onUpload(file) as Promise<
            MergedRow | { error: string; detail?: unknown }
          >
        }
        onDelete={onDelete}
        onThumbnailTap={onThumbnailTap}
        disabled={!journalId}
        disabledMessage="Save the journal entry first to add images."
        emptyStateLabel="Add captures"
        nonDeletable={(row) => row._kind === "eod"}
        renderBadge={(row) =>
          row._kind === "eod" ? <EodBadge /> : null
        }
      />
    </section>
  );
}

function EodBadge() {
  return (
    <span
      data-testid="eod-badge"
      className="rounded-[3px] bg-m-purple px-1 py-px text-[8px] font-semibold tracking-wider text-m-bg"
    >
      EOD
    </span>
  );
}

function TagsSection({
  assignments,
  addableTags,
  atCap,
  disabled,
  onAddTag,
}: {
  assignments: TagAssignment[];
  addableTags: Tag[];
  atCap: boolean;
  disabled: boolean;
  onAddTag: (tag: Tag) => void;
}) {
  const [sheetOpen, setSheetOpen] = useState(false);

  return (
    <section data-testid="tags-section" className="mx-5 pb-6">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-[11px] font-semibold tracking-[0.06em] text-m-text-dim">
          TAGS
        </span>
        {assignments.length >= 10 && (
          <span className="text-[10px] text-m-text-faint">10 / 10</span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {assignments.map((a) => (
          <TagChip key={a.id} assignment={a} />
        ))}
        <button
          type="button"
          onClick={() => setSheetOpen(true)}
          disabled={disabled || atCap || addableTags.length === 0}
          data-testid="tag-add-button"
          className="inline-flex items-center gap-1 rounded-m-pill border border-dashed border-m-border-strong px-2 py-0.5 text-[11px] text-m-text-dim disabled:opacity-40"
        >
          <Plus size={11} strokeWidth={1.6} aria-hidden="true" />
          Add tag
        </button>
        {disabled && (
          <span className="text-[10px] text-m-text-faint">Save entry first</span>
        )}
      </div>
      {sheetOpen && (
        <TagAddSheet
          options={addableTags}
          onSelect={(tag) => {
            onAddTag(tag);
            setSheetOpen(false);
          }}
          onClose={() => setSheetOpen(false)}
        />
      )}
    </section>
  );
}

function TagChip({ assignment }: { assignment: TagAssignment }) {
  const palette = TAG_PALETTE[assignment.tag_color as TagTone];
  if (!palette) {
    return (
      <span
        data-testid={`tag-chip-${assignment.id}`}
        className="rounded-m-pill border-[0.5px] border-m-border bg-m-surface-2 px-2 py-px text-[11px] text-m-text-dim"
      >
        {assignment.tag_name}
      </span>
    );
  }
  return (
    <span
      data-testid={`tag-chip-${assignment.id}`}
      className="rounded-m-pill px-2 py-px text-[11px] font-medium"
      style={{
        background: palette.body,
        color: palette.text,
        border: `0.5px solid ${palette.ring}`,
      }}
    >
      {assignment.tag_name}
    </span>
  );
}

function TagAddSheet({
  options,
  onSelect,
  onClose,
}: {
  options: Tag[];
  onSelect: (tag: Tag) => void;
  onClose: () => void;
}) {
  // MobileSelectSheet's trigger tile is the wrong shape for inline use
  // (it owns its own button); we render a parallel bottom sheet inline
  // here, matching the existing visual + interaction model. Mirrors the
  // pattern in mobile-weekly-retro.tsx's PillSelectSheet.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  return (
    <>
      <button
        type="button"
        aria-label="Close tag picker"
        onClick={onClose}
        className="fixed inset-0 z-40 bg-black/50"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Add tag"
        data-testid="tag-add-sheet"
        className="fixed inset-x-0 bottom-0 z-50 flex max-h-[70vh] flex-col border-t-[0.5px] border-m-border bg-m-bg"
        style={{
          borderTopLeftRadius: "var(--m-radius-xl)",
          borderTopRightRadius: "var(--m-radius-xl)",
        }}
      >
        <div className="flex shrink-0 items-center justify-between border-b-[0.5px] border-m-border px-5 pt-4 pb-3">
          <h2 className="text-base font-medium text-m-text">Add tag</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-8 w-8 items-center justify-center text-m-text-dim"
          >
            <ChevronDown size={20} strokeWidth={1.5} aria-hidden="true" />
          </button>
        </div>
        <div
          role="listbox"
          className="min-h-0 flex-1 overflow-y-auto"
          style={{ paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))" }}
        >
          {options.length === 0 ? (
            <div className="px-5 py-6 text-center text-[13px] text-m-text-faint">
              All available tags are already assigned.
            </div>
          ) : (
            options.map((tag) => {
              const palette = TAG_PALETTE[tag.color as TagTone];
              return (
                <button
                  key={tag.id}
                  type="button"
                  role="option"
                  data-testid={`tag-option-${tag.id}`}
                  onClick={() => onSelect(tag)}
                  className="flex min-h-[48px] w-full items-center gap-2 border-b-[0.5px] border-m-border px-5 py-3 text-left last:border-b-0"
                >
                  {palette && (
                    <span
                      aria-hidden="true"
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ background: palette.dot }}
                    />
                  )}
                  <span className="text-[14px] text-m-text">{tag.name}</span>
                </button>
              );
            })
          )}
        </div>
      </div>
    </>
  );
}

// ── Loading + disabled wrappers ─────────────────────────────────────

function LoadingSkeleton({
  date,
  portfolio,
  onBack,
}: {
  date: string;
  portfolio: string;
  onBack: () => void;
}) {
  return (
    <div className="flex flex-col gap-3 pt-2">
      <SimpleHeader date={date} portfolio={portfolio} onBack={onBack} />
      <div className="grid grid-cols-3 gap-2 px-5">
        <div className="h-[60px] animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-[60px] animate-pulse rounded-m-md bg-m-surface-2" />
        <div className="h-[60px] animate-pulse rounded-m-md bg-m-surface-2" />
      </div>
      <div className="mx-5 h-[68px] animate-pulse rounded-m-md bg-m-surface-2" />
      <div className="mx-5 h-[120px] animate-pulse rounded-m-md bg-m-surface-2" />
      <div className="mx-5 h-[100px] animate-pulse rounded-m-md bg-m-surface-2" />
    </div>
  );
}

function DisabledState({
  date,
  portfolio,
  onBack,
}: {
  date: string;
  portfolio: string;
  onBack: () => void;
}) {
  return (
    <div className="flex flex-col gap-3 pt-2">
      <SimpleHeader date={date} portfolio={portfolio} onBack={onBack} />
      <div
        data-testid="no-entry-state"
        className="mx-5 flex flex-col items-center gap-2 rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-5 py-10 text-center"
      >
        <AlertCircle size={26} strokeWidth={1.4} className="text-m-text-faint" aria-hidden="true" />
        <div className="text-[14px] font-medium text-m-text">
          No journal entry for {fmtDateLong(date)}
        </div>
        <div className="text-[12px] text-m-text-dim">
          Save a Daily Routine for this date first, or create the entry on desktop.
        </div>
        <button
          type="button"
          onClick={onBack}
          data-testid="no-entry-back-button"
          className="mt-3 inline-flex items-center gap-1 rounded-m-pill bg-m-accent px-4 py-1.5 text-[12px] font-medium text-m-accent-text-on"
        >
          <ChevronLeft size={12} strokeWidth={1.8} aria-hidden="true" />
          Back to journal list
        </button>
      </div>
    </div>
  );
}

function SimpleHeader({
  date,
  portfolio,
  onBack,
}: {
  date: string;
  portfolio: string;
  onBack: () => void;
}) {
  return (
    <div className="flex flex-col gap-1 px-5 pt-2">
      <div className="flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={onBack}
          aria-label="Back to Daily Journal"
          data-testid="report-back-button"
          className="-ml-1 flex h-9 w-9 items-center justify-center rounded-m-pill text-m-text-dim active:opacity-80"
        >
          <ChevronLeft size={22} strokeWidth={1.6} aria-hidden="true" />
        </button>
        <h1
          data-testid="report-title"
          className="min-w-0 flex-1 truncate text-center text-[20px] font-medium text-m-text"
        >
          {fmtDateHeader(date)}
        </h1>
        <span className="h-9 w-9" aria-hidden="true" />
      </div>
      <div className="text-center text-[11px] text-m-text-dim">{portfolio}</div>
    </div>
  );
}

// Suppress unused-import warnings for icons reserved for follow-up
// affordances (chevron-right used by future detail-nav, plus-icon used
// in the populated empty-add CTA inside the primitive). Keep imports
// warm so the follow-up doesn't have to re-add them.
// eslint-disable-next-line @typescript-eslint/no-unused-expressions
void ChevronRight;
