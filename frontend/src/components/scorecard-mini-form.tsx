"use client";

// Scorecard mini-form modal — Phase 2 merger. The Journal checklist item
// "captures" via this modal instead of the multi-portfolio NLV Entry form.
// Save writes to `trading_journal.highlights` (JSON per-category scores),
// `trading_journal.score` (coarse 1-5 letter-derived value), and
// `trading_journal.mistakes` (notes textarea) — same shape NLV Entry
// writes today, so both call sites remain compatible.
//
// Auto-tick: on save success, the caller (Daily Journal) auto-ticks the
// "Journal" routine item. Handled outside this component so the modal
// stays focused on capture; the caller owns refetches too.

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import {
  SCORECARD_CATEGORIES,
  SCORECARD_MAX_TOTAL,
  defaultScores,
  gradeToScore,
  letterGrade,
  parseHighlightsScores,
  scoreColor,
  type ScorecardCategory,
} from "@/lib/scorecard";

type ScoresMap = Record<ScorecardCategory["key"], number>;

export interface ScorecardMiniFormProps {
  /** Whether the modal is visible. Parent owns this state. */
  open: boolean;
  /** Portfolio to write to (the active portfolio on Daily Journal). */
  portfolio: string;
  /** Day the score applies to (YYYY-MM-DD, matches selectedDate on the
   *  parent Daily Journal page). */
  day: string;
  /** Existing values to pre-fill the form. Null / undefined fields
   *  fall back to per-field defaults. */
  initial: {
    highlights?: string | null;
    mistakes?: string | null;
  };
  /** Called on successful save (after journalEdit resolves ok). Parent
   *  refetches history + auto-ticks the Journal checklist item. */
  onSaved: () => void;
  /** Called when the user cancels / clicks the backdrop / hits Escape. */
  onClose: () => void;
}

export function ScorecardMiniForm({
  open,
  portfolio,
  day,
  initial,
  onSaved,
  onClose,
}: ScorecardMiniFormProps) {
  const [scores, setScores] = useState<ScoresMap>(() =>
    parseHighlightsScores(initial.highlights),
  );
  const [notes, setNotes] = useState<string>(() =>
    (initial.mistakes && initial.mistakes !== "nan") ? initial.mistakes : "",
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>("");

  // Rehydrate when the modal is (re)opened with fresh initial values.
  useEffect(() => {
    if (!open) return;
    setScores(parseHighlightsScores(initial.highlights));
    setNotes((initial.mistakes && initial.mistakes !== "nan") ? initial.mistakes : "");
    setError("");
  }, [open, initial.highlights, initial.mistakes]);

  // Escape closes when not saving.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !saving) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, saving, onClose]);

  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  const grade = letterGrade(total, SCORECARD_MAX_TOTAL);
  const coarseScore = gradeToScore(grade);

  const save = useCallback(async () => {
    if (saving) return;
    setSaving(true);
    setError("");
    try {
      const res = await api.journalEdit({
        portfolio,
        day,
        score: coarseScore,
        highlights: JSON.stringify(scores),
        mistakes: notes,
      });
      if (res.status === "ok") {
        onSaved();
      } else {
        setError(res.detail || `Save failed (${res.status})`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [saving, portfolio, day, coarseScore, scores, notes, onSaved]);

  if (!open) return null;

  return (
    <div data-testid="scorecard-modal-backdrop"
         className="fixed inset-0 z-[100] grid place-items-start justify-center pt-[10vh]"
         style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
         onClick={() => { if (!saving) onClose(); }}>
      <div className="w-[520px] max-w-[92vw] rounded-[14px] overflow-hidden"
           style={{ background: "var(--surface)", boxShadow: "0 20px 48px rgba(0,0,0,0.2), 0 0 0 1px var(--border)" }}
           onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="px-[18px] py-3.5 flex items-center justify-between"
             style={{ borderBottom: "1px solid var(--border)" }}>
          <div>
            <div className="text-[14px] font-semibold">Grade the day</div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>
              {day} · {portfolio}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[26px] font-semibold"
                  style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: scoreColor(coarseScore), lineHeight: 1 }}>
              {grade}
            </span>
            <kbd className="text-[10px] rounded px-1.5 py-0.5"
                 style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>ESC</kbd>
          </div>
        </div>

        {/* Category rows */}
        <div className="divide-y" style={{ borderColor: "var(--border)" }}>
          {SCORECARD_CATEGORIES.map((cat) => (
            <div key={cat.key} className="flex items-center justify-between px-4 py-3">
              <span className="text-[13px] font-medium">{cat.label}</span>
              <div className="flex items-center gap-1.5">
                {[1, 2, 3, 4, 5].map((v) => {
                  const active = scores[cat.key] === v;
                  return (
                    <button
                      key={v}
                      type="button"
                      disabled={saving}
                      onClick={() => setScores({ ...scores, [cat.key]: v })}
                      aria-pressed={active}
                      aria-label={`${cat.label}: ${v} of 5`}
                      data-testid={`scorecard-${cat.key}-${v}`}
                      className="w-[30px] h-[30px] rounded-[8px] text-[11px] font-semibold transition-all disabled:opacity-50"
                      style={{
                        background: active ? scoreColor(v) : "transparent",
                        color: active ? "white" : "var(--ink-3)",
                        border: `1px solid ${active ? scoreColor(v) : "var(--border)"}`,
                      }}>
                      {v}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Notes */}
        <div className="px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5"
                 style={{ color: "var(--ink-4)" }}>Notes (optional)</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            disabled={saving}
            placeholder="What surprised you? What would you do differently?"
            className="w-full px-3 py-2 rounded-[8px] text-[13px] outline-none"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", minHeight: 72, fontFamily: "inherit", lineHeight: 1.5 }}
          />
        </div>

        {/* Actions */}
        <div className="px-4 py-3 flex items-center gap-2"
             style={{ borderTop: "1px solid var(--border)", background: "var(--surface-2)" }}>
          <button type="button" onClick={onClose} disabled={saving}
                  className="px-4 py-2 rounded-[8px] text-[12px] transition-colors"
                  style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
            Cancel
          </button>
          <button type="button" onClick={() => void save()} disabled={saving}
                  className="ml-auto px-5 py-2 rounded-[8px] text-[12px] font-semibold text-white transition-all disabled:opacity-50"
                  data-testid="scorecard-save"
                  style={{ background: "#08a86b" }}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>

        {error && (
          <div className="px-4 py-2 text-[12px]"
               style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))", color: "#e5484d", borderTop: "1px solid var(--border)" }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

/** Default fallback for scores when the parent doesn't have a journal
 *  entry yet. Exported so parents can pass a stable object without
 *  having to re-derive on every render. */
export function emptyScorecardInitial(): { highlights: null; mistakes: null } {
  return { highlights: null, mistakes: null };
}

// Re-export for consumers that want to compute defaults without importing
// two modules.
export { defaultScores };
