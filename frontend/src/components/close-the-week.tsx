"use client";

// Phase 4.6 — "Close the Week" component. Replaces the inline Weekly
// Summary block at weekly-retro.tsx:600-665 + the outer save button at
// :661-664. Visual contract from
// Design/design_handoff_weekly_retro/design/sections-bottom.jsx
// (WeeklySummary, lines 99-462).
//
// Owned state lives in the parent (weekly-retro.tsx) so the existing
// debounced auto-save effect keeps firing. This component is presentation
// + onChange — it never mutates a hidden store.

import { useEffect, useMemo, useRef, useState } from "react";
import type { NotesRailItem } from "@/lib/api";
import {
  GRADE_OPTIONS, GRADE_DEFS, deriveOverall,
  gradeColor, gradeTint,
} from "@/lib/grade-helpers";

export interface CloseTheWeekState {
  week_grade: string;             // canonical overall (derived or overridden)
  execution_grade: string;
  process_grade: string;
  pnl_grade: string;
  overall_override: boolean;
  reviewed_at: string | null;
  best_decision: string;
  worst_decision: string;
  rule_change: boolean;
  rule_change_text: string;
}

interface CloseTheWeekProps {
  state: CloseTheWeekState;
  onChange: (patch: Partial<CloseTheWeekState>) => void;
  onSave: () => void;
  saving?: boolean;
  /** Last N weeks' overall grades for the Recent Overall trend.
   *  Derived in the parent from railItems so this component stays
   *  presentation-only. */
  recentOverall: string[];
}

// ─── AxisGrade — one of three column selectors ─────────────────────────
function AxisGrade({
  label, help, value, onChange, disabled,
}: {
  label: string;
  help: string;
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
}) {
  return (
    <div data-testid={`axis-${label.toLowerCase()}`}
         style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-2)" }}>
          {label}
        </div>
        <div style={{ fontSize: 10.5, color: "var(--ink-4)", marginTop: 1 }}>
          {help}
        </div>
      </div>
      <div style={{ position: "relative" }}>
        <select value={value} onChange={(e) => onChange(e.target.value)}
                disabled={disabled}
                title={GRADE_DEFS[value] || "Select a grade"}
                aria-label={`${label} grade`}
                data-testid={`axis-${label.toLowerCase()}-select`}
                style={{
                  appearance: "none",
                  width: "100%", height: 38,
                  paddingLeft: 14, paddingRight: 36,
                  borderRadius: 10,
                  background: gradeTint(value),
                  border: `1px solid color-mix(in oklab, ${gradeColor(value)} 28%, var(--border))`,
                  color: gradeColor(value),
                  fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em",
                  fontFamily: "var(--font-fraunces), Georgia, serif",
                  fontStyle: "italic",
                  outline: "none",
                  cursor: disabled ? "not-allowed" : "pointer",
                  opacity: disabled ? 0.7 : 1,
                }}>
          <option value=""
                  style={{ fontFamily: "var(--font-ui)", fontStyle: "normal" }}>—</option>
          {GRADE_OPTIONS.map(g => (
            <option key={g} value={g}
                    style={{ fontFamily: "var(--font-ui)", fontStyle: "normal" }}>{g}</option>
          ))}
        </select>
        <span style={{
          position: "absolute", right: 12, top: "50%",
          transform: "translateY(-50%)", pointerEvents: "none",
          color: "var(--ink-4)",
        }}>
          <svg width={12} height={12} viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth={2.5}
               strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </span>
      </div>
    </div>
  );
}

// ─── Recent Overall trend strip ────────────────────────────────────────
function RecentOverallCard({
  recent, effectiveOverall,
}: {
  recent: string[];
  effectiveOverall: string;
}) {
  // Empty state: render 4 placeholder dashes so the card has stable size
  // even on a user's first week.
  const display = recent.length > 0
    ? recent
    : (["—", "—", "—", "—"] as string[]);
  return (
    <div data-testid="recent-overall"
         style={{
           display: "flex", flexDirection: "column", gap: 4,
           padding: "8px 14px", borderRadius: 10,
           background: "var(--bg)",
           border: "1px solid var(--border)",
         }}>
      <div style={{
        fontSize: 9, fontWeight: 600, letterSpacing: "0.10em",
        textTransform: "uppercase", color: "var(--ink-4)",
      }}>Recent overall</div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {display.map((g, i) => (
          <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            {i > 0 && (
              <span style={{ color: "var(--ink-4)", fontSize: 10 }}>·</span>
            )}
            <span style={{
              fontFamily: "var(--font-fraunces), Georgia, serif",
              fontStyle: "italic", fontSize: 13, fontWeight: 600,
              color: g === "—" ? "var(--ink-4)" : gradeColor(g),
            }}>{g}</span>
          </span>
        ))}
        <span style={{ color: "var(--ink-4)", fontSize: 10, marginLeft: 4 }}>→</span>
        <span data-testid="recent-overall-current"
              style={{
                fontFamily: "var(--font-fraunces), Georgia, serif",
                fontStyle: "italic", fontSize: 14, fontWeight: 700,
                color: gradeColor(effectiveOverall),
              }}>{effectiveOverall || "?"}</span>
      </div>
    </div>
  );
}

// ─── DecisionField — best / worst reflection textarea ──────────────────
function DecisionField({
  tone, label, hint, value, onChange, placeholder,
}: {
  tone: "best" | "worst";
  label: string;
  hint: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  const c = tone === "best" ? "#08a86b" : "#e5484d";
  return (
    <div data-testid={`decision-${tone}`}
         style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          width: 18, height: 18, borderRadius: 999,
          background: `color-mix(in oklab, ${c} 16%, var(--surface))`,
          color: c, display: "grid", placeItems: "center",
          fontSize: 11, fontWeight: 700,
        }}>{tone === "best" ? "✓" : "✗"}</span>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-2)" }}>
          {label}
        </span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--ink-4)" }}>{hint}</span>
      </div>
      <textarea value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                aria-label={label}
                style={{
                  width: "100%", minHeight: 86,
                  padding: "10px 12px", borderRadius: 10,
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  color: "var(--ink)",
                  fontSize: 13, lineHeight: 1.55,
                  outline: "none", resize: "vertical",
                }} />
    </div>
  );
}

// ─── Main shell ────────────────────────────────────────────────────────
export function CloseTheWeek({
  state, onChange, onSave, saving, recentOverall,
}: CloseTheWeekProps) {
  const [open, setOpen] = useState(true);

  const derivedOverall = useMemo(
    () => deriveOverall(state.execution_grade, state.process_grade, state.pnl_grade),
    [state.execution_grade, state.process_grade, state.pnl_grade],
  );

  // effectiveOverall: when override is on, use the persisted week_grade;
  // otherwise the live-derived value. Falls back to "" so gradeColor /
  // gradeTint return their neutral defaults on a fresh retro.
  const effectiveOverall = state.overall_override
    ? state.week_grade
    : (derivedOverall ?? state.week_grade ?? "");

  const reviewed = state.reviewed_at != null;
  const gradesDisabled = reviewed;

  // When override is off AND axes are all set, mirror the derived value
  // into week_grade so the parent's save payload sends the canonical
  // letter even before the user clicks Save. Backend re-derives on its
  // side (defense), but this keeps the local state consistent for the
  // UI swap (the band, the recent-overall arrow target, etc).
  // Use a ref to avoid an infinite useEffect loop — only fire when the
  // computed value actually differs from what's already in state.
  const lastSyncedRef = useRef<string | null>(null);
  useEffect(() => {
    if (state.overall_override) return;
    if (derivedOverall == null) return;
    if (state.week_grade === derivedOverall) return;
    if (lastSyncedRef.current === derivedOverall) return;
    lastSyncedRef.current = derivedOverall;
    onChange({ week_grade: derivedOverall });
  }, [derivedOverall, state.overall_override, state.week_grade, onChange]);

  const handleOverallSelect = (v: string) => {
    // If the user picks back the derived value, clear the override flag
    // so subsequent axis edits flow through. Otherwise pin the override.
    if (v === derivedOverall) {
      onChange({ week_grade: v, overall_override: false });
    } else {
      onChange({ week_grade: v, overall_override: true });
    }
  };

  const toggleReviewed = () => {
    onChange({
      reviewed_at: reviewed ? null : new Date().toISOString(),
    });
  };

  return (
    <div data-testid="close-the-week"
         style={{
           borderRadius: 14, overflow: "hidden", marginBottom: 20,
           background: "var(--surface)", border: "1px solid var(--border)",
           boxShadow: "var(--card-shadow)",
           position: "relative",
         }}>
      {/* Grade-tinted band */}
      <div data-testid="grade-band"
           style={{
             height: 4, width: "100%",
             background: gradeColor(effectiveOverall),
           }} />

      {/* Header */}
      <button type="button" onClick={() => setOpen(o => !o)}
              aria-expanded={open}
              data-testid="close-the-week-toggle"
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 10,
                padding: "12px 18px", textAlign: "left",
                borderBottom: open ? "1px solid var(--border)" : "none",
                background: "transparent", border: "none", cursor: "pointer",
              }}>
        <span style={{
          transition: "transform 150ms",
          transform: open ? "rotate(90deg)" : "none",
          display: "inline-flex", color: "var(--ink-4)",
        }}>
          <svg width={12} height={12} viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth={2.5}
               strokeLinecap="round" strokeLinejoin="round">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </span>
        <span style={{ width: 6, height: 6, borderRadius: 999, background: "#f59f00" }} />
        <span style={{ fontSize: 13, fontWeight: 600 }}>Close the Week</span>
        <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
          Final verdict · grade execution · commit to next steps
        </span>
        <span style={{ flex: 1 }} />
        {effectiveOverall && (
          <span style={{
            fontSize: 11, color: "var(--ink-4)",
            display: "inline-flex", alignItems: "center", gap: 6,
          }}>
            Overall
            <span style={{
              fontFamily: "var(--font-fraunces), Georgia, serif",
              fontSize: 16, fontStyle: "italic", fontWeight: 600,
              color: gradeColor(effectiveOverall),
            }}>{effectiveOverall}</span>
          </span>
        )}
      </button>

      {open && (
        <div style={{
          padding: 22, display: "flex", flexDirection: "column", gap: 22,
          animation: "slide-up 0.18s ease-out",
        }}>
          {/* ── Grade the week ──────────────────────────── */}
          <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, margin: 0 }}>
                Grade the week
              </h3>
              <span style={{ fontSize: 11, color: "var(--ink-4)" }}>
                Three axes — overall is averaged below
              </span>
            </div>
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12,
            }}>
              <AxisGrade label="Execution"
                         help="Did you follow your system?"
                         value={state.execution_grade}
                         onChange={(v) => onChange({ execution_grade: v })}
                         disabled={gradesDisabled} />
              <AxisGrade label="Process"
                         help="Sizing, stops, entries, exits."
                         value={state.process_grade}
                         onChange={(v) => onChange({ process_grade: v })}
                         disabled={gradesDisabled} />
              <AxisGrade label="P&L"
                         help="Actual result vs. expectation."
                         value={state.pnl_grade}
                         onChange={(v) => onChange({ pnl_grade: v })}
                         disabled={gradesDisabled} />
            </div>

            {/* Overall row — derived + override + recent anchor */}
            <div style={{
              display: "flex", alignItems: "center", gap: 14, paddingTop: 4,
            }}>
              <div data-testid="overall-card"
                   style={{
                     display: "flex", alignItems: "center", gap: 10,
                     padding: "10px 14px", borderRadius: 10,
                     background: gradeTint(effectiveOverall),
                     border: `1px solid color-mix(in oklab, ${gradeColor(effectiveOverall)} 28%, var(--border))`,
                     flex: 1,
                   }}>
                <div style={{ flex: 1 }}>
                  <div style={{
                    fontSize: 9, fontWeight: 600, letterSpacing: "0.10em",
                    textTransform: "uppercase", color: "var(--ink-4)",
                  }}>Overall</div>
                  <div data-testid="overall-state-line"
                       style={{
                         fontSize: 11, color: "var(--ink-3)", marginTop: 1,
                         fontStyle: state.overall_override ? "normal" : "italic",
                       }}>
                    {state.overall_override
                      ? `Overridden — average is ${derivedOverall ?? "—"}`
                      : `Derived from ${derivedOverall ?? "—"}`}
                  </div>
                </div>
                <select value={effectiveOverall}
                        onChange={(e) => handleOverallSelect(e.target.value)}
                        disabled={gradesDisabled}
                        aria-label="Overall week grade"
                        data-testid="overall-select"
                        style={{
                          appearance: "none",
                          width: 80, height: 48,
                          borderRadius: 8,
                          background: "var(--surface)",
                          border: `1px solid color-mix(in oklab, ${gradeColor(effectiveOverall)} 36%, var(--border))`,
                          color: gradeColor(effectiveOverall),
                          fontSize: 26, fontWeight: 600,
                          fontFamily: "var(--font-fraunces), Georgia, serif",
                          fontStyle: "italic",
                          textAlign: "center", outline: "none",
                          cursor: gradesDisabled ? "not-allowed" : "pointer",
                          opacity: gradesDisabled ? 0.7 : 1,
                        }}>
                  {GRADE_OPTIONS.map(g => (
                    <option key={g} value={g}
                            style={{ fontFamily: "var(--font-ui)", fontStyle: "normal" }}>
                      {g}
                    </option>
                  ))}
                </select>
              </div>

              <RecentOverallCard recent={recentOverall}
                                 effectiveOverall={effectiveOverall} />
            </div>

            {/* Grade definition (live) */}
            {effectiveOverall && GRADE_DEFS[effectiveOverall] && (
              <div data-testid="grade-interpretation"
                   style={{
                     fontSize: 11, color: "var(--ink-3)",
                     padding: "6px 10px", borderRadius: 6,
                     background: "var(--bg)",
                     borderLeft: `2px solid ${gradeColor(effectiveOverall)}`,
                   }}>
                <strong style={{
                  color: gradeColor(effectiveOverall),
                  fontFamily: "var(--font-fraunces), Georgia, serif",
                  fontStyle: "italic", fontSize: 12, marginRight: 6,
                }}>{effectiveOverall}</strong>
                {GRADE_DEFS[effectiveOverall]}
              </div>
            )}
          </section>

          {/* ── Reflections ───────────────────────────── */}
          <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <h3 style={{ fontSize: 13, fontWeight: 600, margin: 0 }}>Reflections</h3>
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12,
            }}>
              <DecisionField tone="best"
                             label="What worked?"
                             hint="One win worth repeating."
                             value={state.best_decision}
                             onChange={(v) => onChange({ best_decision: v })}
                             placeholder="One trade or decision worth repeating next week." />
              <DecisionField tone="worst"
                             label="What will I do differently?"
                             hint="One mistake worth fixing."
                             value={state.worst_decision}
                             onChange={(v) => onChange({ worst_decision: v })}
                             placeholder="One mistake to phrase as a rule for next week." />
            </div>
          </section>

          {/* ── Rule change (conditional) ─────────────── */}
          <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <label style={{
              display: "flex", alignItems: "center", gap: 8,
              cursor: "pointer", fontSize: 13,
            }}>
              <span style={{
                width: 16, height: 16, borderRadius: 4,
                border: `1.5px solid ${state.rule_change ? "#f59f00" : "var(--border-2)"}`,
                background: state.rule_change ? "#f59f00" : "transparent",
                display: "grid", placeItems: "center", color: "#fff",
              }}>
                {state.rule_change && (
                  <svg width={11} height={11} viewBox="0 0 24 24" fill="none"
                       stroke="currentColor" strokeWidth={3}
                       strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </span>
              <input type="checkbox" checked={state.rule_change}
                     onChange={(e) => onChange({ rule_change: e.target.checked })}
                     aria-label="Rule change needed"
                     data-testid="rule-change-checkbox"
                     style={{ display: "none" }} />
              <span style={{ fontWeight: 600 }}>Rule change needed?</span>
              <span style={{ color: "var(--ink-4)", fontSize: 11, fontWeight: 400 }}>
                Pattern repeating? Codify it.
              </span>
            </label>

            {state.rule_change && (
              <div style={{ animation: "slide-up 0.18s ease-out" }}>
                <textarea value={state.rule_change_text}
                          onChange={(e) => onChange({ rule_change_text: e.target.value })}
                          placeholder="What rule changes? Be precise — phrase it as you'd write it in your playbook."
                          aria-label="Rule change text"
                          data-testid="rule-change-textarea"
                          style={{
                            width: "100%", minHeight: 80,
                            padding: "10px 12px", borderRadius: 10,
                            background: "var(--bg)",
                            border: "1px solid var(--border)",
                            color: "var(--ink)",
                            fontSize: 13, lineHeight: 1.55,
                            outline: "none", resize: "vertical",
                          }} />
              </div>
            )}
          </section>

          {/* ── Footer: Mark reviewed + Save ──────────── */}
          <div style={{
            display: "flex", alignItems: "center", gap: 12,
            paddingTop: 14, marginTop: 4,
            borderTop: "1px solid var(--border)",
          }}>
            <label style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              cursor: "pointer", fontSize: 12,
              color: "var(--ink-2)", flex: 1,
            }}>
              <span style={{
                width: 16, height: 16, borderRadius: 4,
                border: `1.5px solid ${reviewed ? "#08a86b" : "var(--border-2)"}`,
                background: reviewed ? "#08a86b" : "transparent",
                display: "grid", placeItems: "center", color: "#fff",
              }}>
                {reviewed && (
                  <svg width={11} height={11} viewBox="0 0 24 24" fill="none"
                       stroke="currentColor" strokeWidth={3}
                       strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </span>
              <input type="checkbox" checked={reviewed}
                     onChange={toggleReviewed}
                     aria-label="Mark week as reviewed"
                     data-testid="mark-reviewed-checkbox"
                     style={{ display: "none" }} />
              <span style={{ fontWeight: 600 }}>Mark week as reviewed</span>
              <span style={{ color: "var(--ink-4)", fontWeight: 400 }}>
                · locks the grade & moves the dot to green in the rail
              </span>
            </label>

            <button type="button"
                    onClick={onSave}
                    disabled={saving}
                    data-testid="save-button"
                    style={{
                      height: 44, padding: "0 22px", borderRadius: 12,
                      background: reviewed ? "#08a86b" : "#6366f1",
                      color: "#fff", fontSize: 14, fontWeight: 600,
                      transition: "background 120ms",
                      display: "inline-flex", alignItems: "center", gap: 8,
                      border: "none", cursor: saving ? "default" : "pointer",
                      opacity: saving ? 0.75 : 1,
                    }}>
              {reviewed && (
                <svg width={14} height={14} viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" strokeWidth={2.5}
                     strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              )}
              {reviewed ? "Save & close week" : "Save weekly retro"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
