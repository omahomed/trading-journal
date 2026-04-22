"use client";

import { useState, useEffect, useMemo } from "react";
import { api, getActivePortfolio, type JournalHistoryPoint } from "@/lib/api";

type ViewFilter = "week" | "month" | "all";
type Tab = "view" | "manage";

function scoreColor(score: number) {
  if (score >= 4) return "#08a86b";
  if (score >= 3) return "#f59f00";
  return "#e5484d";
}

function parseReportCard(highlights: string): Record<string, number> | null {
  try {
    if (!highlights || !highlights.startsWith("{")) return null;
    return JSON.parse(highlights);
  } catch { return null; }
}

const RC_KEYS = [
  { key: "plan", label: "Plan" },
  { key: "stops", label: "Stops" },
  { key: "sized", label: "Sized" },
  { key: "fomo", label: "FOMO" },
];

function windowColor(w: string) {
  const wl = (w || "").toLowerCase();
  if (wl.includes("power")) return "#8b5cf6";
  if (wl.includes("open") || wl.includes("confirmed") || wl.includes("grow")) return "#08a86b";
  if (wl.includes("neutral")) return "#f59f00";
  return "#e5484d";
}

function pctColor(v: number) {
  return v > 0 ? "#08a86b" : v < 0 ? "#e5484d" : "var(--ink-3)";
}

export function DailyJournal({ navColor }: { navColor: string }) {
  const [history, setHistory] = useState<JournalHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("view");
  const [viewFilter, setViewFilter] = useState<ViewFilter>("week");
  const [selectedMonth, setSelectedMonth] = useState(() => {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
  });
  const [editIdx, setEditIdx] = useState(-1);
  const [editFields, setEditFields] = useState<Record<string, string>>({});
  const [snapshots, setSnapshots] = useState<Array<{ id?: number; image_type?: string; view_url?: string; uploaded_at?: string }>>([]);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");

  useEffect(() => {
    api.journalHistory(getActivePortfolio(), 0).then(h => {
      setHistory((h as JournalHistoryPoint[]).sort((a, b) => String(b.day).localeCompare(String(a.day))));
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    if (history.length === 0) return [];
    const now = new Date();
    if (viewFilter === "week") {
      const weekAgo = new Date(now.getTime() - 7 * 86400000).toISOString().slice(0, 10);
      return history.filter(h => String(h.day).slice(0, 10) >= weekAgo);
    }
    if (viewFilter === "month") {
      return history.filter(h => String(h.day).slice(0, 7) === selectedMonth);
    }
    return history;
  }, [history, viewFilter, selectedMonth]);

  // Available months for picker
  const months = useMemo(() => {
    const set = new Set(history.map(h => String(h.day).slice(0, 7)));
    return [...set].sort().reverse();
  }, [history]);

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Daily <em className="italic" style={{ color: navColor }}>Journal</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          {history.length} entries · CanSlim
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-5 pb-0.5" style={{ borderBottom: "2px solid var(--border)" }}>
        {([
          { key: "view" as Tab, label: "View Logs" },
          { key: "manage" as Tab, label: "Manage Logs" },
        ]).map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
                  className="px-4 py-2 text-[12px] font-medium transition-all"
                  style={{
                    color: tab === t.key ? navColor : "var(--ink-4)",
                    borderBottom: tab === t.key ? `2px solid ${navColor}` : "2px solid transparent",
                    marginBottom: -2,
                  }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ VIEW LOGS ═══ */}
      {tab === "view" && (
        <>
          {/* Filter bar */}
          <div className="flex items-center gap-3 mb-5">
            <div className="flex p-0.5 rounded-[10px] gap-0.5" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              {([
                { key: "week" as ViewFilter, label: "Current Week" },
                { key: "month" as ViewFilter, label: "By Month" },
                { key: "all" as ViewFilter, label: "All History" },
              ]).map(f => (
                <button key={f.key} onClick={() => setViewFilter(f.key)}
                        className="px-3 py-1.5 rounded-[8px] text-[12px] font-medium transition-all"
                        style={{
                          background: viewFilter === f.key ? "var(--surface)" : "transparent",
                          color: viewFilter === f.key ? "var(--ink)" : "var(--ink-4)",
                          boxShadow: viewFilter === f.key ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
                        }}>
                  {f.label}
                </button>
              ))}
            </div>

            {viewFilter === "month" && (
              <select value={selectedMonth} onChange={e => setSelectedMonth(e.target.value)}
                      className="h-[34px] px-3 rounded-[10px] text-[12px]"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", appearance: "none" as any }}>
                {months.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            )}

            <span className="text-[12px] ml-auto" style={{ color: "var(--ink-4)" }}>{filtered.length} entries</span>
          </div>

          {/* Journal table */}
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
                <thead>
                  <tr>
                    {["Day", "Window", "End NLV", "Grade", "Daily %", "LTD %", "Heat", "SPY %", "SPY ATR", "NDX %", "NDX ATR", "Mkt Notes", "Plan", "Stops", "Sized", "FOMO", "Grade Notes"].map(h => (
                      <th key={h} className="text-left text-[9px] uppercase tracking-[0.06em] font-semibold px-2.5 py-2 whitespace-nowrap sticky top-0"
                          style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((h, i) => {
                    const dailyPct = h.daily_pct_change || 0;
                    const score = h.score || 0;
                    const heat = h.portfolio_heat || 0;
                    const mono = "var(--font-jetbrains), monospace";
                    const mw = (h as any).market_window || "";
                    const spyAtr = (h as any).spy_atr || 0;
                    const ndxAtr = (h as any).nasdaq_atr || 0;
                    const spyPct = (h as any).spy_daily_pct || 0;
                    const ndxPct = (h as any).ndx_daily_pct || 0;

                    return (
                      <tr key={i} className="transition-colors"
                          style={{ borderBottom: i < filtered.length - 1 ? "1px solid var(--border)" : "none" }}
                          onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                        <td className="px-2.5 py-2 whitespace-nowrap" style={{ fontFamily: mono, fontSize: 10, color: "var(--ink-4)" }}>
                          {String(h.day).slice(0, 10)}
                        </td>
                        <td className="px-2.5 py-2">
                          {mw && (
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold" style={{ background: `${windowColor(mw)}15`, color: windowColor(mw) }}>
                              {mw}
                            </span>
                          )}
                        </td>
                        <td className="px-2.5 py-2 privacy-mask" style={{ fontFamily: mono }}>${(h.end_nlv || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        {/* Grade + metrics + report card */}
                        {(() => {
                          const rc = parseReportCard((h as any).highlights || "");
                          const gradeLabel = score >= 5 ? "A+" : score >= 4 ? "A" : score >= 3 ? "B" : score >= 2 ? "C" : score > 0 ? "D" : "";
                          return (
                            <>
                              <td className="px-2.5 py-2 text-center">
                                {gradeLabel && (
                                  <span className="text-[11px] font-bold" style={{ color: scoreColor(score) }}>{gradeLabel}</span>
                                )}
                              </td>
                              <td className="px-2.5 py-2 font-semibold" style={{ fontFamily: mono, color: pctColor(dailyPct) }}>
                                {dailyPct >= 0 ? "+" : ""}{dailyPct.toFixed(2)}%
                              </td>
                              <td className="px-2.5 py-2" style={{ fontFamily: mono }}>{(h.portfolio_ltd || 0).toFixed(2)}%</td>
                              <td className="px-2.5 py-2" style={{ fontFamily: mono, color: heat > 20 ? "#e5484d" : heat > 10 ? "#f59f00" : "var(--ink-3)" }}>
                                {heat.toFixed(1)}%
                              </td>
                              {/* SPY/NDX */}
                              <td className="px-2.5 py-2" style={{ fontFamily: mono, color: pctColor(spyPct) }}>{spyPct.toFixed(2)}%</td>
                              <td className="px-2.5 py-2" style={{ fontFamily: mono, color: spyAtr > 1.25 ? "#e5484d" : spyAtr > 1.0 ? "#f59f00" : "#08a86b" }}>
                                {spyAtr > 0 ? `${spyAtr.toFixed(2)}%` : "—"}
                              </td>
                              <td className="px-2.5 py-2" style={{ fontFamily: mono, color: pctColor(ndxPct) }}>{ndxPct.toFixed(2)}%</td>
                              <td className="px-2.5 py-2" style={{ fontFamily: mono, color: ndxAtr > 1.25 ? "#e5484d" : ndxAtr > 1.0 ? "#f59f00" : "#08a86b" }}>
                                {ndxAtr > 0 ? `${ndxAtr.toFixed(2)}%` : "—"}
                              </td>
                              {/* Market Notes */}
                              <td className="px-2.5 py-2 text-[10px]" style={{ color: "var(--ink-3)", maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {(h as any).market_notes || ""}
                              </td>
                              {/* Report card categories */}
                              {RC_KEYS.map(cat => (
                                <td key={cat.key} className="px-2.5 py-2 text-center" style={{ fontFamily: mono, fontSize: 10 }}>
                                  {rc && rc[cat.key] != null ? (
                                    <span style={{ color: scoreColor(rc[cat.key]) }}>{rc[cat.key]}/5</span>
                                  ) : "—"}
                                </td>
                              ))}
                              {/* Grade Notes */}
                              <td className="px-2.5 py-2 text-[10px]" style={{ color: "#e5484d", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {(h as any).mistakes || ""}
                              </td>
                            </>
                          );
                        })()}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ═══ MANAGE LOGS ═══ */}
      {tab === "manage" && (() => {
        const startEdit = (idx: number) => {
          const h = history[idx];
          setEditIdx(idx);
          setSaveMsg("");
          setSnapshots([]);
          const day = String(h.day).slice(0, 10);
          api.listEodSnapshots(day, getActivePortfolio()).then(res => {
            if (Array.isArray(res)) setSnapshots(res as any);
          }).catch(() => {});
          setEditFields({
            end_nlv: String(h.end_nlv || 0),
            daily_pct_change: String(h.daily_pct_change || 0),
            pct_invested: String(h.pct_invested || 0),
            score: String(h.score || 0),
            market_window: (h as any).market_window || "",
            market_notes: (h as any).market_notes || "",
            market_action: (h as any).market_action || "",
            portfolio_heat: String(h.portfolio_heat || 0),
            spy_atr: String((h as any).spy_atr || 0),
            nasdaq_atr: String((h as any).nasdaq_atr || 0),
            highlights: (h as any).highlights || "",
            lowlights: (h as any).lowlights || "",
            mistakes: (h as any).mistakes || "",
            top_lesson: (h as any).top_lesson || "",
          });
        };

        const handleSave = async () => {
          if (editIdx < 0) return;
          setSaving(true);
          const h = history[editIdx];
          const result = await api.journalEdit({
            portfolio: getActivePortfolio(),
            day: String(h.day).slice(0, 10),
            end_nlv: editFields.end_nlv,
            daily_pct_change: editFields.daily_pct_change,
            pct_invested: editFields.pct_invested,
            score: editFields.score,
            market_window: editFields.market_window,
            market_notes: editFields.market_notes,
            market_action: editFields.market_action,
            portfolio_heat: editFields.portfolio_heat,
            spy_atr: editFields.spy_atr,
            nasdaq_atr: editFields.nasdaq_atr,
          });
          setSaving(false);
          if (result.status === "ok") {
            setSaveMsg("Saved successfully");
          } else {
            setSaveMsg(`Error: ${result.detail || "Unknown error"}`);
          }
        };

        const handleDelete = async () => {
          if (editIdx < 0) return;
          const h = history[editIdx];
          const day = String(h.day).slice(0, 10);
          if (!confirm(`Delete journal entry for ${day}? This cannot be undone.`)) return;
          setSaving(true);
          const result = await api.journalDelete(day, getActivePortfolio());
          setSaving(false);
          if (result.status === "ok") {
            setSaveMsg(`Deleted entry for ${day}`);
            setEditIdx(-1);
            // Reload history
            const h2 = await api.journalHistory(getActivePortfolio(), 0).catch(() => []);
            setHistory(h2 as any);
          } else {
            setSaveMsg(`Error: ${result.detail || "Delete failed"}`);
          }
        };

        const editEntry = editIdx >= 0 ? history[editIdx] : null;
        const inputCls = "w-full h-[38px] px-3 rounded-[10px] text-[12px] outline-none";
        const inputSt: React.CSSProperties = { background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: "var(--font-jetbrains), monospace" };

        return (
          <div className="flex flex-col gap-5">
            <div className="grid gap-5" style={{ gridTemplateColumns: "280px 1fr", alignItems: "start" }}>
              {/* Entry list */}
              <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="text-[13px] font-semibold">Select Entry</span>
                </div>
                <div className="max-h-[550px] overflow-y-auto">
                  {history.slice(0, 60).map((h, i) => {
                    const pct = h.daily_pct_change || 0;
                    const isSelected = editIdx === i;
                    return (
                      <button key={i} onClick={() => startEdit(i)}
                              className="w-full flex items-center justify-between px-4 py-2.5 text-left transition-all"
                              style={{
                                borderBottom: "1px solid var(--border)",
                                background: isSelected ? "var(--surface-2)" : "transparent",
                                borderLeft: isSelected ? `3px solid ${navColor}` : "3px solid transparent",
                              }}
                              onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = "var(--bg)"; }}
                              onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = "transparent"; }}>
                        <span className="text-[11px]" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink-3)" }}>
                          {String(h.day).slice(0, 10)}
                        </span>
                        <span className="text-[11px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: pctColor(pct) }}>
                          {pct >= 0 ? "+" : ""}{pct.toFixed(2)}%
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Edit form */}
              {editEntry ? (
                <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                  <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
                    <span className="text-[13px] font-semibold">Edit: {String(editEntry.day).slice(0, 10)}</span>
                  </div>
                  <div className="p-5 flex flex-col gap-4">
                    <div className="grid grid-cols-3 gap-3">
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>End NLV</label>
                        <input type="number" value={editFields.end_nlv} onChange={e => setEditFields({ ...editFields, end_nlv: e.target.value })} className={inputCls} style={inputSt} /></div>
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Daily %</label>
                        <input type="number" value={editFields.daily_pct_change} onChange={e => setEditFields({ ...editFields, daily_pct_change: e.target.value })} step="0.01" className={inputCls} style={inputSt} /></div>
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>% Invested</label>
                        <input type="number" value={editFields.pct_invested} onChange={e => setEditFields({ ...editFields, pct_invested: e.target.value })} className={inputCls} style={inputSt} /></div>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Score (1-5)</label>
                        <input type="number" value={editFields.score} onChange={e => setEditFields({ ...editFields, score: e.target.value })} min="0" max="5" className={inputCls} style={inputSt} /></div>
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Market Window</label>
                        <input type="text" value={editFields.market_window} onChange={e => setEditFields({ ...editFields, market_window: e.target.value })} className={inputCls} style={{ ...inputSt, fontFamily: "inherit" }} /></div>
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Portfolio Heat</label>
                        <input type="number" value={editFields.portfolio_heat} onChange={e => setEditFields({ ...editFields, portfolio_heat: e.target.value })} step="0.1" className={inputCls} style={inputSt} /></div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>SPY ATR</label>
                        <input type="number" value={editFields.spy_atr} onChange={e => setEditFields({ ...editFields, spy_atr: e.target.value })} step="0.01" className={inputCls} style={inputSt} /></div>
                      <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Nasdaq ATR</label>
                        <input type="number" value={editFields.nasdaq_atr} onChange={e => setEditFields({ ...editFields, nasdaq_atr: e.target.value })} step="0.01" className={inputCls} style={inputSt} /></div>
                    </div>
                    <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Market Notes</label>
                      <input type="text" value={editFields.market_notes} onChange={e => setEditFields({ ...editFields, market_notes: e.target.value })} className={inputCls} style={{ ...inputSt, fontFamily: "inherit" }} /></div>
                    <div><label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Market Action</label>
                      <input type="text" value={editFields.market_action} onChange={e => setEditFields({ ...editFields, market_action: e.target.value })} className={inputCls} style={{ ...inputSt, fontFamily: "inherit" }} /></div>

                    {saveMsg && (
                      <div className="text-[12px] font-medium px-3 py-2 rounded-[8px]"
                           style={{ background: saveMsg.startsWith("Error") ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : "color-mix(in oklab, #08a86b 10%, var(--surface))", color: saveMsg.startsWith("Error") ? "#dc2626" : "#16a34a", border: `1px solid ${saveMsg.startsWith("Error") ? "color-mix(in oklab, #e5484d 30%, var(--border))" : "color-mix(in oklab, #08a86b 30%, var(--border))"}` }}>
                        {saveMsg}
                      </div>
                    )}

                    <div className="flex items-center gap-3">
                      <button onClick={handleSave} disabled={saving}
                              className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold text-white transition-all hover:brightness-110 w-fit disabled:opacity-50"
                              style={{ background: navColor }}>
                        {saving ? "Saving..." : "Save Changes"}
                      </button>
                      <button onClick={handleDelete} disabled={saving}
                              className="h-[42px] px-6 rounded-[10px] text-[13px] font-semibold text-white transition-all hover:brightness-110 w-fit disabled:opacity-50"
                              style={{ background: "#e5484d" }}>
                        Delete Entry
                      </button>
                    </div>

                    {/* EOD Snapshots */}
                    {snapshots.length > 0 && (
                      <div className="mt-4 pt-4" style={{ borderTop: "1px solid var(--border)" }}>
                        <div className="text-[12px] font-semibold mb-3" style={{ color: "var(--ink-3)" }}>
                          End-of-Day Snapshots ({snapshots.length})
                        </div>
                        <div className="grid grid-cols-1 gap-3">
                          {snapshots.map((snap, idx) => (
                            <div key={snap.id ?? idx} className="rounded-[10px] overflow-hidden" style={{ border: "1px solid var(--border)", background: "var(--bg)" }}>
                              <div className="px-3 py-2 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
                                <span className="text-[11px] uppercase font-semibold" style={{ color: "var(--ink-4)" }}>
                                  {snap.image_type?.replace("eod_", "") || "Snapshot"}
                                </span>
                                {snap.uploaded_at && (
                                  <span className="text-[10px]" style={{ color: "var(--ink-4)", fontFamily: "var(--font-jetbrains), monospace" }}>
                                    {String(snap.uploaded_at).slice(0, 19)}
                                  </span>
                                )}
                              </div>
                              {snap.view_url && (
                                <a href={snap.view_url} target="_blank" rel="noopener noreferrer">
                                  <img src={snap.view_url} alt={snap.image_type}
                                       style={{ width: "100%", display: "block", cursor: "zoom-in" }} />
                                </a>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center text-sm rounded-[14px] py-20" style={{ color: "var(--ink-4)", background: "var(--bg)", border: "1px solid var(--border)" }}>
                  Select an entry to edit
                </div>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
