"use client";

import { useState, useEffect, useMemo } from "react";
import { api, type TradeDetail } from "@/lib/api";

const EXEC_GRADES = ["A (Perfect)", "B (Good)", "C (Sloppy)", "D (Bad)", "F (Impulse)"];
const BEHAVIOR_TAGS = [
  "Followed Plan", "FOMO Entry", "Caught Knife", "Late Stop",
  "Hesitated", "Boredom Trade", "Sized Too Big", "Revenge Trade", "Panic Sell",
];
const WEEK_GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"];

type Tab = "grade" | "history";

function gradeColor(g: string) {
  if (g.startsWith("A")) return "#08a86b";
  if (g.startsWith("B")) return "#3b82f6";
  if (g.startsWith("C")) return "#f59f00";
  return "#e5484d";
}

// Persist retros to localStorage
function getMonday(dateStr: string): string {
  const d = new Date(dateStr + "T12:00:00");
  const day = d.getDay();
  const offset = day === 0 ? -6 : 1 - day;
  const mon = new Date(d);
  mon.setDate(d.getDate() + offset);
  return `${mon.getFullYear()}-${String(mon.getMonth() + 1).padStart(2, "0")}-${String(mon.getDate()).padStart(2, "0")}`;
}

function loadRetros(): Record<string, any> {
  try {
    const raw = JSON.parse(localStorage.getItem("mo-weekly-retros") || "{}");
    // Deduplicate: normalize all keys to Monday
    const cleaned: Record<string, any> = {};
    for (const [key, val] of Object.entries(raw)) {
      const monKey = getMonday(key);
      // Keep the one with more data (later savedAt wins)
      if (!cleaned[monKey] || (val as any).savedAt > (cleaned[monKey] as any).savedAt) {
        cleaned[monKey] = val;
      }
    }
    if (Object.keys(cleaned).length !== Object.keys(raw).length) {
      localStorage.setItem("mo-weekly-retros", JSON.stringify(cleaned));
    }
    return cleaned;
  } catch { return {}; }
}
function saveRetros(data: Record<string, any>) {
  localStorage.setItem("mo-weekly-retros", JSON.stringify(data));
}

export function WeeklyRetro({ navColor }: { navColor: string }) {
  const [details, setDetails] = useState<TradeDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("grade");
  const [weekDate, setWeekDate] = useState(() => {
    const n = new Date();
    const day = n.getDay(); // 0=Sun
    const offset = day === 0 ? -6 : 1 - day;
    const mon = new Date(n);
    mon.setDate(n.getDate() + offset);
    return `${mon.getFullYear()}-${String(mon.getMonth() + 1).padStart(2, "0")}-${String(mon.getDate()).padStart(2, "0")}`;
  });

  // Ticker-level grades
  const [tickerGrades, setTickerGrades] = useState<Record<string, { grade: string; behavior: string; notes: string }>>({});
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  // Week summary
  const [weekGrade, setWeekGrade] = useState("");
  const [bestDecision, setBestDecision] = useState("");
  const [worstDecision, setWorstDecision] = useState("");
  const [ruleChange, setRuleChange] = useState(false);
  const [ruleChangeText, setRuleChangeText] = useState("");
  const [saveMsg, setSaveMsg] = useState("");

  // History
  const [retros, setRetros] = useState<Record<string, any>>({});

  useEffect(() => {
    api.tradesRecent("CanSlim", 1000).then(d => {
      setDetails(d as TradeDetail[]);
      setLoading(false);
    }).catch(() => setLoading(false));
    setRetros(loadRetros());
  }, []);

  // Load existing retro when week changes
  useEffect(() => {
    const existing = retros[monStr];
    if (existing) {
      setWeekGrade(existing.weekGrade || "");
      setBestDecision(existing.bestDecision || "");
      setWorstDecision(existing.worstDecision || "");
      setRuleChange(existing.ruleChange || false);
      setRuleChangeText(existing.ruleChangeText || "");
      setTickerGrades(existing.tickerGrades || {});
    } else {
      setWeekGrade(""); setBestDecision(""); setWorstDecision("");
      setRuleChange(false); setRuleChangeText(""); setTickerGrades({});
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weekDate]);

  // Week range — always snap to Monday (Mon=1...Sun=0→treat as previous week's end)
  const _wd = new Date(weekDate + "T12:00:00");
  const _dayOfWeek = _wd.getDay(); // 0=Sun, 1=Mon...6=Sat
  // Sunday (0) belongs to the PREVIOUS trading week → snap back to that Monday
  const _monOffset = _dayOfWeek === 0 ? -6 : 1 - _dayOfWeek;
  const monday = new Date(_wd);
  monday.setDate(_wd.getDate() + _monOffset);
  const friday = new Date(monday);
  friday.setDate(monday.getDate() + 4);
  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);
  const monStr = `${monday.getFullYear()}-${String(monday.getMonth() + 1).padStart(2, "0")}-${String(monday.getDate()).padStart(2, "0")}`;
  const sunStr = `${sunday.getFullYear()}-${String(sunday.getMonth() + 1).padStart(2, "0")}-${String(sunday.getDate()).padStart(2, "0")}`;

  const weekTxns = useMemo(() => {
    return details.filter(d => {
      const dt = String(d.date || "").slice(0, 10);
      return dt >= monStr && dt <= sunStr;
    }).sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
  }, [details, monStr, sunStr]);

  const grouped = useMemo(() => {
    const map: Record<string, TradeDetail[]> = {};
    for (const tx of weekTxns) {
      const t = tx.ticker || "Unknown";
      if (!map[t]) map[t] = [];
      map[t].push(tx);
    }
    return Object.entries(map).sort((a, b) => a[0].localeCompare(b[0]));
  }, [weekTxns]);

  const totalTx = weekTxns.length;
  const uniqueTickers = grouped.length;
  const buys = weekTxns.filter(d => String(d.action).toUpperCase() === "BUY");
  const sells = weekTxns.filter(d => String(d.action).toUpperCase() === "SELL");
  const isOveractive = totalTx > 15;
  const gradedTickers = Object.values(tickerGrades).filter(g => g.grade).length;

  const getGrade = (ticker: string) => tickerGrades[ticker] || { grade: "", behavior: "", notes: "" };
  const setGradeField = (ticker: string, field: string, value: string) => {
    setTickerGrades(prev => ({ ...prev, [ticker]: { ...getGrade(ticker), [field]: value } }));
  };

  const handleSave = () => {
    const data = {
      weekGrade, bestDecision, worstDecision, ruleChange, ruleChangeText,
      tickerGrades, savedAt: new Date().toISOString(),
    };
    const all = { ...retros, [monStr]: data };
    saveRetros(all);
    setRetros(all);
    setSaveMsg("Weekly retro saved!");
    setTimeout(() => setSaveMsg(""), 3000);
  };

  // History entries sorted newest first
  const historyEntries = useMemo(() => {
    return Object.entries(retros)
      .sort((a, b) => b[0].localeCompare(a[0]));
  }, [retros]);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  const inputStyle: React.CSSProperties = {
    background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  };

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Weekly <em className="italic" style={{ color: navColor }}>Retro</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Grade execution · Identify patterns · Refine rules</div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-5 pb-0.5" style={{ borderBottom: "2px solid var(--border)" }}>
        {([
          { key: "grade" as Tab, label: "Grade Week" },
          { key: "history" as Tab, label: "Review History" },
        ]).map(t => (
          <button key={t.key} onClick={() => {
            // Auto-save when switching away from grade tab
            if (tab === "grade" && t.key === "history" && (weekGrade || gradedTickers > 0)) {
              handleSave();
            }
            setTab(t.key);
          }}
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

      {/* ═══════════ GRADE WEEK TAB ═══════════ */}
      {tab === "grade" && (
        <>
          {/* Week selector */}
          <div className="flex items-center gap-4 mb-5">
            <div>
              <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Select Week</label>
              <input type="date" value={weekDate} onChange={e => setWeekDate(e.target.value)}
                     className="h-[38px] px-3 rounded-[10px] text-[13px] outline-none"
                     style={{ ...inputStyle, fontFamily: "var(--font-jetbrains), monospace" }} />
            </div>
            <div className="mt-5 px-4 py-2 rounded-[10px] text-[12px] font-medium"
                 style={{ background: "color-mix(in oklab, #1e40af 10%, var(--surface))", color: "#3b82f6", border: "1px solid color-mix(in oklab, #1e40af 30%, var(--border))" }}>
              Reviewing: <strong>{monday.toLocaleDateString("en-US", { month: "short", day: "numeric" })}</strong> → <strong>{friday.toLocaleDateString("en-US", { month: "short", day: "numeric" })}</strong>
            </div>
          </div>

          {/* Activity Monitor */}
          <div className="grid grid-cols-4 gap-3 mb-5">
            {[
              { k: "Total Tickets", v: totalTx, alert: isOveractive },
              { k: "Unique Tickers", v: uniqueTickers },
              { k: "Buys", v: buys.length },
              { k: "Sells / Trims", v: sells.length },
            ].map(m => (
              <div key={m.k} className="p-3 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>{m.k}</div>
                <div className="text-[22px] font-semibold mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: m.alert ? "#e5484d" : "var(--ink)" }}>{m.v}</div>
              </div>
            ))}
          </div>

          {/* Progress */}
          {uniqueTickers > 0 && (
            <div className="mb-5 flex items-center gap-3">
              <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "var(--bg)" }}>
                <div className="h-full rounded-full transition-all" style={{ width: `${(gradedTickers / uniqueTickers) * 100}%`, background: navColor }} />
              </div>
              <span className="text-[11px] font-medium" style={{ color: "var(--ink-4)" }}>{gradedTickers}/{uniqueTickers} tickers graded</span>
            </div>
          )}

          {/* Ticker cards */}
          {grouped.length > 0 ? (
            <div className="flex flex-col gap-3 mb-6">
              {grouped.map(([ticker, txns]) => {
                const isExpanded = expandedTicker === ticker;
                const g = getGrade(ticker);
                const txBuys = txns.filter(t => String(t.action).toUpperCase() === "BUY");
                const txSells = txns.filter(t => String(t.action).toUpperCase() === "SELL");

                return (
                  <div key={ticker} className="rounded-[14px] overflow-hidden"
                       style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
                    {/* Ticker header */}
                    <button onClick={() => setExpandedTicker(isExpanded ? null : ticker)}
                            className="w-full flex items-center justify-between px-5 py-3 text-left transition-colors hover:brightness-[0.98]">
                      <div className="flex items-center gap-3">
                        <span className="text-[16px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{ticker}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: "color-mix(in oklab, #08a86b 12%, var(--surface))", color: "#16a34a" }}>
                          {txBuys.length}B
                        </span>
                        {txSells.length > 0 && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: "color-mix(in oklab, #e5484d 12%, var(--surface))", color: "#dc2626" }}>
                            {txSells.length}S
                          </span>
                        )}
                        {g.grade && (
                          <span className="text-[10px] px-2 py-0.5 rounded font-bold"
                                style={{ background: `${gradeColor(g.grade)}15`, color: gradeColor(g.grade) }}>
                            {g.grade}
                          </span>
                        )}
                      </div>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2"
                           style={{ transform: isExpanded ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}>
                        <path d="M6 9l6 6 6-6"/>
                      </svg>
                    </button>

                    {isExpanded && (
                      <div style={{ borderTop: "1px solid var(--border)", animation: "slide-up 0.12s ease-out" }}>
                        {/* Transactions (read-only context) */}
                        <div className="px-5 py-3" style={{ background: "var(--bg)" }}>
                          {txns.map((tx, i) => {
                            const isSell = String(tx.action).toUpperCase() === "SELL";
                            const mono = "var(--font-jetbrains), monospace";
                            return (
                              <div key={i} className="flex items-center gap-3 py-1.5 text-[11px]">
                                <span style={{ fontFamily: mono, color: "var(--ink-4)", width: 90 }}>{String(tx.date || "").slice(5, 16)}</span>
                                <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold"
                                      style={{ background: isSell ? "color-mix(in oklab, #e5484d 12%, var(--surface))" : "color-mix(in oklab, #08a86b 12%, var(--surface))", color: isSell ? "#dc2626" : "#16a34a" }}>
                                  {tx.action}
                                </span>
                                <span style={{ fontFamily: mono }}>{tx.trx_id || ""}</span>
                                <span style={{ fontFamily: mono }}>{tx.shares} shs</span>
                                <span className="privacy-mask" style={{ fontFamily: mono }}>@ ${parseFloat(String(tx.amount || 0)).toFixed(2)}</span>
                                <span style={{ color: "var(--ink-4)" }}>{tx.rule || ""}</span>
                              </div>
                            );
                          })}
                        </div>

                        {/* Ticker-level grading */}
                        <div className="px-5 py-4">
                          <div className="grid grid-cols-3 gap-3">
                            <div>
                              <label className="block text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Grade</label>
                              <select value={g.grade} onChange={e => setGradeField(ticker, "grade", e.target.value)}
                                      className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                                      style={{ ...inputStyle, appearance: "none" as any }}>
                                <option value="">Select...</option>
                                {EXEC_GRADES.map(gr => <option key={gr} value={gr}>{gr}</option>)}
                              </select>
                            </div>
                            <div>
                              <label className="block text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Behavior</label>
                              <select value={g.behavior} onChange={e => setGradeField(ticker, "behavior", e.target.value)}
                                      className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                                      style={{ ...inputStyle, appearance: "none" as any }}>
                                <option value="">Select...</option>
                                {BEHAVIOR_TAGS.map(bt => <option key={bt} value={bt}>{bt}</option>)}
                              </select>
                            </div>
                            <div>
                              <label className="block text-[9px] uppercase tracking-[0.06em] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>Analysis / Lesson</label>
                              <input type="text" value={g.notes} onChange={e => setGradeField(ticker, "notes", e.target.value)}
                                     placeholder="What did you learn?"
                                     className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                                     style={{ ...inputStyle, fontFamily: "inherit" }} />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12 text-sm mb-6" style={{ color: "var(--ink-4)" }}>No trades found for this week.</div>
          )}

          {/* Weekly Summary */}
          <div className="rounded-[14px] overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center gap-2 px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Weekly Summary</span>
            </div>
            <div className="p-5 flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Overall Week Grade</label>
                  <select value={weekGrade} onChange={e => setWeekGrade(e.target.value)}
                          className="w-full h-[42px] px-3 rounded-[10px] text-[14px] font-semibold outline-none"
                          style={{ ...inputStyle, appearance: "none" as any, color: weekGrade ? gradeColor(weekGrade) : "var(--ink)" }}>
                    <option value="">Select grade...</option>
                    {WEEK_GRADES.map(g => <option key={g} value={g}>{g}</option>)}
                  </select>
                </div>
                <div className="flex items-end">
                  {weekGrade && (
                    <span className="text-[36px] font-semibold" style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: gradeColor(weekGrade), lineHeight: 1 }}>
                      {weekGrade}
                    </span>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Best Decision This Week</label>
                  <input type="text" value={bestDecision} onChange={e => setBestDecision(e.target.value)}
                         placeholder="One win to repeat..." className="w-full h-[42px] px-3 rounded-[10px] text-[13px] outline-none"
                         style={{ ...inputStyle, fontFamily: "inherit" }} />
                </div>
                <div>
                  <label className="block text-[10px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Worst Decision This Week</label>
                  <input type="text" value={worstDecision} onChange={e => setWorstDecision(e.target.value)}
                         placeholder="One mistake to fix..." className="w-full h-[42px] px-3 rounded-[10px] text-[13px] outline-none"
                         style={{ ...inputStyle, fontFamily: "inherit" }} />
                </div>
              </div>
              <div>
                <label className="flex items-center gap-2 mb-2 cursor-pointer text-[13px]">
                  <input type="checkbox" checked={ruleChange} onChange={e => setRuleChange(e.target.checked)} className="rounded" />
                  <span className="font-medium">Rule Change Needed?</span>
                </label>
                {ruleChange && (
                  <input type="text" value={ruleChangeText} onChange={e => setRuleChangeText(e.target.value)}
                         placeholder="e.g., New rule: no buying on Day 1 of FTD..."
                         className="w-full h-[42px] px-3 rounded-[10px] text-[13px] outline-none"
                         style={{ ...inputStyle, fontFamily: "inherit" }} />
                )}
              </div>
            </div>
          </div>

          {saveMsg && (
            <div className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
                 style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
              {saveMsg}
            </div>
          )}

          <button onClick={handleSave}
                  className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110"
                  style={{ background: "#6366f1" }}>
            Save Weekly Retro
          </button>
        </>
      )}

      {/* ═══════════ REVIEW HISTORY TAB ═══════════ */}
      {tab === "history" && (
        <div>
          {historyEntries.length > 0 ? (
            <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
              <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["Week Of", "Grade", "Best Decision", "Worst Decision", "Rule Change", "Tickers Graded"].map(h => (
                      <th key={h} className="text-left text-[10px] uppercase tracking-[0.06em] font-semibold px-4 py-2.5 whitespace-nowrap"
                          style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {historyEntries.map(([weekKey, data], i) => {
                    const tg = data.tickerGrades || {};
                    const gradedList = Object.entries(tg).filter(([, v]: [string, any]) => v.grade);
                    return (
                      <tr key={weekKey} style={{ borderBottom: i < historyEntries.length - 1 ? "1px solid var(--border)" : "none" }}
                          className="transition-colors"
                          onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                        <td className="px-4 py-3 font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{weekKey}</td>
                        <td className="px-4 py-3">
                          {data.weekGrade && (
                            <span className="text-[14px] font-bold" style={{ color: gradeColor(data.weekGrade) }}>{data.weekGrade}</span>
                          )}
                        </td>
                        <td className="px-4 py-3" style={{ color: "var(--ink-3)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {data.bestDecision || "—"}
                        </td>
                        <td className="px-4 py-3" style={{ color: "#e5484d", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {data.worstDecision || "—"}
                        </td>
                        <td className="px-4 py-3 text-[11px]" style={{ color: data.ruleChange ? "#e5484d" : "var(--ink-4)" }}>
                          {data.ruleChange ? data.ruleChangeText || "Yes" : "—"}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex gap-1 flex-wrap">
                            {gradedList.map(([t, v]: [string, any]) => (
                              <span key={t} className="text-[9px] px-1.5 py-0.5 rounded font-semibold"
                                    style={{ background: `${gradeColor(v.grade)}15`, color: gradeColor(v.grade) }}>
                                {t}: {v.grade.split(" ")[0]}
                              </span>
                            ))}
                            {gradedList.length === 0 && <span style={{ color: "var(--ink-4)" }}>—</span>}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-16 text-sm" style={{ color: "var(--ink-4)" }}>No weekly retros saved yet. Start by grading this week.</div>
          )}
        </div>
      )}
    </div>
  );
}
