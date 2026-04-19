"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";

const REPORT_CATEGORIES = [
  { key: "plan", label: "Followed plan" },
  { key: "stops", label: "Respected stops" },
  { key: "sized", label: "Sized correctly" },
  { key: "fomo", label: "No FOMO entries" },
  { key: "journaled", label: "Journaled" },
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
function gradeToScore(g: string) { return g.startsWith("A") ? 5 : g.startsWith("B") ? 4 : g.startsWith("C") ? 3 : g.startsWith("D") ? 2 : 1; }
function gradeColor(g: string) { return g.startsWith("A") ? "#08a86b" : g.startsWith("B") ? "#3b82f6" : g.startsWith("C") ? "#f59f00" : "#e5484d"; }
function scoreColor(v: number) { return v >= 4 ? "#08a86b" : v >= 3 ? "#f59f00" : "#e5484d"; }

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>{label}</label>
      {children}
    </div>
  );
}

const inputCls = "w-full h-[38px] px-3 rounded-[10px] text-[13px] outline-none";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

export function DailyRoutine({ navColor }: { navColor: string }) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");

  // Market
  const [spyClose, setSpyClose] = useState("");
  const [ndxClose, setNdxClose] = useState("");
  const [marketNotes, setMarketNotes] = useState("");
  const [entryDate, setEntryDate] = useState(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  });

  // CanSlim
  const [csEnabled, setCsEnabled] = useState(true);
  const [csNlv, setCsNlv] = useState("");
  const [csHold, setCsHold] = useState("");
  const [csCash, setCsCash] = useState("0");
  const [csAction, setCsAction] = useState("");
  const [csPrev, setCsPrev] = useState(0);

  // 457B
  const [b4Enabled, setB4Enabled] = useState(true);
  const [b4Nlv, setB4Nlv] = useState("");
  const [b4Hold, setB4Hold] = useState("");
  const [b4Cash, setB4Cash] = useState("0");
  const [b4Action, setB4Action] = useState("");
  const [b4Prev, setB4Prev] = useState(0);

  // Report Card
  const [scores, setScores] = useState<Record<string, number>>({ plan: 5, stops: 5, sized: 5, fomo: 5, journaled: 5 });
  const [gradeNotes, setGradeNotes] = useState("");
  const [forceOverwrite, setForceOverwrite] = useState(false);

  useEffect(() => {
    Promise.all([
      api.journalLatest("CanSlim").catch(() => ({ end_nlv: 0 })),
      api.journalLatest("457B Plan").catch(() => ({ end_nlv: 0 })),
      api.batchPrices(["SPY", "^IXIC"]).catch(() => ({})),
      api.rallyPrefix().catch(() => ({ prefix: "" })),
    ]).then(([csJ, b4J, prices, rally]) => {
      setCsPrev(parseFloat(String((csJ as any).end_nlv || 0)));
      setB4Prev(parseFloat(String((b4J as any).end_nlv || 0)));
      const p = prices as Record<string, number>;
      if (p["SPY"]) setSpyClose(p["SPY"].toFixed(2));
      if (p["^IXIC"]) setNdxClose(p["^IXIC"].toFixed(2));
      const prefix = (rally as any).prefix || "";
      if (prefix) setMarketNotes(prefix);
      setLoading(false);
    });
  }, []);

  // Computed
  const csNlvN = parseFloat(csNlv) || 0;
  const csCashN = parseFloat(csCash) || 0;
  const csDailyChg = csPrev > 0 ? csNlvN - csPrev - csCashN : 0;
  const csAdj = csPrev + csCashN;
  const csDailyPct = csAdj > 0 ? (csDailyChg / csAdj) * 100 : 0;
  const csInvPct = csNlvN > 0 ? ((parseFloat(csHold) || 0) / csNlvN) * 100 : 0;

  const b4NlvN = parseFloat(b4Nlv) || 0;
  const b4CashN = parseFloat(b4Cash) || 0;
  const b4DailyChg = b4Prev > 0 ? b4NlvN - b4Prev - b4CashN : 0;
  const b4Adj = b4Prev + b4CashN;
  const b4DailyPct = b4Adj > 0 ? (b4DailyChg / b4Adj) * 100 : 0;
  const b4InvPct = b4NlvN > 0 ? ((parseFloat(b4Hold) || 0) / b4NlvN) * 100 : 0;

  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const grade = letterGrade(totalScore, REPORT_CATEGORIES.length * 5);
  const overallScore = gradeToScore(grade);

  const savePortfolio = async (portfolio: string, nlv: number, prev: number, holdings: string, cash: number, action: string, dc: number, dp: number, ip: number) => {
    if (nlv <= 0 && cash === 0) return { status: "skip" };
    return api.journalEdit({
      portfolio, day: entryDate, end_nlv: nlv, beg_nlv: prev,
      cash_change: cash, daily_dollar_change: dc, daily_pct_change: dp,
      pct_invested: ip, spy: parseFloat(spyClose) || 0, nasdaq: parseFloat(ndxClose) || 0,
      market_notes: marketNotes, market_action: action, score: overallScore,
      highlights: JSON.stringify(scores), mistakes: gradeNotes, market_window: "",
    });
  };

  const handleSubmit = async () => {
    setSaving(true); setSaveMsg("");
    let ok = 0;
    if (csEnabled && (csNlvN > 0 || csCashN !== 0)) {
      const r = await savePortfolio("CanSlim", csNlvN, csPrev, csHold, csCashN, csAction, csDailyChg, csDailyPct, csInvPct);
      if (r.status === "ok") ok++;
    }
    if (b4Enabled && (b4NlvN > 0 || b4CashN !== 0)) {
      const r = await savePortfolio("457B Plan", b4NlvN, b4Prev, b4Hold, b4CashN, b4Action, b4DailyChg, b4DailyPct, b4InvPct);
      if (r.status === "ok") ok++;
    }
    setSaving(false);
    setSaveMsg(ok > 0 ? `Successfully updated ${ok} portfolio(s)!` : "Error: Enter NLV for at least one portfolio");
  };

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  // Computed preview helper
  const Preview = ({ prev, dc, dp, ip }: { prev: number; dc: number; dp: number; ip: number }) => (
    <div className="grid grid-cols-2 gap-2 mt-3">
      {[
        { k: "Prev NLV", v: `$${prev.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
        { k: "Daily $", v: `${dc >= 0 ? "+" : ""}$${dc.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, c: dc >= 0 ? "#08a86b" : "#e5484d" },
        { k: "Daily %", v: `${dp >= 0 ? "+" : ""}${dp.toFixed(2)}%`, c: dp >= 0 ? "#08a86b" : "#e5484d" },
        { k: "% Invested", v: `${ip.toFixed(1)}%` },
      ].map(s => (
        <div key={s.k} className="p-2 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
          <div className="text-[8px] uppercase tracking-[0.06em] font-semibold" style={{ color: "var(--ink-4)" }}>{s.k}</div>
          <div className="text-[13px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: (s as any).c || "var(--ink)" }}>{s.v}</div>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Daily <em className="italic" style={{ color: navColor }}>Routine</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Master Blotter · End-of-Day</div>
      </div>

      {/* ═══ 4 COLUMNS: Market | CanSlim | 457B | Report Card ═══ */}
      <div className="grid grid-cols-4 gap-4 mb-6" style={{ alignItems: "start" }}>

        {/* ── Column 1: General Market Data ── */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Market</span>
          </div>
          <div className="p-4 flex flex-col gap-3">
            <Field label="Date">
              <input type="date" value={entryDate} onChange={e => setEntryDate(e.target.value)} className={inputCls} style={inputStyle} />
            </Field>
            <Field label="SPY Close">
              <input type="number" value={spyClose} onChange={e => setSpyClose(e.target.value)} step="0.01" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Nasdaq Close">
              <input type="number" value={ndxClose} onChange={e => setNdxClose(e.target.value)} step="0.01" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Market Notes">
              <input type="text" value={marketNotes} onChange={e => setMarketNotes(e.target.value)}
                     placeholder="Day 14 UPTREND: ..." className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
            </Field>
          </div>
        </div>

        {/* ── Column 1: CanSlim ── */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center justify-between px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#6366f1" }} />
              <span className="text-[13px] font-semibold">CanSlim</span>
            </div>
            <label className="flex items-center gap-1.5 cursor-pointer text-[11px]" style={{ color: "var(--ink-4)" }}>
              <input type="checkbox" checked={csEnabled} onChange={e => setCsEnabled(e.target.checked)} className="rounded" />
              Update
            </label>
          </div>
          {csEnabled && (
            <div className="p-4 flex flex-col gap-3">
              <Field label="Closing NLV">
                <input type="number" value={csNlv} onChange={e => setCsNlv(e.target.value)} step="100" placeholder="0.00" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Total Holdings">
                <input type="number" value={csHold} onChange={e => setCsHold(e.target.value)} step="100" placeholder="0.00" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Cash +/-">
                <input type="number" value={csCash} onChange={e => setCsCash(e.target.value)} step="100" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Actions">
                <input type="text" value={csAction} onChange={e => setCsAction(e.target.value)} placeholder="BUY: NVDA"
                       className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
              </Field>
              {csNlvN > 0 && <Preview prev={csPrev} dc={csDailyChg} dp={csDailyPct} ip={csInvPct} />}
            </div>
          )}
        </div>

        {/* ── Column 2: 457B ── */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center justify-between px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#f59f00" }} />
              <span className="text-[13px] font-semibold">457B Plan</span>
            </div>
            <label className="flex items-center gap-1.5 cursor-pointer text-[11px]" style={{ color: "var(--ink-4)" }}>
              <input type="checkbox" checked={b4Enabled} onChange={e => setB4Enabled(e.target.checked)} className="rounded" />
              Update
            </label>
          </div>
          {b4Enabled && (
            <div className="p-4 flex flex-col gap-3">
              <Field label="Closing NLV">
                <input type="number" value={b4Nlv} onChange={e => setB4Nlv(e.target.value)} step="100" placeholder="0.00" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Total Holdings">
                <input type="number" value={b4Hold} onChange={e => setB4Hold(e.target.value)} step="100" placeholder="0.00" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Cash +/-">
                <input type="number" value={b4Cash} onChange={e => setB4Cash(e.target.value)} step="100" className={inputCls} style={inputStyle} />
              </Field>
              <Field label="Actions">
                <input type="text" value={b4Action} onChange={e => setB4Action(e.target.value)} placeholder="Rebalance"
                       className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
              </Field>
              {b4NlvN > 0 && <Preview prev={b4Prev} dc={b4DailyChg} dp={b4DailyPct} ip={b4InvPct} />}
            </div>
          )}
        </div>

        {/* ── Column 3: Daily Report Card ── */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
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
            {REPORT_CATEGORIES.map(cat => (
              <div key={cat.key} className="flex items-center justify-between px-4 py-3">
                <span className="text-[12px] font-medium">{cat.label}</span>
                <div className="flex items-center gap-2.5">
                  <input type="range" min="1" max="5" value={scores[cat.key]}
                         onChange={e => setScores({ ...scores, [cat.key]: parseInt(e.target.value) })}
                         className="w-[80px] h-1 rounded-full appearance-none cursor-pointer"
                         style={{ accentColor: scoreColor(scores[cat.key]) }} />
                  <span className="text-[11px] font-semibold w-[28px] text-right" style={{ fontFamily: "var(--font-jetbrains), monospace", color: scoreColor(scores[cat.key]) }}>
                    {scores[cat.key]}/5
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
            <Field label="Grade Notes">
              <input type="text" value={gradeNotes} onChange={e => setGradeNotes(e.target.value)}
                     placeholder="Optional..." className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
            </Field>
          </div>
        </div>
      </div>

      {/* ═══ Bottom: Overwrite + Submit ═══ */}
      <label className="flex items-center gap-2 mb-4 cursor-pointer text-[12px]" style={{ color: "var(--ink-3)" }}>
        <input type="checkbox" checked={forceOverwrite} onChange={e => setForceOverwrite(e.target.checked)} className="rounded" />
        Force Overwrite Existing Entry
      </label>

      {saveMsg && (
        <div className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]"
             style={{
               background: saveMsg.startsWith("Error") ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : "color-mix(in oklab, #08a86b 10%, var(--surface))",
               color: saveMsg.startsWith("Error") ? "#dc2626" : "#16a34a",
               border: `1px solid ${saveMsg.startsWith("Error") ? "color-mix(in oklab, #e5484d 30%, var(--border))" : "color-mix(in oklab, #08a86b 30%, var(--border))"}`,
             }}>
          {saveMsg}
        </div>
      )}

      <button onClick={handleSubmit} disabled={saving}
              className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
              style={{ background: "#6366f1" }}>
        {saving ? "Saving..." : "LOG SELECTED ACCOUNTS"}
      </button>
    </div>
  );
}
