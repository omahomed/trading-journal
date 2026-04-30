"use client";

import { useState, useEffect } from "react";
import { api, getActivePortfolio } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";

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
  const { activePortfolio } = usePortfolio();
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

  // Active portfolio (single-portfolio UI; multi-portfolio switching will
  // come with the top-nav selector post-beta)
  const [portNlv, setPortNlv] = useState("");
  const [portHold, setPortHold] = useState("");
  const [portCash, setPortCash] = useState("0");
  const [portAction, setPortAction] = useState("");
  const [portPrev, setPortPrev] = useState(0);

  // IBKR auto-fill state. `nlvSource` and `holdingsSource` flip manual →
  // ibkr_auto on a successful pull and to ibkr_override the moment the user
  // types over the auto-filled value. They're tracked independently — user
  // can accept IBKR's NLV but override Holdings (or vice versa). Both go
  // into the save payload (diagnostic-only, not displayed on dashboard).
  // Single banner / loading flag covers both fields since they fail together.
  type IbkrSource = "manual" | "ibkr_auto" | "ibkr_override";
  const [nlvSource, setNlvSource] = useState<IbkrSource>("manual");
  const [holdingsSource, setHoldingsSource] = useState<IbkrSource>("manual");
  const [nlvLoading, setNlvLoading] = useState(true);
  const [ibkrError, setIbkrError] = useState<string>("");
  const [ibkrAutoFilledNlv, setIbkrAutoFilledNlv] = useState<string>("");
  const [ibkrAutoFilledHoldings, setIbkrAutoFilledHoldings] = useState<string>("");

  // Report Card
  const [scores, setScores] = useState<Record<string, number>>({ plan: 5, stops: 5, sized: 5, fomo: 5 });
  const [gradeNotes, setGradeNotes] = useState("");
  const [forceOverwrite, setForceOverwrite] = useState(false);

  // portPrev + SPY/NDX must reflect the *entryDate*, not the calendar today.
  // - portPrev: latest journal entry strictly BEFORE entryDate (so editing a
  //   past day doesn't diff against that day's own prior estimate). Without
  //   the `before` param, journalLatest returns the most recent entry overall
  //   and Daily % gets computed against the wrong baseline.
  // - SPY/NDX: when entryDate is in the past, use that date's CLOSE from
  //   yfinance instead of the current live price. Otherwise editing a past
  //   day silently overwrites SPY/NDX with whatever the index is doing right
  //   now — which can be the same value across consecutive days and zero out
  //   the Daily %% column.
  useEffect(() => {
    const today = new Date();
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
    const isPastDate = entryDate < todayStr;
    Promise.all([
      api.journalLatest(getActivePortfolio(), entryDate).catch(() => ({ end_nlv: 0 })),
      api.batchPrices(["SPY", "^IXIC"], undefined, isPastDate ? entryDate : undefined).catch(() => ({})),
    ]).then(([latestJ, prices]) => {
      setPortPrev(parseFloat(String((latestJ as any).end_nlv || 0)));
      const p = prices as Record<string, number>;
      if (p["SPY"]) setSpyClose(p["SPY"].toFixed(2));
      if (p["^IXIC"]) setNdxClose(p["^IXIC"].toFixed(2));
      setLoading(false);
    });
  }, [entryDate]);

  // Rally prefix lives in its own date-keyed effect so changing the date
  // picker refetches it. Also passes the entry date as `as_of_date` so the
  // prefix reflects the routine's date — not the latest ingested bar, which
  // can lag by a day. Backend projects forward when our date is past the
  // latest bar (see _project_rally_prefix_for_data_lag in api/main.py).
  // Cancellation guard mirrors the IBKR effect's contract.
  useEffect(() => {
    let cancelled = false;
    api.rallyPrefix(entryDate).catch(() => ({ prefix: "" })).then(rally => {
      if (cancelled) return;
      const prefix = (rally as any).prefix || "";
      if (prefix) setMarketNotes(prefix);
    });
    return () => { cancelled = true; };
  }, [entryDate]);

  // Auto-fill End NLV + Total Holdings from IBKR Flex Query. Runs on mount
  // and on date change. Decoupled from the rest of the load — failures
  // surface as one banner above the form (NLV and Holdings always succeed
  // or fail together since they share the same response). Cancellation
  // guard protects against the user changing the date mid-pull.
  useEffect(() => {
    let cancelled = false;
    setNlvLoading(true);
    setIbkrError("");
    api.ibkrNavForDate(entryDate).then(res => {
      if (cancelled) return;
      if (res.success) {
        const navStr = String(res.nav);
        const holdStr = String(res.position_value);
        setPortNlv(navStr);
        setPortHold(holdStr);
        setIbkrAutoFilledNlv(navStr);
        setIbkrAutoFilledHoldings(holdStr);
        setNlvSource("ibkr_auto");
        setHoldingsSource("ibkr_auto");
        setIbkrError("");
      } else {
        // Soft failure (no_data_for_date, ibkr_not_configured, etc.) —
        // both fields stay empty/editable, both source tags stay 'manual'.
        // One banner covers the whole pull; we don't render it twice.
        setIbkrAutoFilledNlv("");
        setIbkrAutoFilledHoldings("");
        setNlvSource("manual");
        setHoldingsSource("manual");
        setIbkrError(res.message || res.error || "IBKR pull failed");
      }
      setNlvLoading(false);
    }).catch((e: unknown) => {
      if (cancelled) return;
      setIbkrAutoFilledNlv("");
      setIbkrAutoFilledHoldings("");
      setNlvSource("manual");
      setHoldingsSource("manual");
      setIbkrError(e instanceof Error ? e.message : "IBKR pull failed");
      setNlvLoading(false);
    });
    return () => { cancelled = true; };
  }, [entryDate]);

  // Track override per field: any user edit to an auto-filled value flips
  // its source tag. Compare strings so re-typing the same number still
  // counts (the act of editing is what we record, not numeric divergence).
  const handleNlvChange = (v: string) => {
    setPortNlv(v);
    if (nlvSource === "ibkr_auto" && v !== ibkrAutoFilledNlv) {
      setNlvSource("ibkr_override");
    }
  };
  const handleHoldingsChange = (v: string) => {
    setPortHold(v);
    if (holdingsSource === "ibkr_auto" && v !== ibkrAutoFilledHoldings) {
      setHoldingsSource("ibkr_override");
    }
  };

  // Auto-populate Actions from today's detail rows.
  // Format: "SELL: GOOG, AAPL | BUY: NVDA".
  useEffect(() => {
    const buildActions = (details: any[]) => {
      const grouped: Record<string, string[]> = {};
      for (const d of details) {
        const dDate = String(d.date || "").slice(0, 10);
        if (dDate !== entryDate) continue;
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
    };

    api.tradesRecent(getActivePortfolio(), 1000).catch(() => []).then(det => {
      setPortAction(buildActions(det as any[]));
    });
  }, [entryDate]);

  // Computed
  const portNlvN = parseFloat(portNlv) || 0;
  const portCashN = parseFloat(portCash) || 0;
  const portDailyChg = portPrev > 0 ? portNlvN - portPrev - portCashN : 0;
  const portAdj = portPrev + portCashN;
  const portDailyPct = portAdj > 0 ? (portDailyChg / portAdj) * 100 : 0;
  const portInvPct = portNlvN > 0 ? ((parseFloat(portHold) || 0) / portNlvN) * 100 : 0;

  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const grade = letterGrade(totalScore, REPORT_CATEGORIES.length * 5);
  const overallScore = gradeToScore(grade);

  const handleSubmit = async () => {
    setSaving(true); setSaveMsg("");
    if (portNlvN <= 0 && portCashN === 0) {
      setSaveMsg("Error: Enter closing NLV");
      setSaving(false);
      return;
    }
    const r = await api.journalEdit({
      portfolio: getActivePortfolio(), day: entryDate,
      end_nlv: portNlvN, beg_nlv: portPrev,
      cash_change: portCashN, daily_dollar_change: portDailyChg, daily_pct_change: portDailyPct,
      pct_invested: portInvPct, spy: parseFloat(spyClose) || 0, nasdaq: parseFloat(ndxClose) || 0,
      market_notes: marketNotes, market_action: portAction, score: overallScore,
      // market_window: "" was sent here in the V10 era, overwriting the
      // column with empty string on every save. Phase 3a stopped reading
      // the field; the backend defaults a missing market_window to 'Open'
      // when persisting (historical preservation). No payload entry needed.
      highlights: JSON.stringify(scores), mistakes: gradeNotes,
      nlv_source: nlvSource,
      holdings_source: holdingsSource,
    });
    setSaving(false);
    setSaveMsg(r.status === "ok" ? "Saved" : `Error: ${r.detail || "Failed to save"}`);
  };

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Daily <em className="italic" style={{ color: navColor }}>Routine</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Master Blotter · End-of-Day</div>
      </div>

      {/* IBKR fallback banner — visible only when the auto-pull failed and the
          user hasn't yet typed an NLV. Once they fill the field, the
          remediation is implicit (they're doing it manually) so we hide it. */}
      {ibkrError && !nlvLoading && !portNlv && (
        <div className="mb-4 text-[12px] font-medium px-4 py-2.5 rounded-[10px]" role="alert" data-testid="ibkr-warning-banner"
             style={{
               background: "color-mix(in oklab, #f59f00 10%, var(--surface))",
               color: "#b45309",
               border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))",
             }}>
          ⚠ Could not auto-fill NLV from IBKR — please enter manually. Reason: {ibkrError}
        </div>
      )}

      {/* 3 columns: Market | Portfolio | Report Card */}
      <div className="grid grid-cols-3 gap-4 mb-6" style={{ alignItems: "start" }}>

        {/* Market */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Market</span>
          </div>
          <div className="p-4 flex flex-col gap-3">
            <Field label="Date">
              <input type="date" value={entryDate} onChange={e => setEntryDate(e.target.value)} className={inputCls} style={inputStyle} aria-label="Entry date" />
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

        {/* Active Portfolio */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-4 py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#6366f1" }} />
            <span className="text-[13px] font-semibold">{activePortfolio?.name ?? "Portfolio"}</span>
          </div>
          <div className="p-4 flex flex-col gap-3">
            <Field label="Closing NLV">
              <div className="relative">
                <input type="number" value={portNlv} onChange={e => handleNlvChange(e.target.value)} step="100"
                       placeholder={nlvLoading ? "Pulling NLV from IBKR…" : "0.00"}
                       className={inputCls} style={inputStyle} aria-label="Closing NLV" />
                {/* Status badge: live spinner while loading, ✓ once auto-filled
                    (and unchanged by the user), nothing once they override. */}
                {nlvLoading && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px]"
                        style={{ color: "var(--ink-4)" }} aria-label="Pulling NLV from IBKR">
                    Pulling…
                  </span>
                )}
                {!nlvLoading && nlvSource === "ibkr_auto" && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold px-1.5 py-0.5 rounded-[4px]"
                        style={{ background: "color-mix(in oklab, #08a86b 14%, var(--surface))", color: "#08a86b", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}
                        aria-label="Auto-filled from IBKR" data-testid="nlv-auto-badge">
                    ✓ IBKR
                  </span>
                )}
                {!nlvLoading && nlvSource === "ibkr_override" && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold px-1.5 py-0.5 rounded-[4px]"
                        style={{ background: "color-mix(in oklab, #f59f00 14%, var(--surface))", color: "#f59f00", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}
                        aria-label="Overrode IBKR auto-fill" data-testid="nlv-override-badge">
                    Override
                  </span>
                )}
              </div>
            </Field>
            <Field label="Total Holdings">
              <div className="relative">
                <input type="number" value={portHold} onChange={e => handleHoldingsChange(e.target.value)} step="100"
                       placeholder={nlvLoading ? "Pulling Holdings from IBKR…" : "0.00"}
                       className={inputCls} style={inputStyle} aria-label="Total Holdings" />
                {nlvLoading && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px]"
                        style={{ color: "var(--ink-4)" }} aria-label="Pulling Holdings from IBKR">
                    Pulling…
                  </span>
                )}
                {!nlvLoading && holdingsSource === "ibkr_auto" && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold px-1.5 py-0.5 rounded-[4px]"
                        style={{ background: "color-mix(in oklab, #08a86b 14%, var(--surface))", color: "#08a86b", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}
                        aria-label="Auto-filled from IBKR" data-testid="holdings-auto-badge">
                    ✓ IBKR
                  </span>
                )}
                {!nlvLoading && holdingsSource === "ibkr_override" && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold px-1.5 py-0.5 rounded-[4px]"
                        style={{ background: "color-mix(in oklab, #f59f00 14%, var(--surface))", color: "#f59f00", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}
                        aria-label="Overrode IBKR auto-fill" data-testid="holdings-override-badge">
                    Override
                  </span>
                )}
              </div>
            </Field>
            <Field label="Cash +/-">
              <input type="number" value={portCash} onChange={e => setPortCash(e.target.value)} step="100" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Actions">
              <input type="text" value={portAction} onChange={e => setPortAction(e.target.value)} placeholder="BUY: NVDA"
                     className={inputCls} style={{ ...inputStyle, fontFamily: "inherit" }} />
            </Field>
            {portNlvN > 0 && (
              <div className="grid grid-cols-2 gap-2 mt-1">
                {[
                  { k: "Prev NLV", v: `$${portPrev.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                  { k: "Daily $", v: `${portDailyChg >= 0 ? "+" : ""}$${portDailyChg.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, c: portDailyChg >= 0 ? "#08a86b" : "#e5484d" },
                  { k: "Daily %", v: `${portDailyPct >= 0 ? "+" : ""}${portDailyPct.toFixed(2)}%`, c: portDailyPct >= 0 ? "#08a86b" : "#e5484d" },
                  { k: "% Invested", v: `${portInvPct.toFixed(1)}%` },
                ].map(s => (
                  <div key={s.k} className="p-2 rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
                    <div className="text-[8px] uppercase tracking-[0.06em] font-semibold" style={{ color: "var(--ink-4)" }}>{s.k}</div>
                    <div className="text-[13px] font-semibold mt-0.5 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: (s as any).c || "var(--ink)" }}>{s.v}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Report Card */}
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

      {/* Submit */}
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
        {saving ? "Saving..." : "Save Daily Routine"}
      </button>
    </div>
  );
}
