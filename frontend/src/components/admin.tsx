"use client";

import { useState, useEffect, useCallback } from "react";
import { api, getActivePortfolio } from "@/lib/api";

const mono = "var(--font-jetbrains), monospace";

function Section({ title, icon, defaultOpen = false, children }: { title: string; icon: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-[14px] overflow-hidden mb-4" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
      <button onClick={() => setOpen(!open)} className="w-full flex items-center gap-2 px-5 py-3.5 text-left cursor-pointer">
        <span>{icon}</span>
        <span className="text-[14px] font-bold flex-1" style={{ color: "var(--ink)" }}>{title}</span>
        <span className="text-[11px]" style={{ color: "var(--ink-4)" }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && <div className="px-5 pb-5 pt-0" style={{ borderTop: "1px solid var(--border)" }}>{children}</div>}
    </div>
  );
}

function SaveBtn({ onClick, label = "Save", loading = false }: { onClick: () => void; label?: string; loading?: boolean }) {
  return (
    <button onClick={onClick} disabled={loading}
      className="px-4 py-2 rounded-[8px] text-[12px] font-semibold transition-all cursor-pointer"
      style={{ background: "#6366f1", color: "#fff", opacity: loading ? 0.6 : 1 }}>
      {loading ? "Saving..." : label}
    </button>
  );
}

function Toast({ msg, type }: { msg: string; type: "ok" | "err" }) {
  return (
    <div className="mt-2 text-[12px] font-medium px-3 py-1.5 rounded-[8px] inline-block"
         style={{ background: type === "ok" ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : "color-mix(in oklab, #e5484d 12%, var(--surface))", color: type === "ok" ? "#08a86b" : "#e5484d" }}>
      {type === "ok" ? "Done" : "Error"}: {msg}
    </div>
  );
}

function InputRow({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <div className="text-[11px] font-bold uppercase tracking-[0.08em] mb-1" style={{ color: "var(--ink-3)" }}>{label}</div>
      {help && <div className="text-[10px] mb-1.5" style={{ color: "var(--ink-4)" }}>{help}</div>}
      {children}
    </div>
  );
}

function TextInput({ value, onChange, placeholder, mono: useMono, type = "text", min, max, step }: {
  value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean; type?: string; min?: number; max?: number; step?: number;
}) {
  return (
    <input type={type} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
      min={min} max={max} step={step}
      className="h-[34px] px-3 rounded-[8px] text-[12px] w-full"
      style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: useMono ? mono : undefined }} />
  );
}

/* ══════════════════════════════════════════════════════════ */
/* ██ MAIN EXPORT                                          ██ */
/* ══════════════════════════════════════════════════════════ */
export function Admin({ navColor }: { navColor: string }) {
  const [toast, setToast] = useState<{ msg: string; type: "ok" | "err" } | null>(null);
  const [saving, setSaving] = useState(false);

  // Config state
  const [resetDate, setResetDate] = useState("");
  const [hardDecks, setHardDecks] = useState({ L1: { pct: 7.5, action: "Remove Margin" }, L2: { pct: 12.5, action: "Max 30% Invested" }, L3: { pct: 15, action: "Go to Cash" } });
  const [maxPositions, setMaxPositions] = useState(12);
  const [heatThreshold, setHeatThreshold] = useState(2.5);
  const [earningsCushion, setEarningsCushion] = useState({ pass_pct: 5, fail_pct: -2, default_max_risk_pct: 1 });
  const [pyramidRules, setPyramidRules] = useState({ trigger_pct: 5, alloc_pct: 20 });

  // Events
  const [events, setEvents] = useState<any[]>([]);
  const [evDate, setEvDate] = useState(new Date().toISOString().slice(0, 10));
  const [evLabel, setEvLabel] = useState("");
  const [evCat, setEvCat] = useState("market");
  const [evNotes, setEvNotes] = useState("");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editFields, setEditFields] = useState<any>({});

  // Audit
  const [auditData, setAuditData] = useState<any[]>([]);
  const [auditFilter, setAuditFilter] = useState("All");
  const [auditLimit, setAuditLimit] = useState(100);

  // Backfill state
  const [backfillPortfolio, setBackfillPortfolio] = useState(getActivePortfolio());
  const [backfillStart, setBackfillStart] = useState("");
  const [backfillEnd, setBackfillEnd] = useState("");
  const [backfillForce, setBackfillForce] = useState(false);
  const [backfillRunning, setBackfillRunning] = useState(false);
  const [backfillResult, setBackfillResult] = useState<{ checked: number; updated: number; errors: string[] } | null>(null);

  const runBackfill = async () => {
    setBackfillRunning(true);
    setBackfillResult(null);
    try {
      const res = await api.journalBackfillMetrics({
        portfolio: backfillPortfolio,
        start_date: backfillStart || undefined,
        end_date: backfillEnd || undefined,
        force: backfillForce,
      });
      if (res.status === "ok") {
        setBackfillResult({ checked: res.checked || 0, updated: res.updated || 0, errors: res.errors || [] });
        flash(`Backfilled ${res.updated}/${res.checked} entries`, "ok");
      } else {
        flash(res.detail || "Backfill failed", "err");
      }
    } catch (err: any) {
      flash(err.message || "Backfill failed", "err");
    }
    setBackfillRunning(false);
  };

  const flash = (msg: string, type: "ok" | "err") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  // Load config on mount
  useEffect(() => {
    Promise.all([
      api.config("reset_date").catch(() => ({ value: null })),
      api.config("hard_decks").catch(() => ({ value: null })),
      api.config("max_positions").catch(() => ({ value: null })),
      api.config("heat_threshold_pct").catch(() => ({ value: null })),
      api.config("earnings_cushion").catch(() => ({ value: null })),
      api.config("pyramid_rules").catch(() => ({ value: null })),
      api.events().catch(() => []),
    ]).then(([rd, hd, mp, ht, ec, pr, ev]) => {
      if (rd.value) setResetDate(String(rd.value));
      if (hd.value) setHardDecks(hd.value);
      if (mp.value) setMaxPositions(Number(mp.value));
      if (ht.value) setHeatThreshold(Number(ht.value));
      if (ec.value) setEarningsCushion(ec.value);
      if (pr.value) setPyramidRules(pr.value);
      setEvents(Array.isArray(ev) ? ev : []);
    });
  }, []);

  const loadEvents = useCallback(() => {
    api.events().then(ev => setEvents(Array.isArray(ev) ? ev : [])).catch(() => {});
  }, []);

  const loadAudit = useCallback(() => {
    const filterMap: Record<string, string | undefined> = { "All": undefined, "Config": "CONFIG", "Events": "EVENT", "Trades": "BUY" };
    api.audit(auditLimit, filterMap[auditFilter]).then(d => setAuditData(Array.isArray(d) ? d : [])).catch(() => {});
  }, [auditFilter, auditLimit]);

  const saveConfig = async (key: string, value: any, opts?: any) => {
    setSaving(true);
    try {
      const r = await api.setConfig(key, value, opts);
      if (r.status === "ok") flash(`${key} saved`, "ok");
      else flash(r.detail || "Save failed", "err");
    } catch { flash("Network error", "err"); }
    setSaving(false);
  };

  const catMeta = (cat: string) => {
    if (cat === "market") return { emoji: "M", color: "#dc2626" };
    if (cat === "macro") return { emoji: "P", color: "#9333ea" };
    return { emoji: "S", color: "#6b7280" };
  };

  return (
    <div>
      <div className="text-[11px] mb-5" style={{ color: "var(--ink-4)" }}>Settings here override hardcoded defaults app-wide. Changes are logged to the audit trail.</div>
      {toast && <div className="fixed top-4 right-4 z-50"><Toast msg={toast.msg} type={toast.type} /></div>}

      {/* ── 1. Risk Management ── */}
      <Section title="Risk Management" icon="R" defaultOpen>
        {/* Reset Date */}
        <InputRow label="Reset Date" help="Drawdown peak is calculated from the highest End NLV on or after this date.">
          <div className="flex gap-2 items-center">
            <input type="date" value={resetDate} onChange={e => setResetDate(e.target.value)}
              className="h-[34px] px-3 rounded-[8px] text-[12px]"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
            <SaveBtn label="Save Reset Date" loading={saving} onClick={() => saveConfig("reset_date", resetDate, { value_type: "date", category: "risk", description: "Date from which drawdown peak is calculated." })} />
          </div>
        </InputRow>

        <div className="h-px my-4" style={{ background: "var(--border)" }} />

        {/* Hard Decks */}
        <InputRow label="Hard Decks" help="Drawdown thresholds (% from peak NLV). L1 < L2 < L3.">
          <div className="grid grid-cols-3 gap-3 mb-3">
            {(["L1", "L2", "L3"] as const).map((lvl, i) => {
              const colors = ["#eab308", "#f97316", "#dc2626"];
              const labels = ["Level 1", "Level 2", "Level 3"];
              return (
                <div key={lvl} className="p-3 rounded-[10px]" style={{ background: "var(--bg)", border: `2px solid ${colors[i]}20` }}>
                  <div className="text-[10px] font-bold mb-2" style={{ color: colors[i] }}>{labels[i]}</div>
                  <input type="number" value={hardDecks[lvl].pct} step={0.5} min={0.5} max={50}
                    onChange={e => setHardDecks(prev => ({ ...prev, [lvl]: { ...prev[lvl], pct: parseFloat(e.target.value) || 0 } }))}
                    className="h-[30px] px-2 rounded-[6px] text-[12px] w-full mb-1.5"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                  <input type="text" value={hardDecks[lvl].action}
                    onChange={e => setHardDecks(prev => ({ ...prev, [lvl]: { ...prev[lvl], action: e.target.value } }))}
                    className="h-[30px] px-2 rounded-[6px] text-[11px] w-full"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                </div>
              );
            })}
          </div>
          <SaveBtn label="Save Hard Decks" loading={saving} onClick={() => {
            if (!(hardDecks.L1.pct < hardDecks.L2.pct && hardDecks.L2.pct < hardDecks.L3.pct)) {
              flash("Hard decks must increase: L1 < L2 < L3", "err"); return;
            }
            saveConfig("hard_decks", hardDecks, { value_type: "json", category: "risk", description: "Hard deck drawdown thresholds and action labels." });
          }} />
        </InputRow>

        <div className="h-px my-4" style={{ background: "var(--border)" }} />

        {/* Max Positions */}
        <InputRow label="Max Positions" help="Maximum concurrent open positions. Shown on Dashboard as X/N Pos.">
          <div className="flex gap-2 items-center">
            <input type="number" value={maxPositions} min={1} max={100} step={1}
              onChange={e => setMaxPositions(parseInt(e.target.value) || 12)}
              className="h-[34px] px-3 rounded-[8px] text-[12px] w-[100px]"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            <SaveBtn label="Save" loading={saving} onClick={() => saveConfig("max_positions", maxPositions, { value_type: "number", category: "risk" })} />
          </div>
        </InputRow>
      </Section>

      {/* ── 2. Event Log ── */}
      <Section title="Event Log — Dashboard EC Markers" icon="E" defaultOpen>
        <div className="text-[10px] mb-3" style={{ color: "var(--ink-4)" }}>
          Events show as vertical lines on the EC. <span style={{ color: "#dc2626" }}>Market</span> = technical. <span style={{ color: "#9333ea" }}>Macro</span> = news/political. <span style={{ color: "#6b7280" }}>System</span> = auto (locked).
        </div>

        {/* Add form */}
        <div className="grid grid-cols-[1fr_2fr_1fr] gap-2 mb-2">
          <input type="date" value={evDate} onChange={e => setEvDate(e.target.value)}
            className="h-[34px] px-3 rounded-[8px] text-[12px]"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          <input type="text" value={evLabel} onChange={e => setEvLabel(e.target.value)} placeholder="Label (e.g. FTD, Trump Tweet)"
            className="h-[34px] px-3 rounded-[8px] text-[12px]"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          <select value={evCat} onChange={e => setEvCat(e.target.value)}
            className="h-[34px] px-2.5 rounded-[8px] text-[12px]"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
            <option value="market">Market</option>
            <option value="macro">Macro</option>
          </select>
        </div>
        <div className="flex gap-2 mb-4">
          <input type="text" value={evNotes} onChange={e => setEvNotes(e.target.value)} placeholder="Notes (optional)"
            className="h-[34px] px-3 rounded-[8px] text-[12px] flex-1"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          <SaveBtn label="Add Event" loading={saving} onClick={async () => {
            if (!evLabel.trim()) { flash("Label required", "err"); return; }
            setSaving(true);
            const r = await api.addEvent({ event_date: evDate, label: evLabel.trim(), category: evCat, notes: evNotes.trim() });
            if (r.status === "ok") { flash("Event added", "ok"); setEvLabel(""); setEvNotes(""); loadEvents(); }
            else flash("Add failed", "err");
            setSaving(false);
          }} />
        </div>

        {/* Events list */}
        <div className="text-[12px] font-semibold mb-2" style={{ color: "var(--ink)" }}>Existing Events ({events.length})</div>
        <div style={{ maxHeight: 400, overflow: "auto" }}>
          {events
            .sort((a, b) => String(b.event_date || "").localeCompare(String(a.event_date || "")))
            .map((ev) => {
              const id = ev.id;
              const isAuto = Boolean(ev.auto_generated);
              const { emoji, color } = catMeta(ev.category);
              const isEditing = editingId === id;

              if (isEditing && !isAuto) {
                return (
                  <div key={id} className="p-3 mb-1 rounded-[8px]" style={{ background: "var(--bg)", border: `2px solid ${color}40` }}>
                    <div className="grid grid-cols-[1fr_2fr_1fr] gap-2 mb-2">
                      <input type="date" value={editFields.event_date || ""} onChange={e => setEditFields((p: any) => ({ ...p, event_date: e.target.value }))}
                        className="h-[30px] px-2 rounded-[6px] text-[11px]"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                      <input type="text" value={editFields.label || ""} onChange={e => setEditFields((p: any) => ({ ...p, label: e.target.value }))}
                        className="h-[30px] px-2 rounded-[6px] text-[11px]"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                      <select value={editFields.category || "market"} onChange={e => setEditFields((p: any) => ({ ...p, category: e.target.value }))}
                        className="h-[30px] px-2 rounded-[6px] text-[11px]"
                        style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                        <option value="market">Market</option>
                        <option value="macro">Macro</option>
                      </select>
                    </div>
                    <input type="text" value={editFields.notes || ""} onChange={e => setEditFields((p: any) => ({ ...p, notes: e.target.value }))}
                      placeholder="Notes" className="h-[30px] px-2 rounded-[6px] text-[11px] w-full mb-2"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
                    <div className="flex gap-2">
                      <button className="px-3 py-1 rounded-[6px] text-[11px] font-semibold cursor-pointer" style={{ background: "#6366f1", color: "#fff" }}
                        onClick={async () => {
                          const r = await api.updateEvent(id, editFields);
                          if (r.status === "ok") { flash("Updated", "ok"); setEditingId(null); loadEvents(); }
                          else flash("Update failed", "err");
                        }}>Save</button>
                      <button className="px-3 py-1 rounded-[6px] text-[11px] font-semibold cursor-pointer" style={{ background: "var(--bg)", color: "var(--ink-3)", border: "1px solid var(--border)" }}
                        onClick={() => setEditingId(null)}>Cancel</button>
                    </div>
                  </div>
                );
              }

              return (
                <div key={id} className="flex items-center gap-2 px-3 py-2 mb-0.5 rounded-[6px] transition-colors hover:brightness-95"
                     style={{ borderLeft: `3px solid ${color}` }}>
                  <span className="text-[10px] font-bold uppercase w-[14px]" style={{ color }}>{emoji}</span>
                  <span className="text-[11px] font-semibold" style={{ color }}>{String(ev.event_date || "").slice(0, 10)}</span>
                  <span className="text-[12px] font-semibold flex-1" style={{ color: "var(--ink)" }}>{ev.label}{isAuto ? " (auto)" : ""}</span>
                  <span className="text-[10px] flex-1" style={{ color: "var(--ink-4)" }}>{ev.notes || ""}</span>
                  {!isAuto && (
                    <>
                      <button className="text-[10px] px-1.5 py-0.5 rounded cursor-pointer" style={{ color: "var(--ink-4)" }}
                        onClick={() => { setEditingId(id); setEditFields({ event_date: String(ev.event_date || "").slice(0, 10), label: ev.label, category: ev.category, notes: ev.notes || "" }); }}>Edit</button>
                      <button className="text-[10px] px-1.5 py-0.5 rounded cursor-pointer" style={{ color: "#e5484d" }}
                        onClick={async () => { const r = await api.deleteEvent(id); if (r.status === "ok") { flash("Deleted", "ok"); loadEvents(); } else flash("Delete failed", "err"); }}>Del</button>
                    </>
                  )}
                </div>
              );
            })}
        </div>
      </Section>

      {/* ── 3. Portfolio Heat ── */}
      <Section title="Portfolio Heat" icon="H">
        <InputRow label="Heat Alert Threshold (%)" help="Total portfolio heat target. Values at or above this trigger an alert.">
          <div className="flex gap-2 items-center">
            <input type="number" value={heatThreshold} min={0.5} max={20} step={0.1}
              onChange={e => setHeatThreshold(parseFloat(e.target.value) || 2.5)}
              className="h-[34px] px-3 rounded-[8px] text-[12px] w-[100px]"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            <SaveBtn label="Save" loading={saving} onClick={() => saveConfig("heat_threshold_pct", heatThreshold, { value_type: "number", category: "heat" })} />
          </div>
        </InputRow>
      </Section>

      {/* ── 4. Earnings Planner ── */}
      <Section title="Earnings Planner" icon="$">
        <InputRow label="Cushion Thresholds" help="PASS / THIN ICE / FAIL verdicts before earnings.">
          <div className="grid grid-cols-3 gap-3 mb-3">
            <div>
              <div className="text-[10px] font-semibold mb-1" style={{ color: "#08a86b" }}>PASS threshold (%)</div>
              <input type="number" value={earningsCushion.pass_pct} step={0.5}
                onChange={e => setEarningsCushion(p => ({ ...p, pass_pct: parseFloat(e.target.value) || 0 }))}
                className="h-[30px] px-2 rounded-[6px] text-[12px] w-full"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
            <div>
              <div className="text-[10px] font-semibold mb-1" style={{ color: "#e5484d" }}>FAIL threshold (%)</div>
              <input type="number" value={earningsCushion.fail_pct} step={0.5}
                onChange={e => setEarningsCushion(p => ({ ...p, fail_pct: parseFloat(e.target.value) || 0 }))}
                className="h-[30px] px-2 rounded-[6px] text-[12px] w-full"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
            <div>
              <div className="text-[10px] font-semibold mb-1" style={{ color: "var(--ink-3)" }}>Default max risk (%)</div>
              <input type="number" value={earningsCushion.default_max_risk_pct} step={0.05}
                onChange={e => setEarningsCushion(p => ({ ...p, default_max_risk_pct: parseFloat(e.target.value) || 0 }))}
                className="h-[30px] px-2 rounded-[6px] text-[12px] w-full"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
          </div>
          <SaveBtn label="Save Earnings Settings" loading={saving} onClick={() => {
            if (earningsCushion.fail_pct >= earningsCushion.pass_pct) { flash("FAIL must be less than PASS", "err"); return; }
            saveConfig("earnings_cushion", earningsCushion, { value_type: "json", category: "earnings" });
          }} />
        </InputRow>
      </Section>

      {/* ── 5. Pyramid Pace ── */}
      <Section title="Pyramid Pace" icon="P">
        <InputRow label="Pyramid Sizer Rules" help="trigger_pct = profit needed on last buy for a full add. alloc_pct = max share allocation per add.">
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <div className="text-[10px] font-semibold mb-1" style={{ color: "var(--ink-3)" }}>Trigger profit % (full add)</div>
              <input type="number" value={pyramidRules.trigger_pct} step={0.5} min={0.5} max={50}
                onChange={e => setPyramidRules(p => ({ ...p, trigger_pct: parseFloat(e.target.value) || 5 }))}
                className="h-[30px] px-2 rounded-[6px] text-[12px] w-full"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
            <div>
              <div className="text-[10px] font-semibold mb-1" style={{ color: "var(--ink-3)" }}>Max allocation % per add</div>
              <input type="number" value={pyramidRules.alloc_pct} step={1} min={1} max={100}
                onChange={e => setPyramidRules(p => ({ ...p, alloc_pct: parseFloat(e.target.value) || 20 }))}
                className="h-[30px] px-2 rounded-[6px] text-[12px] w-full"
                style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
          </div>
          <SaveBtn label="Save Pyramid Rules" loading={saving} onClick={() =>
            saveConfig("pyramid_rules", pyramidRules, { value_type: "json", category: "sizing" })} />
        </InputRow>
      </Section>

      {/* ── 6. Audit Trail ── */}
      <Section title="Audit Trail Viewer" icon="A">
        <div className="flex items-center gap-3 mb-3">
          <select value={auditFilter} onChange={e => setAuditFilter(e.target.value)}
            className="h-[34px] px-2.5 rounded-[8px] text-[12px]"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
            {["All", "Config", "Events", "Trades"].map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          <input type="number" value={auditLimit} min={10} max={1000} step={10}
            onChange={e => setAuditLimit(parseInt(e.target.value) || 100)}
            className="h-[34px] px-3 rounded-[8px] text-[12px] w-[80px]"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
          <button onClick={loadAudit} className="h-[34px] px-4 rounded-[8px] text-[12px] font-semibold cursor-pointer"
            style={{ background: navColor, color: "#fff" }}>Load</button>
        </div>

        {auditData.length > 0 ? (
          <div className="overflow-auto rounded-[10px]" style={{ border: "1px solid var(--border)", maxHeight: 400 }}>
            <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
              <thead className="sticky top-0">
                <tr style={{ background: "var(--bg-2)" }}>
                  {["Timestamp", "User", "Action", "Portfolio", "Trade ID", "Ticker", "Details"].map(h => (
                    <th key={h} className="text-left px-3 py-2 text-[9px] uppercase font-semibold" style={{ color: "var(--ink-4)", borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {auditData.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td className="px-3 py-1.5" style={{ color: "var(--ink-4)", fontSize: 10 }}>{String(r.timestamp || "").slice(0, 19)}</td>
                    <td className="px-3 py-1.5">{r.username}</td>
                    <td className="px-3 py-1.5 font-semibold">{r.action}</td>
                    <td className="px-3 py-1.5" style={{ color: "var(--ink-4)" }}>{r.portfolio}</td>
                    <td className="px-3 py-1.5" style={{ fontFamily: mono, fontSize: 10 }}>{r.trade_id}</td>
                    <td className="px-3 py-1.5 font-semibold">{r.ticker}</td>
                    <td className="px-3 py-1.5 text-[10px]" style={{ color: "var(--ink-3)", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>Click Load to fetch audit entries.</div>
        )}
      </Section>

      {/* ── 7. Journal Metrics Backfill ── */}
      <Section title="Journal Metrics Backfill" icon="B">
        <div className="text-[11px] mb-3" style={{ color: "var(--ink-4)" }}>
          Compute <strong>market_window</strong>, <strong>market_cycle</strong>, <strong>portfolio_heat</strong>, <strong>spy_atr</strong>, and <strong>nasdaq_atr</strong> for existing journal entries that are missing these values. Re-runs yfinance lookups, so may take time.
        </div>
        <div className="grid grid-cols-3 gap-3 mb-3">
          <InputRow label="Portfolio">
            <select value={backfillPortfolio} onChange={e => setBackfillPortfolio(e.target.value)}
              className="h-[34px] px-2.5 rounded-[8px] text-[12px] w-full"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
              {["CanSlim", "457B Plan", "TQQQ Strategy"].map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </InputRow>
          <InputRow label="Start Date (optional)">
            <TextInput type="date" value={backfillStart} onChange={setBackfillStart} mono />
          </InputRow>
          <InputRow label="End Date (optional)">
            <TextInput type="date" value={backfillEnd} onChange={setBackfillEnd} mono />
          </InputRow>
        </div>
        <label className="flex items-center gap-2 text-[12px] mb-3 cursor-pointer" style={{ color: "var(--ink-3)" }}>
          <input type="checkbox" checked={backfillForce} onChange={e => setBackfillForce(e.target.checked)} />
          <span>Force recompute (overwrite existing non-zero values)</span>
        </label>
        <SaveBtn label={backfillRunning ? "Running..." : "Run Backfill"} loading={backfillRunning} onClick={runBackfill} />
        {backfillResult && (
          <div className="mt-3 text-[12px] px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
            Checked <strong style={{ color: "var(--ink)" }}>{backfillResult.checked}</strong> entries, updated <strong style={{ color: "#08a86b" }}>{backfillResult.updated}</strong>.
            {backfillResult.errors.length > 0 && (
              <div className="mt-2 text-[11px]" style={{ color: "#e5484d" }}>
                First errors:<br />
                {backfillResult.errors.map((e, i) => <div key={i} style={{ fontFamily: mono }}>{e}</div>)}
              </div>
            )}
          </div>
        )}
      </Section>
    </div>
  );
}
