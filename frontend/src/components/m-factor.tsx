"use client";

import { useState, useEffect, useMemo } from "react";
import { api } from "@/lib/api";

const STATE_COLORS: Record<string, { bg: string; fg: string }> = {
  POWERTREND: { bg: "#8A2BE2", fg: "#fff" },
  UPTREND: { bg: "#08a86b", fg: "#fff" },
  "RALLY MODE": { bg: "#f59f00", fg: "#000" },
  CORRECTION: { bg: "#e5484d", fg: "#fff" },
};

function LockIcon({ size = 11 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
         aria-label="Capped at 100%" role="img">
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </svg>
  );
}

export function MFactor({ navColor }: { navColor: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadData = () => {
    setLoading(true);
    api.rallyPrefix().then(d => { setData(d); setLoading(false); }).catch(() => setLoading(false));
  };

  useEffect(() => { loadData(); }, []);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  if (!data || data.error) return <div className="text-center py-16" style={{ color: "var(--ink-4)" }}>{data?.error || "Unable to compute cycle state."}</div>;

  const state = data.state || "CORRECTION";
  const sc = STATE_COLORS[state] || STATE_COLORS.CORRECTION;
  const dayNum = data.day_num || 0;
  const entryStep = data.entry_step ?? -1;
  const entryExp = data.entry_exposure || 0;
  const price = data.price || 0;
  const ladder = data.entry_ladder || [];
  const mono = "var(--font-jetbrains), monospace";

  const subtitles: Record<string, string> = {
    POWERTREND: "8 EMA > 21 EMA > 50 SMA > 200 SMA — all systems go",
    UPTREND: "Market structure intact — confirmed uptrend",
    "RALLY MODE": `Day ${dayNum} of rally attempt`,
    CORRECTION: `NASDAQ down ${Math.abs(data.drawdown_pct || 0).toFixed(1)}% from high`,
  };

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px] flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            M <em className="italic" style={{ color: navColor }}>Factor</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            NASDAQ cycle analysis · Entry & exit ladder
            {data.data_as_of && <span className="ml-2 opacity-70">· Data as of {data.data_as_of}</span>}
          </div>
        </div>
        <button onClick={loadData} className="flex items-center gap-1.5 h-[32px] px-3.5 rounded-[10px] text-xs font-medium"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
          </svg>
          Refresh
        </button>
      </div>

      {/* ═══ State Banner ═══ */}
      <div className="rounded-[14px] p-6 text-center mb-6" style={{ background: sc.bg, color: sc.fg, boxShadow: "0 4px 12px rgba(0,0,0,0.15)" }}>
        <div className="text-[12px] uppercase tracking-[0.15em] opacity-80 mb-1">NASDAQ M Factor</div>
        <div className="text-[52px] font-extrabold tracking-tight" style={{ lineHeight: 1.1 }}>{state}</div>
        {data.cap_at_100 && (
          <div className="inline-flex items-center gap-1 mt-2 px-2.5 py-1 rounded-full text-[11px] font-semibold"
               style={{ background: "rgba(0,0,0,0.2)" }}>
            <LockIcon /> Capped at 100%
          </div>
        )}
        <div className="text-[15px] mt-2 opacity-90">{subtitles[state] || ""}</div>
        <div className="text-[18px] font-bold mt-2">Suggested Exposure: {entryExp}%</div>
        {data.ftd_date && <div className="text-[12px] mt-1 opacity-70">FTD: {data.ftd_date}</div>}
        {state === "POWERTREND" && data.power_trend_on_since && (
          <div className="text-[12px] mt-1 opacity-70">Power-Trend ON since {data.power_trend_on_since}</div>
        )}
        {data.cycle_start_date && dayNum > 0 && (
          <div className="text-[12px] mt-1 opacity-70">Cycle started {data.cycle_start_date} (Day {dayNum})</div>
        )}
      </div>

      {/* ═══ Key Metrics ═══ */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        {[
          { label: "NASDAQ", value: `${price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, gradient: "linear-gradient(135deg, #6366f1, #818cf8)" },
          { label: "REF HIGH", value: `${(data.reference_high || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`, sub: data.reference_high_date || "", gradient: "linear-gradient(135deg, #7c3aed, #a78bfa)" },
          { label: "21 EMA", value: `${(data.ema21 || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`, sub: `${((price - (data.ema21 || 1)) / (data.ema21 || 1) * 100).toFixed(2)}%`, gradient: "linear-gradient(135deg, #10b981, #34d399)" },
          { label: "50 SMA", value: `${(data.sma50 || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`, sub: `${((price - (data.sma50 || 1)) / (data.sma50 || 1) * 100).toFixed(2)}%`, gradient: "linear-gradient(135deg, #f97316, #fb923c)" },
          { label: "200 SMA", value: `${(data.sma200 || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`, sub: `${((price - (data.sma200 || 1)) / (data.sma200 || 1) * 100).toFixed(2)}%`, gradient: "linear-gradient(135deg, #1e40af, #3b82f6)" },
        ].map(k => (
          <div key={k.label} className="relative overflow-hidden rounded-[14px] p-[12px_14px] text-white flex flex-col justify-between h-[80px]"
               style={{ background: k.gradient }}>
            <div className="text-[9px] font-semibold uppercase tracking-[0.10em] opacity-85">{k.label}</div>
            <div className="text-[18px] font-semibold tracking-tight privacy-mask" style={{ fontFamily: mono }}>{k.value}</div>
            {k.sub && <div className="text-[10px] font-medium opacity-80">{k.sub}</div>}
          </div>
        ))}
      </div>

      {/* ═══ Exit Alerts ═══ */}
      <div className="rounded-[14px] overflow-hidden mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="px-5 py-3 text-[14px] font-bold" style={{ borderBottom: "1px solid var(--border)" }}>EXIT ALERTS</div>
        <div className="px-5 py-1 text-[12px] mb-2" style={{ color: "var(--ink-4)" }}>Non-negotiable action rules — when a violation fires, act immediately.</div>

        {(!data.active_exits || data.active_exits.length === 0) ? (
          <>
            <div className="mx-5 mb-3 p-4 rounded-[10px]" style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", borderLeft: "5px solid #08a86b" }}>
              <div className="text-[16px] font-bold" style={{ color: "#08a86b" }}>No Active Violations</div>
              <div className="text-[13px] mt-1" style={{ color: "#555" }}>Market structure intact — all exit signals clear.</div>
            </div>
            <div className="grid grid-cols-3 gap-3 mx-5 mb-4">
              {[
                { label: "21 EMA", ok: (data.consecutive_below_21 || 0) === 0, detail: (data.consecutive_below_21 || 0) === 0 ? "Holding" : `${data.consecutive_below_21} close(s) below` },
                { label: "50 SMA", ok: price > (data.sma50 || 0), detail: price > (data.sma50 || 0) ? "Above" : "Below" },
                { label: "200 SMA", ok: price > (data.sma200 || 0), detail: price > (data.sma200 || 0) ? "Above" : "Below" },
              ].map(m => (
                <div key={m.label} className="px-4 py-2.5 rounded-[10px] text-[13px] font-medium"
                     style={{ background: m.ok ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #e5484d 10%, var(--surface))", color: m.ok ? "#16a34a" : "#dc2626", border: `1px solid ${m.ok ? "color-mix(in oklab, #08a86b 30%, var(--border))" : "color-mix(in oklab, #e5484d 30%, var(--border))"}` }}>
                  {m.label} — {m.detail}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="mx-5 mb-4 flex flex-col gap-3">
            {data.active_exits.map((alert: any, i: number) => {
              const sevStyle = alert.severity === "CRITICAL" ? { bg: "color-mix(in oklab, #e5484d 10%, var(--surface))", border: "#dc2626" }
                : alert.severity === "SERIOUS" ? { bg: "color-mix(in oklab, #e5484d 10%, var(--surface))", border: "#e5484d" }
                : { bg: "color-mix(in oklab, #f59f00 10%, var(--surface))", border: "#f59f00" };
              return (
                <div key={i} className="p-4 rounded-[10px]" style={{ background: sevStyle.bg, borderLeft: `5px solid ${sevStyle.border}` }}>
                  <div className="text-[18px] font-bold">{alert.signal}</div>
                  <div className="text-[13px] mt-1">{alert.detail}</div>
                  <div className="text-[16px] font-bold mt-2">TARGET EXPOSURE: {alert.target}</div>
                  <div className="text-[11px] mt-1 opacity-70">Take profits on extended positions. Tighten all stops immediately.</div>
                </div>
              );
            })}
          </div>
        )}

        {/* Exit Ladder Rules */}
        <details className="mx-5 mb-4">
          <summary className="text-[12px] font-medium cursor-pointer px-3 py-2 rounded-[8px]" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
            Exit Ladder Rules
          </summary>
          <div className="mt-2 overflow-hidden rounded-[8px]" style={{ border: "1px solid var(--border)" }}>
            <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Signal", "Condition", "Target", "Severity"].map(h => (
                    <th key={h} className="text-left px-3 py-2 font-semibold" style={{ background: "var(--surface-2)", borderBottom: "1px solid var(--border)", color: "var(--ink-4)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ["21 EMA Violation", "Close below 21 EMA + next day undercuts >0.2%", "50%", "WARNING"],
                  ["21 EMA Confirmed Break", "Two consecutive closes below 21 EMA", "30%", "SERIOUS"],
                  ["50 SMA Violation", "Close below 50 SMA + next day undercuts >0.2%", "0%", "CRITICAL"],
                ].map(([s, c, t, sv], i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td className="px-3 py-2 font-semibold">{s}</td>
                    <td className="px-3 py-2">{c}</td>
                    <td className="px-3 py-2 font-bold">{t}</td>
                    <td className="px-3 py-2">{sv}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>

      {/* ═══ Two columns: Entry Ladder + MA Stack ═══ */}
      <div className="grid grid-cols-2 gap-5 mb-6" style={{ alignItems: "start" }}>

        {/* Entry Ladder */}
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="text-[13px] font-semibold">Entry Ladder</span>
            <span className="text-[12px] font-semibold" style={{ color: navColor }}>Step {entryStep} · {entryExp}%</span>
          </div>
          {/* Progress bar */}
          <div className="px-5 pt-3">
            <div className="h-2.5 rounded-full overflow-hidden" style={{ background: "var(--bg)" }}>
              <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(entryExp / 200, 1) * 100}%`, background: sc.bg }} />
            </div>
          </div>
          <div className="p-4 flex flex-col gap-1.5">
            {ladder.map((item: any) => {
              const achieved = item.achieved;
              const isNext = !achieved && (item.step === 0 || (ladder[item.step - 1]?.achieved));
              return (
                <div key={item.step} className="flex items-center justify-between px-3 py-2.5 rounded-[8px]"
                     style={{
                       background: achieved ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : isNext ? "color-mix(in oklab, #f59f00 10%, var(--surface))" : "var(--bg)",
                       border: `1px solid ${achieved ? "color-mix(in oklab, #08a86b 30%, var(--border))" : isNext ? "color-mix(in oklab, #f59f00 30%, var(--border))" : "var(--border)"}`,
                       color: !achieved && !isNext ? "var(--ink-4)" : "var(--ink)",
                     }}>
                  <div className="flex items-center gap-2">
                    <span className="text-[13px]">{achieved ? "✅" : isNext ? "⏳" : "🔒"}</span>
                    <span className="text-[12px] font-semibold">Step {item.step}:</span>
                    <span className="text-[12px]">{item.label}</span>
                  </div>
                  <span className="text-[13px] font-bold" style={{ fontFamily: mono }}>{item.exposure}%</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* MA Stack + Streaks */}
        <div className="flex flex-col gap-4">
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="px-5 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Moving Average Stack</div>
            <div className="p-4 flex flex-col gap-2">
              {[
                { label: "8 EMA", value: data.ema8, dist: price > 0 && data.ema8 > 0 ? ((price - data.ema8) / data.ema8 * 100) : 0 },
                { label: "21 EMA", value: data.ema21, dist: price > 0 && data.ema21 > 0 ? ((price - data.ema21) / data.ema21 * 100) : 0 },
                { label: "50 SMA", value: data.sma50, dist: price > 0 && data.sma50 > 0 ? ((price - data.sma50) / data.sma50 * 100) : 0 },
                { label: "200 SMA", value: data.sma200, dist: price > 0 && data.sma200 > 0 ? ((price - data.sma200) / data.sma200 * 100) : 0 },
              ].map(ma => (
                <div key={ma.label} className="flex items-center justify-between py-2" style={{ borderBottom: "1px dashed var(--border)" }}>
                  <span className="text-[13px] font-medium">{ma.label}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-[13px] privacy-mask" style={{ fontFamily: mono }}>${(ma.value || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    <span className="text-[12px] font-semibold" style={{ fontFamily: mono, color: ma.dist >= 0 ? "#08a86b" : "#e5484d" }}>
                      {ma.dist >= 0 ? "+" : ""}{ma.dist.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
              {/* Stack order */}
              <div className="flex gap-2 mt-2 flex-wrap">
                {[
                  { label: "8E > 21E", ok: data.stack_8_21 },
                  { label: "21E > 50S", ok: data.stack_21_50 },
                  { label: "50S > 200S", ok: data.stack_50_200 },
                ].map(s => (
                  <span key={s.label} className="text-[10px] px-2 py-0.5 rounded font-semibold"
                        style={{ background: s.ok ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : "color-mix(in oklab, #e5484d 12%, var(--surface))", color: s.ok ? "#16a34a" : "#dc2626" }}>
                    {s.label} {s.ok ? "✅" : "❌"}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Streaks */}
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="px-5 py-3 text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>Streak Monitor</div>
            <div className="p-4 grid grid-cols-2 gap-3">
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Low {">"} 21 EMA</div>
                <div className="text-[22px] font-semibold mt-0.5" style={{ fontFamily: mono, color: (data.low_above_21_streak || 0) >= 3 ? "#08a86b" : "#f59f00" }}>
                  {data.low_above_21_streak || 0} days
                </div>
                <div className="text-[10px]" style={{ color: (data.low_above_21_streak || 0) >= 3 ? "#08a86b" : "var(--ink-4)" }}>
                  {(data.low_above_21_streak || 0) >= 3 ? "UPTREND ✓" : `${3 - (data.low_above_21_streak || 0)} more needed`}
                </div>
              </div>
              <div className="p-3 rounded-[10px]" style={{ border: "1px solid var(--border)" }}>
                <div className="text-[10px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>Low {">"} 50 SMA</div>
                <div className="text-[22px] font-semibold mt-0.5" style={{ fontFamily: mono, color: (data.low_above_50_streak || 0) >= 3 ? "#08a86b" : "#f59f00" }}>
                  {data.low_above_50_streak || 0} days
                </div>
                <div className="text-[10px]" style={{ color: (data.low_above_50_streak || 0) >= 3 ? "#08a86b" : "var(--ink-4)" }}>
                  {(data.low_above_50_streak || 0) >= 3 ? "Strong ✓" : `${3 - (data.low_above_50_streak || 0)} more needed`}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ Recent Signal Log ═══ */}
      <SignalLog mono={mono} />

      {/* ═══ Methodology ═══ */}
      <details className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <summary className="px-5 py-3 cursor-pointer text-[13px] font-semibold">Cycle Tracker Methodology</summary>
        <div className="p-5 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-[13px] font-semibold mb-2" style={{ color: "var(--ink)" }}>Market States</h4>
              <table className="w-full text-[11px]" style={{ borderCollapse: "collapse" }}>
                <tbody>
                  {[
                    ["CORRECTION", "NASDAQ down 7%+ — Day 0, waiting for rally"],
                    ["RALLY MODE", "Steps 0-3: rally day → low above 21 EMA"],
                    ["UPTREND", "Steps 4-6: holds 21 EMA 3 days → MA crossovers"],
                    ["POWERTREND", "Step 7: 8E > 21E > 50S > 200S (200%)"],
                  ].map(([s, d]) => (
                    <tr key={s} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td className="py-1.5 font-semibold" style={{ color: "var(--ink)" }}>{s}</td>
                      <td className="py-1.5 pl-3">{d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <h4 className="text-[13px] font-semibold mb-2" style={{ color: "var(--ink)" }}>Key Rules</h4>
              <div className="flex flex-col gap-1.5">
                <div>Entry ladder = suggestions — actual exposure follows trade performance</div>
                <div>Exit ladder = rules — must act when triggered</div>
                <div>All signals are NASDAQ only</div>
                <div>This page sits on top of M Factor, not a replacement</div>
              </div>
            </div>
          </div>
        </div>
      </details>
    </div>
  );
}

type Signal = {
  trade_date: string;
  signal_type: string;
  signal_label: string;
  exposure_before: number | null;
  exposure_after: number | null;
};

function SignalLog({ mono }: { mono: string }) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    api.marketSignals(30)
      .then((r) => setSignals(r.signals || []))
      .catch(() => setSignals([]))
      .finally(() => setLoading(false));
  }, []);

  const types = useMemo(
    () => Array.from(new Set(signals.map((s) => s.signal_type))).sort(),
    [signals]
  );

  const filtered = filter ? signals.filter((s) => s.signal_type === filter) : signals;

  return (
    <div className="rounded-[14px] overflow-hidden mb-6"
         style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
      <div className="px-5 py-3 flex items-center justify-between gap-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <div className="text-[13px] font-semibold">Recent Signals</div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>Last 30 days from M Factor engine</div>
        </div>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter signal log by signal type"
          className="text-[11px] px-2.5 py-1.5 rounded-[8px]"
          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink-2)" }}
        >
          <option value="">All types ({signals.length})</option>
          {types.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>

      {loading ? (
        <div className="p-6 text-center text-[12px]" style={{ color: "var(--ink-4)" }}>Loading…</div>
      ) : filtered.length === 0 ? (
        <div className="p-8 text-center text-[12px]" style={{ color: "var(--ink-4)" }}>
          {filter ? "No signals match this filter." : "No signals fired in the last 30 days."}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["Date", "Signal", "Description", "Exposure"].map((h) => (
                  <th key={h} className="text-left px-4 py-2 text-[10px] uppercase tracking-[0.06em] font-semibold"
                      style={{ background: "var(--surface-2)", borderBottom: "1px solid var(--border)", color: "var(--ink-4)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => (
                <tr key={`${s.trade_date}-${s.signal_type}-${i}`} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td className="px-4 py-2" style={{ fontFamily: mono }}>{s.trade_date}</td>
                  <td className="px-4 py-2 font-semibold">{s.signal_type}</td>
                  <td className="px-4 py-2" style={{ color: "var(--ink-2)" }}>{s.signal_label}</td>
                  <td className="px-4 py-2" style={{ fontFamily: mono }}>
                    {s.exposure_before != null && s.exposure_after != null
                      ? `${s.exposure_before}% → ${s.exposure_after}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
