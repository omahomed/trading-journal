"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";

interface MFData {
  price: number; ema21: number; sma50: number; sma200: number;
  above_21ema: boolean; above_50sma: boolean; above_200sma: boolean;
  d21: number; d50: number; d200: number;
  state: string; powertrend_streak: number;
}

const STATE_CONFIG: Record<string, { bg: string; exp: string; desc: string }> = {
  POWERTREND: { bg: "#8A2BE2", exp: "200% (Margin Enabled)", desc: "Either index in super cycle — aggressive positioning" },
  OPEN: { bg: "#08a86b", exp: "100% (Full Exposure)", desc: "Both indices above 21 EMA — healthy market, normal buying" },
  NEUTRAL: { bg: "#f59f00", exp: "50% Max (Caution)", desc: "1+ index violated 21 EMA — hold winners, avoid new buys" },
  CLOSED: { bg: "#e5484d", exp: "0% (Defensive)", desc: "Both indices violated 50 SMA — protect capital" },
};

function Arrow({ val }: { val: number }) {
  return <span style={{ color: val >= 0 ? "#08a86b" : "#e5484d" }}>{val >= 0 ? "▲" : "▼"}</span>;
}

function MARow({ label, sub, value, dist, above }: { label: string; sub: string; value: number; dist: number; above: boolean }) {
  return (
    <div className="flex items-center justify-between py-3" style={{ borderBottom: "1px dashed var(--border)" }}>
      <div>
        <span className="text-[14px] font-semibold">{label}</span>
        <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>({value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })})</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="px-2 py-0.5 rounded text-[10px] font-bold"
              style={{ background: above ? "color-mix(in oklab, #08a86b 12%, var(--surface))" : "color-mix(in oklab, #e5484d 12%, var(--surface))", color: above ? "#16a34a" : "#dc2626" }}>
          {above ? "ABOVE" : "BELOW"}
        </span>
        <span className="text-[13px] font-bold" style={{ fontFamily: "var(--font-jetbrains), monospace", color: dist >= 0 ? "#08a86b" : "#e5484d" }}>
          <Arrow val={dist} /> {Math.abs(dist).toFixed(2)}%
        </span>
      </div>
    </div>
  );
}

export function MFactor({ navColor }: { navColor: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadData = () => {
    setLoading(true);
    api.mfactor().then(d => { setData(d); setLoading(false); }).catch(() => setLoading(false));
  };

  useEffect(() => { loadData(); }, []);

  if (loading) return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;

  const nasdaq: MFData | null = data?.nasdaq || null;
  const spy: MFData | null = data?.spy || null;
  const combined = data?.combined_state || "CLOSED";
  const cfg = STATE_CONFIG[combined] || STATE_CONFIG.CLOSED;

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px] flex items-center justify-between" style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            M <em className="italic" style={{ color: navColor }}>Factor</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Market Health Assessment</div>
        </div>
        <button onClick={loadData} className="flex items-center gap-1.5 h-[32px] px-3.5 rounded-[10px] text-xs font-medium transition-colors"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
          </svg>
          Refresh
        </button>
      </div>

      {data?.error && (
        <div className="mb-5 p-3 rounded-[10px] text-[12px]" style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", color: "#d97706", border: "1px solid color-mix(in oklab, #f59f00 30%, var(--border))" }}>
          {data.error}
        </div>
      )}

      {/* ═══ Market Window Banner ═══ */}
      <div className="rounded-[14px] p-6 text-center text-white mb-6" style={{ background: cfg.bg, boxShadow: "0 4px 12px rgba(0,0,0,0.15)" }}>
        <div className="text-[12px] uppercase tracking-[0.15em] opacity-80 mb-1">Market Window</div>
        <div className="text-[48px] font-extrabold tracking-tight" style={{ lineHeight: 1.1 }}>{combined}</div>
        <div className="text-[15px] font-medium mt-2 opacity-90">Recommended Exposure: {cfg.exp}</div>
        <div className="text-[12px] mt-1 opacity-70">{cfg.desc}</div>
      </div>

      {/* ═══ Index Cards (2 columns) ═══ */}
      <div className="grid grid-cols-2 gap-5 mb-6">
        {[
          { label: "NASDAQ", d: nasdaq, color: "#8b5cf6" },
          { label: "S&P 500 (SPY)", d: spy, color: "#0d6efd" },
        ].map(({ label, d, color }) => {
          if (!d) return null;
          const stCfg = STATE_CONFIG[d.state] || STATE_CONFIG.CLOSED;
          return (
            <div key={label} className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
              {/* Header: title + price */}
              <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "2px solid var(--bg)" }}>
                <span className="text-[18px] font-bold">{label}</span>
                <span className="text-[18px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${d.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
              {/* State badge */}
              <div className="mx-5 mt-3 mb-2 py-2 rounded-[8px] text-center text-white font-bold text-[13px]" style={{ background: stCfg.bg }}>
                {d.state}
              </div>
              {/* MA rows */}
              <div className="px-5 pb-4">
                <MARow label="Short (21e)" sub="EMA" value={d.ema21} dist={d.d21} above={d.above_21ema} />
                <MARow label="Med (50s)" sub="SMA" value={d.sma50} dist={d.d50} above={d.above_50sma} />
                <MARow label="Long (200s)" sub="SMA" value={d.sma200} dist={d.d200} above={d.above_200sma} />
              </div>
            </div>
          );
        })}
      </div>

      {/* ═══ Methodology ═══ */}
      <details className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <summary className="px-5 py-3 cursor-pointer text-[13px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
          M Factor Methodology
        </summary>
        <div className="p-5 text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-[13px] font-semibold mb-2" style={{ color: "var(--ink)" }}>Market Phases</h4>
              <div className="flex flex-col gap-2">
                {[
                  { icon: "🟣", label: "POWERTREND (200%)", desc: "Low > 21 EMA for 3+ days — max aggression" },
                  { icon: "🟢", label: "OPEN (100%)", desc: "Both above 21 EMA — normal buying" },
                  { icon: "🟡", label: "NEUTRAL (50%)", desc: "1+ violated 21 EMA — hold winners only" },
                  { icon: "🔴", label: "CLOSED (0%)", desc: "Both violated 50 SMA — protect capital" },
                ].map(p => (
                  <div key={p.label} className="flex gap-2">
                    <span>{p.icon}</span>
                    <div><strong>{p.label}</strong> — {p.desc}</div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-[13px] font-semibold mb-2" style={{ color: "var(--ink)" }}>Philosophy</h4>
              <div className="flex flex-col gap-2">
                <div><strong>21 EMA is the primary filter</strong> — violation signals immediate caution</div>
                <div><strong>Dual-index confirmation</strong> — requires both indices to agree before downgrading</div>
                <div><strong>Stock-first approach</strong> — don{"'"}t panic sell strong holdings because indices are weak</div>
              </div>
            </div>
          </div>
        </div>
      </details>
    </div>
  );
}
