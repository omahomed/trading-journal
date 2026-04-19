"use client";

import { useState, useEffect } from "react";
import { api, type TradePosition } from "@/lib/api";

const SELL_RULES = [
  "sr1 Capital Protection", "sr2 Trailing Stop", "sr3 Portfolio Management",
  "sr4 Time Stop", "sr5 Climax Top", "sr6 Exhaustion Gap",
  "sr7 200d Moving Avg Break", "sr8 Living Below 50d", "sr9 Failed Breakout",
  "sr10 Scale-Out T1 (-3%)", "sr11 Scale-Out T2 (-5%)", "sr12 Scale-Out T3 (-8%)",
  "sr13 Earnings Exit", "sr14 Market Correction Exit",
];

function FormField({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <div>
      <label className="block text-[11px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
        {label}
      </label>
      {children}
      {hint && <div className="text-[11px] mt-1" style={{ color: "var(--ink-4)" }}>{hint}</div>}
    </div>
  );
}

const inputStyle = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

export function LogSell({ navColor }: { navColor: string }) {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [selectedTrade, setSelectedTrade] = useState("");
  const [shares, setShares] = useState("");
  const [price, setPrice] = useState("");
  const [rule, setRule] = useState(SELL_RULES[0]);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.tradesOpen("CanSlim").then(trades => {
      setOpenTrades(trades);
      setLoading(false);

      // Prefill from Position Sizer Trim (via localStorage)
      try {
        const raw = localStorage.getItem("ps_prefill_sell");
        if (!raw) return;
        const data = JSON.parse(raw);
        localStorage.removeItem("ps_prefill_sell");
        if (data.trade_id) setSelectedTrade(data.trade_id);
        if (data.shares) setShares(String(data.shares));
        if (data.price) setPrice(String(data.price));
      } catch { /* ignore */ }
    }).catch(() => setLoading(false));
  }, []);

  const selected = openTrades.find(t => t.trade_id === selectedTrade);
  const sharesNum = parseFloat(shares) || 0;
  const priceNum = parseFloat(price) || 0;
  const proceeds = sharesNum * priceNum;
  const avgEntry = selected?.avg_entry || 0;
  const returnPct = avgEntry > 0 && priceNum > 0 ? ((priceNum - avgEntry) / avgEntry) * 100 : 0;
  const realizedPl = avgEntry > 0 ? (priceNum - avgEntry) * sharesNum : 0;

  const handleSubmit = () => {
    if (!selectedTrade) return alert("Select a campaign");
    if (sharesNum <= 0) return alert("Shares must be > 0");
    if (priceNum <= 0) return alert("Price must be > 0");
    if (selected && sharesNum > selected.shares) return alert(`Max shares: ${selected.shares}`);
    // TODO: POST /api/trades/sell endpoint needed
    alert("Backend write endpoint not yet available. Will save via POST /api/trades/sell");
  };

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Log <em className="italic" style={{ color: navColor }}>Sell</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Record a sell order against an open campaign
        </div>
      </div>

      <div className="grid gap-6" style={{ gridTemplateColumns: "2fr 1fr" }}>
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Sell Order</span>
          </div>
          <div className="p-5 flex flex-col gap-5">
            <FormField label="Select Campaign" hint={selected ? `${selected.shares} shares @ $${selected.avg_entry?.toFixed(2)} avg` : undefined}>
              <select value={selectedTrade} onChange={e => setSelectedTrade(e.target.value)}
                      className="w-full h-[38px] px-3 rounded-[10px] text-[13px] appearance-none" style={inputStyle}>
                <option value="">Choose an open campaign...</option>
                {openTrades.map(t => (
                  <option key={t.trade_id} value={t.trade_id}>
                    {t.trade_id} — {t.ticker} ({t.shares} shares)
                  </option>
                ))}
              </select>
            </FormField>

            <FormField label="Sell Rule">
              <select value={rule} onChange={e => setRule(e.target.value)}
                      className="w-full h-[38px] px-3 rounded-[10px] text-[13px] appearance-none" style={inputStyle}>
                {SELL_RULES.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Shares to Sell" hint={selected ? `Max: ${selected.shares}` : undefined}>
                <input type="number" value={shares} onChange={e => setShares(e.target.value)}
                       placeholder="0" className="w-full h-[38px] px-3 rounded-[10px] text-[13px]" style={inputStyle} />
              </FormField>
              <FormField label="Sell Price ($)">
                <input type="number" value={price} onChange={e => setPrice(e.target.value)} step="0.01"
                       placeholder="0.00" className="w-full h-[38px] px-3 rounded-[10px] text-[13px]" style={inputStyle} />
              </FormField>
            </div>

            <FormField label="Sell Context / Notes">
              <input type="text" value={notes} onChange={e => setNotes(e.target.value)}
                     placeholder="Why did you sell?"
                     className="w-full h-[38px] px-3 rounded-[10px] text-[13px]"
                     style={{ ...inputStyle, fontFamily: "inherit" }} />
            </FormField>

            {/* Position Changes (Optional) */}
            <div>
              <div className="flex items-center gap-2 mb-2 mt-1">
                <span className="text-[14px]">📸</span>
                <span className="text-[13px] font-semibold">Position Changes (Optional)</span>
              </div>
              <div className="text-[11px] mb-2 flex items-center gap-1" style={{ color: "var(--ink-3)" }}>
                <span>🔄</span> Upload charts (Partial Sells / Full Exits)
              </div>
              <label className="flex items-center gap-2.5 h-[48px] px-4 rounded-[10px] cursor-pointer transition-colors hover:brightness-95"
                     style={{ border: "1.5px dashed var(--border)", background: "var(--bg)" }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>Upload</span>
                <span className="text-[10px]" style={{ color: "var(--ink-4)", opacity: 0.6 }}>200MB per file · PNG, JPG, JPEG</span>
                <input type="file" accept="image/png,image/jpeg" multiple className="hidden" />
              </label>
            </div>

            <button onClick={handleSubmit}
                    className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110"
                    style={{ background: "#6366f1" }}>
              LOG SELL ORDER
            </button>
          </div>
        </div>

        {/* Side panel */}
        <div className="flex flex-col gap-4">
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Sell Preview</span>
            </div>
            <div className="p-4 flex flex-col gap-3">
              {[
                { k: "Proceeds", v: `$${proceeds.toLocaleString(undefined, { maximumFractionDigits: 2 })}` },
                { k: "Avg Entry", v: avgEntry > 0 ? `$${avgEntry.toFixed(2)}` : "—" },
                { k: "Return", v: returnPct !== 0 ? `${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(1)}%` : "—", color: returnPct >= 0 ? "#08a86b" : "#e5484d" },
                { k: "Realized P&L", v: realizedPl !== 0 ? `$${realizedPl >= 0 ? "+" : ""}${realizedPl.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—", color: realizedPl >= 0 ? "#08a86b" : "#e5484d" },
                { k: "Remaining", v: selected ? `${selected.shares - sharesNum} shares` : "—" },
              ].map(s => (
                <div key={s.k} className="flex items-center justify-between">
                  <span className="text-[11px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>{s.k}</span>
                  <span className="text-[14px] font-semibold privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: s.color || "var(--ink)" }}>
                    {s.v}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Open positions quick reference */}
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="text-[13px] font-semibold">Open Campaigns</span>
              <span className="text-xs" style={{ color: "var(--ink-4)" }}>{openTrades.length}</span>
            </div>
            <div className="max-h-[300px] overflow-y-auto">
              {openTrades.map(t => {
                const isSelected = selectedTrade === t.trade_id;
                return (
                  <button key={t.trade_id} onClick={() => setSelectedTrade(t.trade_id)}
                          className="w-full flex items-center justify-between px-4 py-2.5 text-left transition-all"
                          style={{
                            borderBottom: "1px solid var(--border)",
                            background: isSelected ? "var(--surface-2)" : "transparent",
                            borderLeft: isSelected ? `3px solid ${navColor}` : "3px solid transparent",
                          }}
                          onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = "var(--bg)"; }}
                          onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = "transparent"; }}>
                    <div>
                      <span className="text-[12px] font-semibold" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.ticker}</span>
                      <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>{t.shares} sh</span>
                    </div>
                    <span className="text-[11px]" style={{ color: isSelected ? navColor : "var(--ink-4)", fontWeight: isSelected ? 600 : 400 }}>{t.trade_id}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
