"use client";

import { useState, useEffect } from "react";
import { api, getActivePortfolio, type TradePosition } from "@/lib/api";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.10em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>{label}</label>
      {children}
    </div>
  );
}

function MetricCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="p-4 rounded-[12px]" style={{ border: "1px solid var(--border)" }}>
      <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[22px] font-semibold mt-1 privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace", color: color || "var(--ink)" }}>{value}</div>
      {sub && <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{sub}</div>}
    </div>
  );
}

const inputCls = "w-full h-[42px] px-3.5 rounded-[10px] text-[13px] outline-none";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

export function EarningsPlanner({ navColor }: { navColor: string }) {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [equity, setEquity] = useState(0);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  // Form
  const [selectedTicker, setSelectedTicker] = useState("");
  const [currPrice, setCurrPrice] = useState("");
  const [nlv, setNlv] = useState("");
  const [sharesHeld, setSharesHeld] = useState("");
  const [riskTolPct, setRiskTolPct] = useState(0.50);
  const [impliedMove, setImpliedMove] = useState("5.00");
  const [stressMult, setStressMult] = useState(2.0);
  const [calculated, setCalculated] = useState(false);

  useEffect(() => {
    Promise.all([
      api.tradesOpen(getActivePortfolio()).catch(() => []),
      api.journalLatest(getActivePortfolio()).catch(() => ({ end_nlv: 100000 })),
    ]).then(async ([open, journal]) => {
      const openArr = open as TradePosition[];
      setOpenTrades(openArr);
      const eq = parseFloat(String((journal as any).end_nlv || 100000));
      setEquity(eq);
      setNlv(String(eq));

      // Fetch live prices
      const tickers = openArr.map(t => t.ticker).filter(Boolean);
      if (tickers.length > 0) {
        try {
          const prices = await api.batchPrices(tickers);
          if (prices && !("error" in prices)) setLivePrices(prices);
        } catch { /* */ }
      }
      setLoading(false);
    });
  }, []);

  // When ticker changes, populate fields
  const selected = openTrades.find(t => t.ticker === selectedTicker);
  const handleTickerChange = (ticker: string) => {
    setSelectedTicker(ticker);
    setCalculated(false);
    const pos = openTrades.find(t => t.ticker === ticker);
    if (pos) {
      setSharesHeld(String(pos.shares || 0));
      const lp = livePrices[ticker] || parseFloat(String(pos.avg_entry || 0));
      setCurrPrice(String(lp.toFixed(2)));
    }
  };

  // Computed
  const price = parseFloat(currPrice) || 0;
  const avgCost = parseFloat(String(selected?.avg_entry || 0));
  const shares = parseFloat(sharesHeld) || 0;
  const nlvVal = parseFloat(nlv) || equity;
  const expMove = parseFloat(impliedMove) || 0;

  const unrealizedPct = avgCost > 0 && price > 0 ? ((price - avgCost) / avgCost) * 100 : 0;
  const unrealizedDlr = (price - avgCost) * shares;

  // Stress test
  const gapDlr = expMove * stressMult;
  const disasterPrice = price - gapDlr;
  const totalDropEquity = gapDlr * shares;
  const principalRiskDlr = disasterPrice < avgCost ? (avgCost - disasterPrice) * shares : 0;
  const pctImpactPrincipal = nlvVal > 0 ? (principalRiskDlr / nlvVal) * 100 : 0;
  const maxAllowedLoss = nlvVal * (riskTolPct / 100);

  // Trim calculation
  const lossPerShare = avgCost - disasterPrice;
  const excessLoss = principalRiskDlr - maxAllowedLoss;
  const sharesToTrim = lossPerShare > 0 && excessLoss > 0 ? Math.ceil(excessLoss / lossPerShare) : 0;
  const safeShares = shares - sharesToTrim;

  // Verdict
  const verdict = principalRiskDlr <= maxAllowedLoss
    ? (principalRiskDlr === 0 ? "safe" : "approved")
    : "exceeded";

  // Cushion check thresholds
  const cushionPass = unrealizedPct >= 10;
  const cushionFail = unrealizedPct <= 0;

  if (loading) {
    return <div className="animate-pulse"><div className="h-[90px] rounded-[14px]" style={{ background: "var(--bg-2)" }} /></div>;
  }

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Earnings <em className="italic" style={{ color: navColor }}>Planner</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>Binary Event Logic · Principal Protection</div>
      </div>

      {/* 1. Select Ticker */}
      <div className="mb-5">
        <Field label="Select Ticker into Earnings">
          <select value={selectedTicker} onChange={e => handleTickerChange(e.target.value)}
                  className={inputCls} style={{ ...inputStyle, appearance: "none" as any }}>
            <option value="">Select position...</option>
            {openTrades.map(t => (
              <option key={t.trade_id} value={t.ticker}>{t.ticker} — {t.shares} shs @ ${parseFloat(String(t.avg_entry || 0)).toFixed(2)}</option>
            ))}
          </select>
        </Field>
      </div>

      {selected && (
        <>
          {/* Section 1: Setup & Cushion Check */}
          <h3 className="text-[14px] font-semibold mb-3">1. Setup & Cushion Check</h3>
          <div className="grid grid-cols-4 gap-4 mb-4">
            <Field label="Current Price ($)">
              <input type="number" value={currPrice} onChange={e => { setCurrPrice(e.target.value); setCalculated(false); }}
                     step="0.01" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Account Equity (NLV)">
              <input type="number" value={nlv} onChange={e => { setNlv(e.target.value); setCalculated(false); }}
                     step="1000" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Shares Held">
              <input type="number" value={sharesHeld} onChange={e => { setSharesHeld(e.target.value); setCalculated(false); }}
                     step="1" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Avg Cost ($)">
              <div className={inputCls + " flex items-center"} style={{ ...inputStyle, background: "var(--bg)" }}>
                ${avgCost.toFixed(2)}
              </div>
            </Field>
          </div>

          {/* Cushion verdict */}
          {price > 0 && (
            <div className="mb-5 px-4 py-3 rounded-[10px] text-[13px] font-medium"
                 style={{
                   background: cushionPass ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : cushionFail ? "color-mix(in oklab, #e5484d 10%, var(--surface))" : "color-mix(in oklab, #f59f00 10%, var(--surface))",
                   color: cushionPass ? "#16a34a" : cushionFail ? "#dc2626" : "#d97706",
                   border: `1px solid ${cushionPass ? "color-mix(in oklab, #08a86b 30%, var(--border))" : cushionFail ? "color-mix(in oklab, #e5484d 30%, var(--border))" : "color-mix(in oklab, #f59f00 30%, var(--border))"}`,
                 }}>
              {cushionPass && `PASS: Cushion is ${unrealizedPct.toFixed(2)}% ($${unrealizedDlr.toLocaleString(undefined, { maximumFractionDigits: 0 })}). You have earned the right to hold.`}
              {!cushionPass && !cushionFail && `THIN ICE: Cushion is only ${unrealizedPct.toFixed(2)}%. Any gap will likely eat principal.`}
              {cushionFail && `FAIL: You are underwater (-$${Math.abs(unrealizedDlr).toLocaleString(undefined, { maximumFractionDigits: 0 })}). Strategy Rule: SELL ALL before earnings.`}
            </div>
          )}

          {/* Section 2: Stress Test Parameters */}
          <h3 className="text-[14px] font-semibold mb-3">2. Stress Test Parameters</h3>
          <div className="grid grid-cols-3 gap-4 mb-5">
            <Field label="Max Capital Risk %">
              <input type="number" value={riskTolPct} onChange={e => { setRiskTolPct(parseFloat(e.target.value) || 0.5); setCalculated(false); }}
                     step="0.05" min="0.1" max="1.0" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Implied Move (+/- $)">
              <input type="number" value={impliedMove} onChange={e => { setImpliedMove(e.target.value); setCalculated(false); }}
                     step="0.50" className={inputCls} style={inputStyle} />
            </Field>
            <Field label="Stress Multiplier">
              <div className="flex gap-2 mt-1">
                {[1.5, 2.0].map(m => (
                  <button key={m} onClick={() => { setStressMult(m); setCalculated(false); }}
                          className="flex-1 h-[38px] rounded-[10px] text-[13px] font-semibold transition-all"
                          style={{
                            background: stressMult === m ? `color-mix(in oklab, ${navColor} 10%, transparent)` : "var(--bg)",
                            color: stressMult === m ? navColor : "var(--ink-4)",
                            border: `1.5px solid ${stressMult === m ? navColor : "var(--border)"}`,
                          }}>
                    {m}x
                  </button>
                ))}
              </div>
            </Field>
          </div>

          {/* Calculate */}
          <button onClick={() => setCalculated(true)}
                  className="h-[44px] px-8 rounded-[10px] text-[13px] font-semibold text-white mb-6 transition-all hover:brightness-110"
                  style={{ background: "#6366f1" }}>
            Run Stress Test
          </button>

          {/* Section 3: The Verdict */}
          {calculated && price > 0 && (
            <div style={{ animation: "slide-up 0.15s ease-out" }}>
              <h3 className="text-[14px] font-semibold mb-3">3. The Verdict</h3>
              <div className="grid grid-cols-4 gap-3 mb-5">
                <MetricCard label="Disaster Price" value={`$${disasterPrice.toFixed(2)}`} sub={`-$${gapDlr.toFixed(2)} Gap`} />
                <MetricCard label="Profit Buffer" value={`$${unrealizedDlr.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="Your Cushion" />
                <MetricCard label="Projected Drawdown" value={`-$${totalDropEquity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="Equity Drop" color="#e5484d" />
                <MetricCard label="Risk to Principal" value={`$${principalRiskDlr.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                            sub={`${pctImpactPrincipal.toFixed(2)}% of NLV`}
                            color={principalRiskDlr === 0 ? "#08a86b" : "#e5484d"} />
              </div>

              {/* Verdict */}
              {verdict === "safe" && (
                <div className="px-4 py-3 rounded-[10px] text-[13px] font-medium mb-4"
                     style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
                  <strong>SAFE (HOUSE MONEY):</strong> Even with a ${gapDlr.toFixed(2)} gap, price (${disasterPrice.toFixed(2)}) stays above your cost (${avgCost.toFixed(2)}). No principal at risk.
                </div>
              )}
              {verdict === "approved" && (
                <div className="px-4 py-3 rounded-[10px] text-[13px] font-medium mb-4"
                     style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#16a34a", border: "1px solid color-mix(in oklab, #08a86b 30%, var(--border))" }}>
                  <strong>APPROVED:</strong> Principal risk is ${principalRiskDlr.toLocaleString(undefined, { maximumFractionDigits: 0 })} ({pctImpactPrincipal.toFixed(2)}%), which is within your {riskTolPct}% budget.
                </div>
              )}
              {verdict === "exceeded" && (
                <div className="px-4 py-3 rounded-[10px] text-[13px] font-medium mb-4"
                     style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#dc2626", border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))" }}>
                  <strong>RISK EXCEEDED:</strong> You risk losing {pctImpactPrincipal.toFixed(2)}% of your starting capital.
                </div>
              )}

              {/* Trim recommendation */}
              {verdict === "exceeded" && sharesToTrim > 0 && (
                <div className="grid grid-cols-2 gap-3">
                  <MetricCard label="Required Trim" value={`-${sharesToTrim} Shares`} sub="Sell Before Close" color="#dc2626" />
                  <MetricCard label="Max Safe Hold" value={`${safeShares} Shares`} sub={`Protects ${riskTolPct}% Principal`} color="#08a86b" />
                </div>
              )}
            </div>
          )}
        </>
      )}

      {openTrades.length === 0 && (
        <div className="text-center py-16 text-sm" style={{ color: "var(--ink-4)" }}>No open positions found to analyze.</div>
      )}
    </div>
  );
}
