"use client";

import { useEffect, useMemo, useState } from "react";
import { api, getActivePortfolio, type TradePosition } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

/**
 * Mobile-native Earnings Planner. Same math as the desktop version
 * (binary event stress test → principal-at-risk verdict), presented as
 * a vertical form suited for a phone. Uses --m-* tokens throughout.
 *
 * Flow:
 *   1. Pick ticker (open positions only)
 *   2. Enter/adjust: current price, NLV, shares, implied move, stress mult
 *   3. Verdict card: SAFE / APPROVED / EXCEEDED with disaster price,
 *      cushion, projected drawdown, principal risk
 */
export function MobileEarningsPlanner() {
  const [openTrades, setOpenTrades] = useState<TradePosition[]>([]);
  const [equity, setEquity] = useState(0);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);

  const [selectedTicker, setSelectedTicker] = useState("");
  const [currPrice, setCurrPrice] = useState("");
  const [nlv, setNlv] = useState("");
  const [sharesHeld, setSharesHeld] = useState("");
  const [riskTolPct, setRiskTolPct] = useState(0.50);
  const [impliedMove, setImpliedMove] = useState("5.00");
  const [stressMult, setStressMult] = useState<1.5 | 2.0>(2.0);

  useEffect(() => {
    Promise.all([
      api.tradesOpen(getActivePortfolio()).catch((err) => {
        log.error("mobile-earnings-planner", "tradesOpen failed", err);
        return [];
      }),
      api.journalLatest(getActivePortfolio()).catch((err) => {
        log.error("mobile-earnings-planner", "journalLatest failed", err);
        return null;
      }),
    ]).then(async ([open, journal]) => {
      const openArr = open as TradePosition[];
      setOpenTrades(openArr);
      const raw = journal && "end_nlv" in (journal as object)
        ? (journal as { end_nlv?: number }).end_nlv
        : null;
      const eq = raw != null ? Number(raw) : 0;
      setEquity(eq);
      setNlv(eq > 0 ? String(eq) : "");

      const tickers = openArr.map(t => t.ticker).filter(Boolean);
      if (tickers.length > 0) {
        try {
          const prices = await api.batchPrices(tickers);
          if (prices && !("error" in prices)) setLivePrices(prices);
        } catch { /* fall back to entry */ }
      }
      setLoading(false);
    });
  }, []);

  const selected = openTrades.find(t => t.ticker === selectedTicker);
  const handleTickerChange = (ticker: string) => {
    setSelectedTicker(ticker);
    const pos = openTrades.find(t => t.ticker === ticker);
    if (pos) {
      setSharesHeld(String(pos.shares || 0));
      const lp = livePrices[ticker] || parseFloat(String(pos.avg_entry || 0));
      setCurrPrice(lp.toFixed(2));
    }
  };

  // Math — identical to desktop earnings-planner.tsx to keep verdicts
  // aligned across surfaces.
  const price = parseFloat(currPrice) || 0;
  const avgCost = parseFloat(String(selected?.avg_entry || 0));
  const shares = parseFloat(sharesHeld) || 0;
  const nlvVal = parseFloat(nlv) || equity;
  const expMove = parseFloat(impliedMove) || 0;

  const unrealizedPct = avgCost > 0 && price > 0 ? ((price - avgCost) / avgCost) * 100 : 0;
  const unrealizedDlr = (price - avgCost) * shares;

  const gapDlr = expMove * stressMult;
  const disasterPrice = price - gapDlr;
  const totalDropEquity = gapDlr * shares;
  const principalRiskDlr = disasterPrice < avgCost ? (avgCost - disasterPrice) * shares : 0;
  const pctImpactPrincipal = nlvVal > 0 ? (principalRiskDlr / nlvVal) * 100 : 0;
  const maxAllowedLoss = nlvVal * (riskTolPct / 100);

  const verdict = useMemo<"safe" | "approved" | "exceeded" | "idle">(() => {
    if (!selected || price <= 0) return "idle";
    if (principalRiskDlr <= maxAllowedLoss) {
      return principalRiskDlr === 0 ? "safe" : "approved";
    }
    return "exceeded";
  }, [selected, price, principalRiskDlr, maxAllowedLoss]);

  const cushionPass = unrealizedPct >= 10;
  const cushionFail = unrealizedPct <= 0;

  if (loading) {
    return (
      <div className="pb-4 flex flex-col gap-3">
        {[0, 1, 2].map(i => (
          <div key={i} className="rounded-m-md animate-pulse h-[100px]"
               style={{ background: "var(--m-surface)" }} />
        ))}
      </div>
    );
  }

  return (
    <div className="pb-4 flex flex-col gap-3" data-testid="mobile-earnings-planner-root">
      {/* Ticker picker */}
      <div className="rounded-m-md p-4"
           style={{
             background: "var(--m-surface)",
             border: "0.5px solid var(--m-border)",
           }}>
        <Label>Ticker (open positions)</Label>
        <select
          value={selectedTicker}
          onChange={(e) => handleTickerChange(e.target.value)}
          className="w-full rounded-m-sm px-3 py-2 text-[14px]"
          style={{
            background: "var(--m-surface-2)",
            color: "var(--m-text)",
            border: "0.5px solid var(--m-border)",
            minHeight: 44,
            fontFamily: "var(--font-jetbrains), monospace",
          }}>
          <option value="">Select position…</option>
          {openTrades.map(t => (
            <option key={t.ticker} value={t.ticker}>
              {t.ticker} — {t.shares} sh @ {formatCurrency(parseFloat(String(t.avg_entry || 0)))}
            </option>
          ))}
        </select>
        {selected && (
          <div className="mt-2 text-[11px] text-m-text-dim">
            Avg cost {formatCurrency(avgCost)}
            {price > 0 && (
              <span className={unrealizedPct >= 0 ? "text-m-accent" : "text-m-down"}>
                {" · "}unrealized {unrealizedPct.toFixed(2)}%
              </span>
            )}
          </div>
        )}
      </div>

      {selected && (
        <>
          {/* Setup inputs */}
          <div className="rounded-m-md p-4 flex flex-col gap-3"
               style={{
                 background: "var(--m-surface)",
                 border: "0.5px solid var(--m-border)",
               }}>
            <div className="text-[11px] font-semibold uppercase tracking-[0.06em] text-m-text-dim">
              Setup
            </div>

            <MobileInput label="Current price ($)" value={currPrice}
                         step="0.01"
                         onChange={setCurrPrice} />
            <MobileInput label="NLV ($)" value={nlv} step="1000"
                         onChange={setNlv} />
            <MobileInput label="Shares held" value={sharesHeld} step="1"
                         onChange={setSharesHeld} />

            {/* Cushion verdict */}
            {price > 0 && (
              <div className="px-3 py-2 rounded-m-sm text-[12px] font-medium"
                   style={{
                     background: cushionPass
                       ? "color-mix(in oklab, var(--m-accent) 12%, var(--m-surface))"
                       : cushionFail
                         ? "color-mix(in oklab, var(--m-down) 12%, var(--m-surface))"
                         : "color-mix(in oklab, var(--m-warn) 12%, var(--m-surface))",
                     color: cushionPass ? "var(--m-accent)" : cushionFail ? "var(--m-down)" : "var(--m-warn)",
                     border: `1px solid ${cushionPass
                       ? "var(--m-accent-border-soft)"
                       : "var(--m-warn-border-soft)"}`,
                   }}>
                {cushionPass && `PASS — cushion ${unrealizedPct.toFixed(2)}% (${formatCurrency(unrealizedDlr, { decimals: 0 })}). Earned the right to hold.`}
                {!cushionPass && !cushionFail &&
                  `THIN ICE — cushion only ${unrealizedPct.toFixed(2)}%. Any gap eats principal.`}
                {cushionFail && `FAIL — underwater (${formatCurrency(unrealizedDlr, { decimals: 0 })}). Rule: SELL ALL before earnings.`}
              </div>
            )}
          </div>

          {/* Stress test parameters */}
          <div className="rounded-m-md p-4 flex flex-col gap-3"
               style={{
                 background: "var(--m-surface)",
                 border: "0.5px solid var(--m-border)",
               }}>
            <div className="text-[11px] font-semibold uppercase tracking-[0.06em] text-m-text-dim">
              Stress test
            </div>

            <MobileInput label="Max capital risk %" value={String(riskTolPct)}
                         step="0.05"
                         onChange={(v) => setRiskTolPct(parseFloat(v) || 0.5)} />
            <MobileInput label="Implied move (± $)" value={impliedMove}
                         step="0.50"
                         onChange={setImpliedMove} />

            <div>
              <Label>Stress multiplier</Label>
              <div className="flex gap-2">
                {[1.5, 2.0].map(m => {
                  const active = stressMult === m;
                  return (
                    <button key={m} type="button"
                            onClick={() => setStressMult(m as 1.5 | 2.0)}
                            className="flex-1 rounded-m-sm text-[13px] font-semibold"
                            style={{
                              minHeight: 44,
                              background: active
                                ? "color-mix(in oklab, var(--m-accent) 14%, var(--m-surface))"
                                : "var(--m-surface-2)",
                              color: active ? "var(--m-accent)" : "var(--m-text-muted)",
                              border: `1px solid ${active
                                ? "var(--m-accent-border)"
                                : "var(--m-border)"}`,
                            }}>
                      {m}x
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Verdict card */}
          {verdict !== "idle" && (
            <VerdictCard
              verdict={verdict}
              disasterPrice={disasterPrice}
              cushion={unrealizedDlr}
              projectedDrawdown={totalDropEquity}
              principalRisk={principalRiskDlr}
              pctImpactPrincipal={pctImpactPrincipal}
              riskTolPct={riskTolPct}
              maxAllowedLoss={maxAllowedLoss}
              gapDlr={gapDlr}
              avgCost={avgCost}
            />
          )}
        </>
      )}

      {!selected && (
        <div className="rounded-m-md p-8 text-center text-[13px]"
             style={{
               background: "var(--m-surface)",
               border: "0.5px solid var(--m-border)",
               color: "var(--m-text-muted)",
             }}>
          Pick a ticker above to stress-test its exposure into earnings.
        </div>
      )}
    </div>
  );
}

// ── Small UI atoms ─────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim mb-1.5">
      {children}
    </div>
  );
}

function MobileInput({ label, value, onChange, step }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  step?: string;
}) {
  return (
    <div>
      <Label>{label}</Label>
      <input
        type="number"
        value={value}
        step={step ?? "0.01"}
        inputMode="decimal"
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-m-sm px-3 py-2 text-[14px] outline-none"
        style={{
          background: "var(--m-surface-2)",
          color: "var(--m-text)",
          border: "0.5px solid var(--m-border)",
          minHeight: 44,
          fontFamily: "var(--font-jetbrains), monospace",
        }}
      />
    </div>
  );
}

function VerdictCard({
  verdict,
  disasterPrice,
  cushion,
  projectedDrawdown,
  principalRisk,
  pctImpactPrincipal,
  riskTolPct,
  maxAllowedLoss,
  gapDlr,
  avgCost,
}: {
  verdict: "safe" | "approved" | "exceeded";
  disasterPrice: number;
  cushion: number;
  projectedDrawdown: number;
  principalRisk: number;
  pctImpactPrincipal: number;
  riskTolPct: number;
  maxAllowedLoss: number;
  gapDlr: number;
  avgCost: number;
}) {
  const color = verdict === "exceeded" ? "var(--m-down)" : "var(--m-accent)";
  const bgColor = `color-mix(in oklab, ${color} 14%, var(--m-surface))`;
  const borderColor = verdict === "exceeded"
    ? "color-mix(in oklab, var(--m-down) 30%, var(--m-border))"
    : "var(--m-accent-border)";

  const headline = verdict === "safe" ? "SAFE — House Money"
                 : verdict === "approved" ? "APPROVED"
                 : "EXCEEDED — Trim before earnings";

  return (
    <div className="rounded-m-md p-4 flex flex-col gap-3"
         data-testid="mobile-ep-verdict"
         data-verdict={verdict}
         style={{
           background: bgColor,
           border: `1px solid ${borderColor}`,
         }}>
      <div className="text-[15px] font-semibold" style={{ color }}>
        {headline}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <VerdictCell label="Disaster price"
                     main={formatCurrency(disasterPrice)}
                     sub={`-${formatCurrency(gapDlr)} gap`} />
        <VerdictCell label="Profit buffer"
                     main={formatCurrency(cushion, { decimals: 0, showSign: true })}
                     color={cushion >= 0 ? "var(--m-accent)" : "var(--m-down)"} />
        <VerdictCell label="Projected drawdown"
                     main={`-${formatCurrency(projectedDrawdown, { decimals: 0 })}`}
                     color="var(--m-down)" />
        <VerdictCell label="Risk to principal"
                     main={formatCurrency(principalRisk, { decimals: 0 })}
                     sub={`${pctImpactPrincipal.toFixed(2)}% of NLV`}
                     color={principalRisk === 0 ? "var(--m-accent)" : "var(--m-down)"} />
      </div>

      {verdict === "exceeded" && (
        <div className="text-[12px] leading-snug"
             style={{ color: "var(--m-text-muted)" }}>
          Principal risk {formatCurrency(principalRisk, { decimals: 0 })} exceeds
          the {riskTolPct}% budget of {formatCurrency(maxAllowedLoss, { decimals: 0 })}.
          Trim on desktop to bring it under, or exit before the print.
        </div>
      )}

      {verdict === "safe" && (
        <div className="text-[12px] leading-snug"
             style={{ color: "var(--m-text-muted)" }}>
          Even a {formatCurrency(gapDlr)} gap keeps price ({formatCurrency(disasterPrice)})
          above cost ({formatCurrency(avgCost)}). No principal at risk.
        </div>
      )}
    </div>
  );
}

function VerdictCell({ label, main, sub, color }: {
  label: string;
  main: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.06em] font-semibold text-m-text-dim">
        {label}
      </div>
      <div className="mt-0.5 text-[14px] font-semibold privacy-mask"
           style={{ color: color ?? "var(--m-text)", fontFamily: "var(--font-jetbrains), monospace" }}>
        {main}
      </div>
      {sub && (
        <div className="mt-0.5 text-[10px] text-m-text-faint"
             style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
          {sub}
        </div>
      )}
    </div>
  );
}
