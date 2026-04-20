"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export function ImportTrades({ navColor }: { navColor: string }) {
  const [pulling, setPulling] = useState(false);
  const [executions, setExecutions] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const handlePull = async () => {
    setPulling(true);
    setError("");
    setMessage("");
    try {
      const result = await api.importTrades();
      if (result.error) {
        setError(result.error);
      } else {
        setExecutions(result.trades || []);
        if (result.count === 0) {
          setMessage(result.message || "No trades found");
        } else {
          setMessage(`Pulled ${result.count} execution(s) from IBKR`);
        }
      }
    } catch (e: any) {
      setError(e.message || "Failed to connect to IBKR");
    } finally {
      setPulling(false);
    }
  };

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Import <em className="italic" style={{ color: navColor }}>Trades</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Pull executions from IBKR Flex Web Service
        </div>
      </div>

      {/* Connection status */}
      <div className="flex items-center gap-3 mb-6 p-4 rounded-[14px]" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="w-10 h-10 rounded-[12px] flex items-center justify-center text-lg"
             style={{ background: `color-mix(in oklab, ${navColor} 12%, transparent)` }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={navColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
          </svg>
        </div>
        <div className="flex-1">
          <div className="text-[13px] font-semibold">Interactive Brokers Flex Query</div>
          <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>
            Configured via server-side credentials · Stocks & Options
          </div>
        </div>
        <button onClick={handlePull} disabled={pulling}
                className="flex items-center gap-2 h-[36px] px-5 rounded-[10px] text-[13px] font-semibold text-white transition-all disabled:opacity-50"
                style={{ background: navColor }}>
          {pulling ? (
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
          )}
          {pulling ? "Pulling..." : "Pull IBKR Trades"}
        </button>
      </div>

      {error && (
        <div className="mb-5 flex items-center gap-2 px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
             style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#e5484d", border: "1px solid #e5484d30" }}>
          {error}
        </div>
      )}

      {message && !error && (
        <div className="mb-5 flex items-center gap-2 px-4 py-2.5 rounded-[10px] text-[12px] font-medium"
             style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", color: "#08a86b", border: "1px solid #08a86b30" }}>
          {message}
        </div>
      )}

      {/* Workflow guide when no data */}
      {executions.length === 0 && !pulling && (
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Import Workflow</span>
          </div>
          <div className="p-6">
            <div className="flex flex-col gap-4">
              {[
                { step: "1", title: "Pull Executions", desc: "Fetch today's trade confirmations from IBKR Flex Query" },
                { step: "2", title: "Review & Validate", desc: "Check quantities, prices, and partial fill consolidation" },
                { step: "3", title: "Log to Journal", desc: "Send each execution to Log Buy or Log Sell, or Quick Log directly" },
              ].map(s => (
                <div key={s.step} className="flex items-start gap-3">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-semibold text-white shrink-0"
                       style={{ background: navColor }}>
                    {s.step}
                  </div>
                  <div>
                    <div className="text-[13px] font-semibold">{s.title}</div>
                    <div className="text-[12px]" style={{ color: "var(--ink-4)" }}>{s.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Executions table would render here after pull */}
      {executions.length > 0 && (
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Executions</span>
            <span className="text-xs" style={{ color: "var(--ink-4)" }}>{executions.length} trades</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr>
                  {["Time", "Symbol", "Type", "Action", "Qty", "Price", "Amount", "Commission", "Net"].map(h => (
                    <th key={h} className="text-left text-[10px] uppercase tracking-[0.08em] font-semibold px-3 py-2.5 whitespace-nowrap"
                        style={{ color: "var(--ink-4)", background: "var(--surface-2)", borderBottom: "1px solid var(--border)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {executions.map((t: any, i: number) => (
                  <tr key={i}>
                    <td className="px-3 py-2.5" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: 11 }}>{t.order_time}</td>
                    <td className="px-3 py-2.5 font-semibold">{t.symbol}</td>
                    <td className="px-3 py-2.5 text-[11px]">{t.asset_class}</td>
                    <td className="px-3 py-2.5">
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                            style={{ background: t.action === "BUY" ? "#e5f7ee" : "#fdecec", color: t.action === "BUY" ? "#08a86b" : "#e5484d" }}>
                        {t.action}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>{t.quantity}</td>
                    <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${t.price}</td>
                    <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${t.amount}</td>
                    <td className="px-3 py-2.5 text-right" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${t.commission}</td>
                    <td className="px-3 py-2.5 text-right privacy-mask" style={{ fontFamily: "var(--font-jetbrains), monospace" }}>${t.net_cash}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
