"use client";

import { useState, useEffect, useRef } from "react";
import { api, getActivePortfolio, type TradePosition } from "@/lib/api";

const SELL_RULES = [
  "sr1 Capital Protection", "sr2 Trailing Stop", "sr3 Portfolio Management",
  "sr4 Time Stop", "sr5 Climax Top", "sr6 Exhaustion Gap",
  "sr7 200d Moving Avg Break", "sr8 Living Below 50d", "sr9 Failed Breakout",
  "sr10 Scale-Out T1 (-3%)", "sr11 Scale-Out T2 (-5%)", "sr12 Scale-Out T3 (-8%)",
  "sr13 Earnings Exit", "sr14 Market Correction Exit",
  "sr15 BE Stop Out (moved at +10%)",
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

function DropZone({ label, icon, files, onFiles, multiple, accept }: {
  label: string; icon: string; files: File[]; onFiles: (f: File[]) => void; multiple?: boolean; accept?: string;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter(f => !accept || accept.split(",").some(a => f.type === a.trim()));
    if (dropped.length > 0) onFiles(multiple ? [...files, ...dropped] : [dropped[0]]);
  };

  const hasFiles = files.length > 0;
  const borderColor = dragging ? "#6366f1" : hasFiles ? "#08a86b" : "var(--border)";
  const bgColor = dragging ? "color-mix(in oklab, #6366f1 5%, var(--bg))" : hasFiles ? "color-mix(in oklab, #08a86b 5%, var(--bg))" : "var(--bg)";
  const textColor = hasFiles ? "#08a86b" : "var(--ink-4)";

  return (
    <div>
      {label && (
        <div className="text-[11px] font-medium mb-1.5 flex items-center gap-1" style={{ color: "var(--ink-3)" }}>
          {icon && <span>{icon}</span>} {label}
        </div>
      )}
      <div
        className="flex flex-col items-center justify-center rounded-[10px] cursor-pointer transition-all"
        style={{ border: `1.5px dashed ${borderColor}`, background: bgColor, padding: hasFiles ? "10px 16px" : "16px", minHeight: 70 }}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        {hasFiles ? (
          <div className="flex flex-wrap gap-1.5 w-full">
            {files.map((f, i) => (
              <div key={i} className="flex items-center gap-1.5 px-2 py-1 rounded-[6px] text-[10px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}>
                <span className="truncate" style={{ maxWidth: 120 }}>{f.name}</span>
                <button onClick={e => { e.stopPropagation(); onFiles(files.filter((_, j) => j !== i)); }}
                        className="text-[10px] font-bold cursor-pointer" style={{ color: "#e5484d" }}>x</button>
              </div>
            ))}
          </div>
        ) : (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={dragging ? "#6366f1" : "var(--ink-4)"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-1.5 opacity-60">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span className="text-[11px]" style={{ color: textColor }}>{dragging ? "Drop files here" : "Drag & drop or click to upload"}</span>
            <span className="text-[9px] mt-0.5" style={{ color: "var(--ink-5)" }}>PNG, JPG, PDF</span>
          </>
        )}
        <input ref={inputRef} type="file" accept={accept} multiple={multiple} className="hidden"
               onChange={e => { if (e.target.files) onFiles(multiple ? [...files, ...Array.from(e.target.files)] : Array.from(e.target.files)); e.target.value = ""; }} />
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
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
  const [grade, setGrade] = useState<number | null>(null);
  const [date, setDate] = useState(() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
  });
  const [time, setTime] = useState(() => {
    const n = new Date();
    return `${String(n.getHours()).padStart(2, "0")}:${String(n.getMinutes()).padStart(2, "0")}`;
  });
  const [positionCharts, setPositionCharts] = useState<File[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<{ ok: boolean; msg: string } | null>(null);

  useEffect(() => {
    api.tradesOpen(getActivePortfolio()).then(trades => {
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
        if (data.date) setDate(String(data.date));
        if (data.time) setTime(String(data.time));
      } catch { /* ignore */ }
    }).catch(() => setLoading(false));
  }, []);

  const selected = openTrades.find(t => t.trade_id === selectedTrade);

  // Prefill Grade from the selected trade's existing grade, if any
  useEffect(() => {
    if (!selected) return;
    const g = (selected as any).grade;
    setGrade(typeof g === "number" && g >= 1 && g <= 5 ? g : null);
  }, [selectedTrade]); // intentionally not `selected` to avoid thrash
  const sharesNum = parseFloat(shares) || 0;
  const priceNum = parseFloat(price) || 0;
  const proceeds = sharesNum * priceNum;
  const avgEntry = selected?.avg_entry || 0;
  const returnPct = avgEntry > 0 && priceNum > 0 ? ((priceNum - avgEntry) / avgEntry) * 100 : 0;
  const realizedPl = avgEntry > 0 ? (priceNum - avgEntry) * sharesNum : 0;

  const handleSubmit = async () => {
    if (!selectedTrade || !selected) return;
    if (sharesNum <= 0) return setSubmitResult({ ok: false, msg: "Shares must be > 0" });
    if (priceNum <= 0) return setSubmitResult({ ok: false, msg: "Price must be > 0" });
    if (sharesNum > selected.shares) return setSubmitResult({ ok: false, msg: `Max shares: ${selected.shares}` });

    setSubmitting(true);
    setSubmitResult(null);

    try {
      const body = {
        portfolio: getActivePortfolio(),
        trade_id: selectedTrade,
        shares: sharesNum,
        price: priceNum,
        rule,
        notes,
        grade,
        date,
        time,
      };

      const result = await api.logSell(body);

      if (result.error) {
        setSubmitResult({ ok: false, msg: result.error });
      } else {
        // Upload position change charts
        if (positionCharts.length > 0) {
          for (const file of positionCharts) {
            await api.uploadImage(file, getActivePortfolio(), selectedTrade, selected.ticker, "position_change");
          }
        }

        const plStr = result.realized_pl != null ? ` | P&L: $${result.realized_pl.toFixed(2)}` : "";
        const closedStr = result.is_closed ? " (CLOSED)" : ` (${result.remaining_shares} remaining)`;
        setSubmitResult({ ok: true, msg: `Sold ${result.trx_id || "S1"}: ${shares} shs of ${selected.ticker} @ $${price}${plStr}${closedStr}` });

        // Reset form
        setShares(""); setPrice(""); setNotes(""); setGrade(null); setPositionCharts([]);

        // Refresh open trades
        const trades = await api.tradesOpen(getActivePortfolio()).catch(() => []);
        setOpenTrades(trades);
        if (result.is_closed) setSelectedTrade("");
      }
    } catch (err: any) {
      setSubmitResult({ ok: false, msg: err.message || "Failed to log sell" });
    }

    setSubmitting(false);
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

      {submitResult && (
        <div className="mb-4 px-4 py-3 rounded-[10px] text-[13px]"
             style={{ background: submitResult.ok ? "color-mix(in oklab, #08a86b 10%, var(--surface))" : "color-mix(in oklab, #e5484d 10%, var(--surface))",
                      border: `1px solid ${submitResult.ok ? "#08a86b30" : "#e5484d30"}`,
                      color: submitResult.ok ? "#08a86b" : "#e5484d" }}>
          {submitResult.msg}
        </div>
      )}

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

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Date">
                <input type="date" value={date} onChange={e => setDate(e.target.value)}
                       className="w-full h-[38px] px-3 rounded-[10px] text-[13px]" style={inputStyle} />
              </FormField>
              <FormField label="Time">
                <input type="time" value={time} onChange={e => setTime(e.target.value)}
                       className="w-full h-[38px] px-3 rounded-[10px] text-[13px]" style={inputStyle} />
              </FormField>
            </div>

            <FormField label="Sell Context / Notes">
              <input type="text" value={notes} onChange={e => setNotes(e.target.value)}
                     placeholder="Why did you sell?"
                     className="w-full h-[38px] px-3 rounded-[10px] text-[13px]"
                     style={{ ...inputStyle, fontFamily: "inherit" }} />
            </FormField>

            <FormField label="Grade" hint="Rate the trade 1-5 stars. Click again to clear.">
              <div className="flex items-center gap-1 h-[38px]" onMouseLeave={() => { /* no-op */ }}>
                {[1, 2, 3, 4, 5].map(n => {
                  const filled = grade != null && n <= grade;
                  return (
                    <button key={n} type="button"
                            onClick={() => setGrade(grade === n ? null : n)}
                            className="p-1 bg-transparent border-0 cursor-pointer transition-transform hover:scale-110"
                            aria-label={`${n} star${n > 1 ? "s" : ""}`}>
                      <svg width="22" height="22" viewBox="0 0 24 24"
                           fill={filled ? "#f59f00" : "none"}
                           stroke={filled ? "#f59f00" : "var(--ink-4)"}
                           strokeWidth="2" strokeLinejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                      </svg>
                    </button>
                  );
                })}
                {grade != null && (
                  <span className="ml-2 text-[12px]" style={{ color: "var(--ink-3)" }}>{grade}/5</span>
                )}
              </div>
            </FormField>

            {/* Position Changes — drag & drop */}
            <DropZone label="Position Changes (Partial Sells / Full Exits)" icon="🔄"
                      files={positionCharts} onFiles={setPositionCharts}
                      multiple accept="image/png,image/jpeg,application/pdf" />

            <button onClick={handleSubmit} disabled={submitting || !selectedTrade}
                    className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
                    style={{ background: "#e5484d" }}>
              {submitting ? "Saving..." : "LOG SELL ORDER"}
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
