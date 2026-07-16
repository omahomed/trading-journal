"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { api, getActivePortfolio, type TradePosition, type TradeDetail } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";
import { SELL_RULE_LABELS as SELL_RULES } from "@/lib/trade-rules";
import { SellRuleGlossary } from "./sell-rule-glossary";
import { SearchSelect } from "./search-select";
import { uploadWithTimeout } from "@/lib/upload-with-timeout";
import { UploadTracker, type UploadEntry, type UploadKind } from "./upload-tracker";
import { LESSON_CATEGORIES, CAT_COLORS, CAT_FALLBACK } from "@/lib/lesson-categories";

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
  // Details for open trades — needed to read B1's stop_ladder for the
  // Scale-Out auto-detect / manual-override on the sell price. Fetched
  // once on mount alongside the summaries.
  const [openDetails, setOpenDetails] = useState<TradeDetail[]>([]);
  const [selectedTrade, setSelectedTrade] = useState("");
  const [shares, setShares] = useState("");
  const [price, setPrice] = useState("");
  // "auto" defers to price-based leg matching; explicit "leg1|leg2|leg3"
  // forces attribution to that leg (fills the shares input from that
  // leg on click); "none" opts out of the pre-fill flow entirely.
  const [ladderAttribution, setLadderAttribution] = useState<"auto" | "leg1" | "leg2" | "leg3" | "none">("auto");
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
  // Trade lesson — same shape Trade Review writes (`trade_lessons.category`
  // is pipe-joined). Shown only when the current sell will fully close
  // the position; saved fire-and-forget after the sell submit so a
  // lesson-save hiccup never blocks the trade record. Pre-filled from
  // any existing lesson on campaign selection so the user can refine
  // (rather than overwrite blindly) when they revisit at exit time.
  const [lessonCats, setLessonCats] = useState<string[]>([]);
  const [lessonNote, setLessonNote] = useState("");
  const [lessonExisted, setLessonExisted] = useState(false);

  // Background upload tracker — populated when a submit succeeds, then
  // updated as each upload resolves or fails. Submit button re-enables
  // as soon as the DB write is done; uploads run independently and
  // surface their status here. See [[upload-tracker]].
  const [uploadEntries, setUploadEntries] = useState<UploadEntry[]>([]);
  const uploadEntriesRef = useRef<UploadEntry[]>([]);
  uploadEntriesRef.current = uploadEntries;

  const fireUpload = useCallback((entry: UploadEntry) => {
    uploadWithTimeout(entry.file, entry.portfolio, entry.tradeId, entry.ticker, entry.kind)
      .then(result => {
        setUploadEntries(prev => prev.map(e =>
          e.id === entry.id
            ? { ...e, status: result.ok ? "done" : "failed", error: result.error }
            : e,
        ));
      });
  }, []);

  const onRetryUpload = useCallback((id: string) => {
    const entry = uploadEntriesRef.current.find(e => e.id === id);
    if (!entry) return;
    setUploadEntries(prev => prev.map(e =>
      e.id === id ? { ...e, status: "uploading", error: undefined } : e,
    ));
    fireUpload({ ...entry, status: "uploading", error: undefined });
  }, [fireUpload]);

  const onDismissTracker = useCallback(() => setUploadEntries([]), []);

  useEffect(() => {
    api.tradesOpen(getActivePortfolio()).then(trades => {
      setOpenTrades(trades);
      setLoading(false);
      // Piggyback on the same page load to grab open-trade details for
      // ladder auto-detect. Failure is non-fatal — the sell flow works
      // without the ladder hint.
      api.tradesOpenDetails(getActivePortfolio())
        .then(bundle => setOpenDetails(bundle.details || []))
        .catch(err => log.debug.devOnly("log-sell", "tradesOpenDetails skipped", err));

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
    }).catch((err) => {
      log.error("log-sell", "tradesOpen fetch failed", err);
      setLoading(false);
    });
  }, []);

  const selected = openTrades.find(t => t.trade_id === selectedTrade);

  // Prefill Grade from the selected trade's existing grade, if any
  useEffect(() => {
    if (!selected) return;
    const g = (selected as any).grade;
    setGrade(typeof g === "number" && g >= 1 && g <= 5 ? g : null);
  }, [selectedTrade]); // intentionally not `selected` to avoid thrash

  // Pre-fill any existing trade lesson (Trade Review may have written
  // one earlier). Pipe-joined category string is split into chips. If
  // no lesson exists yet, state stays empty — first save on full exit
  // creates the row. The endpoint is per-portfolio + bulk, so the
  // payload is tiny.
  useEffect(() => {
    if (!selectedTrade) { setLessonCats([]); setLessonNote(""); setLessonExisted(false); return; }
    api.getTradeLessons(getActivePortfolio()).then(r => {
      const existing = r.lessons?.[selectedTrade];
      if (existing) {
        setLessonCats(String(existing.category || "").split("|").filter(Boolean));
        setLessonNote(String(existing.note || ""));
        setLessonExisted(true);
      } else {
        setLessonCats([]); setLessonNote(""); setLessonExisted(false);
      }
    }).catch((err) => {
      log.debug.devOnly("log-sell", "getTradeLessons prefill skipped", err);
      setLessonCats([]); setLessonNote(""); setLessonExisted(false);
    });
  }, [selectedTrade]);

  const sharesNum = parseFloat(shares) || 0;
  const priceNum = parseFloat(price) || 0;

  // Selected trade's B1 ladder (Phase 3 sub-scope B). Prices track
  // current avg entry — same convention as Trade Journal display.
  const selectedB1 = openDetails.find(d => d.trade_id === selectedTrade && String(d.action).toUpperCase() === "BUY");
  const selectedB1Ladder = (() => {
    const raw = selectedB1 ? (selectedB1 as any).stop_ladder : null;
    if (!raw || typeof raw !== "object") return null;
    const legs = (raw as { legs?: unknown }).legs;
    if (!Array.isArray(legs) || legs.length !== 3) return null;
    const parsed = legs.map(l => {
      const leg = l as { pct?: unknown; shares?: unknown };
      return { pct: Number(leg.pct), shares: Number(leg.shares) };
    });
    if (parsed.some(l => !Number.isFinite(l.pct) || !Number.isFinite(l.shares))) return null;
    return { legs: parsed };
  })();
  const selectedAvgEntry = selected?.avg_entry || 0;
  const ladderLegPrices = selectedB1Ladder && selectedAvgEntry > 0
    ? selectedB1Ladder.legs.map(l => ({ pct: l.pct, shares: l.shares, price: selectedAvgEntry * (1 - l.pct / 100) }))
    : null;

  // Auto-detect: sell price within 1% of a leg's stop price. First
  // match wins (legs are ordered -3, -5, -7 so tighter stops win ties).
  const autoLegMatch = (() => {
    if (!ladderLegPrices || priceNum <= 0) return null;
    for (let i = 0; i < ladderLegPrices.length; i++) {
      const leg = ladderLegPrices[i];
      if (leg.price > 0 && Math.abs(priceNum - leg.price) / leg.price < 0.01) {
        return { index: i, leg };
      }
    }
    return null;
  })();

  // Resolved leg the UI is pointing at — obeys manual override, else
  // falls back to autoLegMatch. "none" and no auto-match both resolve
  // to null (no hint shown, no pre-fill button).
  const resolvedLeg = (() => {
    if (!ladderLegPrices) return null;
    if (ladderAttribution === "none") return null;
    if (ladderAttribution === "auto") return autoLegMatch;
    const idx = ladderAttribution === "leg1" ? 0 : ladderAttribution === "leg2" ? 1 : 2;
    return { index: idx, leg: ladderLegPrices[idx] };
  })();

  // Reset attribution to auto whenever the selected trade changes so a
  // stale manual pick doesn't carry over to a different position.
  useEffect(() => { setLadderAttribution("auto"); }, [selectedTrade]);

  // Detect option from the selected campaign's metadata first (preferred — set
  // by Migration 016), with a ticker-shape fallback for any legacy row that
  // hasn't been recomputed yet. Multiplier scales proceeds + realized P&L
  // into notional dollars; return % is per-contract and stays invariant.
  const isOption = String((selected as any)?.instrument_type || "").toUpperCase() === "OPTION"
    || /^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$/.test(String(selected?.ticker || ""));
  const multiplier = isOption
    ? Math.max(parseFloat(String((selected as any)?.multiplier || 0)) || 100, 1)
    : 1;
  const unitLabel = isOption ? "Contracts" : "Shares";
  const proceeds = sharesNum * priceNum * multiplier;
  const avgEntry = selected?.avg_entry || 0;
  const returnPct = avgEntry > 0 && priceNum > 0 ? ((priceNum - avgEntry) / avgEntry) * 100 : 0;
  const realizedPl = avgEntry > 0 ? (priceNum - avgEntry) * sharesNum * multiplier : 0;

  const handleSubmit = async () => {
    if (!selectedTrade || !selected) return;
    if (sharesNum <= 0) return setSubmitResult({ ok: false, msg: `${unitLabel} must be > 0` });
    if (priceNum <= 0) return setSubmitResult({ ok: false, msg: "Price must be > 0" });
    if (sharesNum > selected.shares) return setSubmitResult({ ok: false, msg: `Max ${unitLabel.toLowerCase()}: ${selected.shares}` });

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
        // Snapshot attached files into the background upload tracker — same
        // pattern as Log Buy. The submit chain no longer awaits uploads, so
        // a stalled R2 call can't hang the "Saving…" button. Per-file
        // status (uploading / done / failed) is rendered by <UploadTracker>.
        const portfolio = getActivePortfolio();
        const entriesToFire: UploadEntry[] = [];
        const kind: UploadKind = "position_change";
        for (const f of positionCharts) {
          entriesToFire.push({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${entriesToFire.length}`,
            file: f, fileName: f.name, kind, portfolio,
            tradeId: selectedTrade, ticker: selected.ticker, status: "uploading",
          });
        }
        if (entriesToFire.length > 0) {
          setUploadEntries(prev => [...prev, ...entriesToFire]);
          for (const entry of entriesToFire) fireUpload(entry);
        }

        const plStr = result.realized_pl != null ? ` | P&L: ${formatCurrency(result.realized_pl)}` : "";
        const closedStr = result.is_closed ? " (CLOSED)" : ` (${result.remaining_shares} remaining)`;
        setSubmitResult({ ok: true, msg: `Sold ${result.trx_id || "S1"}: ${shares} shs of ${selected.ticker} @ $${price}${plStr}${closedStr}` });

        // Lesson save (fire-and-forget). Only on the close-out sell, and
        // only when the user actually picked categories or wrote a note —
        // an empty lesson is treated as "nothing to record", same intent
        // as the Trade Review path. Failure is non-fatal: the sell
        // already went through; the user can still fill the lesson from
        // Trade Review later.
        const lessonHasContent = lessonCats.length > 0 || lessonNote.trim().length > 0;
        if (result.is_closed && lessonHasContent) {
          api.saveTradeLessons({
            portfolio,
            trade_id: selectedTrade,
            note: lessonNote,
            category: lessonCats.join("|"),
          }).catch((err) => {
            log.error("log-sell", "lesson save failed (non-fatal — sell already recorded)", err);
          });
        }

        // Reset form — File refs are already captured in uploadEntries.
        setShares(""); setPrice(""); setNotes(""); setGrade(null); setPositionCharts([]);
        setLessonCats([]); setLessonNote(""); setLessonExisted(false);

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

      <UploadTracker entries={uploadEntries} onRetry={onRetryUpload} onDismiss={onDismissTracker} />

      <div className="grid gap-6" style={{ gridTemplateColumns: "2fr 1fr" }}>
        <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
          <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
            <span className="text-[13px] font-semibold">Sell Order</span>
          </div>
          <div className="p-5 flex flex-col gap-5">
            <FormField label="Select Campaign" hint={selected ? `${selected.shares} ${unitLabel.toLowerCase()} @ ${formatCurrency(selected.avg_entry || 0)} avg` : undefined}>
              {/* Searchable combobox shared with Log Buy's Scale In picker.
                  Option labels lead with ticker + unit count so the user can
                  scan the open-campaigns list visually; trade_id sits on
                  the right after the pipe, and the SearchSelect filter
                  matches on BOTH halves of the string (the ticker for "AAPL"
                  searches and the trade_id for "202604" searches). On
                  select we split on " | " and keep the right side — that's
                  the bare trade_id the rest of this component expects in
                  selectedTrade state. */}
              <SearchSelect
                value={selectedTrade ? `${openTrades.find(t => t.trade_id === selectedTrade)?.ticker || ""} (${selected?.shares ?? ""} ${unitLabel.toLowerCase()}) | ${selectedTrade}` : ""}
                onChange={(v) => {
                  const id = v.split(" | ")[1]?.trim() || "";
                  setSelectedTrade(id);
                }}
                options={openTrades.map(t => {
                  const unit = String((t as any).instrument_type || "").toUpperCase() === "OPTION" ? "contracts" : "shares";
                  return `${t.ticker} (${t.shares} ${unit}) | ${t.trade_id}`;
                })}
                placeholder="Choose an open campaign..."
              />
            </FormField>

            {isOption && (
              <div className="-mt-1 text-[12px] px-3 py-2 rounded-[8px] flex items-center gap-2"
                   style={{ background: "color-mix(in oklab, #f59f00 10%, var(--surface))", color: "#92400e" }}>
                <span className="font-semibold">OPTION ×{multiplier}</span>
                <span style={{ color: "var(--ink-3)" }}>·</span>
                <span>Proceeds and realized P&L shown as notional</span>
              </div>
            )}

            {(() => {
              const beStamp = (selected as any)?.be_stop_moved_at;
              if (!beStamp) return null;
              const beDate = String(beStamp).slice(0, 10);
              const isSr15 = rule.startsWith("sr15");
              return (
                <div className="-mt-1 px-3 py-2 rounded-[10px] flex items-center justify-between gap-3"
                     style={{
                       background: "color-mix(in oklab, #f59f00 12%, var(--surface))",
                       border: "1px solid color-mix(in oklab, #f59f00 35%, var(--border))",
                     }}>
                  <div className="text-[12px]" style={{ color: "var(--ink-2)" }}>
                    <span className="font-semibold">BE rule flagged</span>
                    {" "}
                    <span style={{ color: "var(--ink-4)" }}>
                      on {beDate} — pick <span style={{ fontFamily: "var(--font-jetbrains), monospace" }}>sr15</span> if this exit is the BE stop hitting.
                    </span>
                  </div>
                  {!isSr15 && (
                    <button type="button"
                            onClick={() => setRule("sr15 BE Stop Out (moved at +10%)")}
                            className="text-[11px] font-semibold px-2.5 py-1 rounded-[6px] whitespace-nowrap"
                            style={{ background: "#f59f00", color: "white", border: "none", cursor: "pointer" }}>
                      Use sr15
                    </button>
                  )}
                </div>
              );
            })()}

            <FormField label="Sell Rule">
              <SearchSelect value={rule} onChange={setRule} options={SELL_RULES} placeholder="Type to search rules..." />
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label={`${unitLabel} to Sell`} hint={selected ? `Max: ${selected.shares}` : undefined}>
                <input type="number" value={shares} onChange={e => setShares(e.target.value)}
                       placeholder="0" className="w-full h-[38px] px-3 rounded-[10px] text-[13px]" style={inputStyle} />
              </FormField>
              <FormField label={isOption ? "Premium per Contract ($)" : "Sell Price ($)"}>
                <input type="number" value={price} onChange={e => setPrice(e.target.value)} step="0.01"
                       placeholder="0.00" className="w-full h-[38px] px-3 rounded-[10px] text-[13px]" style={inputStyle} />
              </FormField>
            </div>

            {/* Scale-Out Stops — auto-detect that this sell price is close
                to a ladder leg's stop. Only renders when the selected
                trade has a persisted ladder. Manual override dropdown lets
                the user force attribution (Leg 1/2/3) or opt out (None).
                No leg attribution is persisted on the sell row — this is
                pre-fill convenience only. */}
            {ladderLegPrices && (
              <div className="rounded-[10px] px-3 py-2.5 flex items-center justify-between gap-3 text-[12px] flex-wrap"
                   style={{ background: "color-mix(in oklab, #0ea5a4 5%, var(--surface))", border: "1px solid color-mix(in oklab, #0ea5a4 25%, var(--border))", color: "#0ea5a4" }}>
                <div className="flex items-center gap-2 flex-wrap">
                  <span>🪜</span>
                  {resolvedLeg ? (
                    <span>
                      <strong>Matches Leg {resolvedLeg.index + 1}</strong>
                      <span style={{ color: "var(--ink-3)" }}>
                        {" · "}−{resolvedLeg.leg.pct}% · {resolvedLeg.leg.shares} shs @ {formatCurrency(resolvedLeg.leg.price)}
                      </span>
                    </span>
                  ) : (
                    <span style={{ color: "var(--ink-3)" }}>
                      Scale-Out plan active. Enter a price near a leg to auto-attribute, or pick one manually.
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {resolvedLeg && ladderAttribution !== "none" && (
                    <button type="button"
                            onClick={() => setShares(String(resolvedLeg.leg.shares))}
                            className="h-[26px] px-2.5 rounded-[6px] text-[11px] font-semibold text-white cursor-pointer transition-all hover:brightness-110"
                            style={{ background: "#0ea5a4" }}>
                      Fill {resolvedLeg.leg.shares} shs
                    </button>
                  )}
                  <select value={ladderAttribution}
                          onChange={e => setLadderAttribution(e.target.value as typeof ladderAttribution)}
                          className="h-[26px] px-2 rounded-[6px] text-[11px] appearance-none cursor-pointer"
                          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}>
                    <option value="auto">Auto</option>
                    <option value="leg1">Leg 1 (−3%)</option>
                    <option value="leg2">Leg 2 (−5%)</option>
                    <option value="leg3">Leg 3 (−7%)</option>
                    <option value="none">None</option>
                  </select>
                </div>
              </div>
            )}

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

            {/* Lesson — only shown when this sell will close the position
                fully (sharesNum >= selected.shares). Saves alongside the
                sell submit; surfaces verbatim in the Trade Review tab and
                Trade Journal trade cards via the existing trade_lessons
                table. See [[lesson-categories]]. */}
            {selected && sharesNum > 0 && sharesNum >= (selected.shares || 0) && (
              <div className="rounded-[12px] p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
                <div className="flex items-center justify-between mb-1">
                  <div className="text-[13px] font-semibold">🎓 Exit Lesson</div>
                  {lessonExisted && (
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-full" style={{ background: "var(--bg)", color: "var(--ink-4)", border: "1px solid var(--border)" }}>
                      pre-filled from Trade Review
                    </span>
                  )}
                </div>
                <div className="text-[11px] mb-3" style={{ color: "var(--ink-4)" }}>
                  Capture the takeaway now — shows up automatically in Trade Review &amp; the Trade Journal card.
                </div>
                <div className="text-[10px] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>Category (pick one or more)</div>
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {LESSON_CATEGORIES.map(cat => {
                    const active = lessonCats.includes(cat);
                    const cc = CAT_COLORS[cat] || CAT_FALLBACK;
                    return (
                      <button key={cat} type="button"
                              onClick={() => setLessonCats(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat])}
                              className="text-[10px] font-bold px-2.5 py-1 rounded-full transition-all"
                              style={{ background: active ? cc.bg : "var(--bg)", color: active ? cc.fg : "var(--ink-4)", border: `1px solid ${active ? cc.fg + "40" : "var(--border)"}` }}>
                        {active ? "✓ " : ""}{cat}
                      </button>
                    );
                  })}
                </div>
                <div className="text-[10px] font-semibold mb-1" style={{ color: "var(--ink-4)" }}>What did you learn from this trade?</div>
                <textarea rows={3} value={lessonNote}
                          onChange={e => setLessonNote(e.target.value)}
                          placeholder="e.g. Scaled in too fast on the third add..."
                          className="w-full px-3 py-2 rounded-[8px] text-[11px] outline-none resize-none"
                          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
              </div>
            )}

            <button onClick={handleSubmit} disabled={submitting || !selectedTrade}
                    className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
                    style={{ background: "#e5484d" }}>
              {submitting ? "Saving..." : "LOG SELL ORDER"}
            </button>
          </div>
        </div>

        {/* Side panel — h-full so it fills the grid cell to match the
            Sell Order column's height; Open Campaigns then flex-1's
            to consume the remaining vertical space below Sell Preview. */}
        <div className="flex flex-col gap-4 h-full">
          <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
              <span className="text-[13px] font-semibold">Sell Preview</span>
            </div>
            <div className="p-4 flex flex-col gap-3">
              {[
                { k: "Proceeds", v: formatCurrency(proceeds) },
                { k: "Avg Entry", v: avgEntry > 0 ? formatCurrency(avgEntry) : "—" },
                { k: "Return", v: returnPct !== 0 ? `${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(1)}%` : "—", color: returnPct >= 0 ? "#08a86b" : "#e5484d" },
                { k: "Realized P&L", v: realizedPl !== 0 ? formatCurrency(realizedPl, { showSign: true, decimals: 0 }) : "—", color: realizedPl >= 0 ? "#08a86b" : "#e5484d" },
                { k: "Remaining", v: selected ? `${selected.shares - sharesNum} ${unitLabel.toLowerCase()}` : "—" },
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

          {/* Open positions quick reference — flex-1 lets the card grow
              into the remaining height of the side panel; the inner
              scroll uses flex-1 + min-h-0 so it bounds its scroll
              region to the card's remaining height instead of the
              old fixed 300px cap. */}
          <div className="rounded-[14px] overflow-hidden flex flex-col flex-1 min-h-0" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span className="text-[13px] font-semibold">Open Campaigns</span>
              <span className="text-xs" style={{ color: "var(--ink-4)" }}>{openTrades.length}</span>
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto">
              {[...openTrades].sort((a, b) => a.ticker.localeCompare(b.ticker)).map(t => {
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
                      <span className="text-[11px] ml-2" style={{ color: "var(--ink-4)" }}>
                        {t.shares} {String((t as any).instrument_type || "").toUpperCase() === "OPTION" ? "ct" : "sh"}
                      </span>
                    </div>
                    <span className="text-[11px]" style={{ color: isSelected ? navColor : "var(--ink-4)", fontWeight: isSelected ? 600 : 400 }}>{t.trade_id}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <SellRuleGlossary />
    </div>
  );
}
