"use client";

// Robinhood CSV importer UI — companion to the IBKR Flex Query pull on
// the Import Trades page. The IBKR path is API-driven (server pulls
// from their web service); Robinhood has no first-class API so the
// user downloads the transaction-history CSV and drops it here.
//
// Flow: paste or upload → Preview (dry-run parse) → Commit (writes).
// The preview payload is the same JSON that
// scripts/import_robinhood_csv.py CLI reports produce; render it as a
// summary + campaigns table.

import { useState } from "react";
import { api } from "@/lib/api";
import type { RobinhoodImportPreview, RobinhoodImportCommit } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";
import { log } from "@/lib/log";

// Default cutoff — align with the campaign-review "since 2026-01-01"
// convention already used across Deep Dive pages. User can override.
const DEFAULT_SINCE = "2026-01-01";

const mono = "var(--font-jetbrains), monospace";

export function RobinhoodImport({ navColor }: { navColor: string }) {
  const { portfolios } = usePortfolio();

  const [csvText, setCsvText] = useState("");
  const [since, setSince] = useState(DEFAULT_SINCE);
  const [portfolio, setPortfolio] = useState<string>(() => {
    // Default: user's Long-Term Growth if it exists, else the active
    // portfolio. Robinhood is most commonly the LTG data source per the
    // build brief; but leave the picker mutable so CanSlim etc. work too.
    return "Long-Term Growth";
  });
  const [strategy, setStrategy] = useState("LongTerm");
  const [resetCashLedger, setResetCashLedger] = useState(false);

  const [previewing, setPreviewing] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [preview, setPreview] = useState<RobinhoodImportPreview | null>(null);
  const [commitResult, setCommitResult] = useState<RobinhoodImportCommit | null>(null);
  const [error, setError] = useState<string>("");

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => setCsvText(String(reader.result || ""));
    reader.onerror = () => setError("Failed to read file");
    reader.readAsText(f);
  };

  const handlePreview = async () => {
    setPreviewing(true); setError(""); setCommitResult(null);
    try {
      const res = await api.robinhoodImportPreview(csvText, since, portfolio);
      if ("detail" in res) {
        setError(res.detail);
        setPreview(null);
      } else {
        setPreview(res);
      }
    } catch (e) {
      log.error("robinhood-import", "preview failed", e);
      setError(String(e));
      setPreview(null);
    } finally {
      setPreviewing(false);
    }
  };

  const handleCommit = async () => {
    if (!preview) return;
    setCommitting(true); setError("");
    try {
      const res = await api.robinhoodImportCommit({
        csv_text: csvText, since, portfolio, strategy,
        reset_cash_ledger: resetCashLedger,
      });
      if ("detail" in res) setError(res.detail);
      else setCommitResult(res);
    } catch (e) {
      log.error("robinhood-import", "commit failed", e);
      setError(String(e));
    } finally {
      setCommitting(false);
    }
  };

  const handleReset = () => {
    setCsvText(""); setPreview(null); setCommitResult(null); setError("");
    setResetCashLedger(false);
  };

  const canPreview = csvText.trim().length > 0 && portfolio.trim().length > 0 && !previewing;
  const canCommit = preview !== null && !committing && !commitResult;

  const totalCampaigns = (preview?.equity_campaigns.length || 0) + (preview?.option_campaigns.length || 0);
  const totalCash = preview?.cash_rows.length || 0;
  const hasWarnings = (preview?.warnings.length || 0) > 0;
  const dupWarning = (preview?.existing_trades || 0) > 0;

  return (
    <div className="flex flex-col gap-5">
      {/* Import form */}
      <div className="rounded-[14px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "var(--card-shadow)" }}>
        <div className="flex items-center gap-2 px-[18px] py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: navColor }} />
          <span className="text-[13px] font-semibold">Import from Robinhood</span>
          <span className="text-xs" style={{ color: "var(--ink-4)" }}>
            Paste or upload the transaction-history CSV from your Robinhood web account
          </span>
        </div>
        <div className="p-4 flex flex-col gap-4">
          {/* Row 1: Portfolio + Since + Strategy */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="block text-[9px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
                Portfolio
              </label>
              <select value={portfolio} onChange={e => setPortfolio(e.target.value)}
                      className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                      style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }}>
                {portfolios.map(p => <option key={p.id} value={p.name}>{p.name}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-[9px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
                Since (YYYY-MM-DD)
              </label>
              <input type="date" value={since} onChange={e => setSince(e.target.value)}
                     className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
            <div>
              <label className="block text-[9px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
                Strategy tag
              </label>
              <input type="text" value={strategy} onChange={e => setStrategy(e.target.value)}
                     placeholder="LongTerm"
                     className="w-full h-[36px] px-2.5 rounded-[8px] text-[12px] outline-none"
                     style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
            </div>
          </div>

          {/* Row 2: CSV input */}
          <div>
            <label className="block text-[9px] uppercase tracking-[0.08em] font-semibold mb-1.5" style={{ color: "var(--ink-4)" }}>
              CSV — paste content OR upload file
            </label>
            <div className="flex gap-2 mb-2">
              <input type="file" accept=".csv,text/csv" onChange={handleFileUpload}
                     className="text-[12px]"
                     style={{ color: "var(--ink-3)" }} />
              {csvText && (
                <span className="text-[11px] self-center" style={{ color: "var(--ink-4)" }}>
                  {csvText.length.toLocaleString()} characters loaded
                </span>
              )}
            </div>
            <textarea value={csvText} onChange={e => setCsvText(e.target.value)}
                      placeholder='Paste CSV content here — starts with "Activity Date","Process Date",…'
                      rows={8}
                      className="w-full px-3 py-2 rounded-[8px] text-[11px] outline-none resize-y"
                      style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)", fontFamily: mono }} />
          </div>

          {/* Row 3: Options + Actions */}
          <div className="flex items-center gap-3 flex-wrap">
            <label className="flex items-center gap-2 text-[12px] cursor-pointer">
              <input type="checkbox" checked={resetCashLedger}
                     onChange={e => setResetCashLedger(e.target.checked)} />
              <span style={{ color: "var(--ink-3)" }}>
                Reset cash ledger before import (deletes existing deposits/transfers on this portfolio)
              </span>
            </label>
            <div className="ml-auto flex gap-2">
              {(preview || commitResult) && (
                <button onClick={handleReset}
                        className="h-[36px] px-4 rounded-[10px] text-[12px] font-semibold cursor-pointer"
                        style={{ background: "var(--surface)", color: "var(--ink-3)", border: "1px solid var(--border)" }}>
                  Reset
                </button>
              )}
              <button onClick={handlePreview} disabled={!canPreview}
                      className="h-[36px] px-4 rounded-[10px] text-[12px] font-semibold cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{ background: `color-mix(in oklab, ${navColor} 12%, transparent)`, color: navColor, border: `1px solid ${navColor}` }}>
                {previewing ? "Parsing…" : "Preview"}
              </button>
              <button onClick={handleCommit} disabled={!canCommit}
                      className="h-[36px] px-5 rounded-[10px] text-[12px] font-semibold text-white cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{ background: navColor }}>
                {committing ? "Committing…" : "Commit Import"}
              </button>
            </div>
          </div>

          {error && (
            <div className="text-[12px] px-3 py-2 rounded-[8px]"
                 style={{ background: "color-mix(in oklab, #e5484d 10%, var(--surface))", color: "#dc2626", border: "1px solid #e5484d30" }}>
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Commit success */}
      {commitResult && (
        <div className="rounded-[14px] p-4"
             style={{ background: "color-mix(in oklab, #08a86b 10%, var(--surface))", border: "1px solid #08a86b40" }}>
          <div className="text-[13px] font-semibold" style={{ color: "#16a34a" }}>
            Import committed successfully
          </div>
          <div className="text-[12px] mt-1" style={{ color: "var(--ink-3)" }}>
            Wrote <strong>{commitResult.written.summary}</strong> campaigns ·
            {" "}<strong>{commitResult.written.details}</strong> transactions ·
            {" "}<strong>{commitResult.written.cash}</strong> cash rows
            {commitResult.reset_count !== null && commitResult.reset_count > 0 && (
              <> · reset {commitResult.reset_count} prior cash rows</>
            )}
          </div>
        </div>
      )}

      {/* Preview panel */}
      {preview && !commitResult && (
        <>
          {/* Summary tiles */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <SummaryTile label="Equity campaigns" value={preview.equity_campaigns.length} accent="#08a86b" />
            <SummaryTile label="Option campaigns" value={preview.option_campaigns.length} accent="#8b5cf6" />
            <SummaryTile label="Cash rows" value={totalCash} accent="#0ea5e9" />
            <SummaryTile
              label={dupWarning ? "Existing trades ⚠︎" : "Existing trades"}
              value={preview.existing_trades}
              accent={dupWarning ? "#f59e0b" : "#64748b"}
            />
          </div>

          {/* Duplicate warning banner */}
          {dupWarning && (
            <div className="rounded-[10px] p-3"
                 style={{ background: "color-mix(in oklab, #f59e0b 10%, var(--surface))", color: "#b45309", border: "1px solid #f59e0b40" }}>
              <div className="text-[12px] font-semibold">
                Duplicate risk: {preview.existing_trades} trade{preview.existing_trades === 1 ? "" : "s"} already exist{preview.existing_trades === 1 ? "s" : ""} in {preview.portfolio} since {preview.since}.
              </div>
              <div className="text-[11px] mt-1">
                A previous import may have already run. Verify the counts below match what's still missing before you hit Commit — the writer doesn't dedupe row-by-row.
              </div>
            </div>
          )}

          {/* Warnings */}
          {hasWarnings && (
            <div className="rounded-[10px] p-3"
                 style={{ background: "color-mix(in oklab, #f59e0b 10%, var(--surface))", border: "1px solid #f59e0b40" }}>
              <div className="text-[12px] font-semibold" style={{ color: "#b45309" }}>
                {preview.warnings.length} warning{preview.warnings.length === 1 ? "" : "s"}
              </div>
              <ul className="text-[11px] mt-1 list-disc ml-4" style={{ color: "var(--ink-3)" }}>
                {preview.warnings.slice(0, 20).map((w, i) => <li key={i}>{w}</li>)}
                {preview.warnings.length > 20 && (
                  <li style={{ color: "var(--ink-4)" }}>… and {preview.warnings.length - 20} more</li>
                )}
              </ul>
            </div>
          )}

          {/* Classification breakdown */}
          <div className="rounded-[12px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="px-3 py-2.5 text-[12px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
              Row classification
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 text-[11px]">
              {[
                ["Equity trades", preview.counts.equity_trade || 0, "#08a86b"],
                ["Option trades", preview.counts.option_trade || 0, "#8b5cf6"],
                ["Option expirations", preview.counts.option_expire || 0, "#a78bfa"],
                ["Cash deposits", preview.counts.cash_deposit || 0, "#0ea5e9"],
                ["Income (skipped)", preview.counts.income || 0, "#64748b"],
                ["Fees (skipped)", preview.counts.fee || 0, "#64748b"],
                ["Short options (skipped)", preview.counts.short_option_trade || 0, "#f59e0b"],
                ["Unknown (skipped)", preview.counts.unknown || 0, "#e5484d"],
              ].map(([label, count, color]) => (
                <div key={label as string} className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ background: color as string }} />
                  <span style={{ color: "var(--ink-3)" }}>{label as string}</span>
                  <span className="ml-auto font-semibold" style={{ fontFamily: mono }}>{count as number}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Campaigns preview */}
          {totalCampaigns > 0 && (
            <div className="rounded-[12px] overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="px-3 py-2.5 text-[12px] font-semibold" style={{ borderBottom: "1px solid var(--border)" }}>
                Campaigns to write · {totalCampaigns}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr>
                      <th className="text-left px-3 py-2" style={{ color: "var(--ink-4)" }}>Ticker</th>
                      <th className="text-left px-3 py-2" style={{ color: "var(--ink-4)" }}>Type</th>
                      <th className="text-left px-3 py-2" style={{ color: "var(--ink-4)" }}>Open</th>
                      <th className="text-right px-3 py-2" style={{ color: "var(--ink-4)" }}>Buys</th>
                      <th className="text-right px-3 py-2" style={{ color: "var(--ink-4)" }}>Sells</th>
                      <th className="text-right px-3 py-2" style={{ color: "var(--ink-4)" }}>Remaining</th>
                      <th className="text-left px-3 py-2" style={{ color: "var(--ink-4)" }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...preview.equity_campaigns, ...preview.option_campaigns].map((c, i) => (
                      <tr key={`${c.ticker}-${c.open_date}-${i}`} style={{ borderTop: "1px solid var(--border)" }}>
                        <td className="px-3 py-1.5 font-semibold" style={{ fontFamily: mono }}>{c.ticker}</td>
                        <td className="px-3 py-1.5" style={{ color: c.instrument_type === "OPTION" ? "#8b5cf6" : "#08a86b" }}>
                          {c.instrument_type}
                        </td>
                        <td className="px-3 py-1.5" style={{ fontFamily: mono, color: "var(--ink-3)" }}>{c.open_date}</td>
                        <td className="px-3 py-1.5 text-right" style={{ fontFamily: mono }}>{c.buys}</td>
                        <td className="px-3 py-1.5 text-right" style={{ fontFamily: mono }}>{c.sells}</td>
                        <td className="px-3 py-1.5 text-right" style={{ fontFamily: mono }}>{c.shares_remaining}</td>
                        <td className="px-3 py-1.5">
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                                style={{
                                  background: c.status === "OPEN"
                                    ? "color-mix(in oklab, #08a86b 12%, var(--surface))"
                                    : "color-mix(in oklab, #64748b 12%, var(--surface))",
                                  color: c.status === "OPEN" ? "#16a34a" : "#64748b",
                                }}>
                            {c.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function SummaryTile({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <div className="rounded-[12px] p-3"
         style={{ background: "var(--surface)", border: `1px solid color-mix(in oklab, ${accent} 24%, var(--border))` }}>
      <div className="text-[9px] uppercase tracking-[0.08em] font-semibold" style={{ color: "var(--ink-4)" }}>{label}</div>
      <div className="text-[22px] font-semibold mt-0.5" style={{ color: accent, fontFamily: mono }}>
        {value}
      </div>
    </div>
  );
}
