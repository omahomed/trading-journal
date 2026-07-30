"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Link from "next/link";
import { api, type TickerTaxonomy } from "@/lib/api";
import { log } from "@/lib/log";

interface Props {
  navColor: string;
}

interface EditState {
  ticker: string;
  sector: string;
  theme: string;
  notes: string;
  suggestion?: { sector: string; industry: string };
  saving: boolean;
  error?: string;
}

/** Draws its autocomplete vocabulary from the user's own prior values so
 * they build up a controlled set as they classify (Technology / Memory
 * shows up as a suggestion the moment they've used it once). */
function useAutocomplete(rows: TickerTaxonomy[]): { sectors: string[]; themes: string[] } {
  return useMemo(() => {
    const sectors = new Set<string>();
    const themes = new Set<string>();
    for (const r of rows) {
      if (r.sector) sectors.add(r.sector);
      if (r.theme) themes.add(r.theme);
    }
    return {
      sectors: Array.from(sectors).sort(),
      themes: Array.from(themes).sort(),
    };
  }, [rows]);
}

export function SectorMapping({ navColor }: Props) {
  const [mapped, setMapped] = useState<TickerTaxonomy[]>([]);
  const [unmapped, setUnmapped] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [edit, setEdit] = useState<EditState | null>(null);
  const { sectors: sectorSuggestions, themes: themeSuggestions } = useAutocomplete(mapped);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await api.taxonomyList();
      setMapped(res.mapped ?? []);
      setUnmapped(res.unmapped ?? []);
    } catch (e) {
      log.error("sector-mapping", "load failed", e);
      setLoadError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const openEditor = useCallback(async (ticker: string, existing?: TickerTaxonomy) => {
    const initial: EditState = {
      ticker,
      sector: existing?.sector ?? "",
      theme: existing?.theme ?? "",
      notes: existing?.notes ?? "",
      saving: false,
    };
    setEdit(initial);
    // Fire-and-forget yfinance suggestion. When editing an existing
    // mapping the user's values win — the hint only shows as reference.
    // When creating a NEW mapping (no existing row), pre-fill the empty
    // inputs so most tickers save with one click. sector ← yfinance
    // sector; theme ← yfinance industry as a starter (SNDK's "Computer
    // Hardware" you'll correct, but MU's "Semiconductors" saves as-is).
    try {
      const s = await api.taxonomySuggest(ticker);
      setEdit((cur) => {
        if (!cur || cur.ticker !== ticker) return cur;
        const suggestion = { sector: s.sector ?? "", industry: s.industry ?? "" };
        const prefill = !existing;
        return {
          ...cur,
          suggestion,
          sector: prefill && !cur.sector && suggestion.sector ? suggestion.sector : cur.sector,
          theme:  prefill && !cur.theme  && suggestion.industry ? suggestion.industry : cur.theme,
        };
      });
    } catch {
      // yfinance may not know this ticker (ETFs). Silent — not an error.
    }
  }, []);

  const save = useCallback(async () => {
    if (!edit) return;
    if (!edit.sector.trim()) {
      setEdit({ ...edit, error: "Sector is required." });
      return;
    }
    setEdit({ ...edit, saving: true, error: undefined });
    try {
      const res = await api.taxonomyUpsert(edit.ticker, {
        sector: edit.sector.trim(),
        theme: edit.theme.trim() || undefined,
        notes: edit.notes.trim() || undefined,
      });
      if ("error" in res) throw new Error(res.error);
      setEdit(null);
      await refresh();
    } catch (e) {
      log.error("sector-mapping", "save failed", e);
      setEdit((cur) => cur ? { ...cur, saving: false, error: String(e) } : cur);
    }
  }, [edit, refresh]);

  const remove = useCallback(async (ticker: string) => {
    if (!confirm(`Remove ${ticker} mapping? It'll return to the Unmapped list.`)) return;
    try {
      await api.taxonomyDelete(ticker);
      await refresh();
    } catch (e) {
      log.error("sector-mapping", "delete failed", e);
    }
  }, [refresh]);

  const mappedByTicker = useMemo(() => {
    const m = new Map<string, TickerTaxonomy>();
    for (const r of mapped) m.set(r.ticker, r);
    return m;
  }, [mapped]);

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <header className="mb-6">
        <div className="flex items-center gap-3">
          <span className="w-2 h-6 rounded-sm" style={{ background: navColor }} />
          <h1 className="text-2xl font-semibold" style={{ color: "var(--ink-1)" }}>Sector Mapping</h1>
        </div>
        <p className="mt-2 text-[13px]" style={{ color: "var(--ink-4)" }}>
          Your own sector + theme classification per ticker. Powers{" "}
          <Link href="/concentration-risk" style={{ color: navColor }}>Concentration Risk</Link>{" "}
          rollups. yfinance shows up only as a hint — its taxonomy is unreliable for storage/memory,
          spinoffs, and ETFs.
        </p>
      </header>

      {loadError && (
        <div className="mb-4 p-3 rounded-lg text-[13px]" style={{ background: "#fee", color: "#c00", border: "1px solid #fbb" }}>
          {loadError}
        </div>
      )}

      {loading ? (
        <div className="text-[13px]" style={{ color: "var(--ink-4)" }}>Loading…</div>
      ) : (
        <>
          {/* ─── Unmapped section ─── */}
          <section className="mb-8">
            <div className="flex items-center gap-2 mb-3">
              <h2 className="text-[15px] font-semibold" style={{ color: unmapped.length > 0 ? "#e5484d" : "var(--ink-3)" }}>
                {unmapped.length > 0 ? "⚠" : "✓"} Unmapped ({unmapped.length})
              </h2>
              {unmapped.length > 0 && (
                <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>
                  Every ticker you&apos;ve traded that isn&apos;t yet classified.
                </span>
              )}
            </div>
            {unmapped.length === 0 ? (
              <div className="text-[13px] p-4 rounded-lg text-center" style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                Every ticker in your book is classified. Nice.
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {unmapped.map((t) => (
                  <button
                    key={t}
                    onClick={() => openEditor(t)}
                    className="px-3 py-1.5 rounded-md text-[13px] font-medium transition-colors hover:bg-[var(--surface-2)]"
                    style={{ background: "var(--surface)", border: "1px dashed #e5484d", color: "var(--ink-2)" }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            )}
          </section>

          {/* ─── Mapped section ─── */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <h2 className="text-[15px] font-semibold" style={{ color: "var(--ink-2)" }}>
                ✓ Mapped ({mapped.length})
              </h2>
            </div>
            {mapped.length === 0 ? (
              <div className="text-[13px] p-4 rounded-lg text-center" style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                No mappings yet. Click an unmapped ticker above to start.
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg" style={{ border: "1px solid var(--border)" }}>
                <table className="w-full text-[13px]">
                  <thead>
                    <tr style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                      <th className="px-3 py-2 text-left font-medium">Ticker</th>
                      <th className="px-3 py-2 text-left font-medium">Sector</th>
                      <th className="px-3 py-2 text-left font-medium">Theme</th>
                      <th className="px-3 py-2 text-left font-medium">Notes</th>
                      <th className="px-3 py-2 text-left font-medium">Updated</th>
                      <th className="px-3 py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {mapped.map((r) => (
                      <tr key={r.ticker} style={{ borderTop: "1px solid var(--border)" }}>
                        <td className="px-3 py-2 font-semibold" style={{ color: "var(--ink-1)" }}>{r.ticker}</td>
                        <td className="px-3 py-2" style={{ color: "var(--ink-2)" }}>{r.sector}</td>
                        <td className="px-3 py-2" style={{ color: "var(--ink-2)" }}>{r.theme || "—"}</td>
                        <td className="px-3 py-2 max-w-xs truncate" style={{ color: "var(--ink-4)" }} title={r.notes || ""}>
                          {r.notes || ""}
                        </td>
                        <td className="px-3 py-2" style={{ color: "var(--ink-4)" }}>
                          {new Date(r.updated_at).toLocaleDateString()}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-right">
                          <button
                            onClick={() => openEditor(r.ticker, r)}
                            className="text-[12px] px-2 py-1 rounded hover:bg-[var(--surface-2)]"
                            style={{ color: navColor }}
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => remove(r.ticker)}
                            className="text-[12px] px-2 py-1 rounded hover:bg-[var(--surface-2)] ml-1"
                            style={{ color: "var(--ink-4)" }}
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      {/* ─── Edit modal ─── */}
      {edit && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(0,0,0,0.4)" }}
          onClick={(e) => { if (e.target === e.currentTarget) setEdit(null); }}
        >
          <div className="rounded-xl p-6 max-w-md w-full shadow-2xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="flex items-baseline justify-between mb-4">
              <h3 className="text-lg font-semibold" style={{ color: "var(--ink-1)" }}>
                {mappedByTicker.has(edit.ticker) ? "Edit" : "Classify"} {edit.ticker}
              </h3>
              <button onClick={() => setEdit(null)} className="text-[20px] leading-none" style={{ color: "var(--ink-4)" }}>×</button>
            </div>
            {edit.suggestion && (edit.suggestion.sector || edit.suggestion.industry) && (
              <div className="mb-4 p-2.5 rounded text-[12px]" style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                <span className="opacity-70">yfinance suggests:</span>{" "}
                <span style={{ color: "var(--ink-3)" }}>
                  {edit.suggestion.sector || "—"} / {edit.suggestion.industry || "—"}
                </span>
              </div>
            )}

            <label className="block mb-3">
              <span className="text-[12px] font-medium block mb-1" style={{ color: "var(--ink-3)" }}>Sector *</span>
              <input
                list="sector-suggestions"
                value={edit.sector}
                onChange={(e) => setEdit({ ...edit, sector: e.target.value })}
                placeholder="Technology"
                className="w-full px-2.5 py-1.5 rounded text-[13px]"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}
                autoFocus
              />
              <datalist id="sector-suggestions">
                {sectorSuggestions.map((s) => <option key={s} value={s} />)}
              </datalist>
            </label>

            <label className="block mb-3">
              <span className="text-[12px] font-medium block mb-1" style={{ color: "var(--ink-3)" }}>Theme</span>
              <input
                list="theme-suggestions"
                value={edit.theme}
                onChange={(e) => setEdit({ ...edit, theme: e.target.value })}
                placeholder="Memory, Semis, AI Infra, Leveraged Index…"
                className="w-full px-2.5 py-1.5 rounded text-[13px]"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}
              />
              <datalist id="theme-suggestions">
                {themeSuggestions.map((t) => <option key={t} value={t} />)}
              </datalist>
            </label>

            <label className="block mb-4">
              <span className="text-[12px] font-medium block mb-1" style={{ color: "var(--ink-3)" }}>Notes</span>
              <textarea
                value={edit.notes}
                onChange={(e) => setEdit({ ...edit, notes: e.target.value })}
                rows={2}
                placeholder="Optional — e.g. 'NAND competitor to MU, moves with memory cycle'"
                className="w-full px-2.5 py-1.5 rounded text-[13px]"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}
              />
            </label>

            {edit.error && (
              <div className="mb-3 text-[12px]" style={{ color: "#e5484d" }}>{edit.error}</div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setEdit(null)}
                className="px-3 py-1.5 rounded text-[13px]"
                style={{ background: "var(--surface-2)", color: "var(--ink-2)" }}
              >
                Cancel
              </button>
              <button
                onClick={save}
                disabled={edit.saving}
                className="px-3 py-1.5 rounded text-[13px] font-medium"
                style={{ background: navColor, color: "white", opacity: edit.saving ? 0.5 : 1 }}
              >
                {edit.saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
