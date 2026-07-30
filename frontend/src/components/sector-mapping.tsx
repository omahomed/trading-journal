"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Link from "next/link";
import { api, type TickerTaxonomy } from "@/lib/api";
import { log } from "@/lib/log";
import { KPITile, TILE_GRADIENTS } from "./campaign-detail";

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
 * they build up a controlled set as they classify. */
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
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [edit, setEdit] = useState<EditState | null>(null);
  const { sectors: sectorSuggestions, themes: themeSuggestions } = useAutocomplete(mapped);

  const refresh = useCallback(async (opts?: { manual?: boolean }) => {
    if (opts?.manual) setRefreshing(true); else setLoading(true);
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
      setRefreshing(false);
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
      // yfinance may not know this ticker (ETFs) — silent.
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
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      {/* Page header */}
      <div className="mb-[22px] pb-[14px] flex items-end justify-between gap-4"
           style={{ borderBottom: "1px solid var(--border)" }}>
        <div>
          <h1 className="font-normal text-[32px] tracking-tight m-0"
              style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            Sector <em className="italic" style={{ color: navColor }}>Mapping</em>
          </h1>
          <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
            Your own sector + theme classification per ticker. Powers{" "}
            <Link href="/concentration-risk" style={{ color: navColor }}>Concentration Risk</Link>{" "}
            rollups · yfinance is a hint only
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            type="button" onClick={() => refresh({ manual: true })} disabled={refreshing}
            className="px-3 py-2 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors"
            style={{ background: "var(--surface)", border: "1px solid var(--border)",
                     color: refreshing ? "var(--ink-4)" : "var(--ink-2)" }}
          >
            ⟳ {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {loadError && (
        <div className="mb-4 px-4 py-3 rounded-[10px]"
             style={{ background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
                      border: "1px solid var(--border)", color: "#e5484d" }}>
          Failed to load: {loadError}
        </div>
      )}

      {/* KPI strip */}
      {loading ? (
        <div className="grid grid-cols-3 gap-[14px]">
          {[0, 1, 2].map(i => (
            <div key={i} className="rounded-[14px] animate-pulse min-h-[108px]"
                 style={{ background: "var(--bg-2)" }} />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-[14px]">
            <KPITile
              label="Tickers Classified"
              value={String(mapped.length)}
              sub={`${new Set(mapped.map(r => r.sector)).size} distinct sector${new Set(mapped.map(r => r.sector)).size === 1 ? "" : "s"}`}
              gradient={TILE_GRADIENTS.indigo}
            />
            <KPITile
              label="Themes"
              value={String(new Set(mapped.filter(r => r.theme).map(r => r.theme)).size)}
              sub={mapped.filter(r => r.theme).length + " ticker" + (mapped.filter(r => r.theme).length === 1 ? "" : "s") + " with a theme"}
              gradient={TILE_GRADIENTS.blue}
            />
            <KPITile
              label="Unmapped"
              value={String(unmapped.length)}
              sub={unmapped.length === 0 ? "everything classified" : "need attention"}
              gradient={unmapped.length > 0 ? TILE_GRADIENTS.red : TILE_GRADIENTS.green}
            />
          </div>

          {/* ─── Unmapped section ─── */}
          <section className="mt-6">
            <div className="flex items-center gap-2 mb-3">
              <div className="text-[13px] font-semibold" style={{ color: unmapped.length > 0 ? "#e5484d" : "var(--ink-3)" }}>
                {unmapped.length > 0 ? "⚠" : "✓"} Unmapped ({unmapped.length})
              </div>
              {unmapped.length > 0 && (
                <span className="text-[12px]" style={{ color: "var(--ink-4)" }}>
                  Every ticker you&apos;ve traded that isn&apos;t yet classified · click to open the editor
                </span>
              )}
            </div>
            {unmapped.length === 0 ? (
              <div className="text-[13px] p-4 rounded-[14px] text-center"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
                Every ticker in your book is classified. Nice.
              </div>
            ) : (
              <div className="rounded-[14px] p-[18px] flex flex-wrap gap-2"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)",
                            boxShadow: "var(--card-shadow)" }}>
                {unmapped.map((t) => (
                  <button
                    key={t}
                    onClick={() => openEditor(t)}
                    className="px-3 py-1.5 rounded-[8px] text-[13px] font-medium transition-colors hover:brightness-95"
                    style={{ background: "var(--surface-2)", border: "1px dashed #e5484d",
                             color: "var(--ink-2)", fontFamily: "var(--font-jetbrains), monospace" }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            )}
          </section>

          {/* ─── Mapped section ─── */}
          <section className="mt-6">
            <div className="flex items-center gap-2 mb-3">
              <div className="text-[13px] font-semibold" style={{ color: "var(--ink-2)" }}>
                ✓ Mapped ({mapped.length})
              </div>
            </div>
            {mapped.length === 0 ? (
              <div className="text-[13px] p-4 rounded-[14px] text-center"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
                No mappings yet. Click an unmapped ticker above to start.
              </div>
            ) : (
              <div className="rounded-[14px] overflow-hidden"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)",
                            boxShadow: "var(--card-shadow)" }}>
                <table className="w-full text-[13px]">
                  <thead>
                    <tr style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
                      <th className="px-3 py-2.5 text-left font-medium">Ticker</th>
                      <th className="px-3 py-2.5 text-left font-medium">Sector</th>
                      <th className="px-3 py-2.5 text-left font-medium">Theme</th>
                      <th className="px-3 py-2.5 text-left font-medium">Notes</th>
                      <th className="px-3 py-2.5 text-left font-medium">Updated</th>
                      <th className="px-3 py-2.5"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {mapped.map((r) => (
                      <tr key={r.ticker} style={{ borderTop: "1px solid var(--border)" }}>
                        <td className="px-3 py-2.5 font-semibold"
                            style={{ color: "var(--ink-1)", fontFamily: "var(--font-jetbrains), monospace" }}>
                          {r.ticker}
                        </td>
                        <td className="px-3 py-2.5" style={{ color: "var(--ink-2)" }}>{r.sector}</td>
                        <td className="px-3 py-2.5" style={{ color: "var(--ink-2)" }}>{r.theme || "—"}</td>
                        <td className="px-3 py-2.5 max-w-xs truncate" style={{ color: "var(--ink-4)" }} title={r.notes || ""}>
                          {r.notes || ""}
                        </td>
                        <td className="px-3 py-2.5" style={{ color: "var(--ink-4)" }}>
                          {new Date(r.updated_at).toLocaleDateString()}
                        </td>
                        <td className="px-3 py-2.5 whitespace-nowrap text-right">
                          <button
                            onClick={() => openEditor(r.ticker, r)}
                            className="text-[12px] px-2 py-1 rounded-[6px] hover:bg-[var(--surface-2)]"
                            style={{ color: navColor }}
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => remove(r.ticker)}
                            className="text-[12px] px-2 py-1 rounded-[6px] hover:bg-[var(--surface-2)] ml-1"
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
          <div className="rounded-[14px] p-6 max-w-md w-full"
               style={{ background: "var(--surface)", border: "1px solid var(--border)",
                        boxShadow: "0 20px 60px rgba(0,0,0,0.2)" }}>
            <div className="flex items-baseline justify-between mb-4">
              <h3 className="font-normal text-[22px] tracking-tight m-0"
                  style={{ fontFamily: "var(--font-fraunces), Georgia, serif", color: "var(--ink-1)" }}>
                {mappedByTicker.has(edit.ticker) ? "Edit" : "Classify"}{" "}
                <em className="italic" style={{ color: navColor, fontFamily: "var(--font-jetbrains), monospace" }}>
                  {edit.ticker}
                </em>
              </h3>
              <button onClick={() => setEdit(null)} className="text-[24px] leading-none"
                      style={{ color: "var(--ink-4)" }}>×</button>
            </div>
            {edit.suggestion && (edit.suggestion.sector || edit.suggestion.industry) && (
              <div className="mb-4 p-2.5 rounded-[8px] text-[12px]"
                   style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}>
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
                className="w-full px-3 py-2 rounded-[8px] text-[13px]"
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
                className="w-full px-3 py-2 rounded-[8px] text-[13px]"
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
                className="w-full px-3 py-2 rounded-[8px] text-[13px]"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-1)" }}
              />
            </label>

            {edit.error && (
              <div className="mb-3 text-[12px]" style={{ color: "#e5484d" }}>{edit.error}</div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setEdit(null)}
                className="px-3 py-2 rounded-[10px] text-[13px]"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--ink-2)" }}
              >
                Cancel
              </button>
              <button
                onClick={save}
                disabled={edit.saving}
                className="px-3 py-2 rounded-[10px] text-[13px] font-medium"
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
