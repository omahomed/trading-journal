"use client";

import { useState, useEffect, useRef } from "react";

// Same trigger-button styling as the inline form inputs used in Log Buy
// and Log Sell. Inlined here (not imported) so the search-select file
// is self-contained — the constants are a 6-line snippet, and the
// alternative (extracting them to a shared file too) would broaden the
// blast radius of what was meant to be a mechanical component move.
const inputCls = "w-full h-[42px] px-3.5 rounded-[10px] text-[13px] outline-none transition-colors";
const inputStyle: React.CSSProperties = {
  background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
  fontFamily: "var(--font-jetbrains), monospace",
};

// Structured option for SearchSelect. Lets callers attach a leading visual
// (e.g. a color swatch for the Strategy dropdown) without forking the
// component. `value` is what gets passed to onChange and matched against the
// current selection; `label` is what's rendered if different from value.
// Backwards compatible: callers that pass `string[]` for `options` continue
// to work unchanged — the component normalizes plain strings into structured
// form internally.
export type SearchSelectOption =
  | string
  | { value: string; label?: string; renderPrefix?: () => React.ReactNode };

export function SearchSelect({ value, onChange, options, placeholder, disabled }: {
  value: string; onChange: (v: string) => void;
  // readonly accepted so `as const` arrays from @/lib/trade-rules
  // (BUY_RULE_LABELS, SELL_RULE_LABELS) pass without spreading.
  options: readonly SearchSelectOption[]; placeholder?: string; disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Normalize plain-string options into structured form so the render loop
  // below has one shape to deal with.
  const normalized = options.map(o =>
    typeof o === "string" ? { value: o, label: o } : { label: o.value, ...o }
  );

  const filtered = search
    ? normalized.filter(o =>
        (o.label ?? o.value).toLowerCase().includes(search.toLowerCase())
        || o.value.toLowerCase().includes(search.toLowerCase())
      )
    : normalized;

  // The trigger button's leading visual mirrors the selected option's
  // renderPrefix so the swatch (or any future prefix) stays visible after
  // selection. Falls back to nothing when the current value isn't found.
  const selectedOpt = normalized.find(o => o.value === value);

  return (
    <div ref={ref} className="relative">
      <button type="button" onClick={() => { if (!disabled) setOpen(!open); }}
              disabled={disabled}
              className={inputCls + " flex items-center justify-between text-left"}
              style={{
                ...inputStyle, fontFamily: "inherit",
                cursor: disabled ? "not-allowed" : "pointer",
                opacity: disabled ? 0.6 : 1,
              }}>
        <span className="flex items-center gap-2 min-w-0" style={{ opacity: value ? 1 : 0.5 }}>
          {selectedOpt?.renderPrefix?.()}
          <span className="truncate">{selectedOpt?.label ?? value ?? placeholder ?? "Select..."}</span>
        </span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2"><path d="M6 9l6 6 6-6"/></svg>
      </button>
      {open && !disabled && (
        <div className="absolute z-50 mt-1 w-full rounded-[10px] overflow-hidden shadow-lg"
             style={{ background: "var(--surface)", border: "1px solid var(--border)", maxHeight: 280 }}>
          <div className="p-2" style={{ borderBottom: "1px solid var(--border)" }}>
            <input type="text" value={search} onChange={e => setSearch(e.target.value)}
                   placeholder="Type to search..." autoFocus
                   className="w-full h-[34px] px-3 rounded-[8px] text-[12px] outline-none"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: 220 }}>
            {filtered.map(o => (
              <button key={o.value} type="button"
                      onClick={() => { onChange(o.value); setOpen(false); setSearch(""); }}
                      className="w-full text-left px-3 py-2 text-[12px] transition-colors hover:brightness-95 flex items-center gap-2"
                      style={{ background: o.value === value ? "var(--surface-2)" : "transparent", color: "var(--ink)" }}>
                {o.renderPrefix?.()}
                <span className="truncate">{o.label ?? o.value}</span>
              </button>
            ))}
            {filtered.length === 0 && (
              <div className="px-3 py-3 text-[12px] text-center" style={{ color: "var(--ink-4)" }}>No matches</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default SearchSelect;
