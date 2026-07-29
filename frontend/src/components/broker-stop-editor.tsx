"use client";

// Shared broker_stop_price editor modal (migration 055 — SR14 flag).
// Used by:
//   * active-campaign.tsx — right-click context menu ("Set broker stop...")
//   * trade-manager.tsx — Edit Transaction sidecar (broker_stop_price row)
//   * trade-journal.tsx — inline card action (backfill after the fact)
//
// One source of truth for the input UX, validation, and save call so all
// three edit surfaces agree. See ARCHITECTURE.md §1 (trading_journal) for
// the broker_stop_price contract: presence promotes tier from SR1 → SR14
// in the <10% B1-return window; NULL means classic single-stop model.

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { formatCurrency } from "@/lib/format";
import { log } from "@/lib/log";

// Narrow position shape — accepts both EnrichedPosition (ACS) and a
// raw TradePosition + live-price overlay (Trade Journal), so callers
// don't have to build a full EnrichedPosition just to open the editor.
export interface BrokerStopEditorPosition {
  trade_id: string;
  ticker: string;
  shares: number;
  avg_entry: number;
  current_price?: number;
  broker_stop_price?: number | null;
}

interface BrokerStopEditorProps {
  position: BrokerStopEditorPosition;
  portfolio: string;
  onClose: () => void;
  /** Fired after a successful save OR clear. Parent typically triggers
   *  a data refresh here so the SR14 badge updates immediately. */
  onSaved: () => void;
}

export function BrokerStopEditor({ position, portfolio, onClose, onSaved }: BrokerStopEditorProps) {
  const currentPrice = position.current_price || 0;
  const avgEntry = position.avg_entry || 0;
  const existing = position.broker_stop_price ?? null;
  const [value, setValue] = useState(
    existing != null && existing > 0 ? existing.toFixed(2) : "",
  );
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !saving) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, saving]);

  const parsed = parseFloat(value);
  const parsedOk = value.trim() === "" || (Number.isFinite(parsed) && parsed > 0);
  const belowEntry = value.trim() === "" || (parsedOk && (avgEntry <= 0 || parsed < avgEntry));
  const canSave = parsedOk && belowEntry && !saving;
  // Empty value → clearing the flag (SR14 → SR1). Non-empty → setting/updating.
  const willClear = value.trim() === "";
  // Distance display — % from avg entry to the proposed broker stop.
  const distPct = parsedOk && !willClear && avgEntry > 0
    ? ((avgEntry - parsed) / avgEntry) * 100
    : null;

  const handleSave = async () => {
    setSaving(true);
    setErr(null);
    try {
      const res = await api.updateBrokerStop({
        portfolio,
        trade_id: position.trade_id,
        broker_stop_price: willClear ? null : parsed,
      });
      if (res.error || res.detail) {
        setErr(res.error || res.detail || "Save failed");
      } else {
        onSaved();
      }
    } catch (e: any) {
      log.error("broker-stop-editor", "updateBrokerStop failed", e);
      setErr(e?.message || "Save failed");
    }
    setSaving(false);
  };

  return (
    <div
      role="dialog"
      aria-label="Set broker stop"
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)",
        display: "grid", placeItems: "center", zIndex: 1000,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="rounded-[14px] p-6"
        style={{
          background: "var(--surface)", border: "1px solid var(--border)",
          maxWidth: 420, width: "90%",
          boxShadow: "0 12px 32px rgba(0,0,0,0.2)",
        }}
      >
        <div className="text-[16px] font-semibold mb-1">
          {existing != null ? "Edit" : "Set"} Broker Stop
        </div>
        <div className="text-[12px] mb-4" style={{ color: "var(--ink-3)" }}>
          {position.ticker} · {position.shares.toLocaleString()} shs · avg entry {formatCurrency(avgEntry)}
          {currentPrice > 0 && <> · now {formatCurrency(currentPrice)}</>}
        </div>

        <label className="block text-[11px] font-semibold uppercase tracking-[0.08em] mb-1.5"
               style={{ color: "var(--ink-4)" }}>
          Broker stop price (blank clears)
        </label>
        <input
          type="number"
          step="0.01"
          value={value}
          onChange={e => { setValue(e.target.value); setErr(null); }}
          placeholder={existing != null ? formatCurrency(existing) : "e.g. 248.44"}
          autoFocus
          disabled={saving}
          className="w-full h-[42px] px-3 rounded-[10px] text-[15px]"
          style={{
            background: "var(--bg)", border: "1px solid var(--border)",
            fontFamily: "var(--font-jetbrains), monospace",
          }}
        />

        {distPct !== null && (
          <div className="text-[11px] mt-1.5" style={{ color: "var(--ink-4)" }}>
            {distPct.toFixed(2)}% below avg entry
          </div>
        )}

        {!belowEntry && parsedOk && !willClear && (
          <div className="text-[12px] mt-2" style={{ color: "#dc2626" }}>
            Broker stop must be below avg entry ({formatCurrency(avgEntry)}) — a stop at or above fill fires immediately.
          </div>
        )}

        {err && (
          <div className="text-[12px] mt-2" style={{ color: "#dc2626" }}>
            {err}
          </div>
        )}

        <div className="mt-4 text-[11px]" style={{ color: "var(--ink-4)" }}>
          {willClear
            ? "Clearing removes the SR14 flag — the tier drops back to SR1 while B1 return < 10%."
            : "Setting flags the position as SR14 (0.75× ATR Stop) in the ACS Sell Rule column."}
        </div>

        <div className="mt-5 flex items-center gap-2 justify-end">
          <button
            onClick={onClose}
            disabled={saving}
            className="h-[36px] px-4 rounded-[10px] text-[12px] font-medium"
            style={{
              background: "var(--bg)", border: "1px solid var(--border)",
              color: "var(--ink-2)",
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!canSave}
            className="h-[36px] px-4 rounded-[10px] text-[12px] font-semibold text-white transition-all hover:brightness-110 disabled:opacity-50"
            style={{ background: willClear ? "#dc2626" : "var(--accent, #3b82f6)" }}
          >
            {saving
              ? "Saving…"
              : willClear
                ? "Clear broker stop"
                : existing != null
                  ? "Update"
                  : "Set broker stop"}
          </button>
        </div>
      </div>
    </div>
  );
}
