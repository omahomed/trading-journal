// SR8 Declaration modal — the "promote SR7 → SR8" flow (Migration 062).
//
// The user right-clicks a cushion-qualified (SR7) row in ACS and picks
// "Declare SR8". This modal opens with two reference values so the
// user can make an informed core-share pick:
//
//   * Doctrine 15% reference: 15% × activation NLV ÷ activation price.
//     The "textbook" core size per §1 governing doctrine. May be
//     larger or smaller than the user's current holding.
//   * Current holding: whatever shares the user is holding right now.
//
// The user types the core they want to defend. Submit calls
// api.declareSR8 with the typed value; a successful response flips
// the tier from SR7 → SR8 in the parent list on next refetch.
//
// The Demote flow is a one-click confirm on the right-click menu; no
// modal needed. Anchors + sr8_core_shares persist as historical audit
// per Q4 of the design (see Phase A commit).

"use client";

import { useEffect, useMemo, useState } from "react";
import type { EnrichedPosition } from "@/lib/positions";
import { api } from "@/lib/api";
import { log } from "@/lib/log";

interface Props {
  position: EnrichedPosition;
  portfolio: string;
  /** Called on success — parent should refetch the campaign list so the
   *  tier badge, header counter chip, and Cascade Monitor visibility
   *  all reflect the new is_declared_sr8=TRUE state. */
  onSuccess: () => void;
  onClose: () => void;
}

const mono = "var(--font-jetbrains), monospace";

function fmtMoney(n: number, precision = 2): string {
  const abs = Math.abs(n);
  return `${n < 0 ? "-" : ""}$${abs.toLocaleString(undefined, {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  })}`;
}

function fmtShares(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function SR8DeclareModal({ position, portfolio, onSuccess, onClose }: Props) {
  // Doctrine reference: 15% × activation NLV ÷ activation price. Activation
  // price = the first BUY fill (b1_entry_price). Falls back to current
  // avg_entry when the activation anchor hasn't been stamped yet (rare —
  // the +50% crossing trigger stamps the anchor before this modal is
  // reachable, but defensive).
  const doctrineCore = useMemo(() => {
    const nav = position.sr8_activation_nlv ?? null;
    // b1_entry_price isn't on EnrichedPosition directly; approximate via
    // avg_entry which for pre-add campaigns equals the B1 fill and for
    // scaled-in campaigns is close enough for a display reference.
    const price = position.avg_entry;
    if (nav == null || !Number.isFinite(nav) || nav <= 0) return null;
    if (!Number.isFinite(price) || price <= 0) return null;
    return Math.floor((nav * 0.15) / price);
  }, [position.sr8_activation_nlv, position.avg_entry]);

  const currentShares = position.shares || 0;

  const [coreShares, setCoreShares] = useState<string>(String(currentShares));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const coreNum = parseFloat(coreShares);
  const coreValid = Number.isFinite(coreNum) && coreNum > 0;

  const submit = async () => {
    if (!coreValid || saving) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.declareSR8(position.trade_id, portfolio, coreNum);
      if ("detail" in res) {
        throw new Error(res.detail);
      }
      onSuccess();
      onClose();
    } catch (e) {
      log.error("sr8-declare", "declare failed", e);
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.5)" }}
      onClick={e => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div
        className="rounded-[14px] w-full max-w-[520px] flex flex-col"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.35)",
        }}
        role="dialog"
        aria-modal="true"
        aria-label={`Declare SR8 for ${position.ticker}`}
      >
        <div
          className="px-5 py-4 flex items-end justify-between gap-4"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <h2
            className="font-normal text-[22px] tracking-tight m-0"
            style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}
          >
            Declare{" "}
            <em className="italic" style={{ color: "#16a34a" }}>SR8</em>
            {" — "}
            <span style={{ fontFamily: mono, color: "var(--ink-2)" }}>
              {position.ticker}
            </span>
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-[20px] leading-none px-2 py-1 rounded-[8px]"
            style={{
              background: "transparent",
              color: "var(--ink-3)",
              border: "1px solid var(--border)",
            }}
          >
            ×
          </button>
        </div>

        <div className="p-5 space-y-4">
          <p className="text-[13px]" style={{ color: "var(--ink-3)" }}>
            Promoting this campaign to SR8 locks the weekly MO RS funnel
            ladder and freezes a core share count. Anchors (activation
            date + NLV) are already stamped from the +50% crossing; only
            the core count is your call here. You can demote back to SR7
            any time — anchors persist as historical audit.
          </p>

          <div
            className="rounded-[10px] p-3"
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
            }}
          >
            <div
              className="text-[11px] uppercase tracking-wider mb-2"
              style={{ color: "var(--ink-3)" }}
            >
              Reference values
            </div>
            <div className="grid grid-cols-2 gap-3 text-[12px]">
              <div>
                <div style={{ color: "var(--ink-3)" }}>Current holding</div>
                <div
                  className="text-[16px] font-semibold mt-1"
                  style={{ fontFamily: mono, color: "var(--ink-1)" }}
                >
                  {fmtShares(currentShares)} sh
                </div>
              </div>
              <div>
                <div style={{ color: "var(--ink-3)" }}>
                  Doctrine 15% × activation NLV
                </div>
                <div
                  className="text-[16px] font-semibold mt-1"
                  style={{ fontFamily: mono, color: "var(--ink-1)" }}
                >
                  {doctrineCore != null
                    ? `${fmtShares(doctrineCore)} sh`
                    : "—"}
                </div>
                {position.sr8_activation_nlv != null && (
                  <div
                    className="text-[10px] mt-1"
                    style={{ color: "var(--ink-3)" }}
                  >
                    NLV {fmtMoney(position.sr8_activation_nlv, 0)}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div>
            <label
              htmlFor="sr8-core-input"
              className="text-[12px] font-semibold block mb-1.5"
              style={{ color: "var(--ink-2)" }}
            >
              Core share count to defend
            </label>
            <input
              id="sr8-core-input"
              type="number"
              step="1"
              min="0"
              autoFocus
              value={coreShares}
              onChange={e => setCoreShares(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") submit();
              }}
              className="w-full px-3 py-2 rounded-[8px] text-[14px]"
              style={{
                fontFamily: mono,
                background: "var(--bg)",
                border: "1px solid var(--border)",
                color: "var(--ink-1)",
              }}
            />
            <div
              className="text-[11px] mt-1.5"
              style={{ color: "var(--ink-3)" }}
            >
              This is the fixed share count SR8&apos;s funnel ladder will
              defend. Trims never take you below this floor without an
              explicit demote.
            </div>
          </div>

          {error && (
            <div
              className="px-3 py-2 rounded-[8px] text-[12px]"
              style={{
                background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
                border: "1px solid var(--border)",
                color: "#e5484d",
              }}
            >
              {error}
            </div>
          )}
        </div>

        <div
          className="px-5 py-3 flex justify-end gap-2"
          style={{ borderTop: "1px solid var(--border)" }}
        >
          <button
            onClick={onClose}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              color: "var(--ink-2)",
            }}
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!coreValid || saving}
            className="px-3 py-2 rounded-[10px] text-[13px]"
            style={{
              background: "#16a34a",
              border: "1px solid #16a34a",
              color: "white",
              opacity: !coreValid || saving ? 0.6 : 1,
              cursor: !coreValid || saving ? "not-allowed" : "pointer",
            }}
          >
            {saving ? "Declaring…" : "Declare SR8"}
          </button>
        </div>
      </div>
    </div>
  );
}
