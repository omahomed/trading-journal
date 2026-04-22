"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";

function todayIso() {
  const n = new Date();
  return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
}

export function Onboarding() {
  const [name, setName] = useState("");
  const [capital, setCapital] = useState("");
  const [resetDate, setResetDate] = useState(todayIso());
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const { refetch } = usePortfolio();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setSubmitting(true);
    setErr(null);
    const capitalNum = capital.trim() ? parseFloat(capital) : null;
    if (capital.trim() && (!isFinite(capitalNum as number) || (capitalNum as number) < 0)) {
      setErr("Starting capital must be a non-negative number");
      setSubmitting(false);
      return;
    }
    try {
      const result = await api.createPortfolio({
        name: trimmed,
        starting_capital: capitalNum,
        reset_date: resetDate || null,
      });
      if ("error" in result) {
        setErr(result.error);
        return;
      }
      await refetch();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: "var(--bg)" }}>
      <div
        className="w-full max-w-md bg-[var(--surface)] border border-[var(--border)] rounded-[14px] p-8"
        style={{ boxShadow: "0 1px 2px rgba(14,20,38,0.04), 0 0 0 1px rgba(14,20,38,0.04)" }}
      >
        <h1
          className="text-[28px] font-normal leading-tight"
          style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}
        >
          Welcome to <em className="italic" style={{ color: "#6366f1" }}>MO Trading</em>
        </h1>
        <p className="text-[13px] text-[var(--ink-3)] mt-3 mb-6 leading-relaxed">
          Create your first portfolio to get started. You can add more or edit these values later from Settings.
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <label className="block">
            <span className="text-[11px] uppercase tracking-[0.10em] text-[var(--ink-4)] font-semibold">
              Portfolio name
            </span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={50}
              placeholder="e.g. Main"
              autoFocus
              className="mt-1 w-full h-11 px-3 rounded-[10px] border border-[var(--border)] bg-[var(--bg)] text-sm focus:outline-none focus:border-[#6366f1] transition-colors"
            />
          </label>
          <label className="block">
            <span className="text-[11px] uppercase tracking-[0.10em] text-[var(--ink-4)] font-semibold">
              Starting capital <span className="normal-case font-normal text-[var(--ink-4)]">(optional)</span>
            </span>
            <div className="mt-1 relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--ink-4)] text-sm">$</span>
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(e.target.value)}
                placeholder="100000"
                min={0}
                step="0.01"
                className="w-full h-11 pl-7 pr-3 rounded-[10px] border border-[var(--border)] bg-[var(--bg)] text-sm focus:outline-none focus:border-[#6366f1] transition-colors"
                style={{ fontFamily: "var(--font-jetbrains), monospace" }}
              />
            </div>
            <span className="block mt-1 text-[11px] text-[var(--ink-4)]">
              Your baseline NLV — used for return and drawdown calculations.
            </span>
          </label>
          <label className="block">
            <span className="text-[11px] uppercase tracking-[0.10em] text-[var(--ink-4)] font-semibold">
              Reset date
            </span>
            <input
              type="date"
              value={resetDate}
              onChange={(e) => setResetDate(e.target.value)}
              className="mt-1 w-full h-11 px-3 rounded-[10px] border border-[var(--border)] bg-[var(--bg)] text-sm focus:outline-none focus:border-[#6366f1] transition-colors"
              style={{ fontFamily: "var(--font-jetbrains), monospace" }}
            />
            <span className="block mt-1 text-[11px] text-[var(--ink-4)]">
              Anchor for drawdown tracking — defaults to today.
            </span>
          </label>
          {err && (
            <div className="text-[12px] text-[#e5484d] bg-[#fdf2f2] border border-[#f5c2c2] rounded-[8px] px-3 py-2">
              {err}
            </div>
          )}
          <button
            type="submit"
            disabled={submitting || !name.trim()}
            className="w-full h-11 rounded-[10px] text-white font-medium text-sm disabled:opacity-50 transition-opacity"
            style={{ background: "#6366f1" }}
          >
            {submitting ? "Creating…" : "Create portfolio"}
          </button>
        </form>
      </div>
    </div>
  );
}
