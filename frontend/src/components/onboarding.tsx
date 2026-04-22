"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";

export function Onboarding() {
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const { refetch } = usePortfolio();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setSubmitting(true);
    setErr(null);
    try {
      const result = await api.createPortfolio(trimmed);
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
          Create your first portfolio to get started. Think of it as an account &mdash;
          for example, <em>Main</em>, <em>IRA</em>, or <em>Growth</em>. You can add more later.
        </p>
        <form onSubmit={handleSubmit} className="space-y-3">
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
