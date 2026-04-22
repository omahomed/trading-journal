"use client";

import { useState } from "react";
import { api, type Portfolio } from "@/lib/api";
import { usePortfolio } from "@/lib/portfolio-context";

export function Settings({ navColor }: { navColor: string }) {
  const { portfolios, refetch, loading } = usePortfolio();
  const [creating, setCreating] = useState(false);

  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0"
            style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Account <em className="italic" style={{ color: navColor }}>Settings</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Manage your portfolios
        </div>
      </div>

      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[16px] font-semibold" style={{ color: "var(--ink)" }}>Portfolios</h2>
        <button
          onClick={() => setCreating(true)}
          className="h-[34px] px-4 rounded-[10px] text-white font-medium text-[13px]"
          style={{ background: navColor }}
        >
          + New portfolio
        </button>
      </div>

      {loading && (
        <div className="text-[13px]" style={{ color: "var(--ink-3)" }}>Loading…</div>
      )}

      {!loading && portfolios.length === 0 && !creating && (
        <div className="border-[1.5px] border-dashed rounded-[14px] p-10 text-center"
             style={{ borderColor: "var(--border)" }}>
          <p className="text-[13px]" style={{ color: "var(--ink-3)" }}>
            No portfolios yet. Click &quot;New portfolio&quot; to create one.
          </p>
        </div>
      )}

      <div className="space-y-3">
        {portfolios.map((p) => (
          <PortfolioCard key={p.id} portfolio={p} onChanged={refetch} navColor={navColor} />
        ))}
        {creating && (
          <NewPortfolioCard
            onCreated={async () => { setCreating(false); await refetch(); }}
            onCancel={() => setCreating(false)}
            navColor={navColor}
          />
        )}
      </div>
    </div>
  );
}

function PortfolioCard({
  portfolio, onChanged, navColor,
}: { portfolio: Portfolio; onChanged: () => Promise<void>; navColor: string }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(portfolio.name);
  const [capital, setCapital] = useState(
    portfolio.starting_capital != null ? String(portfolio.starting_capital) : ""
  );
  const [resetDate, setResetDate] = useState(portfolio.reset_date ?? "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function save() {
    setBusy(true);
    setErr(null);
    const capitalNum = capital.trim() ? parseFloat(capital) : null;
    if (capital.trim() && (!isFinite(capitalNum as number) || (capitalNum as number) < 0)) {
      setErr("Starting capital must be a non-negative number");
      setBusy(false);
      return;
    }
    const res = await api.updatePortfolio(portfolio.id, {
      name: name.trim() || portfolio.name,
      starting_capital: capitalNum,
      reset_date: resetDate || null,
    });
    if ("error" in res) {
      setErr(res.error);
    } else {
      setEditing(false);
      await onChanged();
    }
    setBusy(false);
  }

  async function remove() {
    const confirmed = window.confirm(
      `Delete "${portfolio.name}"?\n\nThis cascades: all trades, journal entries, and snapshots under this portfolio will be permanently deleted. This cannot be undone.`
    );
    if (!confirmed) return;
    setBusy(true);
    const res = await api.deletePortfolio(portfolio.id);
    if (res.error) {
      setErr(res.error);
      setBusy(false);
    } else {
      await onChanged();
    }
  }

  const capitalDisplay = portfolio.starting_capital != null
    ? `$${portfolio.starting_capital.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
    : "—";
  const resetDisplay = portfolio.reset_date ?? "—";

  return (
    <div className="rounded-[14px] p-5"
         style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      {!editing ? (
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full" style={{ background: navColor }} />
              <div className="text-[15px] font-semibold" style={{ color: "var(--ink)" }}>{portfolio.name}</div>
            </div>
            <div className="grid grid-cols-2 gap-4 mt-3 text-[12px]">
              <div>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
                  Starting capital
                </div>
                <div className="mt-0.5 privacy-mask"
                     style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink)" }}>{capitalDisplay}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
                  Reset date
                </div>
                <div className="mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink)" }}>{resetDisplay}</div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setEditing(true)} disabled={busy}
                    className="h-[32px] px-3 rounded-[8px] text-[12px] font-medium"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink)" }}>
              Edit
            </button>
            <button onClick={remove} disabled={busy}
                    className="h-[32px] px-3 rounded-[8px] text-[12px] font-medium"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "#e5484d" }}>
              Delete
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={50}
                   className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Starting capital</span>
              <input type="number" min={0} step="0.01" value={capital}
                     onChange={(e) => setCapital(e.target.value)} placeholder="—"
                     className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                              fontFamily: "var(--font-jetbrains), monospace" }} />
            </label>
            <label className="block">
              <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Reset date</span>
              <input type="date" value={resetDate}
                     onChange={(e) => setResetDate(e.target.value)}
                     className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                     style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                              fontFamily: "var(--font-jetbrains), monospace" }} />
            </label>
          </div>
          {err && <div className="text-[12px] text-[#e5484d]">{err}</div>}
          <div className="flex gap-2">
            <button onClick={save} disabled={busy}
                    className="h-[34px] px-4 rounded-[8px] text-white text-[13px] font-medium"
                    style={{ background: navColor }}>
              {busy ? "Saving…" : "Save"}
            </button>
            <button onClick={() => { setEditing(false); setName(portfolio.name); setErr(null); }}
                    disabled={busy}
                    className="h-[34px] px-4 rounded-[8px] text-[13px] font-medium"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink)" }}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function NewPortfolioCard({
  onCreated, onCancel, navColor,
}: { onCreated: () => Promise<void>; onCancel: () => void; navColor: string }) {
  const [name, setName] = useState("");
  const [capital, setCapital] = useState("");
  const [resetDate, setResetDate] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function create() {
    if (!name.trim()) { setErr("Name is required"); return; }
    setBusy(true); setErr(null);
    const capitalNum = capital.trim() ? parseFloat(capital) : null;
    if (capital.trim() && (!isFinite(capitalNum as number) || (capitalNum as number) < 0)) {
      setErr("Starting capital must be a non-negative number");
      setBusy(false);
      return;
    }
    const res = await api.createPortfolio({
      name: name.trim(),
      starting_capital: capitalNum,
      reset_date: resetDate || null,
    });
    if ("error" in res) {
      setErr(res.error);
      setBusy(false);
    } else {
      await onCreated();
    }
  }

  return (
    <div className="rounded-[14px] p-5"
         style={{ background: "var(--surface)", border: `1px solid ${navColor}` }}>
      <div className="text-[13px] font-semibold mb-3" style={{ color: "var(--ink)" }}>New portfolio</div>
      <div className="space-y-3">
        <label className="block">
          <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} maxLength={50}
                 autoFocus placeholder="e.g. IRA"
                 className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                 style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Starting capital</span>
            <input type="number" min={0} step="0.01" value={capital}
                   onChange={(e) => setCapital(e.target.value)} placeholder="100000"
                   className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </label>
          <label className="block">
            <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Reset date</span>
            <input type="date" value={resetDate}
                   onChange={(e) => setResetDate(e.target.value)}
                   className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </label>
        </div>
        {err && <div className="text-[12px] text-[#e5484d]">{err}</div>}
        <div className="flex gap-2">
          <button onClick={create} disabled={busy}
                  className="h-[34px] px-4 rounded-[8px] text-white text-[13px] font-medium"
                  style={{ background: navColor }}>
            {busy ? "Creating…" : "Create"}
          </button>
          <button onClick={onCancel} disabled={busy}
                  className="h-[34px] px-4 rounded-[8px] text-[13px] font-medium"
                  style={{ background: "var(--bg-2)", border: "1px solid var(--border)" }}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
