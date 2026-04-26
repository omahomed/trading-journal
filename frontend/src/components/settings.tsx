"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type CashAction, type CashTransaction, type Portfolio } from "@/lib/api";
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

  const [cashAction, setCashAction] = useState<CashAction | null>(null);

  const capitalDisplay = portfolio.starting_capital != null
    ? `$${portfolio.starting_capital.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
    : "—";
  const resetDisplay = portfolio.reset_date ?? "—";
  const cashBalance = portfolio.cash_balance ?? 0;
  const cashDisplay = `$${cashBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  return (
    <div className="rounded-[14px] p-5"
         style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      {!editing ? (
        <>
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full" style={{ background: navColor }} />
              <div className="text-[15px] font-semibold" style={{ color: "var(--ink)" }}>{portfolio.name}</div>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-3 text-[12px]">
              <div>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
                  Starting capital
                </div>
                <div className="mt-0.5 privacy-mask"
                     style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink)" }}>{capitalDisplay}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
                  Cash balance
                </div>
                <div className="mt-0.5 privacy-mask"
                     style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink)" }}>{cashDisplay}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
                  Reset date
                </div>
                <div className="mt-0.5" style={{ fontFamily: "var(--font-jetbrains), monospace", color: "var(--ink)" }}>{resetDisplay}</div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            <button onClick={() => setCashAction("deposit")} disabled={busy}
                    className="h-[32px] px-3 rounded-[8px] text-[12px] font-medium"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "#08a86b" }}>
              Deposit
            </button>
            <button onClick={() => setCashAction("withdraw")} disabled={busy}
                    className="h-[32px] px-3 rounded-[8px] text-[12px] font-medium"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "#f59f00" }}>
              Withdraw
            </button>
            <button onClick={() => setCashAction("reconcile")} disabled={busy}
                    className="h-[32px] px-3 rounded-[8px] text-[12px] font-medium"
                    style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
              Reconcile
            </button>
            <span className="w-px h-5" style={{ background: "var(--border)" }} />
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
        {cashAction && (
          <CashActionForm
            portfolioId={portfolio.id}
            cashBalance={cashBalance}
            action={cashAction}
            navColor={navColor}
            onDone={async () => { setCashAction(null); await onChanged(); }}
            onCancel={() => setCashAction(null)}
          />
        )}
        <CashLedger portfolioId={portfolio.id} onChanged={onChanged} />
        </>
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


function CashActionForm({
  portfolioId, cashBalance, action, navColor, onDone, onCancel,
}: {
  portfolioId: number;
  cashBalance: number;
  action: CashAction;
  navColor: string;
  onDone: () => Promise<void>;
  onCancel: () => void;
}) {
  const [amount, setAmount] = useState("");
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const todayIso = new Date().toISOString().slice(0, 10);

  const titles: Record<CashAction, { title: string; subtitle: string; button: string; color: string }> = {
    deposit: {
      title: "Deposit cash",
      subtitle: "Money added from your broker account.",
      button: "Deposit",
      color: "#08a86b",
    },
    withdraw: {
      title: "Withdraw cash",
      subtitle: "Money removed from your portfolio.",
      button: "Withdraw",
      color: "#f59f00",
    },
    reconcile: {
      title: "Reconcile with broker",
      subtitle: `Enter your actual broker cash balance. The system will record the delta as a single adjustment — absorbing any commissions or margin interest drift since your last reconcile.`,
      button: "Reconcile",
      color: navColor,
    },
  };
  const meta = titles[action];

  async function submit() {
    setErr(null);
    setInfo(null);
    const num = parseFloat(amount);
    if (!isFinite(num) || num <= 0) {
      setErr("Amount must be a positive number");
      return;
    }
    setBusy(true);
    const res = await api.createCashTransaction(portfolioId, {
      source: action,
      amount: num,
      date: date || null,
      note: note.trim() || null,
    });
    if ("error" in res) {
      setErr(res.error);
      setBusy(false);
      return;
    }
    if ("status" in res && res.status === "noop") {
      setInfo(res.message || "Already in sync");
      setBusy(false);
      return;
    }
    await onDone();
  }

  return (
    <div className="mt-4 pt-4 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
      <div>
        <div className="text-[13px] font-semibold" style={{ color: "var(--ink)" }}>{meta.title}</div>
        <div className="text-[11px] mt-0.5" style={{ color: "var(--ink-4)" }}>{meta.subtitle}</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
            {action === "reconcile" ? "Broker cash balance" : "Amount"}
          </span>
          <div className="mt-1 relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm" style={{ color: "var(--ink-4)" }}>$</span>
            <input type="number" min={0} step="0.01" value={amount} autoFocus
                   onChange={(e) => setAmount(e.target.value)}
                   className="w-full h-10 pl-7 pr-3 rounded-[8px] text-[13px]"
                   style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </div>
          {action === "reconcile" && (
            <span className="block mt-1 text-[11px]" style={{ color: "var(--ink-4)" }}>
              System currently shows{" "}
              <span style={{ fontFamily: "var(--font-jetbrains), monospace" }}>
                ${cashBalance.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </span>
            </span>
          )}
        </label>
        <label className="block">
          <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
            Date
          </span>
          <input type="date" value={date} max={todayIso}
                 onChange={(e) => setDate(e.target.value)}
                 className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
                 style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)",
                          fontFamily: "var(--font-jetbrains), monospace" }} />
        </label>
      </div>
      <label className="block">
        <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
          Note <span className="normal-case font-normal" style={{ color: "var(--ink-5)" }}>(optional)</span>
        </span>
        <input type="text" value={note} onChange={(e) => setNote(e.target.value)} maxLength={200}
               placeholder="e.g. ACH transfer"
               className="mt-1 w-full h-10 px-3 rounded-[8px] text-[13px]"
               style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }} />
      </label>
      {err && (
        <div className="text-[12px] px-3 py-2 rounded-[8px]"
             style={{ color: "#e5484d", background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
                      border: "1px solid color-mix(in oklab, #e5484d 25%, var(--border))" }}>
          {err}
        </div>
      )}
      {info && (
        <div className="text-[12px] px-3 py-2 rounded-[8px]"
             style={{ color: "var(--ink-2)", background: "color-mix(in oklab, #3b82f6 10%, var(--surface))",
                      border: "1px solid color-mix(in oklab, #3b82f6 25%, var(--border))" }}>
          {info}
        </div>
      )}
      <div className="flex gap-2">
        <button onClick={submit} disabled={busy || !amount.trim()}
                className="h-[34px] px-4 rounded-[8px] text-white text-[13px] font-medium disabled:opacity-50"
                style={{ background: meta.color }}>
          {busy ? "Saving…" : meta.button}
        </button>
        <button onClick={onCancel} disabled={busy}
                className="h-[34px] px-4 rounded-[8px] text-[13px] font-medium"
                style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink)" }}>
          Cancel
        </button>
      </div>
    </div>
  );
}


function isProtectedRow(tx: CashTransaction): { protected: boolean; reason: string } {
  if (tx.source === "buy" || tx.source === "sell") {
    return { protected: true, reason: "Linked to a trade — edit the trade itself." };
  }
  if (tx.source === "deposit" && (tx.note || "").startsWith("Initial capital")) {
    return { protected: true, reason: "Managed via Starting capital + Reset date." };
  }
  return { protected: false, reason: "" };
}

function sourceBadge(source: CashTransaction["source"]): { label: string; color: string; bg: string } {
  switch (source) {
    case "deposit":   return { label: "Deposit",   color: "#08a86b", bg: "color-mix(in oklab, #08a86b 12%, var(--surface))" };
    case "withdraw":  return { label: "Withdraw",  color: "#f59f00", bg: "color-mix(in oklab, #f59f00 12%, var(--surface))" };
    case "reconcile": return { label: "Reconcile", color: "#3b82f6", bg: "color-mix(in oklab, #3b82f6 12%, var(--surface))" };
    case "buy":       return { label: "Buy",       color: "var(--ink-3)", bg: "var(--bg-2)" };
    case "sell":      return { label: "Sell",      color: "var(--ink-3)", bg: "var(--bg-2)" };
  }
}

function CashLedger({ portfolioId, onChanged }: { portfolioId: number; onChanged: () => Promise<void> }) {
  const [rows, setRows] = useState<CashTransaction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);

  const refetch = useCallback(async () => {
    setError(null);
    try {
      const res = await api.listCashTransactions(portfolioId, 100, true) as unknown;
      if (Array.isArray(res)) {
        setRows(res as CashTransaction[]);
      } else {
        const errMsg = res && typeof res === "object" && "error" in res
          ? String((res as { error: unknown }).error)
          : "Unexpected response";
        setError(errMsg);
        setRows([]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setRows([]);
    }
  }, [portfolioId]);

  useEffect(() => { refetch(); }, [refetch]);

  // Hide buy/sell from the activity list — they're a function of trades and
  // would dwarf the cash flows the user actually manages here. Show all
  // deposit/withdraw/reconcile rows (including the protected initial capital).
  const userRows = (rows || []).filter(r => r.source !== "buy" && r.source !== "sell");
  const visible = expanded ? userRows : userRows.slice(0, 5);
  const isLoading = rows === null;

  return (
    <div className="mt-4 pt-4" style={{ borderTop: "1px solid var(--border)" }}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[11px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>
          Cash activity {isLoading ? "" : `(${userRows.length})`}
        </div>
        {userRows.length > 5 && (
          <button onClick={() => setExpanded(v => !v)}
                  className="text-[11px] font-medium" style={{ color: "var(--ink-3)" }}>
            {expanded ? "Show less" : `Show all`}
          </button>
        )}
      </div>
      {error && (
        <div className="text-[11px] px-3 py-2 rounded-[6px] mb-2"
             style={{ color: "#e5484d", background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
                      border: "1px solid color-mix(in oklab, #e5484d 25%, var(--border))" }}>
          Failed to load: {error}
        </div>
      )}
      {isLoading && !error && (
        <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>Loading…</div>
      )}
      {!isLoading && userRows.length === 0 && !error && (
        <div className="text-[11px]" style={{ color: "var(--ink-4)" }}>
          No deposits, withdrawals, or reconciles yet.
        </div>
      )}
      <div className="space-y-1.5">
        {visible.map(tx => (
          editingId === tx.id ? (
            <CashRowEditor key={tx.id} tx={tx} portfolioId={portfolioId}
                           onCancel={() => setEditingId(null)}
                           onDone={async () => { setEditingId(null); await refetch(); await onChanged(); }} />
          ) : (
            <CashRowDisplay key={tx.id} tx={tx} portfolioId={portfolioId}
                            onEdit={() => setEditingId(tx.id)}
                            onDeleted={async () => { await refetch(); await onChanged(); }} />
          )
        ))}
      </div>
    </div>
  );
}

function CashRowDisplay({
  tx, portfolioId, onEdit, onDeleted,
}: {
  tx: CashTransaction;
  portfolioId: number;
  onEdit: () => void;
  onDeleted: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const guard = isProtectedRow(tx);
  const badge = sourceBadge(tx.source);
  const dateOnly = (tx.date || "").slice(0, 10);
  const amountAbs = Math.abs(tx.amount);
  const sign = tx.amount >= 0 ? "+" : "−";
  const amountColor = tx.amount >= 0 ? "#08a86b" : "#e5484d";

  async function remove() {
    if (!window.confirm(`Delete this ${badge.label.toLowerCase()} of $${amountAbs.toLocaleString(undefined, { maximumFractionDigits: 2 })} on ${dateOnly}?`)) return;
    setBusy(true); setErr(null);
    const res = await api.deleteCashTransaction(portfolioId, tx.id);
    if ("error" in res) {
      setErr(res.error);
      setBusy(false);
      return;
    }
    await onDeleted();
  }

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-[8px]"
         style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
      <div className="text-[11px] privacy-mask" style={{ color: "var(--ink-3)", fontFamily: "var(--font-jetbrains), monospace", minWidth: 86 }}>
        {dateOnly}
      </div>
      <div className="text-[10px] font-semibold uppercase tracking-[0.05em] px-2 py-0.5 rounded-[6px]"
           style={{ color: badge.color, background: badge.bg }}>
        {badge.label}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[12px] privacy-mask"
             style={{ color: amountColor, fontFamily: "var(--font-jetbrains), monospace", fontWeight: 600 }}>
          {sign}${amountAbs.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </div>
        {tx.note && (
          <div className="text-[11px] truncate" style={{ color: "var(--ink-4)" }}>{tx.note}</div>
        )}
        {err && (
          <div className="text-[11px]" style={{ color: "#e5484d" }}>{err}</div>
        )}
      </div>
      {guard.protected ? (
        <span className="text-[10px]" style={{ color: "var(--ink-5)" }} title={guard.reason}>locked</span>
      ) : (
        <div className="flex gap-1">
          <button onClick={onEdit} disabled={busy}
                  className="h-[26px] px-2 rounded-[6px] text-[11px]"
                  style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
            Edit
          </button>
          <button onClick={remove} disabled={busy}
                  className="h-[26px] px-2 rounded-[6px] text-[11px]"
                  style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "#e5484d" }}>
            {busy ? "…" : "Delete"}
          </button>
        </div>
      )}
    </div>
  );
}

function CashRowEditor({
  tx, portfolioId, onDone, onCancel,
}: {
  tx: CashTransaction;
  portfolioId: number;
  onDone: () => Promise<void>;
  onCancel: () => void;
}) {
  const [amount, setAmount] = useState(String(Math.abs(tx.amount)));
  const [date, setDate] = useState((tx.date || "").slice(0, 10));
  const [note, setNote] = useState(tx.note || "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const todayIso = new Date().toISOString().slice(0, 10);

  async function save() {
    const num = parseFloat(amount);
    if (!isFinite(num) || num <= 0) { setErr("Amount must be positive"); return; }
    setBusy(true); setErr(null);
    const res = await api.updateCashTransaction(portfolioId, tx.id, {
      amount: num,
      date: date || null,
      note: note.trim() || null,
    });
    if ("error" in res) { setErr(res.error); setBusy(false); return; }
    await onDone();
  }

  return (
    <div className="px-3 py-3 rounded-[8px] space-y-2"
         style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
      <div className="grid grid-cols-2 gap-2">
        <label className="block">
          <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Amount</span>
          <div className="mt-1 relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm" style={{ color: "var(--ink-4)" }}>$</span>
            <input type="number" min={0} step="0.01" value={amount} autoFocus
                   onChange={(e) => setAmount(e.target.value)}
                   className="w-full h-9 pl-7 pr-2 rounded-[6px] text-[12px]"
                   style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
                            fontFamily: "var(--font-jetbrains), monospace" }} />
          </div>
        </label>
        <label className="block">
          <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Date</span>
          <input type="date" value={date} max={todayIso}
                 onChange={(e) => setDate(e.target.value)}
                 className="mt-1 w-full h-9 px-2 rounded-[6px] text-[12px]"
                 style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)",
                          fontFamily: "var(--font-jetbrains), monospace" }} />
        </label>
      </div>
      <label className="block">
        <span className="text-[10px] uppercase tracking-[0.10em] font-semibold" style={{ color: "var(--ink-4)" }}>Note</span>
        <input type="text" value={note} onChange={(e) => setNote(e.target.value)} maxLength={200}
               className="mt-1 w-full h-9 px-2 rounded-[6px] text-[12px]"
               style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }} />
      </label>
      {err && <div className="text-[11px]" style={{ color: "#e5484d" }}>{err}</div>}
      <div className="flex gap-2">
        <button onClick={save} disabled={busy}
                className="h-[28px] px-3 rounded-[6px] text-white text-[11px] font-medium"
                style={{ background: "#3b82f6" }}>
          {busy ? "Saving…" : "Save"}
        </button>
        <button onClick={onCancel} disabled={busy}
                className="h-[28px] px-3 rounded-[6px] text-[11px] font-medium"
                style={{ background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--ink)" }}>
          Cancel
        </button>
      </div>
    </div>
  );
}
