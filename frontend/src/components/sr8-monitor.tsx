"use client";

// SR8 Cascade Monitor — daily position-management screen for positions
// tagged sr8. Build sequence:
//   Commit 1 (this commit): backend endpoint + route stub + nav rename.
//   Commit 2: page scaffold (control strip + summary chips + fetch).
//   Commit 3: Action / Hold sections + Mark-done state.
//   Commit 4: All-clear / Loading / Empty / Retry states.
//
// The cascade math lives in mors/monitor.py (Python). The new
// /api/sr8/monitor endpoint wraps it. This page just renders the
// endpoint's response — no JS reimplementation of the engine.

export function Sr8Monitor({ navColor }: { navColor: string }) {
  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0"
            style={{ fontFamily: "var(--font-fraunces), Georgia, serif", letterSpacing: "-0.02em" }}>
          SR8 Cascade <em className="italic" style={{ color: navColor }}>Monitor</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Weekly-chart signals across positions tagged sr8 · prices cached vs SPY, pulled weekly
        </div>
      </div>

      <div className="px-4 py-8 text-center text-[12px] rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
        Coming soon — control strip + data fetch land in Commit 2, action / hold sections in Commit 3, polish states in Commit 4.
      </div>
    </div>
  );
}
