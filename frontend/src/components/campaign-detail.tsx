"use client";

// Campaign Detail page — fill-by-fill ledger across all open stock campaigns.
// Stub for Commit 1 of 4 of the build plan. Commit 2 wires data + KPI strip;
// Commit 3 adds the ledger table + sort/filter/footer; Commit 4 wires the
// Edit affordance to Trade Journal's existing edit flow.

export function CampaignDetail({ navColor }: { navColor: string }) {
  return (
    <div style={{ animation: "slide-up 0.18s ease-out" }}>
      <div className="mb-[22px] pb-[14px]" style={{ borderBottom: "1px solid var(--border)" }}>
        <h1 className="font-normal text-[32px] tracking-tight m-0" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
          Active Campaign <em className="italic" style={{ color: navColor }}>Detail</em>
        </h1>
        <div className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
          Every fill across all open stock campaigns
        </div>
      </div>

      <div className="px-4 py-8 text-center text-[12px] rounded-[14px]"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-4)" }}>
        Coming soon — page scaffold lands in Commit 2 (KPI strip + data fetch),
        ledger table in Commit 3, edit wiring in Commit 4.
      </div>
    </div>
  );
}
