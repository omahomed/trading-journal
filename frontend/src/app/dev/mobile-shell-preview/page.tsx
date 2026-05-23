// PHASE 1 PREVIEW — REMOVE BEFORE PHASE 2 SHIP
//
// Standalone visual verification target for the mobile shell built in
// Phase 1 Step 4. Lives outside the `(app)` route group so the desktop
// sidebar/header don't wrap it — what you see is the mobile chrome and
// only the mobile chrome. Auth still applies via proxy.ts.
"use client";

import { MobileShell } from "@/components/mobile/mobile-shell";

export default function MobileShellPreview() {
  return (
    <MobileShell
      header={{
        title: "Position",
        italicWord: "Sizer",
        // Phase 2 step 1: explicit empty rightSlot suppresses the default
        // MobilePortfolioPicker, which would otherwise throw here — this
        // page lives outside the `(app)` route group and has no
        // PortfolioProvider above it. This whole preview file is marked
        // for removal before Phase 2 ships per the header comment.
        rightSlot: <></>,
      }}
    >
      <div className="space-y-2 pt-2">
        <div className="rounded-m-lg border-[0.5px] border-m-border bg-m-surface p-m-4">
          <div className="text-xs text-m-text-dim">Preview</div>
          <div className="font-m-num text-m-text">Mobile shell chrome</div>
          <div className="mt-1 text-xs text-m-text-muted">
            Tape pill above, header centered with italic-green wordmark, bottom
            nav pinned. Scroll this middle region freely.
          </div>
        </div>
        <div className="rounded-m-md border-[0.5px] border-m-accent-border-soft bg-m-surface p-m-4">
          <div className="text-xs text-m-accent">accent border + tint</div>
          <div className="text-m-text-muted">
            Demonstrates ready-state token application.
          </div>
        </div>
        <div className="rounded-m-md border-[0.5px] border-m-warn-border bg-m-warn-tint-soft p-m-4">
          <div className="text-xs text-m-warn">at-risk border + wash</div>
          <div className="text-m-text-muted">
            Demonstrates at-risk token application.
          </div>
        </div>
        <div className="rounded-m-md border-[0.5px] border-m-purple-border bg-m-purple-tint p-m-4">
          <div className="text-xs text-m-purple-text">purple border + tint</div>
          <div className="text-m-text-muted">
            Same family as the cycle pill above.
          </div>
        </div>
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={i}
            className="rounded-m-md border-[0.5px] border-m-border bg-m-surface p-m-3 text-xs text-m-text-muted"
          >
            Filler card {i + 1} — proves the middle region scrolls and the
            top/bottom chrome stay pinned.
          </div>
        ))}
      </div>
    </MobileShell>
  );
}
