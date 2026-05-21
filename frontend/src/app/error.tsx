"use client";

import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";
import { handleChunkLoadError } from "@/lib/chunk-reload";

// Route-level error boundary. Catches anything that throws inside the app's
// rendered tree (below RootLayout), reports to Sentry, and offers a recover
// button instead of the default blank screen.
//
// Chunk-load errors (deploy invalidated cached chunk hashes) are handled
// specially: we attempt a single auto-reload — once per session, guarded
// against loops — before falling through to the boundary UI below. After
// one failed auto-reload in this session, the user sees the normal UI
// with "refresh the page" guidance as the fallback.

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Try auto-reload for chunk errors first. If it fires, the page is
    // about to navigate — Sentry's transport still has time to flush
    // the event via beacon API before the reload completes.
    Sentry.captureException(error);
    handleChunkLoadError(error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center"
         style={{ background: "var(--bg)" }}>
      <div className="w-[460px] max-w-[90vw] rounded-[20px] p-8"
           style={{ background: "var(--surface)", border: "1px solid var(--border)",
                    boxShadow: "0 8px 30px rgba(0,0,0,0.08)" }}>
        <div className="text-[22px] font-semibold mb-2" style={{ color: "var(--ink)" }}>
          Something broke on this page.
        </div>
        <div className="text-[13px] leading-relaxed mb-6" style={{ color: "var(--ink-4)" }}>
          The error has been reported. You can try again — if it keeps happening,
          refresh the page.
        </div>

        {error.digest && (
          <div className="text-[11px] mb-6 font-mono" style={{ color: "var(--ink-5)" }}>
            ref: {error.digest}
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={() => reset()}
            className="flex-1 h-[44px] rounded-[10px] text-[13px] font-semibold transition-all hover:brightness-110 cursor-pointer"
            style={{ background: "#6366f1", color: "white" }}>
            Try again
          </button>
          <a
            href="/"
            className="flex-1 h-[44px] rounded-[10px] flex items-center justify-center text-[13px] font-semibold transition-all hover:brightness-95 cursor-pointer"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
            Back to dashboard
          </a>
        </div>
      </div>
    </div>
  );
}
