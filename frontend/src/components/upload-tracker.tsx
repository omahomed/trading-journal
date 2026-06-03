"use client";

// UploadTracker — per-file upload status panel used by Log Buy + Log Sell.
//
// After a successful trade submit, we don't block the page on image uploads
// anymore (they used to hang the "Saving…" button when Claude Vision OCR
// stalled). Each file lands in this tracker with status="uploading"; the
// upload promise then transitions it to "done" or "failed". Failed entries
// expose a Retry button so the user can re-fire just that file. Trade is
// saved regardless — this panel only governs the chart attachments.

export type UploadKind = "entry" | "position_change" | "marketsurge";

export interface UploadEntry {
  id: string;
  file: File;
  fileName: string;
  kind: UploadKind;
  portfolio: string;
  tradeId: string;
  ticker: string;
  status: "uploading" | "done" | "failed";
  error?: string;
}

const KIND_LABEL: Record<UploadKind, string> = {
  entry: "Entry chart",
  position_change: "Position change",
  marketsurge: "MarketSurge",
};

export function UploadTracker({
  entries,
  onRetry,
  onDismiss,
}: {
  entries: UploadEntry[];
  onRetry: (id: string) => void;
  onDismiss?: () => void;
}) {
  if (entries.length === 0) return null;

  const uploading = entries.filter(e => e.status === "uploading").length;
  const failed = entries.filter(e => e.status === "failed").length;
  const done = entries.filter(e => e.status === "done").length;
  const allDone = uploading === 0 && failed === 0 && done === entries.length;
  const anyFailed = failed > 0;

  const headerBg = anyFailed
    ? "color-mix(in oklab, #e5484d 12%, var(--surface))"
    : allDone
      ? "color-mix(in oklab, #08a86b 10%, var(--surface))"
      : "color-mix(in oklab, #f59f00 10%, var(--surface))";
  const headerBorder = anyFailed
    ? "color-mix(in oklab, #e5484d 35%, var(--border))"
    : allDone
      ? "color-mix(in oklab, #08a86b 35%, var(--border))"
      : "color-mix(in oklab, #f59f00 35%, var(--border))";

  return (
    <div
      data-testid="upload-tracker"
      className="mb-4 rounded-[10px] overflow-hidden"
      style={{ background: headerBg, border: `1px solid ${headerBorder}` }}
    >
      <div className="px-4 py-2 flex items-center justify-between text-[12px] font-semibold" style={{ color: "var(--ink-2)" }}>
        <span data-testid="upload-tracker-summary">
          {anyFailed
            ? `${failed} chart${failed === 1 ? "" : "s"} failed${uploading > 0 ? ` · ${uploading} still uploading` : ""}`
            : allDone
              ? `All ${done} chart${done === 1 ? "" : "s"} uploaded`
              : `Uploading charts · ${done}/${entries.length} done`}
        </span>
        {allDone && onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            data-testid="upload-tracker-dismiss"
            className="text-[11px] font-medium px-2 py-0.5 rounded-md hover:brightness-95"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-3)" }}
          >
            Dismiss
          </button>
        )}
      </div>
      <ul className="flex flex-col gap-1 px-3 pb-2">
        {entries.map(e => (
          <li
            key={e.id}
            data-testid={`upload-entry-${e.id}`}
            className="flex items-center gap-2 text-[12px] py-1.5 px-2 rounded-[6px]"
            style={{ background: "var(--surface)", fontFamily: "var(--font-jetbrains), monospace" }}
          >
            <span style={{ color: "var(--ink-4)" }}>{KIND_LABEL[e.kind]}</span>
            <span className="truncate" style={{ color: "var(--ink-2)" }} title={e.fileName}>
              {e.fileName}
            </span>
            <span className="ml-auto flex items-center gap-2">
              {e.status === "uploading" && (
                <span style={{ color: "#f59f00" }}>Uploading…</span>
              )}
              {e.status === "done" && (
                <span style={{ color: "#08a86b" }}>Uploaded</span>
              )}
              {e.status === "failed" && (
                <>
                  <span title={e.error} style={{ color: "#e5484d" }}>
                    Failed{e.error ? `: ${e.error}` : ""}
                  </span>
                  <button
                    type="button"
                    data-testid={`upload-retry-${e.id}`}
                    onClick={() => onRetry(e.id)}
                    className="text-[11px] font-semibold px-2 py-0.5 rounded-md text-white hover:brightness-95"
                    style={{ background: "#e5484d" }}
                  >
                    Retry
                  </button>
                </>
              )}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
