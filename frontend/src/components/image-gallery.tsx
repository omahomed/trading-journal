"use client";

// Phase 4.4 — shared image gallery. The grid + card + filename caption +
// delete affordance pulled out of Trade Journal's chart section pattern
// (rule of two: Trade Journal + Weekly Snapshot now share this).
//
// Card chrome mirrors trade-journal.tsx:330-358 exactly:
//   - 8px border-radius
//   - 1px solid var(--border)
//   - hover:shadow-md (CSS transition + box-shadow)
//   - Image area: w-full, h-auto, max-height: 300px, object-fit: contain,
//     bg: var(--bg)
//   - Caption row: bg: var(--surface-2), filename truncated left,
//     "Delete" button right (native confirm())
//
// Auto-detects PDFs by URL/filename extension and renders the Trade
// Journal PDF variant — clickable <a> card with 📄 + "Open PDF" — so
// Trade Journal's existing PDF handling is preserved at the call site
// without consumers needing to know about the discriminator.
//
// What this component does NOT own:
//   - Section header / title
//   - Subsection grouping (Trade Journal's "Entry Charts" /
//     "Position Changes" expanders)
//   - Upload affordance (Trade Journal's "+ Upload" button is per-
//     group; Weekly Snapshot has a drop zone + paste handler)
//   - Lightbox (consumer wires onImageClick to whatever lightbox
//     state they own — Trade Journal has its own local one,
//     Weekly Snapshot uses the shared ImageLightbox from Phase 4.2)
//   - Empty state copy (passed via emptyState prop)
//   - Optimistic upload placeholders (consumer renders alongside
//     the gallery; the gallery is for SETTLED items only)

import { useMemo } from "react";

export interface ImageGalleryItem {
  /** Stable identifier — used as React key and as the argument to
   *  onDelete. For optimistic-upload placeholders, use a temp id
   *  (negative number from a tempId-to-int coercion is fine). */
  id: number;
  url: string;
  filename: string;
  /** When true, the item renders with the "uploading" visual:
   *  dashed border, no clickable image area, no Delete button.
   *  The consumer owns the rest (when to set/clear this flag —
   *  typically true while the upload promise is in flight, then
   *  the item gets replaced by a real one). */
  uploading?: boolean;
}

interface ImageGalleryProps {
  items: ImageGalleryItem[];
  /** When omitted, the "Delete" button doesn't render. */
  onDelete?: (id: number) => void;
  /** When omitted, images aren't clickable (no cursor: pointer, no
   *  click handler). Index is into the items array as passed. */
  onImageClick?: (index: number) => void;
  /** Rendered when items.length === 0. When omitted, nothing
   *  renders for empty state (consumer handles the empty case
   *  elsewhere). */
  emptyState?: React.ReactNode;
  /** Override the "Delete this image?" confirm message. */
  deleteConfirmMessage?: string;
  /** Single-column mode (1-image grids look weird in 2-col).
   *  Default behavior matches Trade Journal: 1 column when
   *  items.length === 1, 2 columns otherwise. Caller can force
   *  by passing this prop. */
  forceColumns?: 1 | 2;
}

// Helper: is this URL/filename a PDF? Matches Trade Journal's regex.
function isPdf(url: string, filename: string): boolean {
  return /\.pdf($|\?)/i.test(filename) || /\.pdf($|\?)/i.test(url);
}

export function ImageGallery({
  items,
  onDelete,
  onImageClick,
  emptyState,
  deleteConfirmMessage = "Delete this image?",
  forceColumns,
}: ImageGalleryProps) {
  // Trade Journal pattern: single-image grids collapse to one column
  // for visual balance; multi-image grids use two.
  const columns = forceColumns ?? (items.length === 1 ? 1 : 2);
  const gridTemplate = useMemo(
    () => columns === 1 ? "1fr" : "1fr 1fr",
    [columns],
  );

  if (items.length === 0) {
    return emptyState ? <>{emptyState}</> : null;
  }

  return (
    <div
      role="list"
      aria-label="Image gallery"
      className="p-3 grid gap-3"
      style={{ gridTemplateColumns: gridTemplate }}
    >
      {items.map((item, index) => {
        const pdf = isPdf(item.url, item.filename);
        // Uploading state — render a placeholder card. Distinct
        // visual (dashed border, no hover-shadow, no clickable
        // image, no Delete button) so the user understands the
        // item isn't persisted yet.
        if (item.uploading) {
          return (
            <div
              key={item.id}
              role="listitem"
              data-testid="upload-placeholder"
              className="rounded-[8px] overflow-hidden"
              style={{ border: "1px dashed var(--border-2)" }}
            >
              <div
                className="flex items-center justify-center text-[11px]"
                style={{ background: "var(--bg)", color: "var(--ink-4)", minHeight: 140 }}
              >
                Uploading {item.filename}…
              </div>
              <div
                className="px-2.5 py-1.5 text-[10px]"
                style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}
              >
                <span className="truncate block">{item.filename}</span>
              </div>
            </div>
          );
        }
        return (
          <div
            key={item.id}
            role="listitem"
            className="rounded-[8px] overflow-hidden transition-all hover:shadow-md"
            style={{ border: "1px solid var(--border)" }}
          >
            {pdf ? (
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-col items-center justify-center p-8 gap-2 no-underline"
                style={{ background: "var(--bg)", color: "var(--ink)", minHeight: 140 }}
              >
                <span className="text-[32px]">📄</span>
                <span className="text-[12px] font-semibold">Open PDF</span>
                <span className="text-[10px]" style={{ color: "var(--ink-4)" }}>
                  {item.filename || "document.pdf"}
                </span>
              </a>
            ) : (
              <div
                className={onImageClick ? "cursor-pointer" : ""}
                onClick={onImageClick ? () => onImageClick(index) : undefined}
              >
                {item.url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={item.url}
                    alt={item.filename || `Image ${index + 1}`}
                    className="w-full h-auto"
                    style={{
                      maxHeight: 300,
                      objectFit: "contain",
                      background: "var(--bg)",
                      display: "block",
                    }}
                  />
                ) : (
                  <div className="p-4 text-center text-[11px]" style={{ color: "var(--ink-4)" }}>
                    No URL
                  </div>
                )}
              </div>
            )}
            <div
              className="px-2.5 py-1.5 text-[10px] flex items-center justify-between"
              style={{ background: "var(--surface-2)", color: "var(--ink-4)" }}
            >
              <span className="truncate flex-1" title={item.filename}>
                {item.filename || `Image ${index + 1}`}
              </span>
              {onDelete && (
                <button
                  type="button"
                  onClick={() => {
                    if (typeof window !== "undefined" && !window.confirm(deleteConfirmMessage)) return;
                    onDelete(item.id);
                  }}
                  aria-label={`Delete ${item.filename || `image ${index + 1}`}`}
                  className="ml-2 px-1.5 py-0.5 rounded text-[9px] cursor-pointer transition-colors hover:brightness-90"
                  style={{
                    color: "#e5484d",
                    background: "color-mix(in oklab, #e5484d 8%, var(--surface))",
                    border: "none",
                  }}
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
