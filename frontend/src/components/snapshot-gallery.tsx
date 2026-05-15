"use client";

// Phase 7 — shared image-gallery component for entity attachments.
// Extracted from weekly-snapshot.tsx and parameterized by entityType so
// the daily report can mount an identical gallery against a different
// upload endpoint and FK target. Weekly Retro continues to consume this
// via the thin <WeeklySnapshot> wrapper in weekly-snapshot.tsx; Daily
// Report mounts <SnapshotGallery entityType="daily_journal" ...>
// directly.
//
// Phase 4 history (preserved for context): Image gallery attached to
// either a weekly retro or daily journal. Users drop / paste / pick-
// from-disk images and review them later. Ships upload + display +
// delete; captions and drag-to-reorder are pre-provisioned in both
// schemas (columns exist) but deferred to follow-up commits.
//
// Storage: bytes upload to Cloudflare R2 via the backend; the server
// returns a public CDN URL (view_url) in each row dict, so the browser
// fetches bytes directly from R2 — backend is not in the serving hot
// path.
//
// Component contract:
//   - Disabled state when entityId === null (TagPicker idiom — the
//     parent entity must be saved before captures can attach). Drop /
//     paste / click are no-ops in the disabled state.
//   - Optimistic uploads: a placeholder row appears immediately on
//     drop/paste/pick; the placeholder is replaced by the server row
//     on success or removed on error.
//   - Lightbox: click thumbnail → full-screen overlay. Esc / outside
//     click closes; ← / → navigate. Prev/next wrap around.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type SnapshotRow,
  type DailyJournalCaptureRow,
} from "@/lib/api";
import { log } from "@/lib/log";
import { ImageLightbox, type LightboxImage } from "./image-lightbox";
import { ImageGallery, type ImageGalleryItem } from "./image-gallery";

export type SnapshotGalleryEntityType = "weekly_retro" | "daily_journal";

// Unified row shape consumed by the gallery internals. Real rows have an
// id + view_url + file_name; the discriminator (weekly_retro_id vs
// daily_journal_id) is hidden inside the API call layer.
type GalleryRow = SnapshotRow | DailyJournalCaptureRow;

export interface SnapshotGalleryProps {
  /** Discriminator picks upload/list/delete endpoint and disabled-state
   *  copy. */
  entityType: SnapshotGalleryEntityType;
  /** Parent entity row id. `null` puts the gallery into the disabled
   *  state (save-parent-first idiom). */
  entityId: number | null;
  /** Portfolio name forwarded to the upload endpoint. */
  portfolio: string;
  /** Optional callback fired whenever the visible-count (real rows +
   *  uploading placeholders) changes. Used by the parent to render the
   *  "N attached" caption on the SectionExpander header. */
  onCountChange?: (count: number) => void;
  /** Phase 7 — copy override for the disabled-state drop zone. The
   *  default is "Save the {entity} first to add {kind}." */
  disabledMessage?: string;
  /** Phase 7 — copy override for the active-state drop zone hint. */
  activeMessage?: string;
  /** Phase 7 — copy override for the secondary "Charts and snapshots…"
   *  microcopy under the drop zone CTA. */
  microcopy?: string;
  /** Phase 7 — aria-label override for the drop zone role="button". */
  dropZoneAriaLabel?: string;
  /** Phase 7 — aria-label override for the lightbox. Preserved for
   *  testing parity ("Snapshot preview" for weekly, "Capture preview"
   *  for daily). */
  lightboxAriaLabel?: string;
}

const MAX_FILE_BYTES = 5 * 1024 * 1024;
const ALLOWED_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
]);

interface PlaceholderRow {
  tempId: string;
  name: string;
}

function isImageFile(f: File): boolean {
  return ALLOWED_MIMES.has(f.type) && f.size <= MAX_FILE_BYTES;
}

function describeRejection(f: File): string {
  if (!ALLOWED_MIMES.has(f.type))
    return `${f.name}: only PNG / JPEG / GIF / WEBP allowed`;
  if (f.size > MAX_FILE_BYTES)
    return `${f.name}: exceeds 5MB limit`;
  return `${f.name}: rejected`;
}

export function SnapshotGallery({
  entityType,
  entityId,
  portfolio,
  onCountChange,
  disabledMessage,
  activeMessage = "Paste a screenshot or drag an image here",
  microcopy = "Charts and snapshots from your week. PNG, JPEG, GIF, WEBP. Max 5MB.",
  dropZoneAriaLabel = "Upload snapshot",
  lightboxAriaLabel = "Snapshot preview",
}: SnapshotGalleryProps) {
  const [snapshots, setSnapshots] = useState<GalleryRow[]>([]);
  const [uploading, setUploading] = useState<PlaceholderRow[]>([]);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [inlineError, setInlineError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Default disabled message (computed from entityType when not overridden).
  const effectiveDisabledMessage = disabledMessage
    ?? (entityType === "weekly_retro"
      ? "Save the retro first to add snapshots."
      : "Save the journal entry first to add captures.");

  // Count change callback. Run when snapshots or uploading length changes.
  // Wrapped in a ref-style guard to avoid spurious calls on every render.
  const lastCountRef = useRef<number>(-1);
  useEffect(() => {
    const n = snapshots.length + uploading.length;
    if (n !== lastCountRef.current) {
      lastCountRef.current = n;
      onCountChange?.(n);
    }
  }, [snapshots.length, uploading.length, onCountChange]);

  // Initial fetch when entityId becomes non-null; re-fetch on change.
  useEffect(() => {
    if (entityId == null) {
      setSnapshots([]);
      return;
    }
    let cancelled = false;
    const listPromise = entityType === "weekly_retro"
      ? api.listWeeklyRetroSnapshots(entityId, portfolio)
      : api.listDailyJournalCaptures(entityId, portfolio);
    listPromise.then(res => {
      if (cancelled) return;
      if (Array.isArray(res)) setSnapshots(res as GalleryRow[]);
      else {
        log.error(entityType === "weekly_retro" ? "weekly-snapshot" : "daily-captures", "snapshot list returned non-array", res);
        setSnapshots([]);
      }
    }).catch((err) => {
      if (cancelled) return;
      log.error(entityType === "weekly_retro" ? "weekly-snapshot" : "daily-captures", "snapshot list fetch failed", err);
      setSnapshots([]);
    });
    return () => { cancelled = true; };
  }, [entityType, entityId, portfolio]);

  const showInlineError = useCallback((msg: string) => {
    setInlineError(msg);
    window.setTimeout(() => setInlineError(null), 3000);
  }, []);

  // Upload pipeline. Filters client-side, attaches optimistic placeholder,
  // calls API, replaces placeholder or rolls back on error.
  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    if (entityId == null) return;
    const arr = Array.from(files);
    const accepted = arr.filter(isImageFile);
    const rejected = arr.filter(f => !isImageFile(f));
    if (rejected.length > 0) showInlineError(describeRejection(rejected[0]));
    if (accepted.length === 0) return;

    for (const file of accepted) {
      const tempId = `temp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setUploading(prev => [...prev, { tempId, name: file.name }]);
      try {
        const res = entityType === "weekly_retro"
          ? await api.uploadWeeklyRetroSnapshot(entityId, file, portfolio)
          : await api.uploadDailyJournalCapture(entityId, file, portfolio);
        if (res && typeof res === "object" && "id" in res) {
          setSnapshots(prev => [...prev, res as GalleryRow]);
          setUploading(prev => prev.filter(p => p.tempId !== tempId));
        } else {
          setUploading(prev => prev.filter(p => p.tempId !== tempId));
          showInlineError(`${file.name}: upload failed`);
        }
      } catch (err) {
        setUploading(prev => prev.filter(p => p.tempId !== tempId));
        showInlineError(`${file.name}: upload failed`);
      }
    }
  }, [entityType, entityId, portfolio, showInlineError]);

  // --- Drop zone handlers -----------------------------------------------

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (entityId == null) return;
    e.preventDefault();
    setDragOver(true);
  }, [entityId]);

  const handleDragLeave = useCallback(() => { setDragOver(false); }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (entityId == null) return;
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      void uploadFiles(e.dataTransfer.files);
    }
  }, [entityId, uploadFiles]);

  const handleDropZoneClick = useCallback(() => {
    if (entityId == null) return;
    fileInputRef.current?.click();
  }, [entityId]);

  const handleFilePickerChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) void uploadFiles(e.target.files);
    e.target.value = ""; // reset so the same file can be picked again
  }, [uploadFiles]);

  // --- Paste handler (window-level) -------------------------------------
  // NOTE: This component installs a window-level paste handler. For
  // Phase 7's Daily Report, the parent page must scope its own
  // window-level paste handler to not double-fire when focus is inside
  // a ThoughtsEditor (the editor's local handler captures and prevents
  // default). See daily-report-card.tsx for the parent-side guard.

  useEffect(() => {
    if (entityId == null) return;
    const onPaste = (e: ClipboardEvent) => {
      // Phase 7 — cooperate with <ThoughtsEditor> on pages that mount
      // both. The editor handles image pastes locally (inline embed);
      // if we ALSO uploaded the same paste to the gallery the user
      // would get a double upload. Detect the case via the data
      // attribute the editor stamps on its contentEditable.
      const active = document.activeElement as HTMLElement | null;
      if (active?.closest?.("[data-thoughts-editor]")) return;
      const items = e.clipboardData?.items;
      if (!items) return;
      const files: File[] = [];
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (it.kind === "file" && it.type.startsWith("image/")) {
          const f = it.getAsFile();
          if (f) files.push(f);
        }
      }
      if (files.length > 0) {
        e.preventDefault();
        void uploadFiles(files);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [entityId, uploadFiles]);

  // --- Delete -----------------------------------------------------------

  const handleDeleteClick = useCallback((id: number) => {
    const removed = snapshots.find(s => s.id === id);
    setSnapshots(prev => prev.filter(s => s.id !== id));
    const deletePromise = entityType === "weekly_retro"
      ? api.deleteWeeklyRetroSnapshot(id)
      : api.deleteDailyJournalCapture(id);
    deletePromise.then(res => {
      if (!res || (typeof res === "object" && "error" in res)) {
        if (removed) setSnapshots(prev => [...prev, removed]);
        showInlineError("Delete failed");
      }
    }).catch(() => {
      if (removed) setSnapshots(prev => [...prev, removed]);
      showInlineError("Delete failed");
    });
  }, [entityType, snapshots, showInlineError]);

  // --- Lightbox ---------------------------------------------------------

  const openLightbox = useCallback((idx: number) => { setLightboxIndex(idx); }, []);
  const closeLightbox = useCallback(() => { setLightboxIndex(null); }, []);

  const lightboxImages: LightboxImage[] = useMemo(
    () => snapshots.map(s => ({
      url: s.view_url,
      alt: s.file_name || `Snapshot ${s.id}`,
    })),
    [snapshots],
  );

  // --- Render -----------------------------------------------------------

  const disabled = entityId == null;
  const dropZoneText = useMemo(() => {
    if (disabled) return effectiveDisabledMessage;
    return activeMessage;
  }, [disabled, effectiveDisabledMessage, activeMessage]);

  return (
    <div style={{ padding: 16 }}>
      {/* Drop zone */}
      <div
        role="button"
        aria-label={dropZoneAriaLabel}
        aria-disabled={disabled}
        onClick={handleDropZoneClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          borderRadius: 10,
          padding: "20px 16px",
          textAlign: "center",
          border: `1.5px dashed ${dragOver ? "#f59f00" : "var(--border-2)"}`,
          background: dragOver
            ? "color-mix(in oklab, #f59f00 6%, var(--bg))"
            : "var(--bg)",
          color: disabled ? "var(--ink-4)" : "var(--ink-3)",
          cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? 0.6 : 1,
          transition: "background 120ms, border-color 120ms",
        }}
      >
        <div
          aria-hidden
          style={{
            width: 36,
            height: 36,
            borderRadius: 999,
            margin: "0 auto 8px",
            background: "color-mix(in oklab, #f59f00 14%, var(--surface))",
            color: "#b45309",
            display: "grid",
            placeItems: "center",
            fontSize: 16,
            fontWeight: 700,
          }}
        >
          +
        </div>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>
          {dropZoneText}
        </div>
        {!disabled && (
          <div style={{ fontSize: 11, color: "var(--ink-4)" }}>
            {microcopy}
          </div>
        )}
      </div>

      {/* Hidden file picker — triggered by drop-zone click. */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/gif,image/webp"
        multiple
        onChange={handleFilePickerChange}
        style={{ display: "none" }}
      />

      {/* Inline error toast */}
      {inlineError && (
        <div
          role="alert"
          style={{
            marginTop: 8,
            padding: "6px 10px",
            fontSize: 12,
            borderRadius: 6,
            background: "color-mix(in oklab, #e5484d 10%, var(--surface))",
            color: "#dc2626",
            border: "1px solid color-mix(in oklab, #e5484d 30%, var(--border))",
          }}
        >
          {inlineError}
        </div>
      )}

      {/* Thumbnail grid */}
      <ImageGallery
        items={[
          ...snapshots.map((s): ImageGalleryItem => ({
            id: s.id,
            url: s.view_url,
            filename: s.file_name || `Snapshot ${s.id}`,
          })),
          ...uploading.map((p): ImageGalleryItem => ({
            id: -Math.abs(Array.from(p.tempId).reduce(
              (h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0,
            )),
            url: "",
            filename: p.name,
            uploading: true,
          })),
        ]}
        onDelete={handleDeleteClick}
        onImageClick={(index) => {
          if (index < snapshots.length) openLightbox(index);
        }}
      />

      <ImageLightbox
        images={lightboxImages}
        activeIndex={lightboxIndex}
        onClose={closeLightbox}
        onNavigate={setLightboxIndex}
        ariaLabel={lightboxAriaLabel}
      />
    </div>
  );
}
