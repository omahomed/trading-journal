"use client";

// Phase 4 — Weekly Snapshot. Image gallery attached to a weekly retro.
// Users drop / paste / pick-from-disk images during the week and review
// them in the retro. Phase 4 ships upload + display + delete; captions
// and drag-to-reorder are pre-provisioned in the schema (columns exist)
// but deferred to follow-up commits.
//
// Storage: bytes upload to Cloudflare R2 via the backend; the server
// returns a public CDN URL (view_url) in each row dict, so the browser
// fetches bytes directly from R2 — backend is not in the serving hot
// path. Same pattern as daily-report-card EOD snapshots.
//
// Component contract:
//   - Disabled state when retroId === null (TagPicker idiom — the
//     parent retro must be saved before snapshots can attach). Drop /
//     paste / click are no-ops in the disabled state.
//   - Optimistic uploads: a placeholder row appears immediately on
//     drop/paste/pick; the placeholder is replaced by the server row
//     on success or removed on error.
//   - Two-click delete pattern: first click arms (× button stays
//     visible + tint changes), second click within 2s commits.
//     Mirrors the Phase 1 tag-delete idiom.
//   - Lightbox: click thumbnail → full-screen overlay. Esc / outside
//     click closes; ← / → navigate. Prev/next wrap around.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type SnapshotRow } from "@/lib/api";
import { ImageLightbox, type LightboxImage } from "./image-lightbox";
import { ImageGallery, type ImageGalleryItem } from "./image-gallery";

interface WeeklySnapshotProps {
  retroId: number | null;
  portfolio: string;
  /** Optional callback fired whenever the visible-count (real rows +
   *  uploading placeholders) changes. Used by the parent to render the
   *  "N attached" caption on the SectionExpander header. */
  onCountChange?: (count: number) => void;
}

const MAX_FILE_BYTES = 5 * 1024 * 1024;
const ALLOWED_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
]);

// Internal placeholder type for optimistic-upload UI. Distinguishable
// from a real SnapshotRow by the presence of `tempId` and absence of
// `id` (we treat `id` as a real DB row identifier).
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

export function WeeklySnapshot({
  retroId,
  portfolio,
  onCountChange,
}: WeeklySnapshotProps) {
  const [snapshots, setSnapshots] = useState<SnapshotRow[]>([]);
  const [uploading, setUploading] = useState<PlaceholderRow[]>([]);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [inlineError, setInlineError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // Initial fetch when retroId becomes non-null; re-fetch on retroId change.
  useEffect(() => {
    if (retroId == null) {
      setSnapshots([]);
      return;
    }
    let cancelled = false;
    api.listWeeklyRetroSnapshots(retroId, portfolio).then(res => {
      if (cancelled) return;
      if (Array.isArray(res)) setSnapshots(res);
      else setSnapshots([]);
    }).catch(() => { if (!cancelled) setSnapshots([]); });
    return () => { cancelled = true; };
  }, [retroId, portfolio]);

  const showInlineError = useCallback((msg: string) => {
    setInlineError(msg);
    window.setTimeout(() => setInlineError(null), 3000);
  }, []);

  // Upload pipeline. Filters client-side, attaches optimistic placeholder,
  // calls API, replaces placeholder or rolls back on error.
  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    if (retroId == null) return;
    const arr = Array.from(files);
    const accepted = arr.filter(isImageFile);
    const rejected = arr.filter(f => !isImageFile(f));
    if (rejected.length > 0) showInlineError(describeRejection(rejected[0]));
    if (accepted.length === 0) return;

    for (const file of accepted) {
      const tempId = `temp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setUploading(prev => [...prev, { tempId, name: file.name }]);
      try {
        const res = await api.uploadWeeklyRetroSnapshot(retroId, file, portfolio);
        if (res && typeof res === "object" && "id" in res) {
          // Server row. Replace the placeholder by removing it AND
          // pushing the real row.
          setSnapshots(prev => [...prev, res as SnapshotRow]);
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
  }, [retroId, portfolio, showInlineError]);

  // --- Drop zone handlers -----------------------------------------------

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (retroId == null) return;
    e.preventDefault();
    setDragOver(true);
  }, [retroId]);

  const handleDragLeave = useCallback(() => { setDragOver(false); }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    if (retroId == null) return;
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      void uploadFiles(e.dataTransfer.files);
    }
  }, [retroId, uploadFiles]);

  const handleDropZoneClick = useCallback(() => {
    if (retroId == null) return;
    fileInputRef.current?.click();
  }, [retroId]);

  const handleFilePickerChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) void uploadFiles(e.target.files);
    e.target.value = ""; // reset so the same file can be picked again
  }, [uploadFiles]);

  // --- Paste handler (window-level) -------------------------------------

  useEffect(() => {
    if (retroId == null) return;
    const onPaste = (e: ClipboardEvent) => {
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
  }, [retroId, uploadFiles]);

  // --- Delete -----------------------------------------------------------
  // Phase 4.4: simplified to a single-click + native confirm() flow,
  // matching the Trade Journal pattern (handled inside <ImageGallery>'s
  // delete button). Optimistic removal with rollback-on-error stays.

  const handleDeleteClick = useCallback((id: number) => {
    const removed = snapshots.find(s => s.id === id);
    setSnapshots(prev => prev.filter(s => s.id !== id));
    api.deleteWeeklyRetroSnapshot(id).then(res => {
      if (!res || (typeof res === "object" && "error" in res)) {
        if (removed) setSnapshots(prev => [...prev, removed]);
        showInlineError("Delete failed");
      }
    }).catch(() => {
      if (removed) setSnapshots(prev => [...prev, removed]);
      showInlineError("Delete failed");
    });
  }, [snapshots, showInlineError]);

  // --- Lightbox ---------------------------------------------------------
  // Keyboard nav + backdrop click + Esc closing all live inside the shared
  // <ImageLightbox> now. We just track the active index here and map
  // snapshots → LightboxImage[] for the consumer.

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

  const disabled = retroId == null;
  const dropZoneText = useMemo(() => {
    if (disabled) return "Save the retro first to add snapshots.";
    return "Paste a screenshot or drag an image here";
  }, [disabled]);

  return (
    <div style={{ padding: 16 }}>
      {/* Drop zone */}
      <div
        role="button"
        aria-label="Upload snapshot"
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
            Charts and snapshots from your week. PNG, JPEG, GIF, WEBP. Max 5MB.
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

      {/* Thumbnail grid — Phase 4.4 uses the shared <ImageGallery>
          that powers Trade Journal's chart section. Real snapshots
          map by id; in-flight placeholders get synthetic negative
          ids derived from their tempId hash so they're stable
          React keys and won't collide with real DB ids. */}
      <ImageGallery
        items={[
          ...snapshots.map((s): ImageGalleryItem => ({
            id: s.id,
            url: s.view_url,
            filename: s.file_name || `Snapshot ${s.id}`,
          })),
          ...uploading.map((p): ImageGalleryItem => ({
            // Hash tempId to a negative integer — distinct space
            // from positive DB ids; React keys stay stable.
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
          // Index aligns with the snapshots array (placeholders come
          // after). Only open lightbox for real items.
          if (index < snapshots.length) openLightbox(index);
        }}
        deleteConfirmMessage="Delete this snapshot?"
      />

      {/* Lightbox — shared component handles backdrop click, Esc,
          ← / → with wrap-around. Pass onNavigate so multi-snapshot
          galleries get arrow-key navigation. ariaLabel preserved as
          "Snapshot preview" so existing accessibility tests don't
          regress from the refactor. */}
      <ImageLightbox
        images={lightboxImages}
        activeIndex={lightboxIndex}
        onClose={closeLightbox}
        onNavigate={setLightboxIndex}
        ariaLabel="Snapshot preview"
      />
    </div>
  );
}
