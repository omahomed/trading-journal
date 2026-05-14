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
const DELETE_ARM_TIMEOUT_MS = 2000;

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
  const [armedDeleteId, setArmedDeleteId] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [inlineError, setInlineError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const armResetTimer = useRef<number | null>(null);

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

  // --- Delete (two-click + 2s auto-reset) -------------------------------

  const handleDeleteClick = useCallback((id: number) => {
    if (armedDeleteId === id) {
      // Second click — commit.
      setArmedDeleteId(null);
      if (armResetTimer.current) {
        window.clearTimeout(armResetTimer.current);
        armResetTimer.current = null;
      }
      // Optimistic remove; rollback on failure.
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
    } else {
      // First click — arm.
      setArmedDeleteId(id);
      if (armResetTimer.current) window.clearTimeout(armResetTimer.current);
      armResetTimer.current = window.setTimeout(() => {
        setArmedDeleteId(null);
        armResetTimer.current = null;
      }, DELETE_ARM_TIMEOUT_MS);
    }
  }, [armedDeleteId, snapshots, showInlineError]);

  // Cleanup the arm timer on unmount.
  useEffect(() => () => {
    if (armResetTimer.current) window.clearTimeout(armResetTimer.current);
  }, []);

  // --- Lightbox ---------------------------------------------------------

  const openLightbox = useCallback((idx: number) => { setLightboxIndex(idx); }, []);
  const closeLightbox = useCallback(() => { setLightboxIndex(null); }, []);

  // Keyboard nav inside the lightbox.
  useEffect(() => {
    if (lightboxIndex == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); closeLightbox(); }
      else if (e.key === "ArrowRight") {
        e.preventDefault();
        setLightboxIndex(i => (i == null ? null : (i + 1) % snapshots.length));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setLightboxIndex(i => (i == null ? null : (i - 1 + snapshots.length) % snapshots.length));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightboxIndex, snapshots.length, closeLightbox]);

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

      {/* Thumbnail grid */}
      {(snapshots.length > 0 || uploading.length > 0) && (
        <div
          className="grid grid-cols-1 sm:grid-cols-3"
          style={{ gap: 12, marginTop: 12 }}
        >
          {snapshots.map((s, idx) => {
            const armed = armedDeleteId === s.id;
            const showDelete = hoveredId === s.id || armed;
            return (
              <div
                key={s.id}
                onMouseEnter={() => setHoveredId(s.id)}
                onMouseLeave={() => {
                  setHoveredId(prev => (prev === s.id ? null : prev));
                  // Hover-out un-arms — prevents a stale armed thumb from
                  // sitting in the grid if the user moves on.
                  setArmedDeleteId(prev => (prev === s.id ? null : prev));
                }}
                style={{
                  position: "relative",
                  borderRadius: 10,
                  overflow: "hidden",
                  border: "1px solid var(--border)",
                  background: "var(--bg-2)",
                  cursor: "pointer",
                }}
              >
                <button
                  type="button"
                  onClick={() => openLightbox(idx)}
                  aria-label={`Open snapshot ${s.file_name || s.id}`}
                  style={{
                    display: "block",
                    width: "100%",
                    padding: 0,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                  }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={s.view_url}
                    alt={s.file_name || `Snapshot ${s.id}`}
                    style={{
                      display: "block",
                      width: "100%",
                      height: 140,
                      objectFit: "cover",
                    }}
                  />
                </button>

                {/* Hover × — two-click delete */}
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); handleDeleteClick(s.id); }}
                  aria-label={armed
                    ? `Confirm delete snapshot ${s.file_name || s.id}`
                    : `Delete snapshot ${s.file_name || s.id}`}
                  title={armed ? "Click again to confirm" : "Delete"}
                  style={{
                    position: "absolute",
                    top: 6,
                    right: 6,
                    width: 22,
                    height: 22,
                    borderRadius: 999,
                    background: armed
                      ? "#dc2626"
                      : "rgba(15,21,36,0.72)",
                    color: "#fff",
                    display: "grid",
                    placeItems: "center",
                    opacity: showDelete ? 1 : 0,
                    transition: "opacity 120ms, background 120ms",
                    border: "1px solid rgba(255,255,255,0.18)",
                    cursor: "pointer",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  ×
                </button>
              </div>
            );
          })}

          {/* Optimistic placeholders */}
          {uploading.map(p => (
            <div
              key={p.tempId}
              data-testid="upload-placeholder"
              style={{
                position: "relative",
                borderRadius: 10,
                overflow: "hidden",
                border: "1px dashed var(--border-2)",
                background: "var(--bg-2)",
                height: 140,
                display: "grid",
                placeItems: "center",
                fontSize: 11,
                color: "var(--ink-4)",
              }}
            >
              Uploading {p.name}…
            </div>
          ))}
        </div>
      )}

      {/* Lightbox */}
      {lightboxIndex != null && snapshots[lightboxIndex] && (
        <div
          role="dialog"
          aria-label="Snapshot preview"
          onClick={closeLightbox}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 50,
            background: "rgba(0,0,0,0.85)",
            display: "grid",
            placeItems: "center",
            padding: 32,
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={snapshots[lightboxIndex].view_url}
            alt={snapshots[lightboxIndex].file_name || `Snapshot ${snapshots[lightboxIndex].id}`}
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
              borderRadius: 8,
            }}
          />
        </div>
      )}
    </div>
  );
}
