"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import {
  AlertCircle,
  Check,
  ImageOff,
  ImagePlus,
  Plus,
  Trash2,
  X,
} from "lucide-react";

/**
 * Mobile image upload primitive — Phase 2 T2-3.
 *
 * Entity-agnostic by design. The backend exposes 4 divergent upload
 * endpoints (trade / EOD-snapshot / weekly-retro / daily-journal)
 * with different parameter shapes, so this primitive doesn't call any
 * of them directly. Consumers wrap their specific `api.upload*` call
 * via the `onUpload` prop:
 *
 *   <MobileImageUpload
 *     rows={images}
 *     onUpload={(file) => api.uploadDailyJournalCapture(
 *       journalId, file, portfolio,
 *     )}
 *     onDelete={(id) => api.deleteDailyJournalCapture(id)}
 *     onThumbnailTap={(row, idx) => setLightboxIndex(idx)}
 *   />
 *
 * ImageLightbox is consumer-invoked (via `onThumbnailTap`), not
 * embedded — keeps the primitive surface tight. Multi-state UX:
 * empty / populated / uploading / delete-confirm / error /
 * disabled. Click-only file picker (no desktop drag-drop / paste —
 * those don't translate to mobile).
 */

// Minimal row shape every consumer must produce. SnapshotRow,
// DailyJournalCaptureRow, and trade-image rows all satisfy this.
export type ImageUploadRow = {
  id: number;
  view_url: string;
  file_name: string | null;
};

export type ImageUploadResult<TRow extends ImageUploadRow> =
  | TRow
  | { error: string; detail?: unknown };

export type ImageDeleteResult =
  | { deleted: true; id?: number }
  | { error: string; detail?: unknown };

export interface MobileImageUploadProps<TRow extends ImageUploadRow> {
  /** Current uploaded rows, sourced + sorted by the consumer. */
  rows: TRow[];
  /** Fires once per validated file. Consumer wraps its specific
   *  endpoint. Return the new row on success or `{ error }` on failure. */
  onUpload: (file: File) => Promise<ImageUploadResult<TRow>>;
  /** Fires once user confirms delete on a specific row. */
  onDelete: (id: number) => Promise<ImageDeleteResult>;
  /** Disable upload + delete; useful when the parent entity hasn't
   *  been saved yet (e.g., new weekly retro before its first save). */
  disabled?: boolean;
  /** Override the disabled-state title. */
  disabledMessage?: string;
  /** Max bytes per file. Files larger than this are rejected
   *  client-side before `onUpload` fires. Default 5 MB matches the
   *  desktop SnapshotGallery cap. */
  maxFileBytes?: number;
  /** Accepted MIME types. HEIC and other unsupported formats are
   *  rejected with a specific banner message. Defaults to PNG /
   *  JPEG / GIF / WEBP. */
  acceptedMimes?: Set<string>;
  /** Empty-state CTA label. Default "Add images". */
  emptyStateLabel?: string;
  /** Fires when the user taps an uploaded thumbnail. Consumer wires
   *  a lightbox (or other view affordance) here. */
  onThumbnailTap?: (row: TRow, index: number) => void;
}

const DEFAULT_MAX_BYTES = 5 * 1024 * 1024;
const DEFAULT_ACCEPTED: ReadonlySet<string> = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
]);
const ERROR_AUTODISMISS_MS = 3000;

type PendingUpload = {
  tempId: string;
  file: File;
};

type ErrorState = {
  message: string;
  hint: string;
} | null;

export function MobileImageUpload<TRow extends ImageUploadRow>({
  rows,
  onUpload,
  onDelete,
  disabled = false,
  disabledMessage = "Save entry first to add images",
  maxFileBytes = DEFAULT_MAX_BYTES,
  acceptedMimes,
  emptyStateLabel = "Add images",
  onThumbnailTap,
}: MobileImageUploadProps<TRow>) {
  const allowedMimes = acceptedMimes ?? (DEFAULT_ACCEPTED as Set<string>);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  const [confirmingDelete, setConfirmingDelete] = useState<number | null>(null);
  const [deleteFor, setDeleteFor] = useState<TRow | null>(null);
  const [errorState, setErrorState] = useState<ErrorState>(null);

  // Auto-dismiss the error banner after the timeout. Reset whenever
  // a fresh error fires.
  useEffect(() => {
    if (!errorState) return;
    const id = window.setTimeout(() => setErrorState(null), ERROR_AUTODISMISS_MS);
    return () => window.clearTimeout(id);
  }, [errorState]);

  const formatHint = useCallback(() => {
    // Build the format hint from acceptedMimes for self-consistency
    // if the consumer overrides. Default reads as "JPG / PNG / GIF /
    // WEBP · {N} MB max".
    const labels = Array.from(allowedMimes)
      .map((m) => m.replace("image/", "").toUpperCase())
      .map((s) => (s === "JPEG" ? "JPG" : s));
    const mb = Math.round(maxFileBytes / (1024 * 1024));
    return `${labels.join(" / ")} · ${mb} MB max`;
  }, [allowedMimes, maxFileBytes]);

  // ── File validation + dispatch ──────────────────────────────────

  const validate = useCallback(
    (file: File): string | null => {
      if (!allowedMimes.has(file.type)) {
        return `${file.name} — type not supported`;
      }
      if (file.size > maxFileBytes) {
        const mb = Math.round(maxFileBytes / (1024 * 1024));
        return `${file.name} — exceeds ${mb} MB`;
      }
      return null;
    },
    [allowedMimes, maxFileBytes],
  );

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files);
      const rejected: string[] = [];
      const accepted: File[] = [];
      for (const f of arr) {
        const err = validate(f);
        if (err) rejected.push(err);
        else accepted.push(f);
      }

      if (rejected.length > 0) {
        const message =
          rejected.length === 1
            ? rejected[0]
            : `${rejected.length} files rejected: ${rejected.slice(0, 2).join(", ")}${
                rejected.length > 2 ? "…" : ""
              }`;
        setErrorState({
          message,
          hint: `Use ${Array.from(allowedMimes)
            .map((m) => m.replace("image/", "").toUpperCase().replace("JPEG", "JPG"))
            .join(" / ")} under ${Math.round(maxFileBytes / (1024 * 1024))} MB.`,
        });
      }

      if (accepted.length === 0) return;

      // Optimistic placeholders for the strip. Each gets a stable
      // tempId so the cleanup loop below can target it specifically.
      const placeholders: PendingUpload[] = accepted.map((file) => ({
        tempId: `pending-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        file,
      }));
      setPendingUploads((prev) => [...prev, ...placeholders]);

      // Parallel uploads. Each result clears its own placeholder; on
      // error we surface the banner once per failed file.
      await Promise.all(
        placeholders.map(async ({ tempId, file }) => {
          try {
            const res = await onUpload(file);
            setPendingUploads((prev) => prev.filter((p) => p.tempId !== tempId));
            if (res && typeof res === "object" && "error" in res) {
              setErrorState({
                message: `${file.name} — upload failed`,
                hint: typeof res.error === "string" ? res.error : "Try again",
              });
            }
            // On success the consumer adds the row to its `rows` prop;
            // the placeholder removal above handles the strip cleanup.
          } catch (err) {
            setPendingUploads((prev) => prev.filter((p) => p.tempId !== tempId));
            setErrorState({
              message: `${file.name} — upload failed`,
              hint: err instanceof Error ? err.message : "Try again",
            });
          }
        }),
      );
    },
    [allowedMimes, maxFileBytes, onUpload, validate],
  );

  // ── File picker click handler ───────────────────────────────────

  const triggerPicker = useCallback(() => {
    if (disabled) return;
    fileInputRef.current?.click();
  }, [disabled]);

  const handleInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        await handleFiles(files);
      }
      // Reset so picking the same file again re-fires onChange.
      e.target.value = "";
    },
    [handleFiles],
  );

  // ── Delete flow ─────────────────────────────────────────────────

  const startDelete = useCallback((row: TRow) => {
    setConfirmingDelete(row.id);
    setDeleteFor(row);
  }, []);

  const cancelDelete = useCallback(() => {
    setConfirmingDelete(null);
    setDeleteFor(null);
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!deleteFor) return;
    try {
      const res = await onDelete(deleteFor.id);
      if (res && typeof res === "object" && "error" in res) {
        setErrorState({
          message: `Delete failed for ${deleteFor.file_name ?? "image"}`,
          hint: typeof res.error === "string" ? res.error : "Try again",
        });
      }
    } catch (err) {
      setErrorState({
        message: `Delete failed for ${deleteFor.file_name ?? "image"}`,
        hint: err instanceof Error ? err.message : "Try again",
      });
    } finally {
      cancelDelete();
    }
  }, [deleteFor, onDelete, cancelDelete]);

  // ── Render ──────────────────────────────────────────────────────

  const hasContent = rows.length > 0 || pendingUploads.length > 0;

  return (
    <div className="flex flex-col gap-2" data-testid="mobile-image-upload">
      {/* Hidden file input — triggered programmatically via ref. */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleInputChange}
        className="hidden"
        aria-label="Pick image files"
        data-testid="image-upload-input"
      />

      {disabled ? (
        <DisabledState message={disabledMessage} />
      ) : !hasContent ? (
        <EmptyState
          label={emptyStateLabel}
          hint={formatHint()}
          onPick={triggerPicker}
        />
      ) : (
        <PopulatedState
          rows={rows}
          pendingUploads={pendingUploads}
          confirmingDelete={confirmingDelete}
          onPick={triggerPicker}
          onThumbnailTap={onThumbnailTap}
          onStartDelete={startDelete}
        />
      )}

      {/* Delete-confirm action bar appears below the strip. */}
      {deleteFor && (
        <DeleteConfirmBar
          fileName={deleteFor.file_name ?? "image"}
          onCancel={cancelDelete}
          onConfirm={confirmDelete}
        />
      )}

      {errorState && <ErrorBanner state={errorState} />}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────

function EmptyState({
  label,
  hint,
  onPick,
}: {
  label: string;
  hint: string;
  onPick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      data-testid="image-upload-empty-cta"
      className="flex h-[120px] w-full flex-col items-center justify-center gap-1.5 rounded-m-lg border-[0.5px] border-dashed border-m-border-strong bg-m-surface px-4 active:opacity-80"
    >
      <ImagePlus size={22} strokeWidth={1.5} className="text-m-text-muted" aria-hidden="true" />
      <span className="text-[13px] font-medium text-m-text">{label}</span>
      <span className="font-m-num text-[11px] tabular-nums text-m-text-dim">{hint}</span>
    </button>
  );
}

function DisabledState({ message }: { message: string }) {
  return (
    <div
      data-testid="image-upload-disabled"
      className="flex h-[120px] w-full flex-col items-center justify-center gap-1.5 rounded-m-lg border-[0.5px] border-dashed border-m-border bg-m-surface px-4 opacity-60"
    >
      <ImageOff size={22} strokeWidth={1.5} className="text-m-text-faint" aria-hidden="true" />
      <span className="text-center text-[13px] font-medium text-m-text-muted">{message}</span>
      <span className="text-[11px] text-m-text-faint">
        Image attachments unlock after first save
      </span>
    </div>
  );
}

function PopulatedState<TRow extends ImageUploadRow>({
  rows,
  pendingUploads,
  confirmingDelete,
  onPick,
  onThumbnailTap,
  onStartDelete,
}: {
  rows: TRow[];
  pendingUploads: PendingUpload[];
  confirmingDelete: number | null;
  onPick: () => void;
  onThumbnailTap?: (row: TRow, index: number) => void;
  onStartDelete: (row: TRow) => void;
}) {
  const isConfirming = confirmingDelete != null;
  return (
    <>
      <div
        data-testid="image-upload-strip"
        className="-mx-5 flex gap-2 overflow-x-auto whitespace-nowrap px-5"
        style={{ WebkitOverflowScrolling: "touch" }}
      >
        {rows.map((row, idx) => (
          <Thumbnail
            key={row.id}
            row={row}
            index={idx}
            confirming={confirmingDelete === row.id}
            dimmed={isConfirming && confirmingDelete !== row.id}
            onTap={onThumbnailTap}
            onRemove={onStartDelete}
          />
        ))}
        {pendingUploads.map((p) => (
          <PendingThumbnail key={p.tempId} pending={p} />
        ))}
        <AddThumbnail onClick={onPick} disabled={isConfirming} />
      </div>
      {pendingUploads.length > 0 && (
        <div className="font-m-num text-[11px] tabular-nums text-m-text-dim">
          Uploading {pendingUploads[0].file.name}
          {pendingUploads.length > 1 ? ` (+${pendingUploads.length - 1} more)` : ""}…
        </div>
      )}
    </>
  );
}

function Thumbnail<TRow extends ImageUploadRow>({
  row,
  index,
  confirming,
  dimmed,
  onTap,
  onRemove,
}: {
  row: TRow;
  index: number;
  confirming: boolean;
  dimmed: boolean;
  onTap?: (row: TRow, index: number) => void;
  onRemove: (row: TRow) => void;
}) {
  const containerStyle: CSSProperties = {
    width: 84,
    height: 84,
    opacity: dimmed ? 0.4 : 1,
    transition: "opacity 150ms ease",
  };
  return (
    <div
      className="relative shrink-0 overflow-hidden rounded-m-md border-[0.5px] border-m-border bg-m-surface"
      style={containerStyle}
      data-testid={`image-upload-thumb-${row.id}`}
    >
      {/* Tap surface — fires onThumbnailTap (consumer wires lightbox). */}
      <button
        type="button"
        onClick={() => onTap?.(row, index)}
        aria-label={`View ${row.file_name ?? "image"}`}
        className="block h-full w-full"
        disabled={confirming}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={row.view_url}
          alt={row.file_name ?? "Image"}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </button>

      {/* Always-visible X — fires startDelete. */}
      {!confirming && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(row);
          }}
          aria-label={`Remove ${row.file_name ?? "image"}`}
          data-testid={`image-upload-remove-${row.id}`}
          className="absolute right-1 top-1 flex h-[22px] w-[22px] items-center justify-center rounded-full bg-black/65 text-white"
        >
          <X size={12} strokeWidth={2} aria-hidden="true" />
        </button>
      )}

      {/* Delete-target overlay — replaces X with centered trash icon. */}
      {confirming && (
        <div
          data-testid={`image-upload-delete-target-${row.id}`}
          className="absolute inset-0 flex items-center justify-center bg-black/45"
        >
          <Trash2 size={22} strokeWidth={1.8} className="text-m-down" aria-hidden="true" />
        </div>
      )}
    </div>
  );
}

function PendingThumbnail({ pending }: { pending: PendingUpload }) {
  return (
    <div
      data-testid={`image-upload-pending-${pending.tempId}`}
      className="flex shrink-0 items-center justify-center rounded-m-md border-[0.5px] border-m-border bg-m-surface-2"
      style={{ width: 84, height: 84 }}
    >
      <span
        aria-label="Uploading"
        className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-m-accent border-t-transparent"
      />
    </div>
  );
}

function AddThumbnail({
  onClick,
  disabled,
}: {
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label="Add more images"
      data-testid="image-upload-add-more"
      className="flex shrink-0 items-center justify-center rounded-m-md border-[0.5px] border-dashed border-m-border-strong bg-m-surface text-m-text-muted active:opacity-80 disabled:opacity-40"
      style={{ width: 84, height: 84 }}
    >
      <Plus size={20} strokeWidth={1.6} aria-hidden="true" />
    </button>
  );
}

function DeleteConfirmBar({
  fileName,
  onCancel,
  onConfirm,
}: {
  fileName: string;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div
      role="alertdialog"
      aria-label="Confirm delete"
      data-testid="image-upload-delete-confirm-bar"
      className="flex items-center justify-between gap-2 rounded-m-md px-3 py-2"
      style={{
        background: "color-mix(in oklab, var(--m-down) 10%, var(--m-surface))",
        border: "0.5px solid var(--m-down)",
      }}
    >
      <span className="min-w-0 flex-1 truncate text-[12px] text-m-text">
        Delete <strong>{fileName}</strong>?
      </span>
      <div className="flex shrink-0 items-center gap-1.5">
        <button
          type="button"
          onClick={onCancel}
          data-testid="image-upload-delete-cancel"
          className="rounded-m-pill border-[0.5px] border-m-border bg-m-surface px-3 py-1 text-[12px] font-medium text-m-text-muted"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onConfirm}
          data-testid="image-upload-delete-confirm"
          className="rounded-m-pill px-3 py-1 text-[12px] font-medium text-white"
          style={{ background: "var(--m-down)" }}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

function ErrorBanner({ state }: { state: NonNullable<ErrorState> }) {
  return (
    <div
      role="alert"
      data-testid="image-upload-error-banner"
      className="flex items-start gap-2 rounded-m-md px-3 py-2"
      style={{
        background: "color-mix(in oklab, var(--m-down) 10%, var(--m-surface))",
        borderLeft: "3px solid var(--m-down)",
      }}
    >
      <AlertCircle
        size={14}
        strokeWidth={1.6}
        className="mt-0.5 shrink-0 text-m-down"
        aria-hidden="true"
      />
      <div className="min-w-0 flex-1 text-[12px] text-m-text">
        <div className="font-medium">{state.message}</div>
        <div className="mt-0.5 text-m-text-dim">{state.hint}</div>
      </div>
    </div>
  );
}

// Suppress an unused-import warning. The Check icon is reserved for
// the "Added {filename}" success indicator referenced in the directive
// — UX deferred to a follow-up so the active scope ships a clean
// success-by-row-replacement (no separate success row). Keep the
// import path warm so the follow-up doesn't have to re-add it.
// eslint-disable-next-line @typescript-eslint/no-unused-expressions
void Check;
