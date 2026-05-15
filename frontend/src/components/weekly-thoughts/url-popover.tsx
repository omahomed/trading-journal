"use client";

// Phase 3.5: shared popover used by both Insert-Link and Embed-Video
// affordances. The popover is presentational — it collects a URL and
// (optionally) a link-text string, then hands them to the parent's
// onConfirm callback. The parent owns:
//   - When to open it (toolbar button click)
//   - Restoring the editor's Range and running execCommand on submit
//   - Validating the URL (e.g., rejecting non-YouTube/Vimeo for video)
//
// onConfirm returns either null (success — popover may close) or an
// error string, which is displayed inline inside the popover instead
// of as a toast. This keeps validation feedback local to the action.
//
// Positioning: the popover is `position: absolute` and the caller
// wraps the trigger button + this component in a `position: relative`
// container. Defaults to "below the trigger" via top: 100% + 6px gap.
//
// Selection preservation across input focus: the parent saves the
// Range *before* mounting this component (the toolbar button's
// onMouseDown.preventDefault keeps focus in the editor at the moment
// of click, after which the parent grabs the Range and only then
// flips the open flag, mounting this).

import { useCallback, useEffect, useRef, useState } from "react";
import { usePopover } from "@/lib/use-popover";

interface UrlPopoverProps {
  /** Header text shown above the inputs. */
  title: string;
  /** Placeholder text for the URL input. */
  urlPlaceholder?: string;
  /** Initial URL — used when prefilling from an existing <a href>. */
  initialUrl?: string;
  /** Initial link text. */
  initialLinkText?: string;
  /** When true, render an additional "Link text" input below the URL. */
  showLinkTextInput: boolean;
  /** Called with (url, linkText). Return null on success (the popover
   *  closes), or an error string to display inline. */
  onConfirm: (url: string, linkText: string) => string | null;
  /** Called when the user cancels or clicks outside. */
  onClose: () => void;
  /** Button label for the confirm action. Default "OK". */
  okLabel?: string;
  /** CSS override for the popover anchor. Defaults to below the
   *  position-relative parent. */
  anchorStyle?: React.CSSProperties;
}

export function UrlPopover({
  title,
  urlPlaceholder = "https://...",
  initialUrl = "",
  initialLinkText = "",
  showLinkTextInput,
  onConfirm,
  onClose,
  okLabel = "OK",
  anchorStyle,
}: UrlPopoverProps) {
  const [url, setUrl] = useState(initialUrl);
  const [linkText, setLinkText] = useState(initialLinkText);
  const [error, setError] = useState<string | null>(null);
  const urlInputRef = useRef<HTMLInputElement>(null);

  // Autofocus the URL input on mount.
  useEffect(() => { urlInputRef.current?.focus(); }, []);

  // Phase 6.5-followup: outside-click + Escape extracted to usePopover.
  // initialOpen: true because UrlPopover is conditionally mounted by
  // its parent — the parent owns the open state by mount/unmount; the
  // hook just needs its listeners live from first render. onClose
  // passed through so user-dismiss propagates to the parent
  // (which unmounts).
  const { surfaceRef: wrapperRef } = usePopover<HTMLDivElement>({
    initialOpen: true,
    onClose,
  });

  const handleSubmit = useCallback(() => {
    const trimmedUrl = url.trim();
    if (!trimmedUrl) { setError("Please enter a URL"); return; }
    const result = onConfirm(trimmedUrl, linkText.trim());
    if (result) setError(result);
    // onConfirm is responsible for calling onClose on success.
  }, [url, linkText, onConfirm]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") { e.preventDefault(); handleSubmit(); }
  }, [handleSubmit]);

  const style: React.CSSProperties = {
    position: "absolute",
    top: "100%",
    left: 0,
    marginTop: 6,
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 10,
    boxShadow: "var(--card-shadow)",
    padding: 10,
    minWidth: 300,
    zIndex: 30,
    ...anchorStyle,
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    height: 30,
    padding: "0 8px",
    fontSize: 12,
    background: "var(--bg)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--ink)",
    outline: "none",
  };

  return (
    <div
      ref={wrapperRef}
      role="dialog"
      aria-label={title}
      style={style}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 600,
          color: "var(--ink-4)",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          marginBottom: 6,
        }}
      >
        {title}
      </div>
      <input
        ref={urlInputRef}
        type="text"
        value={url}
        onChange={e => { setUrl(e.target.value); setError(null); }}
        onKeyDown={handleKeyDown}
        placeholder={urlPlaceholder}
        aria-label="URL"
        style={inputStyle}
      />
      {showLinkTextInput && (
        <input
          type="text"
          value={linkText}
          onChange={e => setLinkText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Link text (optional)"
          aria-label="Link text"
          style={{ ...inputStyle, marginTop: 6 }}
        />
      )}
      {error && (
        <div
          role="alert"
          style={{
            marginTop: 6,
            fontSize: 11,
            color: "#dc2626",
          }}
        >
          {error}
        </div>
      )}
      <div
        className="flex"
        style={{ gap: 6, marginTop: 8, justifyContent: "flex-end" }}
      >
        <button
          type="button"
          onClick={onClose}
          style={{
            padding: "4px 10px",
            fontSize: 12,
            fontWeight: 600,
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--ink-3)",
            cursor: "pointer",
          }}
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          style={{
            padding: "4px 10px",
            fontSize: 12,
            fontWeight: 600,
            background: "var(--ink)",
            border: "1px solid var(--ink)",
            borderRadius: 6,
            color: "var(--surface)",
            cursor: "pointer",
          }}
        >
          {okLabel}
        </button>
      </div>
    </div>
  );
}
