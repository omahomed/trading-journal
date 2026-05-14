"use client";

// Generic image lightbox extracted from the Phase 4 Weekly Snapshot
// component. Now consumed by:
//   - WeeklySnapshot (gallery) — multi-image mode with prev/next
//   - WeeklyThoughts (editor) — single-image mode when user clicks
//     an inline pasted image
//
// Behavior preserved from the original snapshot lightbox:
//   - Full-screen overlay with rgba(0,0,0,0.85) backdrop
//   - Backdrop click closes (img click stopPropagation inside)
//   - Esc closes
//   - ← / → navigate when onNavigate is provided AND images.length > 1
//   - Wrap-around in both directions
//
// Visual contract for the inner img: object-fit: contain inside a
// max-90vw / max-90vh box. Keeps tall + wide images both visible
// without overflow. Diverges slightly from the original
// snapshot-only implementation (which used 100%/100% with 32px
// container padding) — the 90vw/90vh constraint reads cleaner
// across viewport shapes.

import { useCallback, useEffect } from "react";

export interface LightboxImage {
  url: string;
  alt?: string;
}

interface ImageLightboxProps {
  images: LightboxImage[];
  /** null → lightbox is closed (component returns null). */
  activeIndex: number | null;
  onClose: () => void;
  /** When omitted, arrow keys are no-ops (single-image mode). */
  onNavigate?: (newIndex: number) => void;
  /** aria-label on the dialog. Defaults to "Image preview". Callers
   *  with more specific context (gallery, etc.) can override so the
   *  accessibility tree carries the right name. */
  ariaLabel?: string;
}

export function ImageLightbox({
  images,
  activeIndex,
  onClose,
  onNavigate,
  ariaLabel = "Image preview",
}: ImageLightboxProps) {
  // Wrap-around helper. Used for both ← and → so the wrap math sits
  // in one place. Returns the current index unchanged when there are
  // no images to navigate (defensive — shouldn't happen in practice).
  const navigateBy = useCallback((delta: number) => {
    if (!onNavigate) return;
    if (images.length === 0) return;
    if (activeIndex == null) return;
    const next = ((activeIndex + delta) % images.length + images.length) % images.length;
    onNavigate(next);
  }, [onNavigate, images.length, activeIndex]);

  // Keyboard handler. Esc always closes. Arrows only fire when
  // onNavigate is provided AND there are multiple images.
  useEffect(() => {
    if (activeIndex == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (onNavigate && images.length > 1) {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          navigateBy(1);
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          navigateBy(-1);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeIndex, images.length, onClose, onNavigate, navigateBy]);

  if (activeIndex == null) return null;
  const current = images[activeIndex];
  if (!current) return null;

  return (
    <div
      role="dialog"
      aria-label={ariaLabel}
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        background: "rgba(0,0,0,0.85)",
        display: "grid",
        placeItems: "center",
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={current.url}
        alt={current.alt || ""}
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: "90vw",
          maxHeight: "90vh",
          objectFit: "contain",
          borderRadius: 8,
        }}
      />
    </div>
  );
}
