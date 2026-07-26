"use client";

import { useState, useCallback, useEffect } from "react";
import { api, getActivePortfolio } from "@/lib/api";

interface Props {
  /** CSS selector or element to capture. If not provided, captures document.body. */
  targetSelector?: string;
  snapshotType: "dashboard" | "campaign";
  label: string;
  portfolio?: string;
}

const SAVE_LOCAL_KEY = "captureSnapshot.saveLocal";

export function CaptureSnapshotButton({ targetSelector, snapshotType, label, portfolio = getActivePortfolio() }: Props) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [saveLocal, setSaveLocal] = useState(false);

  useEffect(() => {
    try {
      setSaveLocal(localStorage.getItem(SAVE_LOCAL_KEY) === "1");
    } catch { /* localStorage may be unavailable */ }
  }, []);

  const toggleSaveLocal = (checked: boolean) => {
    setSaveLocal(checked);
    try {
      localStorage.setItem(SAVE_LOCAL_KEY, checked ? "1" : "0");
    } catch { /* ignore */ }
  };

  const capture = useCallback(async () => {
    setBusy(true);
    setMsg(null);
    try {
      const { toBlob } = await import("html-to-image");
      const node = targetSelector ? (document.querySelector(targetSelector) as HTMLElement | null) : document.body;
      if (!node) {
        setMsg({ ok: false, text: "Target not found" });
        setBusy(false);
        return;
      }

      const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#fff";

      // ACS truncation fix 2026-07-26 (v2). html-to-image faithfully
      // reproduces the DOM, which means it also reproduces:
      //   * position: sticky on the table headers (offsets rows in
      //     the raster)
      //   * overflow-x: auto on the equity/options table wrappers
      //     (the wrapper's rendered height ends up shorter than the
      //     tbody, clipping the last row)
      //   * max-h-*/overflow-y-auto on modal panels (usually inert
      //     since modals aren't open during capture, but harmless to
      //     defuse)
      // Neutralize those three during capture via a scoped class +
      // injected stylesheet. Applied before toBlob, removed in
      // finally so a mid-capture throw doesn't leave the DOM styled.
      const styleTag = document.createElement("style");
      styleTag.setAttribute("data-capture-neutralizer", "");
      styleTag.textContent = `
        .capturing-snapshot [class*="sticky"] {
          position: static !important;
          top: auto !important;
        }
        .capturing-snapshot [class*="overflow-x-auto"],
        .capturing-snapshot [class*="overflow-y-auto"],
        .capturing-snapshot [class*="overflow-hidden"] {
          overflow: visible !important;
        }
        .capturing-snapshot [class*="max-h-"] {
          max-height: none !important;
        }
      `;
      document.head.appendChild(styleTag);
      node.classList.add("capturing-snapshot");
      // One rAF so the browser applies the neutralized layout before
      // html-to-image walks the DOM. Without this, the clone can
      // snapshot the pre-neutralization frame on the first tick.
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));

      // Pixel-ratio downgrade guards against browser canvas ceilings:
      // Chrome caps ~65k px per side, Safari ~16k. Drop to 1x when
      // scaled long edge would breach 12k px (leaves headroom).
      const captureWidth = node.scrollWidth;
      const captureHeight = node.scrollHeight;
      const SAFE_CANVAS_EDGE = 12000;
      const rawPixelRatio = 2;
      const scaledLongEdge = Math.max(captureWidth, captureHeight) * rawPixelRatio;
      const pixelRatio = scaledLongEdge > SAFE_CANVAS_EDGE ? 1 : rawPixelRatio;
      let blob: Blob | null = null;
      try {
        blob = await toBlob(node, {
          backgroundColor: bg,
          pixelRatio,
          cacheBust: true,
        });
      } finally {
        node.classList.remove("capturing-snapshot");
        styleTag.remove();
      }
      if (!blob) {
        setMsg({ ok: false, text: "Capture produced no image" });
        setBusy(false);
        return;
      }

      const today = new Date();
      const day = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

      if (saveLocal) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${snapshotType}-${day}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }

      // Replace-if-exists: the DB has a unique constraint on (portfolio,
      // trade_id, image_type), so uploading twice for the same day would
      // collide. Delete any existing snapshots of this exact type for today
      // before the upload so a re-capture Just Works.
      const wantedType = `eod_${snapshotType}`;
      let replaced = false;
      try {
        const existing = await api.listEodSnapshots(day, portfolio);
        if (Array.isArray(existing)) {
          for (const snap of existing) {
            if ((snap as any).image_type === wantedType && (snap as any).id) {
              try {
                await api.deleteImage((snap as any).id);
                replaced = true;
              } catch { /* ignore individual delete failures */ }
            }
          }
        }
      } catch { /* if listing fails, just try the upload */ }

      const res = await api.uploadEodSnapshot(blob, day, snapshotType, portfolio);
      if (res.error) {
        setMsg({ ok: false, text: res.error });
      } else {
        setMsg({ ok: true, text: replaced ? `Replaced ${day}` : `Saved to ${day}` });
      }
    } catch (err: any) {
      setMsg({ ok: false, text: err.message || "Capture failed" });
    }
    setBusy(false);
    setTimeout(() => setMsg(null), 4000);
  }, [targetSelector, snapshotType, portfolio, saveLocal]);

  return (
    <div className="flex items-center gap-2">
      <button onClick={capture} disabled={busy}
              className="flex items-center gap-1.5 h-[32px] px-3.5 rounded-[10px] text-xs font-medium transition-colors hover:brightness-95 disabled:opacity-60"
              style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink-2)" }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
          <circle cx="12" cy="13" r="4" />
        </svg>
        {busy ? "Capturing..." : label}
      </button>
      <label className="flex items-center gap-1.5 text-[11px] font-medium cursor-pointer select-none"
             style={{ color: "var(--ink-3)" }}>
        <input type="checkbox" checked={saveLocal} onChange={(e) => toggleSaveLocal(e.target.checked)}
               className="cursor-pointer" />
        Save copy to Downloads
      </label>
      {msg && (
        <span className="text-[11px] font-medium" style={{ color: msg.ok ? "#16a34a" : "#e5484d" }}>
          {msg.ok ? "✓" : "✗"} {msg.text}
        </span>
      )}
    </div>
  );
}
