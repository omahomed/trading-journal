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
      const blob = await toBlob(node, {
        backgroundColor: bg,
        pixelRatio: 2,
        cacheBust: true,
      });
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
