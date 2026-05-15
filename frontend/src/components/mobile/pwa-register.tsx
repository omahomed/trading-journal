"use client";

import { useEffect } from "react";
import { log } from "@/lib/log";

/**
 * Registers `/sw.js` once on the client. Feature-detects
 * `navigator.serviceWorker` so SSR and unsupported browsers (older
 * iOS Safari pre-PWA, locked-down enterprise browsers) just no-op.
 *
 * The component renders nothing — it exists purely as a mount point.
 * Mounted from `src/app/layout.tsx` so it runs on every authenticated
 * route. Login pages also pick it up, which is fine: registering the
 * SW pre-login means the shell is already cached by the time the
 * user first authenticates.
 */
export function PwaRegister() {
  useEffect(() => {
    if (typeof navigator === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    const onLoad = () => {
      navigator.serviceWorker
        .register("/sw.js", { scope: "/" })
        .catch((err) => {
          // Don't surface this to the user — registration failure
          // degrades the PWA gracefully (offline support disappears,
          // everything else still works). Log only in dev.
          log.warn.devOnly("pwa", "service worker registration failed", err);
        });
    };

    if (document.readyState === "complete") {
      onLoad();
    } else {
      window.addEventListener("load", onLoad, { once: true });
      return () => window.removeEventListener("load", onLoad);
    }
  }, []);

  return null;
}
