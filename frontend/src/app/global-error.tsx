"use client";

import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";
import { handleChunkLoadError } from "@/lib/chunk-reload";

// Root-level error boundary. Catches anything that escapes every layout's
// error.tsx and the root layout itself. Forwards the error to Sentry so we
// see it even if the user's browser can't render the normal error UI.
//
// Backstop chunk-reload handler — error.tsx handles this first for most
// failures, but if a chunk error escapes that boundary (e.g. failure
// inside RootLayout itself), we still get one auto-reload attempt here.

export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  useEffect(() => {
    Sentry.captureException(error);
    handleChunkLoadError(error);
  }, [error]);

  return (
    <html>
      <body>
        <div style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
          <h1 style={{ marginBottom: 8 }}>Something went wrong.</h1>
          <p style={{ color: "#666" }}>
            The error has been reported. Please refresh or try again in a moment.
          </p>
        </div>
      </body>
    </html>
  );
}
