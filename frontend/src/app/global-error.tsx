"use client";

import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";

// Root-level error boundary. Catches anything that escapes every layout's
// error.tsx and the root layout itself. Forwards the error to Sentry so we
// see it even if the user's browser can't render the normal error UI.

export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  useEffect(() => {
    Sentry.captureException(error);
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
