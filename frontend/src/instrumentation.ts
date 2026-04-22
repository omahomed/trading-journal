// Server + edge runtime Sentry init. Next.js calls register() once at server
// boot; we lazy-import the matching runtime's @sentry/nextjs entry so we don't
// drag Node-only code into the Edge bundle (which would break proxy.ts).
//
// onRequestError forwards server-side errors into Sentry. @sentry/nextjs
// exports a ready-made implementation we just re-export.

import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    Sentry.init({
      dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
      tracesSampleRate: 0.1,
      environment: process.env.VERCEL_ENV ?? "development",
    });
  }
  if (process.env.NEXT_RUNTIME === "edge") {
    Sentry.init({
      dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
      tracesSampleRate: 0.1,
      environment: process.env.VERCEL_ENV ?? "development",
    });
  }
}

export const onRequestError = Sentry.captureRequestError;
