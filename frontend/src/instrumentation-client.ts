// Client-side Sentry init (runs before app becomes interactive).
// NEXT_PUBLIC_ prefix makes the DSN available in browser bundles.

import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 0.1,
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: 0,
  environment: process.env.NEXT_PUBLIC_VERCEL_ENV ?? "development",
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
