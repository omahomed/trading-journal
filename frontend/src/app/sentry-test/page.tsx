// TEMPORARY — remove after Sentry is verified. Server component that throws
// on request so we can confirm the SDK is wired up. force-dynamic keeps it
// out of the static pre-render step; without it the build fails on the
// deliberate throw. Delete this folder once the event lands in Sentry.

export const dynamic = "force-dynamic";

export default function SentryTest() {
  throw new Error("Sentry frontend verification — safe to ignore");
}
