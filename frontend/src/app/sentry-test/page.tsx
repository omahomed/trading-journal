"use client";

// TEMPORARY — remove after Sentry is verified. This page throws on render so
// we can confirm the client + server SDKs are wired up. Delete the folder
// once events land in both motrading-web and motrading-api dashboards.

export default function SentryTest() {
  throw new Error("Sentry frontend verification — safe to ignore");
}
