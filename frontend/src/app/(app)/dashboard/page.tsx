import { connection } from "next/server";
import DashboardClient from "./dashboard-client";

// Server-component wrapper for the dashboard. `await connection()` opts
// the route out of prerendering — every /dashboard request becomes a
// fresh server render, so the served HTML always references the chunks
// that exist on this deployment.
//
// Same fix as /login (commit 1e6f14f): pre-fix /dashboard was prerendered
// at build time, cached at the Vercel edge with effectively long TTL.
// When Vercel emitted new chunk hashes on subsequent rebuilds, the
// cached prerender referenced 404'd chunks, no React mounted, and
// authenticated users saw broken/inert pages until hard refresh.
//
// `export const dynamic = "force-dynamic"` on the client-component
// file does not take effect in Next 16; the canonical escape hatch is
// `await connection()` from a server component, which is why the file
// is split: this page (server) owns the dynamic gate, DashboardClient
// (client) owns `usePathname` + the interactive UI.
export default async function DashboardPage() {
  await connection();
  return <DashboardClient />;
}
