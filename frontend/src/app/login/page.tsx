import { connection } from "next/server";
import LoginForm from "./login-form";

// Server-component wrapper for the login form. Calling `connection()`
// inside it opts the route out of prerendering — every /login request
// becomes a fresh server render, and the served HTML always references
// the chunks that exist on this deployment.
//
// Why this matters: prior to this commit, /login was a pure client
// component and Next 16 prerendered it at build time. Vercel cached
// that prerender at the edge with effectively long TTL; when Vercel
// emitted new chunk hashes on subsequent rebuilds, the cached
// prerender referenced 404'd chunks, no React mounted, and users saw
// a broken/inert page until hard refresh.
//
// `export const dynamic = "force-dynamic"` on the client-component
// file did not take effect in Next 16; the canonical Next 16 escape
// hatch is `await connection()` from a server component, which is why
// the file is split: this page (server) owns the dynamic gate,
// LoginForm (client) owns the interactive UI.
export default async function LoginPage() {
  await connection();
  return <LoginForm />;
}
