import { auth } from "@/auth";

export default auth((req) => {
  if (!req.auth) {
    const loginUrl = new URL("/login", req.url);
    return Response.redirect(loginUrl);
  }
});

// Protect all routes except login, auth API routes, and the public
// static assets that the PWA install/offline flow needs *before* the
// user is signed in (manifest, service worker, app icons). Browsers
// fetch these without auth headers; gating them returns the login
// HTML which Chrome's manifest parser then reports as a JSON syntax
// error.
//
// `build-info.json` is the deploy beacon — read by external monitoring
// and by the UpdateBanner poll. Exempted here so unauthenticated
// sessions on /login can poll it without the redirect-then-HTML
// response that would otherwise silently break the JSON parse.
export const config = {
  matcher: [
    "/((?!login|api/auth|_next/static|_next/image|favicon.ico|manifest.json|sw.js|icon-192.png|icon-512.png|icon-maskable-512.png|apple-touch-icon.png|build-info.json).*)",
  ],
};
