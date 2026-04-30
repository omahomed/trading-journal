# Biometric Unlock — Phase 7 Plan

**Status:** Scaffolded in Phase 1, not enforced.
**Owner of enforcement:** Phase 7.
**Files in place:** [`frontend/src/lib/biometric.ts`](../../frontend/src/lib/biometric.ts), [`frontend/src/components/mobile/biometric-prompt.tsx`](../../frontend/src/components/mobile/biometric-prompt.tsx).

## Why scaffold this in Phase 1

The Phase 1 mobile shell exists to make the app installable as a PWA on the home screen. Once installed, the app launches in standalone mode without the browser address bar — which means the standard "I closed the tab" defense for sensitive financial data goes away. We want a credible UX answer ready for that risk. We are *not* deciding the threat model in Phase 1; we are putting the lock on the door so a future phase can decide which doors to lock.

Concretely, the goal of this scaffold is:

- Get the WebAuthn capability check (`isBiometricAvailable`) callable from any client component today, so Phase 7 can write the trigger logic without first plumbing the API surface.
- Get a visual placeholder (`BiometricPrompt`) ready so designers can iterate on the unlock sheet's copy and motion in parallel with the backend work, without it ever rendering in production.
- Lock the function signatures so Phase 7's wiring is a body swap, not a refactor of the call sites that will accumulate between now and then.

## What gates access

The Phase 7 author should pick one of these enforcement points (or a combination), driven by the threat model decided then. The scaffold doesn't pre-commit to any of them.

| Trigger | When it fires | Notes |
|---|---|---|
| **Cold start in standalone** | App opens from home screen icon, no live session | Strongest defense; expected default. |
| **Backgrounded > N minutes** | App returns to foreground after `visibilitychange` away ≥ N min | Tunable. iOS suggests 5–15 min. |
| **Sensitive route entry** | User navigates to a route flagged in a route-list | Fine-grained; e.g. `/log-buy`, `/admin`. |
| **Re-auth before mutation** | Form submit on Log Buy / Log Sell | Belt-and-suspenders for trade entry. |

## Fallback flow

When biometric is unavailable, declined, or fails, the prompt offers an explicit fallback path. Phase 7 chooses one of these — or supports both:

1. **Magic link to email** — reuse the existing Resend-backed magic-link flow from NextAuth. Strongest UX continuity but requires email access.
2. **Password re-entry** — only viable if we add password auth (we currently only have Google + magic link + dev credentials). Adds attack surface; probably not worth it.
3. **Passkey via cross-device flow** — if the user has a passkey on another device synced to iCloud Keychain or Google Password Manager, WebAuthn's `caBLE` flow can use it. Same WebAuthn API as biometric, no extra plumbing.

The default plan is (1) magic link, since the infra is already there.

## Interaction with NextAuth session lifecycle

Two layering options. Phase 7 picks one:

**Option A — Biometric is a *gate over* the session.** The NextAuth session continues to be the source of truth for who the user is and what they can do; biometric is a one-extra-tap unlock that the client requires before showing protected UI. The backend doesn't know or care about biometric state. Easiest to ship. Vulnerable if the JWT is exfiltrated (biometric on the client doesn't help the API).

**Option B — Biometric is a *credential the session depends on*.** A successful biometric verification mints a fresh, short-lived session JWT from a new endpoint (`POST /api/biometric/verify`). The cookie set by NextAuth becomes scoped: long-lived "user is enrolled" credential, short-lived "user is currently unlocked" credential. The API checks the latter for protected calls. Stronger; more moving parts.

Default plan is **Option A** because it's significantly simpler and the marginal threat (exfiltrated JWT) isn't materially mitigated by Option B for a journal app where the value of the data is read-only context, not transferable funds.

## Backend endpoints Phase 7 must add

Both options require:

- `POST /api/biometric/register-challenge` → `{ challenge: base64url }`. The `userId` is taken from the active NextAuth session (no body params; cannot be called by anonymous). Response is a freshly-generated 32-byte random challenge stored server-side keyed to the session for ~60s.
- `POST /api/biometric/register` → body `{ attestation: PublicKeyCredentialJSON }`. Server validates the attestation, persists the credential ID + public key against `userId`. Returns 204.
- `POST /api/biometric/verify-challenge` → same shape, different challenge cache key (`assertion-*`).
- `POST /api/biometric/verify` → body `{ assertion: PublicKeyCredentialJSON }`. Server validates the assertion against the stored public key. **Option A:** returns 204. **Option B:** also rotates the session JWT and returns a fresh `Set-Cookie`.

Same Postgres adapter as auth (`@auth/pg-adapter`) — credentials live in a new `biometric_credentials` table keyed `(user_id, credential_id)`.

## What Phase 7 inherits

- `isBiometricAvailable()` works today — call it from any effect to decide whether to show the prompt.
- `registerBiometric(userId)` and `verifyBiometric()` are stubs that log a marker and return a sensible default. Body swap when the endpoints land.
- `<BiometricPrompt />` renders the warm-dark unlock sheet matching the mobile palette. Hand it `onUnlock` and `onFallback`. Mount it from wherever the chosen trigger fires.

## Out of scope for both Phase 1 and Phase 7's first cut

- Cross-device passkey sync (depends on how Apple / Google handle this in 2026; tracked separately).
- Backup recovery codes (not worth the UX cost until Phase 7 metrics show fallback is hitting frequently).
- Per-route biometric policy in user settings (the trigger logic is hardcoded in Phase 7; configurability is a follow-up).
