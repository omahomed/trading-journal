/**
 * Biometric unlock helpers — Phase 1 scaffold.
 *
 * The signatures and call shapes here are the contract Phase 7+ will
 * wire to a real backend (challenge generation, credential storage,
 * session minting). For now every entry point is a stub:
 *
 *   • `isBiometricAvailable()` does a real feature-detect — this is
 *     the only function that returns useful information today, used
 *     to decide whether the (unmounted) `BiometricPrompt` would even
 *     show on this device.
 *
 *   • `registerBiometric()` and `verifyBiometric()` log a marker and
 *     return a sensible default. Real `navigator.credentials.create`
 *     / `.get` calls require a backend-issued challenge; that endpoint
 *     does not exist yet (see `docs/mobile/01-biometric-plan.md`).
 *
 * Phase 7 will swap the bodies, not the signatures. Callers can
 * import these now without coupling to whatever the final flow ends
 * up looking like.
 */

const STUB_TAG = "[Phase 1 stub]";

/**
 * Returns `true` only if the browser supports WebAuthn AND the device
 * has a user-verifying platform authenticator (Touch ID, Face ID,
 * Windows Hello, Android biometric). Returns `false` everywhere else
 * — non-secure context, unsupported browser, no platform authenticator.
 *
 * Safe to call from a Client Component effect. Does not prompt the
 * user; this is a passive capability check.
 */
export async function isBiometricAvailable(): Promise<boolean> {
  if (typeof window === "undefined") return false;
  if (!("PublicKeyCredential" in window)) return false;

  const PKC = window.PublicKeyCredential as typeof PublicKeyCredential & {
    isUserVerifyingPlatformAuthenticatorAvailable?: () => Promise<boolean>;
  };
  if (typeof PKC.isUserVerifyingPlatformAuthenticatorAvailable !== "function") {
    return false;
  }

  try {
    return await PKC.isUserVerifyingPlatformAuthenticatorAvailable();
  } catch {
    return false;
  }
}

/**
 * Phase 7 will: (1) ask the backend for a registration challenge,
 * (2) call `navigator.credentials.create({ publicKey: { ...opts,
 * challenge } })`, (3) post the resulting attestation to the backend
 * for storage against `userId`.
 *
 * Phase 1 stub: logs and resolves. No credential is actually
 * registered. The backend `POST /api/biometric/register-challenge`
 * endpoint does not exist yet.
 */
export async function registerBiometric(userId: string): Promise<void> {
  if (process.env.NODE_ENV !== "production") {
    console.info(STUB_TAG, "registerBiometric called for userId=", userId);
  }
  return;
}

/**
 * Phase 7 will: (1) ask the backend for an assertion challenge,
 * (2) call `navigator.credentials.get({ publicKey: { ...opts,
 * challenge, allowCredentials } })`, (3) post the assertion to the
 * backend for verification, (4) the backend mints a fresh session
 * token on success.
 *
 * Phase 1 stub: logs and returns `false` because no credential is
 * registered. This default keeps callers on the password / magic-link
 * fallback path (the desired Phase 1 behavior — biometric is
 * scaffolded but never enforced).
 */
export async function verifyBiometric(): Promise<boolean> {
  if (process.env.NODE_ENV !== "production") {
    console.info(STUB_TAG, "verifyBiometric called");
  }
  return false;
}
