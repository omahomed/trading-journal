"use client";

import { ScanFace } from "lucide-react";

/**
 * Phase 1 visual scaffold for the biometric unlock prompt. Not mounted
 * anywhere — exists as the future mount point for Phase 7 enforcement.
 *
 * Render shape: warm-dark sheet centered over the cycle indicator,
 * with the platform-agnostic ScanFace icon (Touch ID and Face ID both
 * read as "biometric" without device-specific copy), an "Unlock with
 * biometrics" CTA, and a fallback "Use password instead" link.
 *
 * Wiring (Phase 7): mount inside `MobileShell` behind a session-age
 * check; on `onUnlock`, call `verifyBiometric()` from `lib/biometric`
 * and either advance to the protected route or fall through to
 * `/login`.
 */
export type BiometricPromptProps = {
  /** Called when the user taps "Unlock with biometrics". */
  onUnlock?: () => void;
  /** Called when the user taps "Use password instead". */
  onFallback?: () => void;
};

export function BiometricPrompt({ onUnlock, onFallback }: BiometricPromptProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-5 rounded-m-xl border-[0.5px] border-m-border bg-m-surface px-6 py-8 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-m-pill bg-m-accent-tint">
        <ScanFace size={32} strokeWidth={1.5} className="text-m-accent" aria-hidden="true" />
      </div>
      <div>
        <div className="text-base font-medium text-m-text">Unlock to continue</div>
        <div className="mt-1 text-xs text-m-text-muted">
          Use Face ID or Touch ID to access your trading journal.
        </div>
      </div>
      <button
        type="button"
        onClick={onUnlock}
        className="w-full rounded-m-lg bg-m-accent px-4 py-3 text-[15px] font-medium text-m-accent-text-on"
      >
        Unlock with biometrics
      </button>
      <button
        type="button"
        onClick={onFallback}
        className="text-xs text-m-text-muted underline-offset-2 hover:underline"
      >
        Use password instead
      </button>
    </div>
  );
}
