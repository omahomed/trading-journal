// uploadWithTimeout: wraps api.uploadImage with an AbortController + timeout
// so a hung upload can't block the UI indefinitely.
//
// Plain fetch() has no built-in timeout — if the backend's R2 call stalls or
// the network blips during a large upload, the promise sits open forever.
// 60s is the default cap; on abort, the promise rejects with AbortError which
// we translate into a friendly {ok:false, error:"Upload timed out (60s)"} so
// the caller's per-file status panel can show it.

import { api } from "./api";

export interface UploadResult {
  ok: boolean;
  error?: string;
}

export const DEFAULT_UPLOAD_TIMEOUT_MS = 60_000;

export async function uploadWithTimeout(
  file: File,
  portfolio: string,
  tradeId: string,
  ticker: string,
  imageType: string,
  timeoutMs: number = DEFAULT_UPLOAD_TIMEOUT_MS,
): Promise<UploadResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const result = await api.uploadImage(file, portfolio, tradeId, ticker, imageType, controller.signal);
    if (result && typeof result === "object" && "error" in result && (result as { error?: string }).error) {
      return { ok: false, error: String((result as { error?: string }).error) };
    }
    return { ok: true };
  } catch (err: unknown) {
    const e = err as { name?: string; message?: string };
    if (e?.name === "AbortError") {
      return { ok: false, error: `Upload timed out (${Math.round(timeoutMs / 1000)}s)` };
    }
    return { ok: false, error: e?.message || String(err) };
  } finally {
    clearTimeout(timeoutId);
  }
}
