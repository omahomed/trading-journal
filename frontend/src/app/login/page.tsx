"use client";

import { signIn } from "next-auth/react";
import { useState } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [sending, setSending] = useState(false);

  async function handleMagicLink(e: React.FormEvent) {
    e.preventDefault();
    if (!email || sending) return;
    setSending(true);
    await signIn("resend", { email, callbackUrl: "/" });
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg)" }}>
      <div className="w-[400px] max-w-[90vw] rounded-[20px] p-8 text-center"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 8px 30px rgba(0,0,0,0.08)" }}>

        <div className="mb-6">
          <div className="text-[36px] font-bold tracking-tight" style={{ fontFamily: "var(--font-fraunces), Georgia, serif" }}>
            MO <em className="italic" style={{ color: "#6366f1" }}>Trading</em>
          </div>
          <div className="text-[13px] mt-1" style={{ color: "var(--ink-4)" }}>Trading Journal & Analytics</div>
          <div className="text-[12px] mt-2" style={{ color: "var(--ink-5)" }}>
            New here? Use either option below — we'll create your account automatically.
          </div>
        </div>

        <div className="h-px mb-6" style={{ background: "var(--border)" }} />

        <button
          onClick={() => signIn("google", { callbackUrl: "/" })}
          className="w-full h-[48px] rounded-[12px] flex items-center justify-center gap-3 text-[14px] font-semibold transition-all hover:brightness-95 cursor-pointer"
          style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}>
          <svg width="20" height="20" viewBox="0 0 24 24">
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
          </svg>
          Sign in with Google
        </button>

        <div className="flex items-center gap-3 my-5">
          <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
          <div className="text-[11px] uppercase tracking-wider" style={{ color: "var(--ink-5)" }}>or</div>
          <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
        </div>

        <form onSubmit={handleMagicLink} className="space-y-3">
          <input
            type="email"
            required
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={sending}
            className="w-full h-[48px] px-4 rounded-[12px] text-[14px] outline-none"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--ink)" }}
          />
          <button
            type="submit"
            disabled={sending || !email}
            className="w-full h-[48px] rounded-[12px] text-[14px] font-semibold transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            style={{ background: "#6366f1", color: "white" }}>
            {sending ? "Sending link…" : "Email me a sign-in link"}
          </button>
        </form>

      </div>
    </div>
  );
}
