export default function CheckEmailPage() {
  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg)" }}>
      <div className="w-[400px] max-w-[90vw] rounded-[20px] p-8 text-center"
           style={{ background: "var(--surface)", border: "1px solid var(--border)", boxShadow: "0 8px 30px rgba(0,0,0,0.08)" }}>

        <div className="mb-4 text-[48px]">📧</div>

        <div className="text-[20px] font-semibold mb-2" style={{ color: "var(--ink)" }}>
          Check your email
        </div>
        <div className="text-[13px] leading-relaxed" style={{ color: "var(--ink-4)" }}>
          We sent a sign-in link to your inbox. Click the link in that email to finish signing in.
        </div>
        <div className="text-[11px] mt-4" style={{ color: "var(--ink-5)" }}>
          The link expires in 24 hours. You can close this tab.
        </div>

        <div className="h-px my-6" style={{ background: "var(--border)" }} />

        <a href="/login" className="text-[12px] hover:underline" style={{ color: "var(--ink-4)" }}>
          ← Back to sign in
        </a>
      </div>
    </div>
  );
}
