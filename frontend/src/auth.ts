import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import Resend from "next-auth/providers/resend";
import Credentials from "next-auth/providers/credentials";
import PostgresAdapter from "@auth/pg-adapter";
import { Pool } from "pg";
import { SignJWT } from "jose";

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

const SECRET_KEY = new TextEncoder().encode(process.env.AUTH_SECRET);
const API_TOKEN_TTL = "1h";

// Staging beta gate — when STAGING_PASSWORD is set, a Credentials provider
// accepts that shared password and signs all testers in as a fixed user.
// Staging-only; prod leaves the env var unset so the provider does nothing.
const STAGING_TESTER_UUID = "5a1a1e57-57e5-4000-8000-000057a1a1e5";
const STAGING_TESTER_EMAIL = "staging-tester@mo-trading.local";
const STAGING_PASSWORD = process.env.STAGING_PASSWORD;

async function ensureStagingTester() {
  await pool.query(
    `INSERT INTO users (id, email, name)
     VALUES ($1::uuid, $2, 'Staging Tester')
     ON CONFLICT (id) DO NOTHING`,
    [STAGING_TESTER_UUID, STAGING_TESTER_EMAIL]
  );
}

async function mintApiToken(userId: string, email: string | null | undefined) {
  return new SignJWT({ sub: userId, email: email ?? undefined })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(API_TOKEN_TTL)
    .sign(SECRET_KEY);
}

// Temporary owner allowlist — blocks anyone else from signing in until Step 3
// (multi-tenant query filtering) is complete. Remove the signIn callback and
// this set once every API query filters by user_id.
//
// Source: AUTH_ALLOWED_EMAILS env var (comma-separated), falling back to the
// owner pair below. Staging can add beta testers via the env var without any
// code change; prod can stay locked down by leaving it unset or listing only
// the owner.
const OWNER_FALLBACK = ["omahomed@gmail.com", "omahomed@icloud.com"];
const envList = process.env.AUTH_ALLOWED_EMAILS
  ?.split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
const ALLOWED_EMAILS = new Set(envList && envList.length > 0 ? envList : OWNER_FALLBACK);
// Staging tester signs in via Credentials; admit them through the allowlist too.
if (STAGING_PASSWORD) ALLOWED_EMAILS.add(STAGING_TESTER_EMAIL);

// Shared-account aliasing — these emails sign in with their own Google account
// (the adapter still creates their `users` row) but operate ON the founder's
// tenant rather than their own. We force their session + minted API token to
// carry the founder UUID as `sub`, so Postgres RLS scopes every query to the
// owner's data. Tradeoff: their actions are attributed to the founder in the
// audit trail — there is no separate actor record. Set via env on the frontend
// host, comma-separated, e.g. SHARED_ACCOUNT_EMAILS="brother@gmail.com".
const FOUNDER_USER_ID =
  process.env.FOUNDER_USER_ID ?? "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a";
const SHARED_ACCOUNT_EMAILS = new Set(
  (process.env.SHARED_ACCOUNT_EMAILS ?? "")
    .split(",").map(s => s.trim().toLowerCase()).filter(Boolean)
);
// Shared-account users must also clear the sign-in allowlist.
for (const e of SHARED_ACCOUNT_EMAILS) ALLOWED_EMAILS.add(e);

export const { handlers, signIn, signOut, auth } = NextAuth({
  adapter: PostgresAdapter(pool),
  session: { strategy: "jwt" },
  trustHost: true,
  providers: [
    Google({ allowDangerousEmailAccountLinking: true }),
    Resend({ from: process.env.EMAIL_FROM }),
    ...(STAGING_PASSWORD ? [Credentials({
      id: "staging-password",
      name: "Staging password",
      credentials: { password: { label: "Beta password", type: "password" } },
      async authorize(credentials) {
        const input = typeof credentials?.password === "string" ? credentials.password : "";
        if (!input || input !== STAGING_PASSWORD) return null;
        await ensureStagingTester();
        return { id: STAGING_TESTER_UUID, email: STAGING_TESTER_EMAIL, name: "Staging Tester" };
      },
    })] : []),
  ],
  callbacks: {
    async signIn({ user }) {
      const email = user.email?.toLowerCase();
      return !!email && ALLOWED_EMAILS.has(email);
    },
    async jwt({ token, user }) {
      if (user) {
        // Shared-account users alias onto the founder's tenant; everyone else
        // keeps their own UUID.
        token.sub =
          user.email && SHARED_ACCOUNT_EMAILS.has(user.email.toLowerCase())
            ? FOUNDER_USER_ID
            : user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (token.sub && session.user) {
        session.user.id = token.sub;
        session.apiToken = await mintApiToken(token.sub, session.user.email);
      }
      return session;
    },
  },
  pages: {
    signIn: "/login",
    verifyRequest: "/login/check-email",
    error: "/login",
  },
});
