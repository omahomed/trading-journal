import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import Resend from "next-auth/providers/resend";
import PostgresAdapter from "@auth/pg-adapter";
import { Pool } from "pg";
import { SignJWT } from "jose";

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

const SECRET_KEY = new TextEncoder().encode(process.env.AUTH_SECRET);
const API_TOKEN_TTL = "1h";

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

export const { handlers, signIn, signOut, auth } = NextAuth({
  adapter: PostgresAdapter(pool),
  session: { strategy: "jwt" },
  trustHost: true,
  providers: [
    Google({ allowDangerousEmailAccountLinking: true }),
    Resend({ from: process.env.EMAIL_FROM }),
  ],
  callbacks: {
    async signIn({ user }) {
      const email = user.email?.toLowerCase();
      return !!email && ALLOWED_EMAILS.has(email);
    },
    async jwt({ token, user }) {
      if (user) token.sub = user.id;
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
