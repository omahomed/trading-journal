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

export const { handlers, signIn, signOut, auth } = NextAuth({
  adapter: PostgresAdapter(pool),
  session: { strategy: "jwt" },
  trustHost: true,
  providers: [
    Google,
    Resend({ from: process.env.EMAIL_FROM }),
  ],
  callbacks: {
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
