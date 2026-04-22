import type { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    apiToken?: string;
    user?: {
      id?: string;
    } & DefaultSession["user"];
  }
}
