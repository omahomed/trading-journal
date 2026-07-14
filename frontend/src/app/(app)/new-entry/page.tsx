import { connection } from "next/server";
import NewEntryClient from "./new-entry-client";

export default async function NewEntryPage() {
  await connection();
  return <NewEntryClient />;
}
