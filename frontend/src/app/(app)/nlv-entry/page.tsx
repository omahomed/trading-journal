import { connection } from "next/server";
import NLVEntryClient from "./nlv-entry-client";

export default async function NLVEntryPage() {
  await connection();
  return <NLVEntryClient />;
}
