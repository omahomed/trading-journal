import { connection } from "next/server";
import RallyContextClient from "./rally-context-client";

export default async function RallyContextPage() {
  await connection();
  return <RallyContextClient />;
}
