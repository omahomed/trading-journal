import { connection } from "next/server";
import CommandCenterClient from "./command-center-client";

export default async function CommandCenterPage() {
  await connection();
  return <CommandCenterClient />;
}
