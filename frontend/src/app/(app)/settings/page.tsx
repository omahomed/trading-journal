import { connection } from "next/server";
import SettingsClient from "./settings-client";

export default async function SettingsPage() {
  await connection();
  return <SettingsClient />;
}
