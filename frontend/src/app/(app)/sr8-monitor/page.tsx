import { connection } from "next/server";
import Sr8MonitorClient from "./sr8-monitor-client";

export default async function Sr8MonitorPage() {
  await connection();
  return <Sr8MonitorClient />;
}
