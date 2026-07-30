import { connection } from "next/server";
import SectorMappingClient from "./sector-mapping-client";

export default async function SectorMappingPage() {
  await connection();
  return <SectorMappingClient />;
}
