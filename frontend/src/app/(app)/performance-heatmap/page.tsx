import { connection } from "next/server";
import PerformanceHeatmapClient from "./performance-heatmap-client";

export default async function PerformanceHeatmapPage() {
  await connection();
  return <PerformanceHeatmapClient />;
}
