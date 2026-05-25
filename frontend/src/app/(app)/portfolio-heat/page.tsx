import { connection } from "next/server";
import PortfolioHeatClient from "./portfolio-heat-client";

export default async function PortfolioHeatPage() {
  await connection();
  return <PortfolioHeatClient />;
}
