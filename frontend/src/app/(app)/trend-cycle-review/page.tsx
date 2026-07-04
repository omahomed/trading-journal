import { connection } from "next/server";
import TrendCycleReviewClient from "./trend-cycle-review-client";

export default async function TrendCycleReviewPage() {
  await connection();
  return <TrendCycleReviewClient />;
}
