import { connection } from "next/server";
import PeriodReviewClient from "./period-review-client";

export default async function PeriodReviewPage({
  searchParams,
}: {
  searchParams: Promise<{ tab?: string }>;
}) {
  await connection();
  const sp = await searchParams;
  return <PeriodReviewClient initialTabProp={sp.tab} />;
}
