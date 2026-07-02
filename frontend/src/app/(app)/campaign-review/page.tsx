import { connection } from "next/server";
import CampaignReviewClient from "./campaign-review-client";

export default async function CampaignReviewPage() {
  await connection();
  return <CampaignReviewClient />;
}
