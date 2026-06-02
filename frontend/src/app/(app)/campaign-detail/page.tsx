import { connection } from "next/server";
import CampaignDetailClient from "./campaign-detail-client";

export default async function CampaignDetailPage() {
  await connection();
  return <CampaignDetailClient />;
}
