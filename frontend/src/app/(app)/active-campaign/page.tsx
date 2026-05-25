import { connection } from "next/server";
import ActiveCampaignClient from "./active-campaign-client";

export default async function ActiveCampaignPage() {
  await connection();
  return <ActiveCampaignClient />;
}
