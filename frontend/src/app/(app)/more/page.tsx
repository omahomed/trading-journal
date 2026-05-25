import { connection } from "next/server";
import MoreClient from "./more-client";

export default async function MorePage() {
  await connection();
  return <MoreClient />;
}
