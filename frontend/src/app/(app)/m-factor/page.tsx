import { connection } from "next/server";
import MFactorClient from "./m-factor-client";

export default async function MFactorPage() {
  await connection();
  return <MFactorClient />;
}
