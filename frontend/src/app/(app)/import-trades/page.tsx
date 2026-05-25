import { connection } from "next/server";
import ImportTradesClient from "./import-trades-client";

export default async function ImportTradesPage() {
  await connection();
  return <ImportTradesClient />;
}
