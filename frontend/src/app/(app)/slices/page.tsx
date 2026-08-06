import { connection } from "next/server";
import SlicesClient from "./slices-client";

export default async function SlicesPage() {
  await connection();
  return <SlicesClient />;
}
