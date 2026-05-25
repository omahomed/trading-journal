import { connection } from "next/server";
import AdminClient from "./admin-client";

export default async function AdminPage() {
  await connection();
  return <AdminClient />;
}
