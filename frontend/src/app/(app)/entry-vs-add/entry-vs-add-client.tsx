"use client";

import { usePathname } from "next/navigation";
import { EntryVsAdd } from "@/components/entry-vs-add";
import { getGroupForHref } from "@/lib/nav";

export default function EntryVsAddClient() {
  const pathname = usePathname();
  const navColor = getGroupForHref(pathname)?.color || "#0d6efd";
  return <EntryVsAdd navColor={navColor} />;
}
