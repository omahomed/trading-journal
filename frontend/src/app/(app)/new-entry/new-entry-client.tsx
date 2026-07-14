"use client";

import { usePathname } from "next/navigation";
import { NewEntry } from "@/components/new-entry";
import { getGroupForHref } from "@/lib/nav";

export default function NewEntryClient() {
  const navColor = getGroupForHref(usePathname())?.color || "#08a86b";
  return <NewEntry navColor={navColor} />;
}
