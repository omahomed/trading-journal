"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { ChevronRight } from "lucide-react";
import { useIsMobile } from "@/lib/use-viewport";
import { useFocusMode } from "@/lib/use-focus-mode";
import { setFocusModeActive } from "@/lib/format";
import { MobileToggleSwitch } from "@/components/mobile/mobile-toggle-switch";

const FOCUS_MODE_KEY = "mo-focus-mode";

/**
 * The fifth bottom-nav destination on mobile. Lists the routes that
 * don't have a dedicated bottom-nav slot. Desktop users have the
 * sidebar for navigation, so this page redirects them to /dashboard
 * after hydration.
 */
export default function MoreClient() {
  const router = useRouter();
  const isMobile = useIsMobile();
  const focusMode = useFocusMode();

  useEffect(() => {
    if (!isMobile) router.replace("/dashboard");
  }, [isMobile, router]);

  const toggleFocus = (next: boolean) => {
    // Same write pattern desktop-shell.tsx uses: flip the module mirror
    // synchronously *before* persisting, so any in-flight render reads
    // the new value.
    setFocusModeActive(next);
    try {
      window.localStorage.setItem(FOCUS_MODE_KEY, next ? "on" : "off");
    } catch {
      /* ignore quota / private-mode errors */
    }
  };

  return (
    <div className="flex flex-col gap-3 pt-2">
      <Section title="Display">
        <MobileToggleSwitch
          id="more-focus-mode"
          checked={focusMode}
          onChange={toggleFocus}
          label="Focus Mode"
          description="Hide dollar amounts across the app."
        />
      </Section>
      <Section title="Dashboards">
        <NavRow href="/dashboard" label="Dashboard" />
        <NavRow href="/overview" label="Trading Overview" />
      </Section>
      <Section title="Trading Ops">
        <NavRow href="/active-campaign" label="Active Campaign" />
        <NavRow href="/trade-manager" label="Trade Manager" />
        <NavRow href="/log-buy" label="Log Buy" />
        <NavRow href="/log-sell" label="Log Sell" />
        <NavRow href="/import-trades" label="Import Trades" />
      </Section>
      <Section title="Risk">
        <NavRow href="/portfolio-heat" label="Portfolio Heat" />
        <NavRow href="/risk-manager" label="Risk Manager" />
        <NavRow href="/earnings" label="Earnings Planner" />
      </Section>
      <Section title="Daily">
        <NavRow href="/daily-routine" label="Daily Routine" />
        <NavRow href="/daily-journal" label="Daily Journal" />
        <NavRow href="/daily-report" label="Daily Report" />
        <NavRow href="/weekly-retro" label="Weekly Retro" />
      </Section>
      <Section title="Market Intel">
        <NavRow href="/rally-context" label="Rally Context" />
      </Section>
      <Section title="Deep Dive">
        <NavRow href="/analytics" label="Analytics" />
        <NavRow href="/performance-heatmap" label="Performance Heat Map" />
        <NavRow href="/period-review" label="Period Review" />
      </Section>
      <Section title="AI">
        <NavRow href="/ai-coach" label="AI Coach" />
      </Section>
      <Section title="Account">
        <NavRow href="/settings" label="Settings" />
        <NavRow href="/admin" label="Admin" />
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="overflow-hidden rounded-m-md border-[0.5px] border-m-border bg-m-surface">
      <div className="border-b-[0.5px] border-m-border px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-m-text-dim">
        {title}
      </div>
      <div>{children}</div>
    </div>
  );
}

function NavRow({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="flex items-center justify-between border-b-[0.5px] border-m-border px-4 py-3 text-[14px] text-m-text last:border-b-0"
    >
      <span>{label}</span>
      <ChevronRight size={16} strokeWidth={1.5} className="text-m-text-faint" aria-hidden="true" />
    </Link>
  );
}
