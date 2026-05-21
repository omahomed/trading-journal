import { SELL_RULES } from "@/lib/trade-rules";
import type { SellRuleTier } from "@/lib/sell-rule";

type SellRuleBadgeProps = {
  tier: SellRuleTier | null;
};

// Tone matches the existing inline-style pill convention used by the
// former Risk Status cell in active-campaign.tsx (color-mix tints over
// surface). sr1 = warn red, sr11 = amber, sr8 = green.
const TONES: Record<SellRuleTier, { bg: string; fg: string }> = {
  sr1: {
    bg: "color-mix(in oklab, #e5484d 14%, var(--surface))",
    fg: "#dc2626",
  },
  sr11: {
    bg: "color-mix(in oklab, #f59f00 12%, var(--surface))",
    fg: "#d97706",
  },
  sr8: {
    bg: "color-mix(in oklab, #08a86b 12%, var(--surface))",
    fg: "#16a34a",
  },
};

export function SellRuleBadge({ tier }: SellRuleBadgeProps) {
  if (!tier) {
    return <span style={{ color: "var(--ink-4)", fontSize: 11 }}>—</span>;
  }

  const rule = SELL_RULES.find((r) => r.code === tier);
  const label = tier.toUpperCase();
  const tooltip = rule
    ? `${label} ${rule.description}\n\n${rule.oneLiner}`
    : label;
  const tone = TONES[tier];

  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold whitespace-nowrap"
      style={{ background: tone.bg, color: tone.fg }}
      title={tooltip}
      data-tier={tier}
    >
      {label}
    </span>
  );
}
