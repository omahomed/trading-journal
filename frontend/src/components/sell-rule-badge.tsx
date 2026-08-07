import { SELL_RULES, SELL_RULE_FAMILIES } from "@/lib/trade-rules";
import type { SellRuleTier } from "@/lib/sell-rule";

type SellRuleBadgeProps = {
  tier: SellRuleTier | null;
};

// Tone matches the existing inline-style pill convention (color-mix
// tints over surface). Ladder is a defensive-progression gradient:
//   sr1  = warn red (no floor; broker stop, if parked, shows as a chip
//                    on the row rather than a distinct tier — SR14 was
//                    retired in the 2026-08-07 cleanup)
//   sr11 = amber (BE stop at entry, 10%-20% band)
//   sr15 = teal (broker stop at +10% profit, 20%-50% band)
//   sr7  = light green (qualified but undeclared, 21 EMA)
//   sr8  = emerald (declared monster hold, weekly MO RS ladder)
const TONES: Record<SellRuleTier, { bg: string; fg: string }> = {
  sr1: {
    bg: "color-mix(in oklab, #e5484d 14%, var(--surface))",
    fg: "#dc2626",
  },
  sr11: {
    bg: "color-mix(in oklab, #f59f00 12%, var(--surface))",
    fg: "#d97706",
  },
  sr15: {
    bg: "color-mix(in oklab, #0891b2 14%, var(--surface))",
    fg: "#0e7490",
  },
  sr7: {
    bg: "color-mix(in oklab, #34d399 14%, var(--surface))",
    fg: "#15803d",
  },
  sr8: {
    bg: "color-mix(in oklab, #08a86b 18%, var(--surface))",
    fg: "#15803d",
  },
};

export function SellRuleBadge({ tier }: SellRuleBadgeProps) {
  if (!tier) {
    return <span style={{ color: "var(--ink-4)", fontSize: 11 }}>—</span>;
  }

  const rule = SELL_RULES.find((r) => r.code === tier);
  // Family-colored stripe on the badge's left edge (2px vertical bar).
  // Grouped at-a-glance without changing the "SR7" / "SR8" text.
  // Falls back to the badge's own fg color if the tier isn't in
  // SELL_RULES for any reason (won't happen in normal flow now that
  // every tier code has a matching rule entry).
  const family = rule?.family;
  const familyMeta = family
    ? SELL_RULE_FAMILIES.find((f) => f.key === family)
    : undefined;
  const stripe = familyMeta?.color;

  const label = tier.toUpperCase();
  const tooltip = rule
    ? `${label} ${rule.description} — ${familyMeta?.label ?? ""}\n\n${rule.oneLiner}`
    : label;
  const tone = TONES[tier];

  return (
    <span
      className="inline-flex items-center rounded-full text-[10px] font-semibold whitespace-nowrap overflow-hidden"
      style={{ background: tone.bg, color: tone.fg }}
      title={tooltip}
      data-tier={tier}
      data-family={family ?? ""}
    >
      {stripe && (
        <span
          aria-hidden="true"
          data-testid="family-stripe"
          className="self-stretch inline-block"
          style={{ width: 2, background: stripe }}
        />
      )}
      <span className="px-2 py-0.5">{label}</span>
    </span>
  );
}
