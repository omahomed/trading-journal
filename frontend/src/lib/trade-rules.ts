// Canonical sell-rule taxonomy. Single source of truth for the
// dropdown options in log-sell, trade-journal, trade-manager, and
// import-trades. DB values are stored as `${code} ${description}`
// (see SELL_RULE_LABELS below).

export type SellRule = { code: string; description: string };

export const SELL_RULES: readonly SellRule[] = [
  { code: "sr1",  description: "Capital Protection" },
  { code: "sr2",  description: "Selling into Strength" },
  { code: "sr3",  description: "Portfolio Management" },
  { code: "sr4",  description: "Time Stop" },
  { code: "sr5",  description: "Climax Top" },
  { code: "sr6",  description: "8e Momentum Trim" },
  { code: "sr7",  description: "Holding Winners - 21e Violation" },
  { code: "sr8",  description: "Big Cushion Sell Rule" },
  { code: "sr9",  description: "Failed Breakout" },
  { code: "sr10", description: "Earnings Exit" },
  { code: "sr11", description: "BE Stop Out (moved at +10%)" },
  { code: "sr12", description: "TQQQ Strategy Exit" },
  { code: "sr13", description: "Change of Character" },
] as const;

export const SELL_RULE_LABELS: readonly string[] = SELL_RULES.map(
  (r) => `${r.code} ${r.description}`,
);
