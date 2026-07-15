// Canonical sell-rule taxonomy. Single source of truth for the
// dropdown options in log-sell, trade-journal, trade-manager, and
// import-trades, and for the in-app glossary rendered on the Log
// Sell page.
//
// DB values are stored as `${code} ${description}` (see
// SELL_RULE_LABELS below). oneLiner and mechanics are
// presentation-only fields rendered by SellRuleGlossary; the
// taxonomy migration uses only code+description.

export type SellRule = {
  code: string;
  description: string;
  /** One-sentence summary shown always (collapsed-card header detail). */
  oneLiner: string;
  /** Markdown body. May contain GFM tables. Optional. */
  mechanics?: string;
};

export const SELL_RULES: readonly SellRule[] = [
  {
    code: "sr1",
    description: "Capital Protection",
    oneLiner:
      "Initial stop on every new position. Size capped by 1% / 0.75% / 0.5% of capital depending on Position Sizer mode. Non-negotiable — when triggered, exit.",
    mechanics: [
      "- Activates on every entry; the foundational rule",
      "- Stop level depends on the entry setup. Examples:",
      "  - Upside reversal entry: low of the day",
      "  - Break of SL entry: low of the day before",
      "- Position sized so that distance-to-stop × shares = max acceptable capital risk",
    ].join("\n"),
  },
  {
    code: "sr2",
    description: "Selling into Strength",
    oneLiner:
      "Trim 25% on ATR extension above the 21 EMA. Typically 4 ATR, sometimes 3 ATR depending on the stock. Same thresholds for options.",
    mechanics: [
      "- Trigger: price extends 3–4 ATR above 21 EMA (stock-specific threshold)",
      "- Action: trim 25% of position",
      "- Exempts SR8 core — SR2 trims only from the ADDS layer; the 15% NAV core is untouched",
      "- Same thresholds apply to options positions",
    ].join("\n"),
  },
  {
    code: "sr3",
    description: "Portfolio Management",
    oneLiner:
      "Trim positions during market-wide pressure or when portfolio drawdown hits L1/L2/L3 levels. SR8 cores are exempt.",
    mechanics: [
      "- Triggered by macro catalysts: Fed/FOMC, war, tariffs, major geopolitical events",
      "- Also triggered by portfolio drawdown thresholds:",
      "  - **L1** (−7.5% from ATH): start managing, reduce exposure",
      "  - **L2** (−12.5% from ATH): take out margin",
      "  - **L3** (−15% from ATH): out of the market",
      "- Core preservation: SR8 cores (15% NAV positions in proven leaders) are exempt from SR3 trims. SR3 reduces exposure by trimming non-leader positions and the ADDS layer of SR8 positions, but does not touch SR8 cores. Cores are only retired by SR8's own signals (Quick / Quicksand / Grateful Dead) or by SR13.",
    ].join("\n"),
  },
  {
    code: "sr4",
    description: "Time Stop",
    oneLiner:
      "Cut after 8 weeks of no meaningful movement (no stop trigger, no progress).",
  },
  {
    code: "sr5",
    description: "Climax Top",
    oneLiner:
      "Exit on climactic action — parabolic move, extreme extension above 50/200 MA, or sustained run-ups.",
    mechanics: [
      "- Multiple ways to identify:",
      "  - Pace of increase becomes vertical",
      "  - Distance above 50-day or 200-day MA reaches extreme levels",
      "  - Many consecutive up days without rest",
      "- Often a judgment call; sometimes multiple signals align",
    ].join("\n"),
  },
  {
    code: "sr6",
    description: "8e Momentum Trim",
    oneLiner:
      "Trim trigger on 8 EMA violation after 10+ consecutive closes above it.",
    mechanics: [
      "- **Activate**: Stock has 10+ consecutive closes above 8 EMA",
      "- **Arm**: First close below 8 EMA after activation",
      "- **Trigger level**: 1% below the arming bar's low",
      "- **Fire**: Subsequent intraday bar's low breaks 1% below trigger",
      "- **Disarm**: Any close back above 8 EMA (single close)",
      "- **Re-arm**: On next close below 8 EMA, at the new bar's level",
      "- **Action on fire**: Trim 25% of position",
    ].join("\n"),
  },
  {
    code: "sr7",
    description: "Holding Winners - 21e Violation",
    oneLiner:
      "Sell on 21 EMA violation. Action scales with cushion: <25% → full exit, 25–50% → trim 50%, >50% → trim to 15% NAV core.",
    mechanics: [
      "- Always armed — no activation requirement",
      "- Same arm/trigger/fire/disarm structure as SR6, but on the 21 EMA",
      "- Action on fire depends on position cushion:",
      "",
      "| Cushion at trigger | Action |",
      "|---|---|",
      "| Up <25% from entry | Full exit |",
      "| Up 25–50% from entry | Trim 50% |",
      "| Up >50% from entry | Trim to 15% NAV core (transitions to SR8) |",
      "",
      "- **Recursion on the remaining position**: SR7 stays armed after any partial trim. If the stock recovers (close back above 21 EMA), SR7 disarms and the remaining shares continue under normal governance. If it doesn't recover, SR7 re-arms on the next close below 21 EMA and the next fire re-evaluates cushion on what's left.",
    ].join("\n"),
  },
  {
    code: "sr8",
    description: "Big Cushion Sell Rule",
    oneLiner:
      "RS-based management of positions up 50%+ that are market leaders. Splits the position into CORE (15% NAV) + ADDS (managed via SR7).",
    mechanics: [
      "- **Activate**: Position up 50%+ AND stock is market leader / has strong fundamentals",
      "- **Structure**:",
      "  - **CORE** (15% NAV): managed via weekly MO RS signals (this rule)",
      "  - **ADDS** (beyond 15%): managed via SR7 (21 EMA violation)",
      "- **Weekly MO RS Triggers** (fire intraweek on live cross, NOT Friday close):",
      "",
      "| Signal | Trigger | Action | Reversible? |",
      "|---|---|---|---|",
      "| 🟡 Quick | RS breaks below 8w MA (orange) | Trim 5% NAV (15% → 10%) | YES — if RS reclaims, rebuild |",
      "| 🟡 Quicksand | RS drifts further below 8w | Trim another 5% (10% → 5%) | YES — if RS reclaims, rebuild |",
      "| 🔴 Grateful Dead | RS breaks below 21w MA (blue) | Full exit | NO — one-way |",
    ].join("\n"),
  },
  {
    code: "sr8.1",
    description: "SR8 Quick Trim",
    oneLiner:
      "First MO RS cascade fire. RS crosses below the 8-week MA — trim the position back to the QUICK cascade target (15% NLV on the 20-cascade, 11.25% on the 15-cascade). Reversible: if RS reclaims the 8w, rebuild.",
    mechanics: [
      "- **Trigger**: weekly RS (relative strength) breaks below its 8w MA (intraweek live cross, NOT Friday close)",
      "- **Cascade selection (per position, based on current % NLV at trigger)**:",
      "  - `≥ 20% NLV` → 20-cascade: trim to **15% NLV**",
      "  - `< 20% NLV` → 15-cascade: trim to **11.25% NLV**",
      "- **Reversible**: if RS reclaims the 8w MA, cascade disarms and position can be rebuilt back toward the core level",
      "- **Purpose**: shave exposure on the FIRST sign of relative-strength weakness; leave the trade with a smaller but still-participating stake",
    ].join("\n"),
  },
  {
    code: "sr8.2",
    description: "SR8 Quicksand Trim",
    oneLiner:
      "Second MO RS cascade fire. RS drifts further below the 8-week MA — trim to the QUICKSAND cascade target (10% NLV on the 20-cascade, 7.5% on the 15-cascade). Still reversible on RS reclaim.",
    mechanics: [
      "- **Trigger**: after SR8.1, RS continues to drift below its 8w MA — the 'quicksand' state on the MO RS engine",
      "- **Cascade target**:",
      "  - `20-cascade` → trim to **10% NLV**",
      "  - `15-cascade` → trim to **7.5% NLV**",
      "- **Reversible**: if RS reclaims the 8w MA, cascade disarms and position can be rebuilt",
      "- **Purpose**: cut exposure further as weakness confirms — but leave a toehold in case RS turns back",
    ].join("\n"),
  },
  {
    code: "sr8.3",
    description: "SR8 Dreadful Dead",
    oneLiner:
      "Final MO RS cascade fire. RS breaks below the 21-week MA — full exit, one-way. Terminates the campaign in `terminate` mode; awaits a fresh GREEN in `revert` mode.",
    mechanics: [
      "- **Trigger**: weekly RS breaks below its 21w MA (the deep-trend line)",
      "- **Action**: **full exit** — cascade target is **0% NLV**",
      "- **Not reversible**: this is the one-way signal. In `terminate` mode (default), the campaign is over. In `revert` mode, position closes and a fresh daily GREEN opens a new sub-entry.",
      "- **Named separately from SR8.1 / SR8.2** so realized outcomes can be analyzed independently — DD exits typically capture the full downside; the analytical question is whether the earlier QUICK/QS trims added or subtracted value.",
    ].join("\n"),
  },
  {
    code: "sr9",
    description: "Failed Breakout",
    oneLiner:
      "Protects existing positions through a failed breakout. Half on close below Day 1, remainder on close below Day 0. Intraday exit OK if breaking bad.",
    mechanics: [
      "- Applies to existing positions, not fresh entries. Typical setup: stock was bought on a trending slope-line break or earlier base; the stock then runs into a new breakout pattern; the breakout fails.",
      "- Day 0 = the breakout bar itself",
      "- Day 1 = the bar following the breakout",
      "- First trigger: close below Day 1's low → sell half",
      "- Second trigger: close below Day 0's low → sell remaining shares",
      "- Discretionary: if the stock is breaking down hard, exit intraday — no need to wait for the close",
      "- **Context**: A breakout failure usually means the stock is going down hard. The exit price may be above your initial entry — SR9 is profit protection during structural failure, not capital protection (SR1's job).",
    ].join("\n"),
  },
  {
    code: "sr10",
    description: "Earnings Exit",
    oneLiner: "Exit before earnings if the stock fails the Earnings Planner test.",
    mechanics: [
      "- Used in conjunction with the Earnings Planner tab",
      "- Run each pre-earnings position through the test",
      "- If it fails, exit before earnings rather than hold through",
    ].join("\n"),
  },
  {
    code: "sr11",
    description: "BE Stop Out (moved at +10%)",
    oneLiner:
      "Move stop to break-even when position is up 10%+ from entry. If price returns to BE, exit. Disengages once SR8 activates.",
    mechanics: [
      "- Trigger: stock appreciates 10%+ from first buy",
      "- Action: move stop to break-even",
      "- Philosophy: protect realized cushion from reverting into a loss",
      "- **Disengagement**: Once the position qualifies for SR8 (up 50%+ AND market leader), the BE stop is removed. The core is no longer BE-defended — it transitions to RS-defended via SR8's weekly MO RS triggers. If price later returns near entry, that fact alone does not trigger an exit; SR8's signals govern instead.",
      "- Tracked as a distinct exit reason — pending analysis on whether maintaining the original risk level would have been the better long-run choice",
    ].join("\n"),
  },
  {
    code: "sr12",
    description: "TQQQ Strategy Exit",
    oneLiner:
      "Exit TQQQ on 21 EMA violation. Distinct rule from SR7 because the TQQQ entry conditions are also distinct.",
    mechanics: [
      "- Entry context (separate buy rule, listed here for completeness): Nasdaq's low is above the 21 EMA for 3 consecutive days",
      "- Exit signal: 21 EMA violation on Nasdaq",
      "  - First close below 21 EMA → arm",
      "  - Trigger: 1% below that bar's low",
      "  - Fire: subsequent intraday low breaks 1% below trigger",
      "- Same violation structure as SR6/SR7",
    ].join("\n"),
  },
  {
    code: "sr13",
    description: "Change of Character",
    oneLiner:
      "Exit on structural shifts — catalyst-driven plunge, lower-low structure, MA break on volume, or scary gap down. Full exit including SR8 core.",
    mechanics: [
      "- Multiple ways to identify a character change:",
      "  - **Catalyst-driven plunge**: new headline causes a 25%+ drop in one day",
      "  - **Lower-low structure**: stock closes below a prior low, breaking the higher-lows pattern",
      "  - **Volume break**: plunge below the 50-day MA on elevated volume",
      "  - **Scary gap down**: unexplained or significant gap down at the open",
      "- **Action**: Full exit of the entire position, including any SR8 core. A true character change voids the SR8 premise (the stock is no longer a leader with strong fundamentals).",
      "- **Bar for triggering must be high**: Market-wide scares (Iran war, circuit breaker, generic selloff) are SR3 events, not SR13 events. SR13 requires a stock-specific structural break.",
    ].join("\n"),
  },
] as const;

export const SELL_RULE_LABELS: readonly string[] = SELL_RULES.map(
  (r) => `${r.code} ${r.description}`,
);


// ────────────────────────────────────────────────────────────────────
// Buy rule labels — single source of truth.
//
// Previously duplicated inline in 4 component files (log-buy,
// trade-manager, trade-journal, campaign-detail). Adding a new rule
// required editing all 4; missing one caused the dropdown to drift
// out of sync with the others. Hoisted here so `import` is the only
// touchpoint.
//
// Order matters — this drives dropdown display order. Sequence is
// major-family (br1..br13) with sub-numbering that groups related
// setups (br1.1..br1.8 = base breakouts; br3.x = reclaims; etc).
//
// br13.x are the MO RS Green entry pair added alongside the SR8
// cascade split (see migration/session notes 2026-07-14).
// ────────────────────────────────────────────────────────────────────
export const BUY_RULE_LABELS: readonly string[] = [
  "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
  "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot", "br1.8 High Tight Flag",
  "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
  "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", "br3.5 Reclaim 8e", "br3.6 Green Line Break",
  "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", "br4.5 PB 8e", "br4.6 VWAP",
  "br5.1 Undercut & Rally", "br5.2 Upside Reversal",
  "br6.1 Gapper", "br6.2 Continuation Gap Up",
  "br7.1 TQQQ Strategy", "br7.2 New High after Gentle PB", "br7.3 JL Century Mark",
  "br8.1 Daily STL Break", "br8.2 Weekly STL Break", "br8.3 Monthly STL Break",
  "br9.1 21e Strategy",
  "br10.1 Hedging with leverage product",
  "br11.1 Shorting",
  "br12.1 Option Play",
  "br13.1 MO RS Green — Initial Entry", "br13.2 MO RS Green — Reset Entry",
] as const;

// Rule Interaction Hierarchy — which rule governs when two could
// fire on the same position. Rendered as a structured table by
// SellRuleGlossary (not markdown).

export type RuleHierarchyEntry = {
  conflict: string;
  winner: string;
  reasoning: string;
};

export const RULE_HIERARCHY: readonly RuleHierarchyEntry[] = [
  {
    conflict: "SR1 vs SR9 (fresh breakout failing)",
    winner:
      "SR9 governs — half on Day 1 close, rest on Day 0 close; SR1 acts as backstop",
    reasoning: "SR9 is more nuanced for active breakout management",
  },
  {
    conflict: "SR2 vs SR8",
    winner: "SR2 trims ADDS only; SR8 core untouched",
    reasoning: "Core is RS-defended, not extension-defended",
  },
  {
    conflict: "SR3 vs SR8",
    winner: "SR3 reduces non-leader exposure first; SR8 cores exempt",
    reasoning: "Cores only retire on SR8 or SR13 signals",
  },
  {
    conflict: "SR11 vs SR8",
    winner: "SR11 disengages once SR8 activates",
    reasoning: "BE protection is for early-trade only",
  },
  {
    conflict: "SR13 vs SR8",
    winner: "SR13 wins — full exit including core",
    reasoning: "Character change voids SR8 premise",
  },
  {
    conflict: "SR7 vs SR8 (in core)",
    winner: "SR8 governs the core; SR7 governs ADDS",
    reasoning: "Each rule's domain is layered",
  },
] as const;
