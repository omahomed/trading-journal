// Canonical sell-rule taxonomy. Single source of truth for the
// dropdown options in log-sell, trade-journal, trade-manager, and
// import-trades, and for the in-app glossary rendered on the Log
// Sell page.
//
// DB values are stored as `${code} ${description}` (see
// SELL_RULE_LABELS below). oneLiner and mechanics are
// presentation-only fields rendered by SellRuleGlossary; the
// taxonomy migration uses only code+description.
//
// Retired codes (2026-08-07 cleanup — migration 063):
//   SR4  — Time Stop. SR3 (Portfolio Management) covers portfolio-time
//          trims. Slot is FREE for future assignment.
//   SR6  — 8e Momentum Trim. Doctrinally retired per the canonical
//          handoff (0-for-5 fire quality). Slot is FREE.
//   SR12 — TQQQ Strategy Exit (original meaning). Same 21 EMA violation
//          shape as SR7, just on the NDX index. Historical stamps
//          retagged to SR7 via migration 063. Slot RE-ASSIGNED to
//          Ratcheting Profit Floor (MCP) by migration 064.
//   SR14 — 0.75× ATR Stop. Collapsed into SR1 — broker-stop presence
//          is now a chip on the row, not a tier promotion. Historical
//          stamps retagged to SR1. Slot is FREE (candidate for SR11-R
//          / MCP when we build it).

// Family = the functional group a sell rule belongs to. Drives the
// colored stripe on SellRuleBadge and the group headers in
// SellRuleGlossary. Purely presentational — the DB stamp is still
// the code+description string.
//
// Groups (2026-08-07 taxonomy pass):
//   defense       Capital defense on a losing thesis
//   trend         Technical / structural violation of the trend
//   floor         Profit-lock / BE stop nudges
//   monster       Declared SR8 cushion management
//   discretionary Judgment trims driven by extension / macro
//   event         Non-price triggers (earnings, character change)
export type SellRuleFamily =
  | "defense" | "trend" | "floor" | "monster" | "discretionary" | "event";

export type SellRule = {
  code: string;
  description: string;
  /** One-sentence summary shown always (collapsed-card header detail). */
  oneLiner: string;
  /** Markdown body. May contain GFM tables. Optional. */
  mechanics?: string;
  /** Functional group — drives the badge stripe + glossary grouping. */
  family: SellRuleFamily;
};

/** Family metadata: display label + colored stripe. Colors reuse the
 *  existing SellRuleBadge palette so the stripe on a "defense" rule
 *  matches the sr1 red fill, "monster" matches sr8 emerald, etc. */
export const SELL_RULE_FAMILIES: readonly {
  key: SellRuleFamily;
  label: string;
  /** Solid color used for the badge stripe + glossary group divider. */
  color: string;
  /** One-liner shown under the group header in the glossary. */
  blurb: string;
}[] = [
  { key: "defense",       label: "Capital defense",  color: "#dc2626",
    blurb: "The thesis died fast — protect the initial risk." },
  { key: "trend",         label: "Trend break",      color: "#4f46e5",
    blurb: "Technical premise gave way — exit or trim." },
  { key: "floor",         label: "Profit floor",     color: "#d97706",
    blurb: "Lock in realized cushion as the position matures." },
  { key: "monster",       label: "Monster hold",     color: "#15803d",
    blurb: "SR8 declared — RS-based management of the core." },
  { key: "discretionary", label: "Discretionary",    color: "#7c3aed",
    blurb: "Judgment trims driven by extension or macro pressure." },
  { key: "event",         label: "Event",            color: "#0e7490",
    blurb: "Non-price triggers — earnings and character change." },
] as const;

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
    family: "defense",
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
    family: "discretionary",
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
    family: "discretionary",
  },
  // SR4 (Time Stop) — retired 2026-08-07. SR3 covers portfolio-time trims.
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
    family: "discretionary",
  },
  // SR6 (8e Momentum Trim) — retired 2026-08-07 (canonical handoff §2:
  // 0-for-5 fire quality; non-activating above ~6% ATR).
  {
    code: "sr7",
    description: "21e Violation",
    oneLiner:
      "Sell on 21 EMA violation. Action scales with cushion: <25% → full exit, 25–50% → trim 50%, >50% → trim to 15% NAV core.",
    mechanics: [
      "- Always armed — no activation requirement",
      "- Arm on first close below 21 EMA · Trigger = 1% below arming bar's low · Fire when subsequent intraday low breaks trigger · Disarm on close back above 21 EMA",
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
    family: "trend",
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
      "| 🟡 Quicksand | RS breaks below 13w MA (mid-trend) | Trim another 5% (10% → 5%) | YES — if RS reclaims, rebuild |",
      "| 🔴 Grateful Dead | RS breaks below 21w MA (blue) | Full exit | NO — one-way |",
    ].join("\n"),
    family: "monster",
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
    family: "monster",
  },
  {
    code: "sr8.2",
    description: "SR8 Quicksand Trim",
    oneLiner:
      "Second MO RS trim. RS breaks below the 13-week MA — trim to the QUICKSAND target (10% NLV on the 20-cascade, 7.5% on the 15-cascade). Still reversible on RS reclaim.",
    mechanics: [
      "- **Trigger**: weekly RS crosses below its 13w MA (the mid-trend line — Fibonacci 8/13/21 weekly stack). Intraweek live cross, NOT Friday close.",
      "- **Cascade target**:",
      "  - `20-cascade` → trim to **10% NLV**",
      "  - `15-cascade` → trim to **7.5% NLV**",
      "- **Reversible**: if RS reclaims the 13w MA, the state disarms and position can be rebuilt",
      "- **Purpose**: cut exposure further as weakness confirms into the mid-trend line — but leave a toehold in case RS turns back",
    ].join("\n"),
    family: "monster",
  },
  {
    code: "sr8.3",
    description: "SR8 Grateful Dead",
    oneLiner:
      "Final MO RS cascade fire. RS breaks below the 21-week MA — full exit, one-way. Terminates the campaign in `terminate` mode; awaits a fresh GREEN in `revert` mode.",
    mechanics: [
      "- **Trigger**: weekly RS breaks below its 21w MA (the deep-trend line)",
      "- **Action**: **full exit** — cascade target is **0% NLV**",
      "- **Not reversible**: this is the one-way signal. In `terminate` mode (default), the campaign is over. In `revert` mode, position closes and a fresh daily GREEN opens a new sub-entry.",
      "- **Named separately from SR8.1 / SR8.2** so realized outcomes can be analyzed independently — GD exits typically capture the full downside; the analytical question is whether the earlier QUICK/QS trims added or subtracted value.",
    ].join("\n"),
    family: "monster",
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
    family: "trend",
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
    family: "event",
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
    family: "floor",
  },
  // SR12 — Ratcheting Profit Floor (MCP). Slot re-assigned by
  // migration 064; the original TQQQ Strategy Exit was collapsed into
  // SR7. Orthogonal to the tier ladder — a position can be SR7-armed
  // and SR12-armed simultaneously. Rendered as a floor chip + amber
  // nudge banner (not a badge stripe promotion).
  {
    code: "sr12",
    description: "Ratcheting Profit Floor",
    oneLiner:
      "MCP disaster backstop. Once peak b1_return crosses +50%, track peak total P&L and park a stop that locks in half of it. Intraday break of the floor = mechanical exit.",
    mechanics: [
      "- **Arm**: peak b1_return crosses +50% → the campaign is cushion-qualified and SR12 becomes active.",
      "- **Anchor**: peak_total_pl = the maximum of (realized_bank + shares × (day_high − avg_cost)) ever observed since B1. Backfilled by scripts/backfill_peak_total_pl.py; ratcheted daily by the b1_reconcile loop.",
      "- **Target price**: avg_entry + (peak_total_pl / 2 − realized_bank) / shares. Firing at this price realizes exactly half of the peak total P&L across the whole campaign (aggregate-anchored, not B1-anchored — corrects the migration-064 bug that misfired on scaled-in positions).",
      "- **Nudge**: amber banner + ⚓ chip on the ACS row when broker_stop_price < target. Clears when you park the stop at/above target. Also auto-clears when realized_bank already exceeds peak_total_pl / 2 (nothing left to protect).",
      "- **Fire**: intraday break of the parked stop = full exit. Same execution shape as SR1 / SR15.",
      "- **Handoff with SR15**: SR15's nudge is band-restricted [20%, 50%). SR12 takes over from 50% up. No overlap.",
      "- **Orthogonal to SR7 / SR8**: a declared SR8 monster hold is *also* SR12-armed. The floor is the disaster backstop for gap-down mornings that beat the trend exits to a worse price. On an orderly SR7 break, SR7 exits first and this never fires.",
      "- **Doctrine**: past +50%, never give back more than half of what you've earned across the whole campaign. Named after LifeCycle Trade's MCP (Mental Capital Preservation).",
    ].join("\n"),
    family: "floor",
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
    family: "event",
  },
  // SR15 — +10% Profit Lock. Automatic tier promotion when peak
  // crosses +20% (migration 062 nudge system). User-selectable here
  // for cases where the sell reason IS the +10% broker stop firing.
  {
    code: "sr15",
    description: "+10% Profit Lock",
    oneLiner:
      "Physical broker stop parked at entry × 1.10, activated once peak crosses +20% from B1. Firing = full exit like SR1 but at a locked +10% profit floor instead of a loss.",
    mechanics: [
      "- **Trigger**: peak b1_return crosses +20% → app nudges you to park a broker stop at entry × 1.10 (B1 fill, not blended avg)",
      "- **Anchor**: B1 fill × 1.10 — scale-ins do NOT walk the target higher",
      "- **Fire**: intraday break of the parked stop → full exit (same shape as SR1)",
      "- **Auto-clears**: once broker_stop_price ≥ entry × 1.10 in the app, the SR15 nudge banner clears (ACS + Risk Manager)",
      "- **Sticks through SR7 / SR8**: the +10% floor stays parked as the hard-price backstop even after tier ratchets higher. 21 EMA / weekly MO RS become the softer trailing tests above it.",
      "- **Tier band**: 20% ≤ peak < 50%. Nudge banner only fires in-band; once peak crosses 50% the floor is expected to already be parked and nagging is stale.",
    ].join("\n"),
    family: "floor",
  },
  // SR14 (0.75× ATR Stop) — retired 2026-08-07. Collapsed into SR1
  // (Capital Protection). Broker-stop presence is now surfaced as a
  // chip on the ACS row rather than a tier promotion — the physical
  // stop mechanism didn't need its own distinct sell-rule tag once
  // the 0.75× ATR backtest premise was validated. Historical sells
  // retagged to SR1 via migration 063.
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
